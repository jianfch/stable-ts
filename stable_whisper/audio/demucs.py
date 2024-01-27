import random
from typing import Union, Optional

import torch

from ..default import cached_model_instances


def is_demucs_available():
    from importlib.util import find_spec
    if find_spec('demucs') is None:
        raise ModuleNotFoundError("Please install Demucs; "
                                  "'pip install -U demucs' or "
                                  "'pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs'; "
                                  "Official Demucs repo: https://github.com/facebookresearch/demucs")


def load_demucs_model(cache: bool = True):
    model_name = 'htdemucs'
    _model_cache = cached_model_instances['demucs'] if cache else None
    if _model_cache is not None and _model_cache[model_name] is not None:
        return _model_cache[model_name]
    is_demucs_available()
    from demucs.pretrained import get_model_from_args
    model = get_model_from_args(type('args', (object,), dict(name=model_name, repo=None))).cpu().eval().models[0]
    if _model_cache is not None:
        _model_cache[model_name] = model
    return model


def apply_demucs_model(
        model,
        mix,
        shifts=0,
        split=True,
        overlap=0.25,
        transition_power=1.,
        progress=False,
        device=None,
        num_workers=0,
        pool=None
):
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
        mix = mix.to(device)
    if pool is None:
        if num_workers > 0 and device.type == 'cpu':
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(num_workers)
        else:
            from demucs.utils import DummyPoolExecutor
            pool = DummyPoolExecutor()

    from demucs.apply import TensorChunk, tensor_chunk
    from demucs.utils import center_trim

    model = model.to(device)
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."

    max_shift = int(0.5 * model.samplerate)

    def _inference(_mix):
        _length = _mix.shape[-1]
        valid_length = model.valid_length(_length) if hasattr(model, 'valid_length') else _length
        padded_mix = tensor_chunk(_mix).padded(valid_length).to(device)
        with torch.no_grad():
            out = model(padded_mix)
        return center_trim(out, _length)

    def _split(_mix):
        batch, channels, length = _mix.shape
        out = torch.zeros(batch, len(model.sources), channels, length, device=device)
        sum_weight = torch.zeros(length, device=device)
        segment = int(model.samplerate * model.segment)
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = torch.cat(
            [torch.arange(1, segment // 2 + 1, device=device),
             torch.arange(segment - segment // 2, 0, -1, device=device)]
        )
        assert len(weight) == segment
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max()) ** transition_power
        futures = []
        for offset in offsets:
            chunk = TensorChunk(_mix, offset, segment)
            future = pool.submit(_inference, chunk)
            futures.append((future, offset))
            offset += segment
        samples_per_future = length / len(futures)
        for future, offset in futures:
            chunk_out = future.result()
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += (weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment] += weight[:chunk_length].to(mix.device)
            update_pbar(samples_per_future)
        assert sum_weight.min() > 0
        out /= sum_weight
        return out

    orig_length = mix.shape[-1]
    if progress:
        import tqdm
        total_duration = round(orig_length / model.samplerate, 2)
        pbar = tqdm.tqdm(total=total_duration, unit='sec', desc='Demucs')

        def update_pbar(samples):
            if samples is None:
                pbar.update(pbar.total - pbar.n)
                return
            if shifts > 1:
                samples /= shifts

            seek_duration = min(round(pbar.n + (samples / model.samplerate), 2), total_duration)
            pbar.update(seek_duration - pbar.n)  # this keeps ``n`` rounded

    else:
        def update_pbar(samples):
            pass

    inference = _split if split else _inference

    if not shifts:
        output = inference(mix)
    else:
        output = 0
        mix = tensor_chunk(mix).padded(orig_length + 2 * max_shift)
        for _ in range(shifts):
            shift_offset = random.randint(0, max_shift)
            shifted = TensorChunk(mix, shift_offset, orig_length + max_shift - shift_offset)
            shifted_out = inference(shifted)
            output += shifted_out[..., max_shift - shift_offset:]
        output /= shifts

    update_pbar(None)

    return output[0, model.sources.index('vocals')].mean(0)


def demucs_audio(
        audio: Union[torch.Tensor, str, bytes],
        input_sr: int = None,
        output_sr: int = None,
        model=None,
        device=None,
        verbose: bool = True,
        save_path: Optional[Union[str, callable]] = None,
        seed: Optional[int] = 1,
        **demucs_options
) -> torch.Tensor:
    """
    Isolates vocals / remove noise from ``audio`` with Demucs.

    Official repo, https://github.com/facebookresearch/demucs.
    """
    if model is None:
        model = load_demucs_model()

    if isinstance(audio, (str, bytes)):
        from .utils import load_audio
        audio = torch.from_numpy(load_audio(audio, model.samplerate))
    elif input_sr != model.samplerate:
        if input_sr is None:
            raise ValueError('No ``input_sr`` specified for audio tensor.')
        from ..audio.utils import resample
        audio = resample(audio, input_sr, model.samplerate)
    audio_dims = audio.dim()
    assert audio_dims <= 3
    if dims_missing := 3 - audio_dims:
        audio = audio[[None]*dims_missing]
    if audio.shape[-2] == 1:
        audio = audio.repeat_interleave(2, -2)

    if 'mix' in demucs_options:
        audio = demucs_options.pop('mix')

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    apply_kwarg = dict(
        model=model,
        mix=audio,
        device=device,
        split=True,
        overlap=.25,
        progress=verbose is not None,
    )
    apply_kwarg.update(demucs_options)
    if seed is not None:
        random.seed(seed)
    vocals = apply_demucs_model(**apply_kwarg)

    if device != 'cpu':
        torch.cuda.empty_cache()

    if output_sr is not None and model.samplerate != output_sr:
        from ..audio.utils import resample
        vocals = resample(vocals, model.samplerate, output_sr)

    if save_path is not None:
        if isinstance(save_path, str):
            from .output import save_audio_tensor
            save_audio_tensor(vocals, save_path, output_sr or model.samplerate, verbose=verbose)
        else:
            save_path(vocals)

    return vocals
