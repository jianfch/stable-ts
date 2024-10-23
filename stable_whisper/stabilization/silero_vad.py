import warnings
from typing import Optional, List

import torch
from tqdm import tqdm

from .utils import SetTorchThread
from ..default import cached_model_instances

VAD_SAMPLE_RATES = (16000, 8000)
VAD_WINDOWS = (256, 512, 768, 1024, 1536)


def load_silero_vad_model(onnx=False, verbose: Optional[bool] = False, cache: bool = True, **kwargs):
    model_cache = cached_model_instances['silero_vad'] if cache else None
    if model_cache is not None and model_cache[onnx] is not None:
        return model_cache[onnx]

    load_kwargs = dict(
        repo_or_dir='snakers4/silero-vad:master',
        model='silero_vad',
        verbose=verbose,
        onnx=onnx,
        trust_repo=True
    )
    load_kwargs.update(kwargs)
    model, utils = torch.hub.load(**load_kwargs)
    get_ts = utils[0]
    if model_cache is not None:
        model_cache[onnx] = (model, get_ts)
    warnings.filterwarnings('ignore', message=r'operator \(\) profile_node.*', category=UserWarning)

    return model, get_ts


def compute_vad_probs(
        model,
        audio: torch.Tensor,
        sampling_rate: int,
        window: int,
        progress: bool = True
) -> List[float]:
    duration = round(audio.shape[-1] / sampling_rate, 2)
    speech_probs = []
    with torch.no_grad(), SetTorchThread(1), tqdm(total=duration, unit='sec', desc='VAD', disable=not progress) as pbar:
        for current_start_sample in range(0, audio.shape[-1], window):
            chunk = audio[current_start_sample: current_start_sample + window]
            if len(chunk) < window:
                chunk = torch.nn.functional.pad(chunk, (0, int(window - len(chunk))))
            prob = model(chunk.cpu(), sampling_rate).item()
            speech_probs.append(prob)
            if not pbar.disable:
                seek_duration = min(
                    round((current_start_sample + window) / sampling_rate, 2),
                    duration
                )
                pbar.update(seek_duration - pbar.n)

    return speech_probs


def assert_sr_window(sr: int, window: int):
    assert sr in VAD_SAMPLE_RATES, f'{sr} not in {VAD_SAMPLE_RATES}'
    assert window in VAD_WINDOWS, f'{window} not in {VAD_WINDOWS}'


def onnx_param_update(vad: (bool, dict), vad_onnx: bool) -> (bool, dict):
    if vad_onnx:
        warnings.warn('``vad_onnx`` is deprecated and will be removed in future versions. '
                      'Use ``onnx=True`` in ``vad`` as a dict (e.g. ``vad=dict(onnx=True)``).',
                      stacklevel=2)
        if vad is not False:
            vad = vad.copy() if isinstance(vad, dict) else {}
            vad.update(onnx=vad_onnx)
    return vad
