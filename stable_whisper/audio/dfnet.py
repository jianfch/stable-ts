from typing import Union, Optional

import torch

from .utils import load_audio
from ..audio.utils import resample
from ..default import cached_model_instances


def is_dfnet_available():
    from importlib.util import find_spec
    if find_spec('df') is None:
        raise ModuleNotFoundError("Please install DeepFilterNet; "
                                  "'pip install -U deepfilternet' or "
                                  "follow installation instructions at https://github.com/Rikorose/DeepFilterNet")


def load_dfnet_model(cache: bool = True, **kwargs):
    model_name = 'dfnet'
    _model_cache = cached_model_instances['dfnet'] if cache else None
    if _model_cache is not None and _model_cache[model_name] is not None:
        return _model_cache[model_name]
    is_dfnet_available()
    from types import MethodType
    from df.enhance import init_df, enhance
    model, df_state, _ = init_df(**kwargs)
    model.df_state = df_state

    def enhance_wrapper(_model, audio, **enhance_kwargs):
        return enhance(model=_model, df_state=_model.df_state, audio=audio, **enhance_kwargs)

    model.enhance = MethodType(enhance_wrapper, model)
    model.samplerate = df_state.sr()
    if _model_cache is not None:
        _model_cache[model_name] = model
    return model


def dfnet_audio(
        audio: Union[torch.Tensor, str, bytes],
        input_sr: int = None,
        output_sr: int = None,
        model=None,
        device=None,
        verbose: bool = True,
        save_path: Optional[Union[str, callable]] = None,
        **dfnet_options
) -> torch.Tensor:
    """
    Remove noise from ``audio`` with DeepFilterNet.

    Official repo: https://github.com/Rikorose/DeepFilterNet.
    """
    if model is None:
        model = load_dfnet_model()
    if isinstance(audio, (str, bytes)):
        audio = torch.from_numpy(load_audio(audio, model.samplerate, mono=False))
    elif input_sr != model.samplerate:
        if input_sr is None:
            raise ValueError('No ``input_sr`` specified for audio tensor.')
        audio = resample(audio, input_sr, model.samplerate)
    audio_dims = audio.dim()
    assert audio_dims <= 2
    if dims_missing := 2 - audio_dims:
        audio = audio[[None]*dims_missing]
    if audio.shape[-2] == 1:
        audio = audio.repeat_interleave(2, -2)

    if device is not None:
        from df import config
        device_str = str(device)
        config.set('DEVICE', device_str, str, section='train')
        audio.to(device=device)
        model.to(device=device)

    dfnet_options.pop('progress', None)  # not implemented
    denoised_audio = model.enhance(audio=audio, **dfnet_options).mean(dim=0)

    if 'cuda' in str(device):
        torch.cuda.empty_cache()

    if output_sr is not None and model.samplerate != output_sr:
        denoised_audio = resample(denoised_audio, model.samplerate, output_sr)

    if save_path is not None:
        if isinstance(save_path, str):
            from .output import save_audio_tensor
            save_audio_tensor(denoised_audio, save_path, output_sr or model.samplerate, verbose=verbose)
        else:
            save_path(denoised_audio)

    return denoised_audio
