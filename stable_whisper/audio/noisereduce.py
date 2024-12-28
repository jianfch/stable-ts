from typing import Union, Optional

import torch

from .utils import load_audio, SAMPLE_RATE
from ..audio.utils import resample


def is_noisereduce_available():
    from importlib.util import find_spec
    if find_spec('noisereduce') is None:
        raise ModuleNotFoundError("Please install noisereduce; "
                                  "'pip install -U noisereduce'"
                                  "Official repo: https://github.com/timsainb/noisereduce")


def load_noisereduce_model(cache: bool = True, **kwargs):
    is_noisereduce_available()
    from noisereduce import reduce_noise

    class NRWrapper:
        samplerate = SAMPLE_RATE

        def __call__(self, audio, **kwargs):
            options = dict(
                n_fft=512,
                stationary=False,
                use_torch=True,
                sr=self.samplerate
            )
            options.update(kwargs)
            return torch.from_numpy(reduce_noise(y=audio, **options))

    model = NRWrapper()
    return model


def noisereduce_audio(
    audio: Union[torch.Tensor, str, bytes],
    input_sr: int = None,
    output_sr: int = None,
    model=None,
    device: str = "cuda",
    verbose: bool = True,
    save_path: Optional[Union[str, callable]] = None,
    **noisereduce_options
) -> torch.Tensor:
    """
    Remove noise from ``audio`` with noisereduce.

    For list of options, see repo: https://github.com/timsainb/noisereduce.
    """
    if model is None:
        model = load_noisereduce_model()
    if noisereduce_options.get("sr", input_sr) != input_sr:
        raise ValueError('Specified conflicting ``input_sr`` and ``sr``.')
    if isinstance(audio, (str, bytes)):
        audio = torch.from_numpy(load_audio(audio, model.samplerate))
        input_sr = model.samplerate
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

    noisereduce_options.pop('progress', None)  # not implemented
    denoised_audio = model(audio=audio, device=device, **noisereduce_options).nan_to_num().mean(dim=0)

    if 'cuda' in str(device):
        torch.cuda.empty_cache()

    if output_sr is not None and input_sr != output_sr:
        denoised_audio = resample(denoised_audio, input_sr, output_sr)

    if save_path is not None:
        if isinstance(save_path, str):
            from .output import save_audio_tensor
            save_audio_tensor(denoised_audio, save_path, output_sr or input_sr, verbose=verbose)
        else:
            save_path(denoised_audio, output_sr)

    return denoised_audio
