import torch
import torchaudio
import numpy as np


def voice_freq_filter(wf: (torch.Tensor, np.ndarray), sr: int,
                      upper_freq: int = None,
                      lower_freq: int = None) -> torch.Tensor:
    if isinstance(wf, np.ndarray):
        wf = torch.from_numpy(wf)
    if upper_freq is None:
        upper_freq = 5000
    if lower_freq is None:
        lower_freq = 200
    assert upper_freq > lower_freq, f'upper_freq {upper_freq} must but greater than lower_freq {lower_freq}'
    return torchaudio.functional.highpass_biquad(torchaudio.functional.lowpass_biquad(wf, sr, upper_freq),
                                                 sr,
                                                 lower_freq)


def is_demucs_available():
    from importlib.util import find_spec
    if find_spec('demucs') is None:
        raise ModuleNotFoundError("Please install Demucs; "
                                  "'pip install -U demucs' or "
                                  "'pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs'; "
                                  "Official Demucs repo: https://github.com/facebookresearch/demucs")


def load_demucs_model():
    is_demucs_available()
    from demucs.pretrained import get_model_from_args
    return get_model_from_args(type('args', (object,), dict(name='htdemucs', repo=None))).cpu().eval()


def demucs_audio(audio: (torch.Tensor, str),
                 input_sr: int = None,
                 output_sr: int = None,
                 model=None,
                 device=None,
                 verbose: bool = True,
                 track_name: str = None,
                 save_path: str = None) -> torch.Tensor:
    """
    Load audio waveform and process to isolate vocals with Demucs
    """
    if model is None:
        model = load_demucs_model()
    else:
        is_demucs_available()
    from demucs.apply import apply_model

    if track_name:
        track_name = f'"{track_name}"'

    if isinstance(audio, str):
        from whisper.audio import load_audio
        if not track_name:
            track_name = f'"{audio}"'
        audio = torch.from_numpy(load_audio(audio, model.samplerate))
    elif input_sr != model.samplerate:
        if input_sr is None:
            raise ValueError('No samplerate provided for audio tensor.')
        audio = torchaudio.functional.resample(audio,
                                               orig_freq=input_sr,
                                               new_freq=model.samplerate,
                                               resampling_method="kaiser_window")
    if not track_name:
        track_name = 'audio track'
    audio_dims = audio.dim()
    if audio_dims == 1:
        audio = audio[None, None].repeat_interleave(2, -2)
    else:
        if audio.shape[-2] == 1:
            audio = audio.repeat_interleave(2, -2)
        if audio_dims < 3:
            audio = audio[None]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vocals_idx = model.sources.index('vocals')
    if verbose:
        print(f'Isolating vocals from {track_name}')
    vocals = apply_model(model, audio, device=device, split=True, overlap=.25, progress=verbose)[0, vocals_idx].mean(0)

    if device != 'cpu':
        torch.cuda.empty_cache()

    if output_sr is not None and model.samplerate != output_sr:
        vocals = torchaudio.functional.resample(vocals,
                                                orig_freq=model.samplerate,
                                                new_freq=output_sr,
                                                resampling_method="kaiser_window")

    if save_path:
        if not save_path.lower().endswith('.wav'):
            save_path += '.wav'
        torchaudio.save(save_path, vocals[None], output_sr or model.samplerate)
        print(f'Saved: {save_path}')

    return vocals
