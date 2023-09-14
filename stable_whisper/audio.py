import subprocess
import warnings
import ffmpeg
import torch
import torchaudio
import numpy as np
from typing import Union

from whisper.audio import SAMPLE_RATE


def is_ytdlp_available():
    return subprocess.run('yt-dlp -h', shell=True, capture_output=True).returncode == 0


def _load_file(file: Union[str, bytes], verbose: bool = False, only_ffmpeg: bool = False):
    if isinstance(file, str) and '://' in file:
        if is_ytdlp_available():
            verbosity = ' -q' if verbose is None else (' --progress' if verbose else ' --progress -q')
            p = subprocess.run(
                f'yt-dlp "{file}" -f ba/w -I 1{verbosity} -o -',
                shell=True,
                stdout=subprocess.PIPE
            )
            if len(p.stdout) == 0:
                raise RuntimeError(f'Failed to download media from "{file}" with yt-dlp')
            return p.stdout
        else:
            warnings.warn('URL detected but yt-dlp not available. '
                          'To handle a greater variety of URLs (i.e. non-direct links), '
                          'install yt-dlp, \'pip install yt-dlp\' (repo: https://github.com/yt-dlp/yt-dlp).')
        if not only_ffmpeg:
            if is_ytdlp_available():
                verbosity = ' -q' if verbose is None else (' --progress' if verbose else ' --progress -q')
                p = subprocess.run(
                    f'yt-dlp "{file}" -f ba/w -I 1{verbosity} -o -',
                    shell=True,
                    stdout=subprocess.PIPE
                )
                if p.returncode != 0 or len(p.stdout) == 0:
                    raise RuntimeError(f'Failed to download media from "{file}" with yt-dlp')
                return p.stdout
            else:
                warnings.warn('URL detected but yt-dlp not available. '
                              'To handle a greater variety of URLs (i.e. non-direct links), '
                              'install yt-dlp, \'pip install yt-dlp\' (repo: https://github.com/yt-dlp/yt-dlp).')
    return file


# modified version of whisper.audio.load_audio
def load_audio(file: Union[str, bytes], sr: int = SAMPLE_RATE, verbose: bool = True, only_ffmpeg: bool = False):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open, bytes of file, or URL to audio/video

    sr: int
        The sample rate to resample the audio if necessary

    verbose: bool
        whether to print yt-dlp log

    only_ffmpeg: bool
        Whether to use only FFmpeg (and not yt-dlp) for URls. (Default: False)

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    file = _load_file(file, verbose=verbose, only_ffmpeg=only_ffmpeg)
    if isinstance(file, bytes):
        inp, file = file, 'pipe:'
    else:
        inp = None
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


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
                 save_path: str = None,
                 **demucs_options) -> torch.Tensor:
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

    if isinstance(audio, (str, bytes)):
        if isinstance(audio, str) and not track_name:
            track_name = f'"{audio}"'
        audio = torch.from_numpy(load_audio(audio, model.samplerate))
    elif input_sr != model.samplerate:
        if input_sr is None:
            raise ValueError('No [input_sr] specified for audio tensor.')
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

    if 'mix' in demucs_options:
        audio = demucs_options.pop('mix')

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vocals_idx = model.sources.index('vocals')
    if verbose:
        print(f'Isolating vocals from {track_name}')
    apply_kwarg = dict(
        model=model,
        mix=audio,
        device=device,
        split=True,
        overlap=.25,
        progress=verbose is not None,
    )
    apply_kwarg.update(demucs_options)
    vocals = apply_model(
        **apply_kwarg
    )[0, vocals_idx].mean(0)

    if device != 'cpu':
        torch.cuda.empty_cache()

    if output_sr is not None and model.samplerate != output_sr:
        vocals = torchaudio.functional.resample(vocals,
                                                orig_freq=model.samplerate,
                                                new_freq=output_sr,
                                                resampling_method="kaiser_window")

    if save_path is not None:
        if isinstance(save_path, str) and not save_path.lower().endswith('.wav'):
            save_path += '.wav'
        torchaudio.save(save_path, vocals[None], output_sr or model.samplerate)
        print(f'Saved: {save_path}')

    return vocals


def get_samplerate(audiofile: (str, bytes)) -> (int, None):
    import re
    if isinstance(audiofile, str):
        metadata = subprocess.run(f'ffmpeg -i {audiofile}', capture_output=True, shell=True).stderr.decode()
    else:
        p = subprocess.Popen(f'ffmpeg -i -',  stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
        try:
            p.stdin.write(audiofile)
        except BrokenPipeError:
            pass
        finally:
            metadata = p.communicate()[-1]
            if metadata is not None:
                metadata = metadata.decode()
    sr = re.findall(r'\n.+Stream.+Audio.+\D+(\d+) Hz', metadata)
    if sr:
        return int(sr[0])
