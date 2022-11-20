import warnings
from copy import deepcopy
from typing import Union
import ffmpeg
import numpy as np
import torch


def load_audio_waveform_img(audio: Union[str, bytes, np.ndarray, torch.Tensor],
                            h: int, w: int, ignore_shift=False) -> np.ndarray:
    """

    Parameters
    ----------
    audio: Union[str, bytes, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or bytes of audio file or a NumPy array or Tensor containing the audio waveform in 16 kHz
    h: int
        Height of waveform image
    w: int
        Width of waveform image
    ignore_shift: bool
        ignore warning if NumpPy array or PyTorch Tensor is used for audio

    Returns
    -------
    Audio waveform image as a NumPy array, in uint8 dtype.
    """

    try:
        if isinstance(audio, str):
            stream = ffmpeg.input(audio, threads=0)
            inp = None

        else:
            if isinstance(audio, bytes):
                stream = ffmpeg.input('pipe:', threads=0)
                inp = audio
            else:
                if not ignore_shift:
                    warnings.warn('A resampled input causes an unexplained temporal shift in waveform image '
                                  'that will skew the timestamp suppression and may result in inaccurate timestamps.\n'
                                  'Use audio_for_mask for transcribe() to provide the original audio track '
                                  'as the path or bytes of the audio file.',
                                  stacklevel=2)
                stream = ffmpeg.input('pipe:', threads=0, ac=1, format='s16le')
                if isinstance(audio, torch.Tensor):
                    audio = np.array(audio)
                inp = (audio * 32768.0).astype(np.int16).tobytes()

        waveform, err = (
            stream.filter('aformat', channel_layouts='mono')
            .filter('highpass', f='200').filter('lowpass', f='3000')
            .filter('showwavespic', s=f'{w}x{h}')
            .output('-', pix_fmt='gray', format='rawvideo', vframes=1)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio in waveform: {e.stderr.decode()}") from e
    else:
        if not waveform:
            partial_file = b'partial file' in err and b'Output file is empty' in err
            add_msg = '\nMetadata for decoding are likely at end of file, try to use path of audio instead.' \
                if partial_file and isinstance(audio, bytes) else ''
            raise RuntimeError(f"Failed to load audio in waveform: {err.decode()}" + add_msg)
        return np.frombuffer(waveform, dtype=np.uint8).reshape(h, w)


def remove_lower_quantile(waveform_img: np.ndarray,
                          upper_quantile: float = None,
                          lower_quantile: float = None,
                          lower_threshold: float = None) -> np.ndarray:
    """
    Removes lower quantile of amplitude from waveform image
    """
    if upper_quantile is None:
        upper_quantile = 0.85
    if lower_quantile is None:
        lower_quantile = 0.15
    if lower_threshold is None:
        lower_threshold = 0.15
    waveform_img = deepcopy(waveform_img)
    wave_sums = waveform_img.sum(0)
    mx = np.quantile(wave_sums, upper_quantile, -1)
    mn = np.quantile(wave_sums, lower_quantile, -1)
    mn_threshold = (mx - mn) * lower_threshold + mn
    waveform_img[:, wave_sums < mn_threshold] = 0
    return waveform_img


def wave_to_ts_filter(waveform_img: np.ndarray, suppress_middle=True,
                      max_index: (list, int) = None) -> np.ndarray:
    """
    Returns A NumPy array mask of sections with amplitude zero
    """
    assert waveform_img.ndim <= 2, f'waveform have at most 2 dims but found {waveform_img.ndim}'
    if waveform_img.ndim == 1:
        wave_sum = waveform_img
    else:
        wave_sum = waveform_img.sum(-2)

    wave_filter = wave_sum.astype(bool)

    if not suppress_middle:
        nonzero_indices = wave_filter.nonzero()[0]
        wave_filter[nonzero_indices[0]:nonzero_indices[-1] + 1] = True
    if max_index is not None:
        wave_filter[max_index + 1:] = False

    return ~wave_filter
