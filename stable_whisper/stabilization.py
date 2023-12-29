import warnings
from typing import List, Union, Tuple, Optional
from itertools import chain

import torch
import torch.nn.functional as F
import numpy as np

from whisper.audio import TOKENS_PER_SECOND, SAMPLE_RATE, N_SAMPLES_PER_TOKEN


NONVAD_SAMPLE_RATES = (16000,)
VAD_SAMPLE_RATES = (16000, 8000)


def is_ascending_sequence(
        seq: List[Union[int, float]],
        verbose=True
) -> bool:
    """
    check if a sequence of numbers are in ascending order
    """
    is_ascending = True
    for idx, (i, j) in enumerate(zip(seq[:-1], seq[1:])):
        if i > j:
            is_ascending = False
            if verbose:
                print(f'[Index{idx}]:{i} > [Index{idx + 1}]:{j}')
            else:
                break

    return is_ascending


def valid_ts(
        ts: List[dict],
        warn=True
) -> bool:
    valid = is_ascending_sequence(list(chain.from_iterable([s['start'], s['end']] for s in ts)), False)
    if warn and not valid:
        warnings.warn(message='Found timestamp(s) jumping backwards in time. '
                              'Use word_timestamps=True to avoid the issue.')
    return valid


def mask2timing(
        silence_mask: (np.ndarray, torch.Tensor),
        time_offset: float = 0.0,
) -> (Tuple[np.ndarray, np.ndarray], None):
    if silence_mask is None or not silence_mask.any():
        return
    assert silence_mask.ndim == 1
    if isinstance(silence_mask, torch.Tensor):
        silences = silence_mask.cpu().numpy().copy()
    elif isinstance(silence_mask, np.ndarray):
        silences = silence_mask.copy()
    else:
        raise NotImplementedError(f'Expected torch.Tensor or numpy.ndarray, but got {type(silence_mask)}')
    silences[0] = False
    silences[-1] = False
    silent_starts = np.logical_and(~silences[:-1], silences[1:]).nonzero()[0] / TOKENS_PER_SECOND
    silent_ends = (np.logical_and(silences[:-1], ~silences[1:]).nonzero()[0] + 1) / TOKENS_PER_SECOND
    if time_offset:
        silent_starts += time_offset
        silent_ends += time_offset
    return silent_starts, silent_ends


def timing2mask(
        silent_starts: np.ndarray,
        silent_ends: np.ndarray,
        size: int,
        time_offset: float = None
) -> torch.Tensor:
    assert len(silent_starts) == len(silent_ends)
    ts_token_mask = torch.zeros(size, dtype=torch.bool)
    if time_offset:
        silent_starts = (silent_starts - time_offset).clip(min=0)
        silent_ends = (silent_ends - time_offset).clip(min=0)
    mask_i = (silent_starts * TOKENS_PER_SECOND).round().astype(np.int16)
    mask_e = (silent_ends * TOKENS_PER_SECOND).round().astype(np.int16)
    for mi, me in zip(mask_i, mask_e):
        ts_token_mask[mi:me+1] = True

    return ts_token_mask


def suppress_silence(
        result_obj,
        silent_starts: np.ndarray,
        silent_ends: np.ndarray,
        min_word_dur: float,
        nonspeech_error: float = 0.3
):
    assert len(silent_starts) == len(silent_ends)
    if len(silent_starts) == 0 or (result_obj.end - result_obj.start) <= min_word_dur:
        return

    start_overlaps = np.all(
        (silent_starts <= result_obj.start, result_obj.start < silent_ends, silent_ends <= result_obj.end),
        axis=0
    ).nonzero()[0].tolist()
    if start_overlaps:
        new_start = silent_ends[start_overlaps[0]]
        result_obj.start = min(new_start, round(result_obj.end - min_word_dur, 3))
        if (result_obj.end - result_obj.start) <= min_word_dur:
            return

    end_overlaps = np.all(
        (result_obj.start <= silent_starts, silent_starts < result_obj.end, result_obj.end <= silent_ends),
        axis=0
    ).nonzero()[0].tolist()
    if end_overlaps:
        new_end = silent_starts[end_overlaps[0]]
        result_obj.end = max(new_end, round(result_obj.start + min_word_dur, 3))
        if (result_obj.end - result_obj.start) <= min_word_dur:
            return

    if nonspeech_error:
        matches = np.logical_and(
            result_obj.start <= silent_starts,
            result_obj.end >= silent_ends,
        ).nonzero()[0].tolist()
        if len(matches) == 0:
            return
        silence_start = np.min(silent_starts[matches])
        silence_end = np.max(silent_ends[matches])
        start_extra = silence_start - result_obj.start
        end_extra = result_obj.end - silence_end
        silent_duration = silence_end - silence_start
        if start_extra <= end_extra:
            if (start_extra / silent_duration) <= nonspeech_error:
                result_obj.start = min(silence_end, round(result_obj.end - min_word_dur, 3))
        else:
            if (end_extra / silent_duration) <= nonspeech_error:
                result_obj.end = max(silence_start, round(result_obj.start + min_word_dur, 3))


def standardize_audio(
        audio: Union[torch.Tensor, np.ndarray, str, bytes],
        resample_sr: Tuple[Optional[int], Union[int, Tuple[int]]] = None
) -> torch.Tensor:
    if isinstance(audio, (str, bytes)):
        from .audio import load_audio
        audio = load_audio(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.float()
    if resample_sr:
        in_sr, out_sr = resample_sr
        if in_sr:
            if isinstance(out_sr, int):
                out_sr = [out_sr]
            if in_sr not in out_sr:
                from torchaudio.functional import resample
                audio = resample(audio, in_sr, out_sr[0], resampling_method="kaiser_window")

    return audio


def audio2loudness(
        audio_tensor: torch.Tensor
) -> (torch.Tensor, None):
    assert audio_tensor.dim() == 1, f'waveform must be 1D, but got {audio_tensor.dim()}D'
    audio_tensor = audio_tensor.abs()
    k = int(audio_tensor.numel() * 0.001)
    if k:
        top_values, _ = torch.topk(audio_tensor, k)
        threshold = top_values[-1]
    else:
        threshold = audio_tensor.quantile(0.999, dim=-1)
    if (token_count := round(audio_tensor.shape[-1] / N_SAMPLES_PER_TOKEN)+1) > 2:
        if threshold < 1e-5:
            return torch.zeros(token_count, dtype=audio_tensor.dtype, device=audio_tensor.device)
        audio_tensor = audio_tensor / min(1., threshold * 1.75)
        audio_tensor = F.interpolate(
            audio_tensor[None, None],
            size=token_count,
            mode='linear',
            align_corners=False
        )[0, 0]
        return audio_tensor


def visualize_mask(
        loudness_tensor: torch.Tensor,
        silence_mask: torch.Tensor = None,
        width: int = 1500,
        height: int = 200,
        output: str = None,
):
    no_silence = silence_mask is None or not silence_mask.any()
    assert no_silence or silence_mask.shape[0] == loudness_tensor.shape[0]
    if loudness_tensor.shape[0] < 2:
        raise NotImplementedError(f'audio size, {loudness_tensor.shape[0]}, is too short to visualize')
    else:
        width = loudness_tensor.shape[0] if width == -1 else width
        im = torch.zeros((height, width, 3), dtype=torch.uint8)
        mid = round(height / 2)
        for i, j in enumerate(loudness_tensor.tolist()):
            j = round(abs(j) * mid)
            if j == 0 or width <= i:
                continue
            im[mid - j:mid + 1, i] = 255
            im[mid + 1:mid + j + 1, i] = 255
        if not no_silence:
            im[:, silence_mask[:width], 1:] = 0
        im = im.cpu().numpy()
        if output and not output.endswith('.png'):
            output += '.png'
        try:
            from PIL import Image
        except ModuleNotFoundError:
            try:
                import cv2
            except ModuleNotFoundError:
                raise ModuleNotFoundError('Failed to import "PIL" or "cv2" to visualize suppression mask. '
                                          'Try "pip install Pillow" or "pip install opencv-python"')
            else:
                im = im[..., [2, 1, 0]]
                if isinstance(output, str):
                    cv2.imwrite(output, im)
                else:
                    cv2.imshow('image', im)
                    cv2.waitKey(0)
        else:
            im = Image.fromarray(im)
            if isinstance(output, str):
                im.save(output)
            else:
                im.show(im)
        if output:
            print(f'Save: {output}')


def wav2mask(
        audio: (torch.Tensor, np.ndarray, str, bytes),
        q_levels: int = 20,
        k_size: int = 5,
        sr: int = None
) -> (Tuple[torch.Tensor, Tuple[np.ndarray, np.ndarray]], None):
    """
    Generate 1D mask from waveform for suppressing timestamp tokens.
    """
    audio = standardize_audio(audio, (sr, NONVAD_SAMPLE_RATES))
    loudness_tensor = audio2loudness(audio)
    if loudness_tensor is None:
        return
    p = k_size // 2 if k_size else 0
    if p and p < loudness_tensor.shape[-1]:
        assert k_size % 2, f'kernel_size must be odd but got {k_size}'
        mask = torch.avg_pool1d(
            F.pad(
                loudness_tensor[None],
                (p, p),
                'reflect'
            ),
            kernel_size=k_size,
            stride=1
        )[0]
    else:
        mask = loudness_tensor.clone()

    if q_levels:
        mask = mask.mul(q_levels).round()

    mask = mask.bool()

    if not mask.any():  # entirely silent
        return ~mask
    temp_timings = mask2timing(mask)
    s, e = temp_timings
    se_mask = (e - s) > 0.1
    s = s[se_mask]
    e = e[se_mask]
    mask = ~timing2mask(s, e, loudness_tensor.shape[-1])

    if not mask.any():  # no silence
        return

    return mask


_model_cache = {}


def get_vad_silence_func(
        onnx=False,
        verbose: (bool, None) = False
):
    if onnx in _model_cache:
        model, get_ts = _model_cache[onnx]
    else:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      verbose=verbose,
                                      onnx=onnx)
        get_ts = utils[0]
        _model_cache[onnx] = (model, get_ts)

    warnings.filterwarnings('ignore', message=r'operator \(\) profile_node.*', category=UserWarning)

    def get_speech_timestamps(wav: torch.Tensor, threshold: float = .35):
        return get_ts(wav, model, threshold, min_speech_duration_ms=100, min_silence_duration_ms=20)

    def vad_silence_timing(
            audio: (torch.Tensor, np.ndarray, str, bytes),
            speech_threshold: float = .35,
            sr: int = None
    ) -> (Tuple[np.ndarray, np.ndarray], None):

        audio = standardize_audio(audio, (sr, VAD_SAMPLE_RATES))

        total_duration = round(audio.shape[-1] / SAMPLE_RATE, 3)
        if not total_duration:
            return
        ori_t = torch.get_num_threads()
        if verbose is not None:
            print('Predicting silences(s) with VAD...\r', end='')
        torch.set_num_threads(1)  # vad was optimized for single performance
        speech_ts = get_speech_timestamps(audio, speech_threshold)
        if verbose is not None:
            print('Predicted silence(s) with VAD.       ')
        torch.set_num_threads(ori_t)
        if len(speech_ts) == 0:  # all silent
            return np.array([0.0]), np.array([total_duration])
        silent_starts = []
        silent_ends = []
        for ts in speech_ts:
            start = round(ts['start'] / SAMPLE_RATE, 3)
            end = round(ts['end'] / SAMPLE_RATE, 3)
            if start != 0:
                silent_ends.append(start)
                if len(silent_starts) == 0:
                    silent_starts.append(0.0)
            if end < total_duration:
                silent_starts.append(end)

        if len(silent_starts) == 0 and len(silent_ends) == 0:
            return

        if len(silent_starts) != 0 and (len(silent_ends) == 0 or silent_ends[-1] < silent_starts[-1]):
            silent_ends.append(total_duration)

        silent_starts = np.array(silent_starts)
        silent_ends = np.array(silent_ends)

        return silent_starts, silent_ends

    return vad_silence_timing


def visualize_suppression(
        audio: Union[torch.Tensor, np.ndarray, str, bytes],
        output: str = None,
        q_levels: int = 20,
        k_size: int = 5,
        vad_threshold: float = 0.35,
        vad: bool = False,
        max_width: int = 1500,
        height: int = 200
):
    """
    Visualize regions on the waveform of ``audio`` detected as silent.

    Regions on the waveform colored red are detected as silent.

    Parameters
    ----------
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
        If audio is ``numpy.ndarray`` or ``torch.Tensor``, the audio must be already at sampled to 16kHz.
    output : str, default None, meaning image will be shown directly via Pillow or opencv-python
        Path to save visualization.
    q_levels : int, default 20
        Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
        Acts as a threshold to marking sound as silent.
        Fewer levels will increase the threshold of volume at which to mark a sound as silent.
    k_size : int, default 5
        Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
        Recommend 5 or 3; higher sizes will reduce detection of silence.
    vad : bool, default False
        Whether to use Silero VAD to generate timestamp suppression mask.
        Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
    vad_threshold : float, default 0.35
        Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
    max_width : int, default 1500
        Maximum width of visualization to avoid overly large image from long audio.
        Each unit of pixel is equivalent  to 1 token.  Use -1 to visualize the entire audio track.
    height : int, default 200
        Height of visualization.
    """
    max_n_samples = None if max_width == -1 else round(max_width * N_SAMPLES_PER_TOKEN)

    audio = standardize_audio(audio)
    if max_n_samples is None:
        max_width = audio.shape[-1]
    else:
        audio = audio[:max_n_samples]
    loudness_tensor = audio2loudness(audio)
    width = min(max_width, loudness_tensor.shape[-1])
    if loudness_tensor is None:
        raise NotImplementedError(f'Audio is too short and cannot visualized.')

    if vad:
        silence_timings = get_vad_silence_func()(audio, vad_threshold)
        silence_mask = None if silence_timings is None else timing2mask(*silence_timings, size=loudness_tensor.shape[0])
    else:
        silence_mask = wav2mask(audio, q_levels=q_levels, k_size=k_size)

    visualize_mask(loudness_tensor, silence_mask, width=width, height=height, output=output)
