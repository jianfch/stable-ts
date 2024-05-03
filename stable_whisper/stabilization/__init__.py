import warnings
from typing import List, Union, Tuple, Optional

import torch
import numpy as np

from .nonvad import NONVAD_SAMPLE_RATES, audio2loudness, wav2mask, visualize_mask
from .silero_vad import VAD_SAMPLE_RATES, load_silero_vad_model, compute_vad_probs, assert_sr_window
from .utils import is_ascending_sequence, valid_ts, mask2timing, timing2mask, filter_timings
from ..audio.utils import audio_to_tensor_resample
from ..default import get_min_word_dur

from ..whisper_compatibility import SAMPLE_RATE, FRAMES_PER_SECOND, N_SAMPLES_PER_TOKEN


class NonSpeechPredictor:
    def __init__(
            self,
            vad: Optional[bool] = False,
            mask_pad_func=None,
            get_mask: bool = False,
            min_word_dur: Optional[float] = None,
            q_levels: int = 20,
            k_size: int = 5,
            vad_threshold: float = 0.35,
            vad_onnx: bool = False,
            vad_window: int = None,
            sampling_rate: int = None,
            verbose: Optional[bool] = True,
            store_timings: bool = False,
            ignore_is_silent: bool = False,
            stream: bool = False,
            units_per_seconds: int = None,
            min_silence_dur: Optional[float] = None
    ):
        min_word_dur = get_min_word_dur(min_word_dur)
        self.min_silence_dur = min_silence_dur
        self.vad = vad
        self.mask_pad_func = mask_pad_func
        self.get_mask = get_mask
        self.q_levels = q_levels
        self.k_size = k_size
        self.vad_threshold = vad_threshold
        self.vad_onnx = vad_onnx
        self.verbose = verbose
        self.store_timings = store_timings
        self.ignore_is_silent = ignore_is_silent
        self._stream = stream
        self._nonspeech_timings = None
        vad_window = 512 if vad_window is None else vad_window
        self.vad_window = vad_window
        if sampling_rate is None:
            sampling_rate = SAMPLE_RATE
        self.sampling_rate = sampling_rate
        self.min_samples_per_word = round(min_word_dur * self.sampling_rate)
        if units_per_seconds is None:
            units_per_seconds = FRAMES_PER_SECOND
        self.min_frames_per_word = max(round(min_word_dur * units_per_seconds), 1)
        if self.vad:
            assert_sr_window(self.sampling_rate, self.vad_window)
        self.min_chunks_per_word = round(min_word_dur * self.sampling_rate / self.vad_window)
        self.second_per_prob = self.vad_window / self.sampling_rate
        self.vad_model = None
        self._prev_speech_probs = []
        self._default_probs = []
        self._using_callback = False
        self._load_vad_model()
        if self.vad is None:
            self._predict = self.predict_with_samples
        else:
            self._predict = self.predict_with_vad if self.vad else self.predict_with_nonvad

    @property
    def nonspeech_timings(self):
        return self._nonspeech_timings

    def predict(
            self,
            audio: torch.Tensor,
            offset: Optional[float] = None
    ) -> dict:
        pred = self._predict(audio, offset)
        if self.min_silence_dur:
            pred['timings'] = filter_timings(pred['timings'], self.min_silence_dur)
        return pred

    def _load_vad_model(self):
        if self.vad:
            self.vad_model = load_silero_vad_model()[0]
            self.reset()

    def reset(self):
        if self.vad_model is not None:
            self.vad_model.reset_states()
        self._prev_speech_probs = []

    def _silent_mask_test(self, mask, min_unit_per_word) -> bool:
        if self.ignore_is_silent or mask is None:
            return False
        non_silent_unit_count = mask.shape[-1] - np.flatnonzero(mask).shape[-1]
        return non_silent_unit_count < min_unit_per_word

    def _append_timings(self, timings: np.ndarray):
        if not self.store_timings or timings is None or not len(timings[0]):
            return
        starts, ends = timings.tolist()
        if not self._nonspeech_timings:
            self._nonspeech_timings = (starts, ends)
            return

        if starts:
            self._nonspeech_timings[0].extend(starts)
            self._nonspeech_timings[1].extend(ends)

    def finalize_timings(self):
        if self._nonspeech_timings is None:
            return

        def _finalize(starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if len(starts) <= 1:
                return starts, ends
            valid_starts = starts[1:] >= ends[:-1]
            if np.all(valid_starts):
                return starts, ends
            starts = starts[np.concatenate(([True], valid_starts))]
            ends = ends[np.concatenate((valid_starts, [True]))]
            return _finalize(starts, ends)

        s, e = self._nonspeech_timings
        s, e = np.array(s), np.array(e)
        s.sort()
        e.sort()
        s, e = _finalize(s, e)
        self._nonspeech_timings = s.tolist(), e.tolist()

    def pad_mask(self, mask):
        if mask is None:
            return
        if self.mask_pad_func is None:
            return mask
        return self.mask_pad_func(mask, 1501)

    def compute_vab_probs(
            self,
            audio: torch.Tensor
    ) -> List[float]:
        return compute_vad_probs(
            model=self.vad_model,
            audio=audio,
            sampling_rate=self.sampling_rate,
            window=self.vad_window,
            progress=self.verbose is not None
        )

    def _nonstream_prep_callback(self, prepped_audio: torch.Tensor, **kwargs):
        if self._default_probs:
            return
        self._default_probs = self.compute_vab_probs(prepped_audio)

    def _stream_prep_callback(self, prepped_audio: torch.Tensor, **kwargs):
        self._default_probs.extend(self.compute_vab_probs(prepped_audio))

    def get_on_prep_callback(self, stream: Optional[bool] = None):
        if not self.vad:
            return
        self._using_callback = True
        if stream is not None:
            self._stream = stream
        if self._stream:
            return self._stream_prep_callback
        return self._nonstream_prep_callback

    def _vad_probs(
            self,
            audio: torch.Tensor,
            offset: Union[float, None] = None
    ) -> Tuple[List[float], Union[float, None]]:
        if self._default_probs:
            assert offset is not None, 'offset is required for default probs'
            sample_offset = offset * self.sampling_rate
            s = np.floor(sample_offset / self.vad_window).astype(np.int32)
            e = np.ceil((sample_offset + audio.shape[-1]) / self.vad_window).astype(np.int32)
            new_offset = s * self.vad_window / self.sampling_rate
            return self._default_probs[s:e], new_offset

        assert not self._using_callback
        speech_probs = self.compute_vab_probs(audio)

        return speech_probs, offset

    def _vad_timings_mask(
            self,
            speech_probs: List[float],
            threshold: float,
            offset: float,
            min_start: float = None,
            max_end: float = None,
            get_mask: bool = False
    ) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], bool]:
        mask = None
        prob_mask = np.less(speech_probs, threshold)
        is_silent = self._silent_mask_test(prob_mask, self.min_chunks_per_word)
        timings = mask2timing(
            prob_mask,
            time_offset=offset,
            second_per_unit=self.second_per_prob,
            min_start=min_start,
            max_end=max_end
        )
        if timings is not None:
            if get_mask:
                mask = timing2mask(*timings, time_offset=offset, size=1501)
            timings = np.stack((timings[0], timings[1]), axis=0)
        return timings, mask, is_silent

    def predict_with_vad(
            self,
            audio: torch.Tensor,
            offset: Optional[float] = None
    ) -> dict:
        if offset is None:
            offset = 0
        max_end = round(offset + audio.shape[-1] / self.sampling_rate, 3)
        speech_probs, new_offset = self._vad_probs(audio, offset=offset)
        timings, mask, is_silent = self._vad_timings_mask(
            speech_probs,
            self.vad_threshold,
            new_offset,
            min_start=offset,
            max_end=max_end,
            get_mask=self.get_mask
        )

        self._append_timings(timings)
        self._prev_speech_probs = speech_probs

        return dict(timings=timings, mask=mask, is_silent=is_silent)

    def predict_with_nonvad(
            self,
            audio: torch.Tensor,
            offset: Optional[float] = None
    ) -> dict:
        mask = wav2mask(audio, q_levels=self.q_levels, k_size=self.k_size, sr=self.sampling_rate)
        timings = mask2timing(mask, time_offset=offset)
        if timings is not None:
            timings = np.stack(timings, axis=0)
        is_silent = self._silent_mask_test(mask, self.min_frames_per_word)
        if mask is not None:
            mask = self.pad_mask(mask)
        self._append_timings(timings)
        return dict(timings=timings, mask=mask, is_silent=is_silent)

    def predict_with_samples(
            self,
            audio: torch.Tensor,
            offset: Optional[float] = None
    ) -> dict:
        if self.get_mask:
            if extra_len := audio.shape[-1] % N_SAMPLES_PER_TOKEN:
                audio = torch.nn.functional.pad(audio, (0, N_SAMPLES_PER_TOKEN - extra_len))
            mask = torch.all(audio.reshape(-1, N_SAMPLES_PER_TOKEN), dim=-1)
            min_unit_per_word = self.min_frames_per_word
        else:
            mask = audio == 0
            min_unit_per_word = self.min_samples_per_word
        is_silent = self._silent_mask_test(mask, min_unit_per_word)
        return dict(timings=None, mask=self.pad_mask(mask) if self.get_mask else None, is_silent=is_silent)


def get_vad_silence_func(
        onnx=False,
        verbose: Optional[bool] = False
):
    predictor = NonSpeechPredictor(
        vad=True,
        vad_onnx=onnx,
        verbose=verbose
    )

    from ..audio import prep_audio

    def vad_silence_timing(
            audio: (torch.Tensor, np.ndarray, str, bytes),
            speech_threshold: float = .35,
            sr: int = None,
            time_offset: Optional[float] = None,
    ) -> (Tuple[np.ndarray, np.ndarray], None):
        predictor.sampling_rate = sr or predictor.sampling_rate
        predictor.vad_threshold = speech_threshold
        audio = prep_audio(audio)
        return predictor.predict_with_vad(audio=audio, offset=time_offset)['timings']

    return vad_silence_timing


def suppress_silence(
        result_obj,
        silent_starts: Union[np.ndarray, List[float]],
        silent_ends: Union[np.ndarray, List[float]],
        min_word_dur: float,
        nonspeech_error: float = 0.1,
        keep_end: Optional[bool] = True
):
    assert len(silent_starts) == len(silent_ends)
    if len(silent_starts) == 0 or (result_obj.end - result_obj.start) <= min_word_dur:
        return
    if isinstance(silent_starts, list):
        silent_starts = np.array(silent_starts)
    if isinstance(silent_ends, list):
        silent_ends = np.array(silent_ends)

    start_overlaps = (keep_end is None or keep_end) and np.all(
        (silent_starts <= result_obj.start, result_obj.start < silent_ends, silent_ends <= result_obj.end),
        axis=0
    ).nonzero()[0].tolist()
    if start_overlaps:
        new_start = silent_ends[start_overlaps[0]]
        result_obj.start = min(new_start, round(result_obj.end - min_word_dur, 3))
        if (result_obj.end - result_obj.start) <= min_word_dur:
            return

    end_overlaps = not keep_end and np.all(
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
        if len(matches) != 1:
            return

        def silence_errors(silence_start, silence_end):
            start_extra = silence_start - result_obj.start
            end_extra = result_obj.end - silence_end
            silent_duration = silence_end - silence_start
            start_error = start_extra / silent_duration
            end_error = end_extra / silent_duration
            return start_error, end_error

        def _adjust(silence_start, silence_end, errors=None):
            if not errors:
                errors = silence_errors(silence_start, silence_end)
            _keep_end = keep_end
            start_within_error = errors[0] <= nonspeech_error
            end_within_error = errors[1] <= nonspeech_error
            if _keep_end is None:
                _keep_end = errors[0] <= errors[1]
            if not (start_within_error or end_within_error):
                return
            if _keep_end:
                result_obj.start = min(silence_end, round(result_obj.end - min_word_dur, 3))
            else:
                result_obj.end = max(silence_start, round(result_obj.start + min_word_dur, 3))

        max_i = len(matches) - 1
        for i in range(len(matches)):
            error = None
            if i == max_i:
                idx = 0
            elif keep_end is None:
                error0 = silence_errors(silent_starts[matches[0]], silent_ends[matches[0]])
                error1 = silence_errors(silent_starts[matches[-1]], silent_ends[matches[-1]])
                idx, error = (0, error0) if min(error0) <= min(error1) else (-1, error1)
            else:
                idx = 0 if keep_end else -1
            idx = matches.pop(idx)
            _adjust(silent_starts[idx], silent_ends[idx], error)


def visualize_suppression(
        audio: Union[torch.Tensor, np.ndarray, str, bytes],
        output: str = None,
        q_levels: int = 20,
        k_size: int = 5,
        vad_threshold: float = 0.35,
        vad: bool = False,
        max_width: int = 1500,
        height: int = 200,
        **kwargs
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
    from ..audio import AudioLoader
    max_n_samples = None if max_width == -1 else round(max_width * N_SAMPLES_PER_TOKEN)
    loader = AudioLoader(audio, max_n_samples, stream=max_width != -1)
    audio = loader.next_chunk(0) if max_width != -1 else loader._buffered_samples
    loader.terminate()

    audio = audio_to_tensor_resample(audio)
    if max_n_samples is None:
        max_width = audio.shape[-1]
    else:
        audio = audio[:max_n_samples]
    loudness_tensor = audio2loudness(audio)
    width = min(max_width, loudness_tensor.shape[-1])
    if loudness_tensor is None:
        raise NotImplementedError(f'Audio is too short and cannot visualized.')

    if vad:
        silence_timings = get_vad_silence_func()(audio, vad_threshold, **kwargs)
        silence_mask = None if silence_timings is None else timing2mask(*silence_timings, size=loudness_tensor.shape[0])
    else:
        silence_mask = wav2mask(audio, q_levels=q_levels, k_size=k_size, **kwargs)

    visualize_mask(loudness_tensor, silence_mask, width=width, height=height, output=output)


class _DeprecatedModelCache:

    @staticmethod
    def warn():
        warnings.warn('_model_cache is deprecated and will be removed in future versions. '
                      'Use stable_whisper.default.cached_model_instances["silero_vad"] instead. '
                      'By default, the Silero-VAD model is loaded into the cache for the first load '
                      'and reused for subsequent loads.',
                      stacklevel=2)

    def __getitem__(self, item):
        from ..default import cached_model_instances
        self.warn()
        return cached_model_instances['silero_vad'][item]

    def __setitem__(self, key, value):
        from ..default import cached_model_instances
        self.warn()
        if key not in cached_model_instances['silero_vad']:
            keys = tuple(cached_model_instances['silero_vad'].keys())
            raise KeyError(f'{key} is not a key in cache for silero-vad model. Keys: {keys}')
        cached_model_instances['silero_vad'][key] = value


_model_cache = _DeprecatedModelCache()
