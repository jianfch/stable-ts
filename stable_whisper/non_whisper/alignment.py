import re
import warnings

import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Callable, Optional, Tuple
from dataclasses import dataclass
import copy

from ..result import WhisperResult, WordTiming
from ..audio import AudioLoader
from ..utils import safe_print, format_timestamp
from ..stabilization import NonSpeechPredictor
from ..default import get_min_word_dur, get_prepend_punctuations, get_append_punctuations
from ..options import AllOptions


@dataclass
class BasicWordTiming:
    word: str
    start: float
    end: float
    tokens: List[int]
    probability: float


@dataclass
class AlignmentTmpData:
    word: Optional[BasicWordTiming] = None
    extra_words: Optional[List[BasicWordTiming]] = None
    mask: Optional[torch.Tensor] = None
    offset: Optional[float] = None

    def clear(self):
        self.word = self.extra_words = self.mask = self.offset = None


@dataclass
class WordToken:
    word: str
    tokens: List[int]
    is_padding: bool = False

    def append(self, other: "WordToken"):
        if self.is_padding or other.is_padding:
            TypeError('append or append to padding')
        self.word += other.word
        self.tokens += other.tokens

    def prepend(self, other: "WordToken"):
        if self.is_padding or other.is_padding:
            TypeError('prepend or prepend to padding')
        self.word = other.word + self.word
        self.tokens = other.tokens + self.tokens


class Aligner:

    def __init__(
            self,
            inference_func: Callable,
            decode: Callable,
            encode: Callable,
            split_words_by_space: bool = True,
            sample_rate: int = 16000,
            max_segment_length: int = '30s',
            time_precision: float = 0.02,
            *,
            remove_instant_words: bool = False,
            token_step: int = 100,
            original_split: bool = False,
            word_dur_factor: Optional[float] = 2.0,
            max_word_dur: Optional[float] = 3.0,
            nonspeech_skip: Optional[float] = 5.0,
            fast_mode: bool = False,
            failure_threshold: Optional[float] = None,
            **options
    ):
        """
        Align plain text or tokens with audio using any compatible models.

        Parameters
        ----------
        inference_func : Callable
            Function that computes the start and end timestamps of an audio segment.
            The function takes two argument: audio segment as a torch.Tensor and a list of WordTokens objects.
            The function must return a list of dictionaries with word, start, end, and probability.
        decode : Callable
            Function of decoding tokens.
        encode : Callable
            Function for tokenizing text.
        split_words_by_space : bool, default True
            Whether to use space to delimit words. This depends on the language of the content (e.g. ``True`` for EN).
        sample_rate : int, default 16000
            The sampling of the audio ``inference_func`` expects.
        max_segment_length : int, default '30s'
            Maximum samples of audio segment ``inference_func`` expects. Use string for seconds followed by 's'.
        time_precision : float, default 0.02
            Time precision of the output from ``inference_func``.
            Use precision no lower than the precision of the model to avoid unnecessary computation.
        remove_instant_words : bool, default False
            Whether to truncate any words with zero duration.
        token_step : int, default 100
            Max number of tokens to align each pass. Use higher values to reduce chance of misalignment.
        original_split : bool, default False
            Whether to preserve the original segment groupings.
            Segments are split by line breaks if ``text`` is plain-text.
        max_word_dur : float or None, default 3.0
            Global maximum word duration in seconds. Re-align words that exceed the global maximum word duration.
        word_dur_factor : float or None, default 2.0
            Factor to compute the Local maximum word duration,
            which is ``word_dur_factor`` * local medium word duration.
            Words that need re-alignment, are re-algined with duration <= local/global maximum word duration.
        nonspeech_skip : float or None, default 5.0
            Skip non-speech sections that are equal or longer than this duration in seconds.
            Disable skipping if ``None``.
        fast_mode : bool, default False
            Whether to speed up alignment by re-alignment with local/global maximum word duration.
            ``True`` tends produce better timestamps when ``text`` is accurate and there are no large speechless gaps.
        stream : bool or None, default None
            Whether to loading ``audio`` in chunks of 30 seconds until the end of file/stream.
            If ``None`` and ``audio`` is a string then set to ``True`` else ``False``.
        failure_threshold : float, optional
            Abort alignment when percentage of words with zero duration exceeds ``failure_threshold``.
        verbose : bool or None, default False
            Whether to display the text being decoded to the console.
            Displays all the details if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.
        regroup : bool or str, default True, meaning the default regroup algorithm
            String for customizing the regrouping algorithm. False disables regrouping.
            Ignored if ``word_timestamps = False``.
        suppress_silence : bool, default True
            Whether to enable timestamps adjustments based on the detected silence.
        suppress_word_ts : bool, default True
            Whether to adjust word timestamps based on the detected silence.
            Only enabled if ``suppress_silence = True``.
        use_word_position : bool, default True
            Whether to use position of the word in its segment to determine whether to keep end or start timestamps if
            adjustments are required. If it is the first word, keep end. Else if it is the last word, keep the start.
        q_levels : int, default 20
            Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
            Acts as a threshold to marking sound as silent.
            Fewer levels will increase the threshold of volume at which to mark a sound as silent.
        k_size : int, default 5
            Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
            Recommend 5 or 3; higher sizes will reduce detection of silence.
        denoiser : str, optional
            String of the denoiser to use for preprocessing ``audio``.
            See ``stable_whisper.audio.SUPPORTED_DENOISERS`` for supported denoisers.
        denoiser_options : dict, optional
            Options to use for ``denoiser``.
        vad : bool or dict, default False
            Whether to use Silero VAD to generate timestamp suppression mask.
            Instead of ``True``, using a dict of keyword arguments will load the VAD with the arguments.
            Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
        vad_threshold : float, default 0.35
            Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
        min_word_dur : float or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
            Shortest duration each word is allowed to reach for silence suppression.
        min_silence_dur : float, optional
            Shortest duration of silence allowed for silence suppression.
        nonspeech_error : float, default 0.1
            Relative error of non-speech sections that appear in between a word for silence suppression.
        only_voice_freq : bool, default False
            Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
        prepend_punctuations : str or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
            Punctuations to prepend to next word.
        append_punctuations : str or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
            Punctuations to append to previous word.
        progress_callback : Callable, optional
            A function that will be called when transcription progress is updated.
            The callback need two parameters.
            The first parameter is a float for seconds of the audio that has been transcribed.
            The second parameter is a float for total duration of audio in seconds.
        presplit : bool or list of str, default True meaning same as ``append_punctuations``
            List of ending punctuation used to split ``text`` into segments for applying ``gap_padding``,
            but segmentation of final output is unnaffected unless ``original_split=True``.
            If ``original_split=True``, the original split is used instead of split from ``presplit``.
            Ignored if ``model`` is a faster-whisper model.
        gap_padding : str, default ' ...'
            Only if ``presplit=True``, ``gap_padding`` is prepended to each segments for word timing alignment.
            Used to reduce the probability of model predicting timestamps earlier than the first utterance.
            Ignored if ``model`` is a faster-whisper model.

        Notes
        -----
        ``regroup`` is ignored if ``original_split = True``.

        """
        if failure_threshold is not None and (failure_threshold < 0 or failure_threshold > 1):
            raise ValueError(f'``failure_threshold`` ({failure_threshold}) must be between 0 and 1.')

        self.options = AllOptions(options)

        if isinstance(max_segment_length, str):
            if not max_segment_length.endswith('s'):
                raise ValueError(f'expect string ``max_segment_length`` to end with "s" but got "{max_segment_length}"')
            max_segment_length = int(float(max_segment_length[:-1]) * sample_rate)

        tokens_per_sec = round(1/time_precision)

        self.sample_rate = sample_rate
        self.n_samples = max_segment_length
        self.tokens_per_sec = tokens_per_sec
        self._prepend_punctuations = get_prepend_punctuations(self.options.post.prepend_punctuations)
        self._append_punctuations = get_append_punctuations(self.options.post.append_punctuations)
        self._all_punctuations = self._prepend_punctuations + self._append_punctuations
        self.options.post.min_word_dur = get_min_word_dur(self.options.post.min_word_dur)

        self.inference_func = inference_func
        self.decode = decode
        self.encode = encode
        self.split_words_by_space = split_words_by_space

        self.remove_instant_words = remove_instant_words
        self.token_step = token_step
        self.original_split = original_split
        self.word_dur_factor = word_dur_factor
        self.max_word_dur = max_word_dur
        self.nonspeech_skip = nonspeech_skip
        self.fast_mode = fast_mode
        self.failure_threshold = failure_threshold

        self._pad_mask = None
        self.failure_count = 0
        self.max_fail = 0
        self._text = ''
        self._split_indices_by_char: List[int] = []
        self._all_word_tokens: List[WordToken] = []
        self._total_words = 0
        self._remaining_len = 0

        self.audio_loader = None
        self.nonspeech_predictor = None
        self._initial_duration = 0

        self._seek_sample = 0
        self._time_offset = 0
        self._temp_data = AlignmentTmpData()
        self._temp_words: List[BasicWordTiming] = []
        self._curr_words: List[BasicWordTiming] = []
        self._nonspeech_preds = {}
        self._seg_word_tokens: List[WordToken] = []

    def align(
            self,
            audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
            text: Union[str, List[int], WhisperResult],
            **options
    ) -> Union[WhisperResult, None]:
        """
        Align plain text or tokens with audio at word-level.

        Parameters
        ----------
        audio : str or numpy.ndarray or torch.Tensor or bytes or AudioLoader
            Path/URL to the audio file, the audio waveform, or bytes of audio file or
            instance of :class:`stable_whisper.audio.AudioLoader`.
            If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
        text : str or list of int or stable_whisper.result.WhisperResult
            String of plain-text, list of tokens, or instance of :class:`stable_whisper.result.WhisperResult`.

        Returns
        -------
        stable_whisper.result.WhisperResult or None
            All timestamps, words, probabilities, and other data from the alignment of ``audio``.
            Return None if alignment fails and ``remove_instant_words = True``.

        """
        self._reset()
        self._load_text(text)
        self._load_audio(audio)
        self._load_nonspeech_detector()
        for k in list(options.keys()):
            if hasattr(self, k):
                setattr(self, k, options.pop(k))
        self.options.update(options)

        with tqdm(
                total=self._initial_duration,
                unit='sec',
                disable=self.options.progress.verbose is not False,
                desc='Align'
        ) as tqdm_pbar:
            result: List[BasicWordTiming] = []
            last_ts = 0.0
            while self._all_word_tokens:

                self._time_offset = self._seek_sample / self.sample_rate
                audio_segment = self.audio_loader.next_chunk(self._seek_sample, self.n_samples)
                if audio_segment is None:
                    break

                self._nonspeech_preds = self.nonspeech_predictor.predict(audio=audio_segment, offset=self._time_offset)

                audio_segment = self._skip_nonspeech(audio_segment)
                if audio_segment is None:
                    continue

                self._curr_words = self._compute_timestamps(audio_segment, *self._get_curr_words())
                self._seg_word_tokens = [WordToken(wts.word, wts.tokens) for wts in self._curr_words]

                last_ts = self._fallback(audio_segment.shape[-1])

                self._update_pbar(tqdm_pbar, last_ts)

                result.extend(self._curr_words)

                if self.options.progress.verbose:
                    line = '\n'.join(
                        f"[{format_timestamp(wts.start)}] -> "
                        f"[{format_timestamp(wts.end)}] \"{wts.word}\""
                        for wts in self._curr_words
                    )
                    safe_print(line)

                if self.failure_threshold is not None:
                    self.failure_count += sum(1 for wts in self._curr_words if wts.end - wts.start == 0)
                    if self.failure_count > self.max_fail:
                        break

            self._update_pbar(tqdm_pbar, last_ts, self.failure_count <= self.max_fail)

        if self._temp_data.word is not None:
            result.append(self._temp_data.word)
        if not result:
            warnings.warn('Failed to align text.', stacklevel=2)
        if self.failure_count > self.max_fail:
            warnings.warn(
                f'Alignment aborted. Failed word percentage exceeded {self.failure_threshold * 100}% at '
                f'{format_timestamp(self._seek_sample / self.sample_rate)}.',
                stacklevel=2
            )
        elif self._all_word_tokens:
            last_ts_str = format_timestamp(result[-1].end if result else 0)
            warnings.warn(
                f'Failed to align the last {len(self._all_word_tokens)}/{self._total_words} words after '
                f'{last_ts_str}.',
                stacklevel=2
            )

        if self._all_word_tokens and not self.remove_instant_words:
            final_total_duration = self.audio_loader.get_duration(3)
            result.extend(
                [
                    BasicWordTiming(
                        word=w.word,
                        start=final_total_duration,
                        end=final_total_duration,
                        tokens=w.tokens,
                        probability=0.0
                    )
                    for w in self._all_word_tokens
                ]
            )

        self.audio_loader.terminate()
        self.nonspeech_predictor.finalize_timings()

        if not result:
            return

        final_result = [
            dict(word=w.word, start=w.start, end=w.end, tokens=w.tokens, probability=w.probability)
            for w in result
        ]
        if len(self._split_indices_by_char):
            word_lens = np.cumsum([len(w.word) for w in result])
            split_indices = [np.flatnonzero(word_lens >= i)[0] + 1 for i in self._split_indices_by_char]
            final_result = WhisperResult(
                [
                    final_result[i:j]
                    for i, j in zip([0] + split_indices[:-1], split_indices)
                    if i != j]
            )
        else:
            final_result = WhisperResult([final_result])

        self._suppress_silence(final_result)

        if not self.original_split:
            final_result.regroup(self.options.post.regroup)

        if fail_segs := len([None for s in final_result.segments if s.end - s.start <= 0]):
            warnings.warn(f'{fail_segs}/{len(final_result.segments)} segments failed to align.', stacklevel=2)

        return final_result

    def align_words(
            self,
            audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
            result: Union[WhisperResult, List[dict]],
            normalize_text: bool = True,
            inplace: bool = True
    ) -> WhisperResult:
        """
        Align segments of plain text or tokens with audio at word-level at specified start and end of each segment.

        This is a version of ``align()`` that confines each segment to a range of timestamps which eliminates the need
        for the fallback mechanisms used in ``align()``. This makes this method is drastically faster than ``align()``
        and reduces word-timstamp errors if the provided start and end timestamps of each segment is accurate.

        Parameters
        ----------
        audio : str or numpy.ndarray or torch.Tensor or bytes or AudioLoader
            Path/URL to the audio file, the audio waveform, or bytes of audio file or
            instance of :class:`stable_whisper.audio.AudioLoader`.
            If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
        result : stable_whisper.result.WhisperResult or list of dict
            Instance of :class:`stable_whisper.result.WhisperResult` or List of dictionaries with start, end, and text.
        normalize_text : bool or dict, default True
            Whether to normalize text of each segment.
        inplace : bool, default True
            Whether to update ``result`` with new timestamps if it is an instance of
            :class:`stable_whisper.result.WhisperResult`.

        Returns
        -------
        stable_whisper.result.WhisperResult
            All timestamps, words, probabilities, and other data from the alignment of ``audio``.
            Same object as ``result`` if ``inplace=True`` (default) and ``result`` is a ``WhisperResult``.

        """
        self._reset()
        result, segment_tokens = self._load_result(result, normalize_text, inplace)
        self._load_audio(audio)
        self._load_nonspeech_detector()

        with tqdm(
                total=self._initial_duration,
                unit='sec',
                disable=self.options.progress.verbose is not False,
                desc='Align Words'
        ) as tqdm_pbar:

            for segment, curr_tokens in zip(result.segments, segment_tokens):
                self._time_offset = segment.start
                self._seek_sample = round(segment.start * self.sample_rate)
                end = segment.end
                if segment.duration == 0:
                    self._update_pbar(tqdm_pbar, end)
                    continue
                segment_samples = round(segment.duration * self.sample_rate)
                audio_segment = self.audio_loader.next_chunk(self._seek_sample, segment_samples)
                if audio_segment is None:
                    break
                self.nonspeech_predictor.predict(audio=audio_segment, offset=self._time_offset)

                curr_word_tokens = tokens_to_word_tokens(
                    curr_tokens,
                    self.decode,
                    self.split_words_by_space,
                    self.options.post.prepend_punctuations,
                    self.options.post.append_punctuations
                )
                words_timings = self._compute_timestamps(audio_segment, curr_word_tokens)
                segment.words = [WordTiming(**w.__dict__) for w in words_timings]
                self._update_pbar(tqdm_pbar, end)
            self._update_pbar(tqdm_pbar, end, True)

        self.audio_loader.terminate()
        self.nonspeech_predictor.finalize_timings()
        result.reassign_ids()
        self._suppress_silence(result)
        result.regroup(self.options.post.regroup)

        return result

    def _reset(self):
        self._seek_sample = 0
        self._time_offset = 0
        self._temp_data.clear()
        self._temp_words: List[dict] = []
        self._curr_words: List[dict] = []
        self._nonspeech_preds = {}
        self._seg_word_tokens: List[WordToken] = []

    @property
    def prepend_punctuations(self):
        return self._prepend_punctuations

    @property
    def append_punctuations(self):
        return self._append_punctuations

    @property
    def all_punctuations(self):
        return self._all_punctuations

    @prepend_punctuations.setter
    def prepend_punctuations(self, punctuations: str):
        self._prepend_punctuations = punctuations
        self._all_punctuations = self._prepend_punctuations + self._append_punctuations

    @append_punctuations.setter
    def append_punctuations(self, punctuations: str):
        self._append_punctuations = punctuations
        self._all_punctuations = self._prepend_punctuations + self._append_punctuations

    @staticmethod
    def _standardize_text(
            text: Union[str, List[int], WhisperResult],
            original_split: bool = False
    ) -> Tuple[Union[str, List[int]], List[int]]:
        split_indices_by_char: List[int] = []
        if isinstance(text, WhisperResult):
            if original_split and len(text.segments) > 1 and text.has_words:
                split_indices_by_char = np.cumsum(
                    [sum(len(w.word) for w in seg.words) for seg in text.segments]
                ).tolist()
            text = text.text
        elif isinstance(text, str):
            if original_split and '\n' in text:
                text_split = [
                    ' ' + norm_line
                    for line in text.splitlines()
                    if (norm_line := re.sub(r'\s', ' ', line).strip())
                ]
                split_indices_by_char = np.cumsum([len(seg) for seg in text_split]).tolist()
                text = ''.join(seg for seg in text_split)
            else:
                text = re.sub(r'\s', ' ', text)
                if not text.startswith(' '):
                    text = ' ' + text
        return text, split_indices_by_char

    def _load_result(
            self,
            result: Union[WhisperResult, List[dict]],
            normalize_text: bool = True,
            inplace: bool = False
    ) -> Tuple[WhisperResult, List[List[int]]]:

        segment_tokens = None
        if isinstance(result, WhisperResult):
            if not inplace:
                result = copy.deepcopy(result)
        else:
            if result and not result[0]['text'] and result[0]['tokens']:
                segment_tokens = [seg['tokens'] for seg in result]
                for seg in result:
                    seg['text'] = self.encode(seg['tokens'])
            result = WhisperResult(result)

        if normalize_text:
            def norm_text(text: str):
                text = re.sub(r'\s', ' ', text)
                if not text.startswith(' '):
                    text = ' ' + text
                return text
        else:
            def norm_text(text: str):
                return text

        max_segment_tokens = self.token_step
        if segment_tokens is None:
            segment_tokens = [self.encode(norm_text(seg.text)) for seg in result]
        segment_exceed_max = [i for i, tokens in enumerate(segment_tokens) if len(tokens) > max_segment_tokens]
        if segment_exceed_max:
            raise RuntimeError(
                f'found segments at following indices exceeding max length for model: {segment_exceed_max}')

        return result, segment_tokens

    def _get_pad_mask(
            self,
            presplit: bool
    ) -> Union[None, List[bool]]:
        if not presplit:
            return
        if isinstance(presplit, bool):
            presplit = get_append_punctuations(self.options.post.append_punctuations)
        if len(self._split_indices_by_char):
            pad_mask = []
            cumsums = self._split_indices_by_char.copy()
            cumsum_len = 0
            for word in self._all_word_tokens:
                cumsum_len += len(word.word)
                if cumsums and cumsum_len >= cumsums[0]:
                    cumsums.pop(0)
                    pad_mask.extend([True] * len(word.word))
                else:
                    pad_mask.extend([False] * len(word.word))
        else:
            pad_mask = [b for w in self._all_word_tokens
                        for b in ([any(map(w.word.endswith, presplit))] * len(w.word))]
        return pad_mask

    def _load_audio(
            self,
            audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader]
    ):
        if isinstance(audio, AudioLoader):
            audio.validate_external_args(
                sr=self.sample_rate,
                vad=self.options.silence.vad,
                stream=self.options.pre.stream,
                denoiser=self.options.pre.denoiser,
                denoiser_options=self.options.pre.denoiser_options,
                only_voice_freq=self.options.pre.only_voice_freq
            )
        else:
            audio = AudioLoader(
                audio,
                sr=self.sample_rate,
                denoiser=self.options.pre.denoiser,
                denoiser_options=self.options.pre.denoiser_options,
                only_voice_freq=self.options.pre.only_voice_freq,
                verbose=self.options.progress.verbose,
                new_chunk_divisor=512,
                stream=self.options.pre.stream,
                only_ffmpeg=self.options.pre.only_ffmpeg
            )

        self.audio_loader = audio
        self._initial_duration = self.audio_loader.get_duration(2)

    def _load_nonspeech_detector(self):

        self.nonspeech_predictor = NonSpeechPredictor(
            vad=self.options.silence.vad if self.options.post.suppress_silence else None,
            get_mask=True,
            min_word_dur=self.options.post.min_word_dur,
            q_levels=self.options.silence.q_levels,
            k_size=self.options.silence.k_size,
            vad_threshold=self.options.silence.vad_threshold,
            vad_window=self.audio_loader.new_chunk_divisor,
            sampling_rate=self.sample_rate,
            verbose=None if self.audio_loader.stream else self.options.progress.verbose,
            store_timings=True,
            ignore_is_silent=True,
            min_silence_dur=self.options.post.min_silence_dur
        )
        self.audio_loader.update_post_prep_callback(
            self.nonspeech_predictor.get_on_prep_callback(self.audio_loader.stream))

    def _load_text(
            self,
            text: Union[str, List[int], WhisperResult]
    ):
        self._text, self._split_indices_by_char = self._standardize_text(text, self.original_split)
        tokens = self.encode(self._text) if isinstance(self._text, str) else self._text
        self._all_word_tokens = tokens_to_word_tokens(tokens, self.decode, self.split_words_by_space)
        self._pad_mask = self._get_pad_mask(self.options.align.presplit)
        self._total_words = len(self._all_word_tokens)
        self._remaining_len = sum(len(w.word) for w in self._all_word_tokens)
        self.failure_count = 0
        self.max_fail = self._total_words * (self.failure_threshold or 1)

    def _compute_timestamps(
            self,
            audio_segment: torch.Tensor,
            word_tokens: List[WordToken],
            split_indices: Optional[List[int]] = None,
            expect_gap: bool = False,
            time_offset: Optional[float] = None
    ) -> List[BasicWordTiming]:
        if split_indices:
            temp_split_indices = [0] + split_indices
            if temp_split_indices[-1] < len(word_tokens):
                temp_split_indices.append(len(word_tokens))
            input_word_tokens = [
                word_tokens[i:j]
                for i, j in zip(temp_split_indices[:-1], temp_split_indices[1:])
            ]
            pad_segment_word_tokens(input_word_tokens, self.options.align.gap_padding, self.encode, expect_gap)
            input_word_tokens = flatten_segment_word_tokens(input_word_tokens)[0]
        else:
            input_word_tokens = word_tokens

        max_curr_ts = round(audio_segment.size(-1) / self.sample_rate, 4)
        output = self.inference_func(audio_segment, input_word_tokens)
        if len(output) < len(input_word_tokens):
            raise RuntimeError(
                f'expected output word count to be at least {len(input_word_tokens)} but got {len(output)}'
            )

        if output[-1]['start'] > max_curr_ts:
            warnings.warn(f'word "{output[-1]}" start later than the max timestamp')

        if time_offset is None:
            time_offset = self._time_offset

        final_output: List[BasicWordTiming] = []
        i = 0
        curr_word = ''
        curr_start = -1
        probs = []
        wti_max = len(output) - 1
        for wti, word_timing in enumerate(output):
            curr_word += word_timing['word']
            if curr_start == -1:
                curr_start = word_timing['start']
            if word_timing.get('probability'):
                probs.append(word_timing['probability'])
            input_word = input_word_tokens[i].word
            if curr_word == input_word:
                if not input_word_tokens[i].is_padding:
                    start = curr_start
                    end = word_timing['end']
                    if start > max_curr_ts:
                        start = max_curr_ts
                    if end > max_curr_ts:
                        end = max_curr_ts
                    start = round(start + time_offset, 3)
                    end = round(end + time_offset, 3)
                    new_wt = BasicWordTiming(
                        input_word,
                        start,
                        end,
                        input_word_tokens[i].tokens,
                        np.mean(probs).item() if probs else 0.0
                    )
                    final_output.append(new_wt)
                curr_word = ''
                curr_start = -1
                probs = []
                i += 1
            elif len(curr_word) > len(input_word) or wti == wti_max:
                raise RuntimeError(f'expect word "{input_word}" but got "{curr_word}"')

        return final_output

    def _get_curr_words(self):
        all_word_tokens = self._all_word_tokens
        pad_mask = self._pad_mask
        curr_tk_count = 0
        word_tokens: List[WordToken] = []
        split_indices: List[int] = []
        is_start_gap = (
            True if self._remaining_len == len(pad_mask) else pad_mask[-(self._remaining_len + 1)]
        ) if pad_mask else True
        for i in range(len(all_word_tokens)):
            tk_count = len(all_word_tokens[0].tokens)
            m_count = 1 if pad_mask and pad_mask[-(self._remaining_len - len(all_word_tokens[0].word) + 1)] else 0
            if curr_tk_count + len(split_indices) + tk_count + m_count > self.token_step and word_tokens:
                break
            if pad_mask and pad_mask[-(self._remaining_len - len(all_word_tokens[0].word) + 1)]:
                split_indices.append(i + 1)
            self._remaining_len -= len(all_word_tokens[0].word)
            word_tokens.append(all_word_tokens.pop(0))
            curr_tk_count += tk_count
        return word_tokens, split_indices, is_start_gap

    def _fix_temp_words(
            self,
            target_word: BasicWordTiming,
            word_sources: List[BasicWordTiming],
            second_target: Optional[BasicWordTiming] = None
    ) -> Tuple[Union[BasicWordTiming, None], List[BasicWordTiming]]:
        first_word_src = word_sources[0]
        assert target_word.word.startswith(first_word_src.word)
        if target_word.word != first_word_src.word:
            if len(word_sources) < 2:
                return None, []
            first_word_src_probs = [first_word_src.probability]
            if first_word_src.word.strip() in self.all_punctuations:
                first_word_src.start, first_word_src.end = word_sources[1].start, word_sources[1].end
            for _ in range(len(word_sources) - 1):
                tw = word_sources.pop(1)
                fullword = first_word_src.word + tw.word
                assert target_word.word.startswith(fullword)
                first_word_src.word = fullword
                first_word_src.tokens += tw.tokens
                first_word_src_probs.append(tw.probability)
                if tw.word.strip() not in self.all_punctuations:
                    first_word_src.end = tw.end
                if target_word.word == first_word_src.word:
                    break
            if target_word.word != first_word_src.word:
                return None, []
            first_word_src.probability = np.mean(first_word_src_probs).item()
        elif second_target:
            if len(word_sources) == 1:
                return first_word_src, []
            second_word_src, word_sources = self._fix_temp_words(second_target, word_sources[1:])
            if second_word_src is not None:
                word_sources = [second_word_src] + word_sources
            return first_word_src, word_sources

        return first_word_src, word_sources[1:]

    def _speech_percentage(
            self,
            word: BasicWordTiming,
            mask: torch.Tensor,
            offset: float
    ) -> float:
        if mask is None:
            return 1
        s, e = word.start, word.end
        s = int((s - offset) * self.tokens_per_sec)
        e = int((e - offset) * self.tokens_per_sec)
        return 1 - mask[s:e].float().mean().nan_to_num().item()

    def _is_new_better(
            self,
            word0: BasicWordTiming,
            mask0: torch.Tensor,
            offset0: float,
            word1: BasicWordTiming,
            mask1: torch.Tensor,
            offest1: float
    ):
        speech0 = round(self._speech_percentage(word0, mask0, offset0), 1)
        speech1 = round(self._speech_percentage(word1, mask1, offest1), 1)
        w0p = word0.probability
        w1p = word1.probability
        return ((w1p ** 0.75 - w0p ** 0.75) < 0.35 and speech0 >= speech1) or w0p >= w1p

    def _update_curr_words(self):
        if self._temp_data.word is None:
            return
        self._temp_words = [self._temp_data.word] + self._temp_data.extra_words[:len(self._curr_words) - 1]
        self._curr_words[:len(self._temp_words)] = self._temp_words
        self._temp_data.word = None

    def _redo_words(self, index: int = None, ):
        if index is not None and self._curr_words and self._temp_data.word is not None:
            self._temp_data.word, self._temp_data.extra_words = self._fix_temp_words(
                self._curr_words[0],
                [self._temp_data.word] + self._temp_data.extra_words,
                self._curr_words[1] if len(self._curr_words) > 1 else None
            )

            if self._temp_data.word:
                use_new = self._is_new_better(
                    self._curr_words[0], self._nonspeech_preds['mask'], self._time_offset,
                    self._temp_data.word, self._temp_data.mask, self._temp_data.offset
                )
                new_extra_words = []
                if use_new:
                    self._temp_data.word = self._curr_words[0]
                else:
                    for wi, (cw, tw) in enumerate(zip(self._curr_words[1:], self._temp_data.extra_words)):
                        assert cw.word.startswith(tw.word)
                        use_new = self._is_new_better(
                            cw, self._nonspeech_preds['mask'], self._time_offset,
                            tw, self._temp_data.mask, self._temp_data.offset
                        )
                        if use_new or cw.word != tw.word or cw.end < tw.end:
                            break
                        new_extra_words.append(tw)
                self._temp_data.extra_words = new_extra_words

        if index is None:  # redo all
            self._remaining_len += sum(len(w.word) for w in self._seg_word_tokens)
            self._all_word_tokens = self._seg_word_tokens + self._all_word_tokens
            self._curr_words = []
            self._temp_data.word = None
        elif index != len(self._seg_word_tokens):  # redo from _idx
            self._remaining_len += sum(len(w.word) for w in self._seg_word_tokens[index:])
            self._all_word_tokens = self._seg_word_tokens[index:] + self._all_word_tokens
            self._curr_words, new_extra_words = self._curr_words[:index], self._curr_words[index:]
            if self._curr_words:
                self._update_curr_words()
                self._remaining_len += sum(len(w.word) for w in self._seg_word_tokens[index - 1:index])
                self._all_word_tokens = self._seg_word_tokens[index - 1:index] + self._all_word_tokens
                self._temp_data.word = self._curr_words.pop(-1)
                self._temp_data.extra_words = new_extra_words
                self._temp_data.mask = self._nonspeech_preds['mask']
                self._temp_data.offset = self._time_offset
        else:
            self._update_curr_words()

    def _skip_nonspeech(
            self,
            audio_segment: torch.Tensor
    ) -> Union[torch.Tensor, None]:
        if self.nonspeech_skip is None:
            return audio_segment

        segment_nonspeech_timings = self._nonspeech_preds['timings']

        if segment_nonspeech_timings is None or len(segment_nonspeech_timings[0]) == 0:
            return audio_segment

        segment_samples = audio_segment.size(-1)
        segment_duration = segment_samples / self.sample_rate

        max_time_offset = self._time_offset + self.options.post.min_word_dur
        min_time_offset = self._time_offset - self.options.post.min_word_dur

        if (
                (segment_nonspeech_timings[0][0] < max_time_offset) and
                (segment_nonspeech_timings[1][0] > min_time_offset + segment_duration)
        ):
            # entire audio segment is within first nonspeech section
            self._seek_sample += segment_samples
            return

        # mask for valid nonspeech sections (i.e. sections with duration >= ``nonspeech_skip``)
        valid_sections = (segment_nonspeech_timings[1] - segment_nonspeech_timings[0]) >= self.nonspeech_skip
        if not valid_sections.any():
            # no valid nonspeech sections
            return audio_segment

        nonspeech_starts = segment_nonspeech_timings[0, valid_sections]
        if max_time_offset < nonspeech_starts[0]:
            # current time is before the first valid nonspeech section
            return audio_segment

        nonspeech_ends = segment_nonspeech_timings[1, valid_sections]
        curr_total_samples = self.audio_loader.get_total_samples()

        # skip to end of the first nonspeech section
        self._seek_sample = round(nonspeech_ends[0] * self.sample_rate)
        if self._seek_sample + (self.options.post.min_word_dur * self.sample_rate) > curr_total_samples:
            # new time is over total duration of the audio
            self._seek_sample = curr_total_samples
            return

        self._time_offset = self._seek_sample / self.sample_rate

        # try to load audio segment from the new timestamp
        audio_segment = self.audio_loader.next_chunk(self._seek_sample, self.n_samples)
        if audio_segment is None:
            # reached eof
            return

        # recompute nonspeech sections for the new audio segment for later use
        self._nonspeech_preds = self.nonspeech_predictor.predict(audio=audio_segment, offset=self._time_offset)
        if len(nonspeech_starts) > 1:
            # remove all audio samples after start of second valid nonspeech section
            new_sample_count = round((nonspeech_starts[1] - nonspeech_ends[0]) * self.sample_rate)
            audio_segment = audio_segment[:new_sample_count]

        return audio_segment

    def _fallback(
            self,
            segment_samples: int
    ) -> float:
        durations = np.array([w.end - w.start for w in self._curr_words]).round(3)
        nonzero_mask = durations > 0
        nonzero_indices = np.flatnonzero(nonzero_mask)
        if len(nonzero_indices):
            redo_index = nonzero_indices[-1] + 1
            if (
                    self._all_word_tokens and
                    len(nonzero_indices) > 1 and
                    (
                            self._curr_words[nonzero_indices[-1]].end
                            >=
                            np.floor(self._time_offset + segment_samples / self.sample_rate)
                    )
            ):
                nonzero_mask[nonzero_indices[-1]] = False
                nonzero_indices = nonzero_indices[:-1]
                redo_index = nonzero_indices[-1] + 1
            med_dur = np.median(durations[:redo_index])

            if self.fast_mode:
                new_start = None
                global_max_dur = None
            else:
                local_max_dur = round(med_dur * self.word_dur_factor, 3) if self.word_dur_factor else None
                if self.max_word_dur:
                    local_max_dur = \
                        min(local_max_dur, self.max_word_dur) if local_max_dur else self.max_word_dur
                    global_max_dur = self.max_word_dur
                else:
                    global_max_dur = local_max_dur or None
                if global_max_dur and med_dur > global_max_dur:
                    med_dur = global_max_dur
                if (
                        local_max_dur and durations[nonzero_indices[0]] > global_max_dur
                ):
                    new_start = round(max(
                        (
                                self._curr_words[nonzero_indices[0]].end
                                -
                                (med_dur * nonzero_indices[0] + local_max_dur)
                        ),
                        self._curr_words[nonzero_indices[0]].start
                    ), 3)
                    if new_start <= self._time_offset:
                        new_start = None
                else:
                    new_start = None
            if new_start is None:
                if global_max_dur:
                    index_offset = nonzero_indices[0] + 1
                    redo_indices = \
                        np.flatnonzero(durations[index_offset:redo_index] > global_max_dur) + index_offset
                    if len(redo_indices):
                        redo_index = redo_indices[0]
                last_ts = self._curr_words[redo_index - 1].end
                self._redo_words(redo_index)
            else:
                last_ts = new_start
                self._redo_words()
            self._seek_sample = round(last_ts * self.sample_rate)
        else:
            self._seek_sample += segment_samples
            last_ts = round(self._seek_sample / self.sample_rate, 2)
            self._redo_words()

        return last_ts

    def _suppress_silence(self, result: WhisperResult):
        if (
                self.options.post.suppress_silence and
                (nonspeech_timings := self.nonspeech_predictor.nonspeech_timings) is not None
        ):
            result.suppress_silence(
                *nonspeech_timings,
                min_word_dur=self.options.post.min_word_dur,
                word_level=self.options.post.suppress_word_ts,
                nonspeech_error=self.options.post.nonspeech_error,
                use_word_position=self.options.post.use_word_position,
                verbose=self.options.progress is not None
            )
            result.update_nonspeech_sections(*nonspeech_timings)
            result.set_current_as_orig()

    def _update_pbar(self, tqdm_pbar: tqdm, last_ts: float, finish: bool = False):
        curr_total = self.audio_loader.get_duration(2)
        if need_refresh := curr_total != tqdm_pbar.total:
            tqdm_pbar.total = curr_total
        tqdm_pbar.update((curr_total if finish else min(round(last_ts, 2), curr_total)) - tqdm_pbar.n)
        if need_refresh:
            tqdm_pbar.refresh()
        if self.options.progress.progress_callback is not None:
            self.options.progress.progress_callback(tqdm_pbar.n, tqdm_pbar.total)


def merge_punctuations(
        word_tokens: List[WordToken],
        prepend_punctuations: Optional[str] = None,
        append_punctuations: Optional[str] = None
):
    if len(word_tokens) < 2:
        return
    prepend_punctuations = get_prepend_punctuations(prepend_punctuations)
    append_punctuations = get_append_punctuations(append_punctuations)
    for i in range(len(word_tokens)-1, -1, -1):
        word_token = word_tokens[i]
        if word_token.is_padding:
            continue
        if (
                word_token is not word_tokens[-1] and
                word_token.word.startswith(' ') and
                word_token.word.strip() in prepend_punctuations
        ):
            word_token = word_tokens.pop(i)
            word_tokens[i].prepend(word_token)
        word_token = word_tokens[i]
        if (
                i != 0 and
                not word_token.word.endswith(' ') and
                word_token.word in append_punctuations
        ):
            word_token = word_tokens.pop(i)
            word_tokens[i-1].append(word_token)


def tokens_to_word_tokens(
        tokens: List[int],
        decode: Callable,
        split_by_space: bool,
        prepend_punctuations: Optional[str] = None,
        append_punctuations: Optional[str] = None
) -> List[WordToken]:
    text: str = decode(tokens)
    final_word_tokens: List[WordToken] = []
    curr_tokens: List[int] = []
    for token in tokens:
        curr_tokens.append(token)
        curr_text = decode(curr_tokens)
        is_whole = text[:len(curr_text)] == curr_text
        if not is_whole:
            continue
        is_append = split_by_space and not curr_text.startswith(" ")
        if is_append and len(final_word_tokens) != 0:
            final_word_tokens[-1].word += curr_text
            final_word_tokens[-1].tokens += curr_tokens
        else:
            final_word_tokens.append(WordToken(curr_text, curr_tokens))
        text = text[len(curr_text):]
        curr_tokens = []

    if curr_tokens:
        final_word_tokens.append(WordToken(text, curr_tokens))
    elif len(text) != 0:
        final_word_tokens[-1].word += text
    merge_punctuations(final_word_tokens, prepend_punctuations, append_punctuations)
    return final_word_tokens


def pad_segment_word_tokens(
        segment_word_tokens: List[List[WordToken]],
        padding: str,
        encode: Callable,
        pad_first_seg: bool = True
):
    if padding is None:
        return
    padding_tokens = encode(padding)
    padding_word_token = WordToken(padding, padding_tokens, True)

    def startswith_padding(_tokens: List[int]):
        if len(padding_tokens) > len(_tokens):
            return False
        return padding_tokens == _tokens[:len(padding_tokens)]

    def endswith_padding(_tokens: List[int]):
        if len(padding_tokens) > len(_tokens):
            return False
        return padding_tokens == _tokens[-len(padding_tokens):]

    for i, word_tokens in enumerate(segment_word_tokens):
        if (
                startswith_padding(word_tokens[0].tokens) or
                (i != 0 and endswith_padding(segment_word_tokens[i-1][-1].tokens)) or
                (i == 0 and not pad_first_seg)
        ):
            continue

        word_tokens.insert(0, padding_word_token)


def flatten_segment_word_tokens(
        segment_word_tokens: List[List[WordToken]],
        track_segment_index: bool = False
) -> Tuple[List[WordToken], List[int]]:
    all_word_tokens: List[WordToken] = []
    segment_indices: List[int] = []
    for i, word_tokens in enumerate(segment_word_tokens):
        all_word_tokens.extend(word_tokens)
        if not track_segment_index:
            continue
        segment_indices.extend(
            [
                -1 if word.is_padding else i
                for word in word_tokens
            ]
        )

    return all_word_tokens, segment_indices
