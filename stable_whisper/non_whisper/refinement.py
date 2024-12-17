import copy

import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Callable, Optional

from ..result import WhisperResult, WordTiming
from ..audio import prep_audio, audioloader_not_supported
from ..options import AllOptions


class Refiner:

    def __init__(
            self,
            inference_func: Callable,
            sample_rate: int = 16000,
            max_segment_length: int = '30s',
            max_inference_tokens: int = 100,
            *,
            steps: str = 'se',
            rel_prob_decrease: float = .03,
            abs_prob_decrease: float = .05,
            rel_rel_prob_decrease: Optional[float] = None,
            prob_threshold: float = .5,
            rel_dur_change: Optional[float] = .5,
            abs_dur_change: Optional[float] = None,
            word_level: bool = True,
            precision: float = None,
            **options
    ):
        """
        Improve existing timestamps with any compatible ASR models.

        This function iteratively muting portions of the audio and monitoring token probabilities to find the most
        precise timestamps. This "most precise" in this case means the latest start and earliest end of a word that
        maintains an acceptable probability determined by the specified arguments.

        This is useful readjusting timestamps when they start too early or end too late.

        Parameters
        ----------
        inference_func : Callable
            Function that computes confidence score of tokens for an audio segment.
            The function takes two argument: two audio segment as a torch.Tensor and tokens as a list of int.
            The function must return a torch.Tensor of token confidence scores of the audio segments.
        sample_rate : int, default 16000
            The sampling of the audio ``inference_func`` expects.
        max_segment_length : int, default '30s'
            Maximum samples of audio segment ``inference_func`` expects. Use string for seconds followed by 's'.
        max_inference_tokens : int, default 100
            Maximum number of tokens ``inference_func`` expects.
        steps : str, default 'se'
            Instructions for refinement. A 's' means refine start-timestamps. An 'e' means refine end-timestamps.
        rel_prob_decrease : float, default 0.3
            Maximum percent decrease in probability relative to original probability which is the probability from
            muting according initial timestamps.
        abs_prob_decrease : float, default 0.05
            Maximum decrease in probability from original probability.
        rel_rel_prob_decrease : float, optional
            Maximum percent decrease in probability relative to previous probability which is the probability from
            previous iteration of muting.
        prob_threshold : float, default 0.5
            Stop refining the timestamp if the probability of its token goes below this value.
        rel_dur_change : float, default 0.5
            Maximum percent change in duration of a word relative to its original duration.
        abs_dur_change : float, optional
            Maximum seconds a word is allowed deviate from its original duration.
        word_level : bool, default True
            Whether to refine timestamps on word-level. If ``False``, only refine start/end timestamps of each segment.
        precision : float, default 0.1
            Precision of refined timestamps in seconds. The lowest precision is 0.02 second.
        single_batch : bool, default False
            Whether to process in only batch size of one to reduce memory usage.
        inplace : bool, default True
            Whether to alter timestamps in-place. Return a deepcopy of ``result`` if ``False``.
        denoiser : str, optional
            String of the denoiser to use for preprocessing ``audio``.
            See ``stable_whisper.audio.SUPPORTED_DENOISERS`` for supported denoisers.
        denoiser_options : dict, optional
            Options to use for ``denoiser``.
        only_voice_freq : bool, default False
            Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
        verbose : bool or None, default False
            Whether to display the text being decoded to the console.
            Displays all the details if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.

        Notes
        -----
        The lower the ``precision``, the longer the processing time.
        """

        if not steps:
            steps = 'se'
        if invalid_steps := steps.replace('s', '').replace('e', ''):
            raise ValueError(f'Invalid step(s): {", ".join(invalid_steps)}')
        if precision is None:
            precision = 0.1

        if isinstance(max_segment_length, str):
            if not max_segment_length.endswith('s'):
                raise ValueError(f'expect string ``max_segment_length`` to end with "s" but got "{max_segment_length}"')
            self.max_segment_seconds = float(max_segment_length[:-1])
        else:
            self.max_segment_seconds = max_segment_length / sample_rate

        self.options = AllOptions(options, silence=False, align=False)
        self.steps = steps
        self.precision = precision
        self.sample_rate = sample_rate

        self.max_inference_tokens = max_inference_tokens
        self.sample_precision = max(round(self.precision * self.sample_rate), 2)

        self.inference_func = inference_func

        self.rel_prob_decrease = rel_prob_decrease
        self.abs_prob_decrease = abs_prob_decrease
        self.rel_rel_prob_decrease = rel_rel_prob_decrease
        self.prob_threshold = prob_threshold
        self.rel_dur_change = rel_dur_change
        self.abs_dur_change = abs_dur_change
        self.word_level = word_level

        self._prev_ts = 0
        self._pbar_step = 0
        self._step_count = 0
        self._tqdm_pbar = None
        self._audio = torch.tensor([])

    def refine(
            self,
            audio: Union[str, np.ndarray, torch.Tensor, bytes],
            result: WhisperResult,
            inplace: bool = True,
            encode: Optional[Callable] = None,
            **options
    ) -> WhisperResult:
        """

        Parameters
        ----------
        audio : str or numpy.ndarray or torch.Tensor or bytes
            Path/URL to the audio file, the audio waveform, or bytes of audio file.
            If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
        result : stable_whisper.result.WhisperResult
            All timestamps, words, probabilities, and other data from the transcription of ``audio``.
        inplace : bool, default True
            Whether to update ``result`` with new timestamps if it is an instance of
            :class:`stable_whisper.result.WhisperResult`.
        encode : Callable, optional
            Function for tokenizing text. Only needed if ``result`` is missing tokens.

        Returns
        -------
        stable_whisper.result.WhisperResult
            All timestamps, words, probabilities, and other data from the alignment of ``audio``.
            Same object as ``result`` if ``inplace=True`` (default) and ``result`` is a ``WhisperResult``..

        """

        if result:
            if not result.has_words:
                raise RuntimeError(f'cannot refine result with missing word-timestamps')
            elif not all(word.tokens for word in result.all_words()):
                if encode is None:
                    raise RuntimeError(f'result must have tokens or provide tokenization function to ``encode``')
                for word in result.all_words():
                    word.tokens = encode(word.word)

        if not inplace:
            result = copy.deepcopy(result)

        self._load_audio(audio)
        for k in list(options.keys()):
            if hasattr(self, k):
                setattr(self, k, options.pop(k))
        self.options.update(options)

        with tqdm(
                total=round(self._audio.size(-1) / self.sample_rate, 2), unit='sec',
                disable=self.options.progress.verbose is not False,
                desc='Refine'
        ) as self._tqdm_pbar:

            self._pbar_step = self._tqdm_pbar.total / len(self.steps)
            for step_count, step in enumerate(self.steps, 1):
                self._step_count = step_count
                self._prev_ts = 0
                self._refine(result, step)
                self.update_pbar(self._tqdm_pbar.total)
            self._tqdm_pbar.update(self._tqdm_pbar.total - self._tqdm_pbar.n)

        result.reassign_ids()

        return result

    def _reset(self):
        self._prev_ts = 0
        self._pbar_step = 0
        self._step_count = 0
        self._tqdm_pbar = None
        self._audio = torch.tensor([])

    def _load_audio(
            self,
            audio: Union[str, np.ndarray, torch.Tensor, bytes]
    ):
        audioloader_not_supported(audio)
        self._audio = prep_audio(
            audio,
            denoiser=self.options.pre.denoiser,
            denoiser_options=self.options.pre.denoiser_options,
            only_voice_freq=self.options.pre.only_voice_freq,
            only_ffmpeg=self.options.pre.only_ffmpeg,
            verbose=self.options.progress.verbose
        )

    def curr_segments(
            self,
            result: WhisperResult,
            total_duration: float
    ):
        all_words = result.all_words()
        seg_edge_mask = np.array([
            1 if _i == 0 else (2 if _i == len(seg.words) - 1 else 0)
            for seg in result.segments
            for _i, w in enumerate(seg.words)
        ])
        start_times = [
            max(
                0 if self.abs_dur_change is None else (w.start - self.abs_dur_change),
                0 if self.rel_dur_change is None else (w.start - w.duration * self.rel_dur_change),
                0 if i == 0 else max(all_words[i - 1].end, w.end - 14.5, 0)
            )
            for i, w in enumerate(all_words)
        ]
        end_times = [
            min(
                total_duration if self.abs_dur_change is None else (w.end + self.abs_dur_change),
                total_duration if self.rel_dur_change is None else (w.end + w.duration * self.rel_dur_change),
                total_duration if i == len(all_words) else min(all_words[i].start, w.start + 14.5, total_duration)
            )
            for i, w in enumerate(all_words, 1)
        ]
        start = start_times[0]

        prev_i = 0
        curr_words, curr_starts, curr_ends = [], [], []
        curr_token_count = 0

        for i, w in enumerate(all_words, 1):
            if (
                    (end_times[0] - start > self.max_segment_seconds) or
                    (curr_token_count + len(w.tokens) > self.max_inference_tokens)
            ):
                if curr_words:
                    yield curr_words, curr_starts, curr_ends, seg_edge_mask[prev_i:prev_i + len(curr_words)]
                    curr_words, curr_starts, curr_ends = [], [], []
                start = start_times[0]
                prev_i = i - 1
                curr_token_count = 0

            curr_words.append(w)
            curr_starts.append(start_times.pop(0))
            curr_ends.append(end_times.pop(0))
            curr_token_count += len(w.tokens)

            if i == len(all_words):
                yield curr_words, curr_starts, curr_ends, seg_edge_mask[prev_i:prev_i + len(curr_words)]

    def second_to_sample(
            self,
            timestamps: Union[np.ndarray, List[float]],
            offset: Optional[float] = None
    ) -> np.ndarray:
        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)
        return ((timestamps - offset) * self.sample_rate).round().astype(np.int32)

    def get_prob(
            self,
            audio_segment: torch.Tensor,
            text_tokens: List[int],
            word_tokens: List[List[int]],
            prob_indices: List[int],
            is_end_ts: bool
    ):

        token_probs: torch.Tensor = self.inference_func(audio_segment, text_tokens)
        if token_probs.size(0) != 2:
            raise RuntimeError(f'expected dim 0 to be length of 2 but got {token_probs.size(0)}')
        if token_probs.size(1) != len(text_tokens):
            raise RuntimeError(f'expected dim 1 to be length of {len(text_tokens)} but got {token_probs.size(1)}')
        if token_probs.ndim != 2 and token_probs.ndim != 3:
            raise RuntimeError(f'expected inference_func output to have 2 or 3 dimensions but got {token_probs.ndim}')
        tokens = torch.tensor(text_tokens, device=token_probs.device)
        word_idxs = torch.arange(len(text_tokens))

        if token_probs.ndim == 2:
            text_token_probs = token_probs
            token_positions = None
        else:
            text_token_probs = token_probs[:, word_idxs, text_tokens]
            token_positions = token_probs[:, word_idxs]

        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens]), (1, 0))

        text_token_probs = text_token_probs[prob_indices, word_idxs].tolist()
        word_probabilities = np.array([
            text_token_probs[j - 1] if is_end_ts else text_token_probs[i]
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ])

        if token_positions is None:
            token_positions = [0] * len(word_tokens)
        else:
            token_positions = token_positions[prob_indices, word_idxs]
            token_positions = token_positions.sort().indices == tokens.unsqueeze(1)
            token_positions = token_positions.nonzero()[:, -1].tolist()
            token_positions = [
                token_positions[j - 1] if is_end_ts else token_positions[i]
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

        return word_probabilities, token_positions

    def update_ts(
            self,
            idx: int,
            is_finish: np.ndarray,
            changes: np.ndarray,
            is_end_ts: bool,
            time_offset: float,
            words: List[WordTiming]
    ):
        if not is_finish[idx] or changes[idx, -1] == -1:
            return
        new_ts = round(time_offset + (float(changes[idx, -1]) / self.sample_rate), 3)
        if changes[idx, 0] and not changes[idx, 1]:
            if is_end_ts:
                if new_ts <= words[idx].end:
                    return
            elif new_ts >= words[idx].start:
                return
        if is_end_ts:
            old_ts = words[idx].end
            words[idx].end = new_ts
        else:
            old_ts = words[idx].start
            words[idx].start = new_ts
        if self.options.progress.verbose and old_ts != new_ts:
            word_info = (f'[Word="{words[idx].word}"] '
                         f'[Segment ID: {words[idx].segment_id}] '
                         f'[Word ID: {words[idx].id}]')
            print(f'{"End" if is_end_ts else "Start"}: {old_ts} -> {new_ts}  {word_info}')

    def _refine(self, result: WhisperResult, step: str):

        total_duration = round(self._audio.shape[-1] / self.sample_rate, 3)

        for words, min_starts, max_ends, edge_mask in self.curr_segments(result, total_duration):

            time_offset = min_starts[0]
            start_sample = round(time_offset * self.sample_rate)
            end_sample = round(max_ends[-1] * self.sample_rate)
            orig_audio_segment = self._audio[start_sample:end_sample + 1].unsqueeze(0)

            max_starts = self.second_to_sample([w.end for w in words], time_offset)
            min_ends = self.second_to_sample([w.start for w in words], time_offset)
            min_starts = self.second_to_sample(min_starts, time_offset)
            max_ends = self.second_to_sample(max_ends, time_offset)

            mid_starts = min_starts + ((max_starts - min_starts) / 2).round().astype(np.int32)
            mid_ends = min_ends + ((max_ends - min_ends) / 2).round().astype(np.int32)

            text_tokens: List[int] = [t for w in words for t in w.tokens]
            word_tokens: List[List[int]] = [w.tokens.copy() for w in words]

            audio_segment = orig_audio_segment.clone().repeat_interleave(2, 0)
            is_end_ts = step == 'e'

            prob_indices = []
            is_finish = np.less([w.probability for w in words], self.prob_threshold)
            is_finish = np.logical_or(is_finish, [w.duration == 0 for w in words])
            if not self.word_level:
                is_finish[edge_mask != (2 if is_end_ts else 1)] = True
            for idx, _i in enumerate(max_starts if is_end_ts else min_ends):
                row = idx % 2
                prob_indices.extend([row] * len(words[idx].tokens))
                if is_finish[idx]:
                    continue
                if is_end_ts:
                    _p = audio_segment.size(-1) if idx == len(words) - 1 else mid_ends[idx + 1]
                    audio_segment[row, _i:_p] = 0
                else:
                    _p = 0 if idx == 0 else mid_starts[idx - 1]
                    audio_segment[row, _p:_i] = 0
            orig_probs, orig_tk_poss = self.get_prob(audio_segment, text_tokens, word_tokens, prob_indices, is_end_ts)
            changes = np.zeros((orig_probs.shape[-1], 3), dtype=np.int32)
            changes[:, -1] = -1
            frame_indices = (mid_ends, max_starts) if is_end_ts else (min_ends, mid_starts)
            for idx, (_s, _e) in enumerate(zip(*frame_indices)):
                row = idx % 2
                if is_finish[idx]:
                    continue
                audio_segment[row, _s:_e] = 0

            new_probs = prev_probs = orig_probs
            while not np.all(is_finish):
                probs, tk_poss = self.get_prob(audio_segment, text_tokens, word_tokens, prob_indices, is_end_ts)
                abs_diffs = orig_probs - probs
                rel_diffs = abs_diffs / orig_probs
                rel_change_diffs = (prev_probs - probs) / prev_probs
                prev_probs = probs
                for idx, (abs_diff, rel_diff, rel_change_diff, prob) \
                        in enumerate(zip(abs_diffs, rel_diffs, rel_change_diffs, probs)):
                    if is_finish[idx]:
                        continue
                    if is_end_ts:
                        curr_min, curr_max, curr_mid = min_ends[idx], max_ends[idx], mid_ends[idx]
                    else:
                        curr_min, curr_max, curr_mid = min_starts[idx], max_starts[idx], mid_starts[idx]

                    row = prob_indices[idx]
                    best_tks_changed = orig_tk_poss[idx] > tk_poss[idx]
                    failed_requirements = (
                            abs_diff > self.abs_prob_decrease or
                            rel_diff > self.rel_prob_decrease or
                            (self.rel_rel_prob_decrease is not None and rel_change_diff > self.rel_rel_prob_decrease) or
                            prob < self.prob_threshold or
                            best_tks_changed
                    )

                    if failed_requirements:
                        changes[idx][0] = 1
                        if is_end_ts:
                            curr_min = curr_mid
                        else:
                            curr_max = curr_mid
                    else:
                        changes[idx][1] = 1
                        if is_end_ts:
                            curr_max = curr_mid
                        else:
                            curr_min = curr_mid

                    if (new_mid_change := round((curr_max - curr_min) / 2)) < self.sample_precision:
                        is_finish[idx] = True
                        self.update_ts(idx, is_finish, changes, is_end_ts, time_offset, words)
                        continue

                    new_mid = curr_min + new_mid_change
                    if failed_requirements:
                        if is_end_ts:
                            audio_segment[row, curr_min:new_mid] = orig_audio_segment[0, curr_min:new_mid]
                        else:
                            audio_segment[row, new_mid:curr_max] = orig_audio_segment[0, new_mid:curr_max]

                    else:
                        if is_end_ts:
                            audio_segment[row, new_mid:curr_max] = 0
                        else:
                            audio_segment[row, curr_min:new_mid] = 0

                    if is_end_ts:
                        min_ends[idx], max_ends[idx], mid_ends[idx] = curr_min, curr_max, new_mid
                    else:
                        min_starts[idx], max_starts[idx], mid_starts[idx] = curr_min, curr_max, new_mid
                    if not best_tks_changed:
                        changes[idx][-1] = new_mid
                    new_probs[idx] = prob

            self.update_pbar(words[-1].end)

    def update_pbar(self, last_ts):
        if self._tqdm_pbar is None:
            return
        if last_ts == self._tqdm_pbar.total:
            new_n = self._pbar_step * self._step_count
        else:
            new_n = ((last_ts - self._prev_ts) / len(self.steps)) + self._tqdm_pbar.n
        self._tqdm_pbar.update(min(round(new_n, 2) - self._tqdm_pbar.n, self._tqdm_pbar.total))
        self._prev_ts = last_ts
        if self.options.progress.progress_callback is not None:
            self.options.progress.progress_callback(self._tqdm_pbar.n, self._tqdm_pbar.total)
