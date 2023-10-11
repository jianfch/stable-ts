import copy
import re
import torch
import numpy as np
from tqdm import tqdm
from typing import TYPE_CHECKING, Union, List, Callable, Optional

from whisper.tokenizer import get_tokenizer
from whisper.utils import format_timestamp, make_safe
from whisper.audio import (
    SAMPLE_RATE, N_FRAMES, N_SAMPLES, N_FFT, pad_or_trim, log_mel_spectrogram, FRAMES_PER_SECOND
)

from .result import WhisperResult
from .timing import add_word_timestamps_stable, split_word_tokens
from .audio import load_audio
from .utils import warn_compatibility_issues

if TYPE_CHECKING:
    from whisper.model import Whisper

__all__ = ['align', 'refine']


def align(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
        text: Union[str, List[int], WhisperResult],
        *,
        language: str = None,
        verbose: str = False,
        regroup: bool = True,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        min_word_dur: bool = 0.1,
        q_levels: int = 20,
        k_size: int = 5,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        demucs: bool = False,
        demucs_output: str = None,
        demucs_options: dict = None,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        progress_callback: Callable = None,
        ignore_compatibility: bool = False,
        remove_instant_words: bool = False,
        token_step: int = 100,
) -> WhisperResult:
    """
    Align plain text with audio at word-level.

    Parameters
    ----------
    text: Union[str, List[int], WhisperResult]
        string of plain-text, list of tokens, or instance of WhisperResult containing words/text

    remove_instant_words: bool
        Whether to truncate any words with zero duration. (Default: False)

    token_step: int
        Max number of tokens to align each pass. (Default: 100)
        If [token_step] is less than 1, than [token_step] will be set to maximum value.
        Use higher values to reduce chance of misalignment.
        Note: maximum value is 442 (model.dims.n_text_ctx - 6).

    Returns
    -------
    An instance of WhisperResult.
    """
    max_token_step = model.dims.n_text_ctx - 6
    if token_step < 1:
        token_step = max_token_step
    elif token_step > max_token_step:
        raise ValueError(f'The max value for [token_step] is {max_token_step} but got {token_step}.')

    warn_compatibility_issues(ignore_compatibility)

    if isinstance(text, WhisperResult):
        if language is None:
            language = text.language
        text = text.all_tokens() if text.has_words else text.text
    elif isinstance(text, str):
        text = re.sub(r'\s', ' ', text)
        if not text.startswith(' '):
            text = ' ' + text
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task='transcribe')
    tokens = tokenizer.encode(text) if isinstance(text, str) else text
    tokens = [t for t in tokens if t < tokenizer.eot]
    _, (words, word_tokens), _ = split_word_tokens([dict(tokens=tokens)], tokenizer)

    if demucs:
        from .audio import demucs_audio, load_demucs_model
        demucs_model = load_demucs_model()
        demucs_kwargs = dict(
            audio=audio,
            output_sr=SAMPLE_RATE,
            model=demucs_model,
            save_path=demucs_output,
            verbose=verbose
        )
        demucs_kwargs.update(demucs_options or {})
        audio = demucs_audio(**demucs_kwargs)
    else:
        audio = torch.from_numpy(load_audio(audio))

    sample_padding = int(N_FFT // 2) + 1
    seek_sample = 0
    total_samples = audio.shape[-1]
    total_tokens = sum(len(wt) for wt in word_tokens)
    finished_tokens = 0

    def get_curr_words():
        nonlocal words, word_tokens
        curr_tk_count = 0
        w, wt = [], []
        for _ in range(len(words)):
            tk_count = len(word_tokens[0])
            if curr_tk_count + tk_count > token_step and w:
                break
            w.append(words.pop(0))
            wt.append(word_tokens.pop(0))
            curr_tk_count += tk_count
        return w, wt
    result = []

    with tqdm(total=total_tokens, unit='token', disable=verbose is not False, desc='Align') as tqdm_pbar:

        def update_pbar(finish: bool = False):
            nonlocal finished_tokens
            if finish:
                finished_tokens = tqdm_pbar.total
            tqdm_pbar.update(finished_tokens - tqdm_pbar.n)
            if progress_callback is not None:
                progress_callback(seek=finished_tokens, total=total_tokens)

        while words and seek_sample < total_samples:
            curr_words, curr_word_tokens = get_curr_words()

            seek_sample_end = seek_sample + N_SAMPLES
            audio_segment = audio[seek_sample:seek_sample_end]
            segment_samples = audio_segment.shape[-1]
            time_offset = seek_sample / SAMPLE_RATE

            mel_segment = log_mel_spectrogram(audio_segment, padding=sample_padding)
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device=model.device)

            segment = dict(
                seek=time_offset,
                tokens=(curr_words, curr_word_tokens)
            )

            add_word_timestamps_stable(
                segments=[segment],
                model=model,
                tokenizer=tokenizer,
                mel=mel_segment,
                num_samples=segment_samples,
                split_callback=(lambda x, _: x),
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations
            )

            break_next = False
            while segment['words']:
                word = segment['words'][-1]
                if break_next or word['end'] - word['start'] == 0:
                    words.insert(0, word['word'])
                    word_tokens.insert(0, word['tokens'])
                    del segment['words'][-1]
                    if break_next:
                        break
                elif words:
                    break_next = True
                else:
                    break

            finished_tokens += sum(len(w['tokens']) for w in segment['words'])
            if segment['words']:
                seek_sample = round(segment['words'][-1]['end'] * SAMPLE_RATE)
            elif seek_sample_end >= total_samples:
                seek_sample = total_samples

            update_pbar()

            result.extend(segment['words'])

            if verbose:
                line = '\n'.join(
                    f"[{format_timestamp(word['start'])}] -> "
                    f"[{format_timestamp(word['end'])}] \"{word['word']}\""
                    for word in segment.get('words', [])
                )
                if line:
                    print(make_safe(line))

        if words and not remove_instant_words:
            total_duration = round(total_samples / SAMPLE_RATE, 3)
            result.extend(
                [
                    dict(word=w, start=total_duration, end=total_duration, probability=0.0, tokens=wt)
                    for w, wt in zip(words, word_tokens)
                ]
            )

        update_pbar(True)

    result = WhisperResult([result])

    if suppress_silence:
        result.adjust_by_silence(
            audio, vad,
            vad_onnx=vad_onnx, vad_threshold=vad_threshold,
            q_levels=q_levels, k_size=k_size,
            sample_rate=SAMPLE_RATE, min_word_dur=min_word_dur,
            word_level=suppress_word_ts, verbose=verbose
        )
    result.regroup(regroup)

    return result


def refine(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
        result: WhisperResult,
        *,
        steps: str = None,
        rel_prob_decrease: float = .07,
        abs_prob_decrease: float = .07,
        rel_rel_prob_decrease: float = .07,
        prob_threshold: float = .5,
        rel_dur_change: Optional[float] = .5,
        abs_dur_change: Optional[float] = .0,
        word_level: bool = True,
        precision: float = None,
        single_batch: bool = False,
        inplace: bool = True,
        verbose: Optional[bool] = False
) -> WhisperResult:
    """

    Iteratively muting portions of the audio and monitoring token probabilities to find the most precise timestamps.
    Note: "most precise" in this case means the latest start and earliest end of a word that
        maintains an acceptable probability determined by the specified arguments.

    Parameters
    ----------
    steps: str
        Instructions for refinement. (Default: 'se')
            's': refine start-timestamps
            'e': refine end-timestamps
            E.g. 'sese' means refine start-timestamps then end-timestamps then repeat both once

    rel_prob_decrease: float
        Maximum percent decrease in probability relative to original probability. (Default: 0.07)
        Note: "original probability" is the probability before any muting.

    abs_prob_decrease: float
        Maximum decrease in probability from original probability. (Default: 0.07)

    rel_rel_prob_decrease: float
        Maximum percent decrease in probability relative to previous probability. (Default: 0.07)
        Note: "previous probability" is the probability from previous iteration of muting.

    prob_threshold: float
        Stop refining the timestamp if the probability of its token goes below this value. (Default: 0.5)

    rel_dur_change: Optional[float]
        Maximum percent change in duration of a word relative to its original duration. (Default: 0.5)

    abs_dur_change: Optional[float]
        Maximum seconds a word is allowed deviate from its original duration. (Default: None)

    word_level: bool
        Whether to refine timestamps on word-level. (Default: True)
        If False, only refine start/end timestamps of each segment.

    precision: float
        Precision of refined timestamps in seconds. (Default: 0.1)
        Note: the lower the precision, the longer the processing time (lowest precision is 0.02 second).

    single_batch: bool
        Whether to process in only batch size of one to reduce memory usage. (Default: False)

    inplace: bool
        Whether to alter timestamps in-place. (Default: True)

    Returns
    -------
    An instance of WhisperResult.
        -if [inplace]=True, returns same object as [result]
        -if [inplace]=False, returns deepcopy of [result]

    """
    if not steps:
        steps = 'se'
    if precision is None:
        precision = 0.1
    if invalid_steps := steps.replace('s', '').replace('e', ''):
        raise ValueError(f'Invalid step(s): {", ".join(invalid_steps)}')
    if not result.has_words:
        raise NotImplementedError(f'Result must have word timestamps.')

    if not inplace:
        result = copy.deepcopy(result)

    max_inference_tokens = model.dims.n_text_ctx - 6
    audio = torch.from_numpy(load_audio(audio))
    sample_padding = int(N_FFT // 2) + 1
    frame_precision = max(round(precision * FRAMES_PER_SECOND), 2)
    total_duration = round(audio.shape[-1] / SAMPLE_RATE, 3)
    tokenizer = get_tokenizer(model.is_multilingual, language=result.language, task='transcribe')

    def ts_to_frames(timestamps: Union[np.ndarray, list]) -> np.ndarray:
        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)
        return (timestamps * FRAMES_PER_SECOND).round().astype(int)

    def curr_segments():
        all_words = result.all_words()
        seg_edge_mask = np.array([
            1 if _i == 0 else (2 if _i == len(seg.words)-1 else 0)
            for seg in result.segments
            for _i, w in enumerate(seg.words)
        ])
        start_times = [
            max(
                0 if abs_dur_change is None else (w.start - abs_dur_change),
                0 if rel_dur_change is None else (w.start - w.duration * rel_dur_change),
                0 if i == 0 else max(all_words[i - 1].end, w.end - 14.5, 0)
            )
            for i, w in enumerate(all_words)
        ]
        end_times = [
            min(
                total_duration if abs_dur_change is None else (w.end + abs_dur_change),
                total_duration if rel_dur_change is None else (w.end + w.duration * rel_dur_change),
                total_duration if i == len(all_words) else min(all_words[i].start, w.start + 14.5, total_duration)
            )
            for i, w in enumerate(all_words, 1)
        ]
        start = start_times[0]

        prev_i = 0
        curr_words, curr_starts, curr_ends = [], [], []

        for i, w in enumerate(all_words, 1):
            if (
                    (end_times[0] - start > 30) or
                    (len(curr_words) + 1 > max_inference_tokens)
            ):
                if curr_words:
                    yield curr_words, curr_starts, curr_ends, seg_edge_mask[prev_i:prev_i+len(curr_words)]
                    curr_words, curr_starts, curr_ends = [], [], []
                start = start_times[0]
                prev_i = i - 1

            curr_words.append(w)
            curr_starts.append(start_times.pop(0))
            curr_ends.append(end_times.pop(0))

            if i == len(all_words):
                yield curr_words, curr_starts, curr_ends, seg_edge_mask[prev_i:prev_i+len(curr_words)]

    def _refine(_step: str):

        for words, min_starts, max_ends, edge_mask in curr_segments():

            time_offset = min_starts[0]
            start_sample = round(time_offset * SAMPLE_RATE)
            end_sample = round(max_ends[-1] * SAMPLE_RATE)
            audio_segment = audio[start_sample:end_sample + 1].unsqueeze(0)

            max_starts = ts_to_frames(np.array([w.end for w in words]) - time_offset)
            min_ends = ts_to_frames(np.array([w.start for w in words]) - time_offset)
            min_starts = ts_to_frames(np.array(min_starts) - time_offset)
            max_ends = ts_to_frames(np.array(max_ends) - time_offset)

            mid_starts = min_starts + ((max_starts - min_starts) / 2).round().astype(int)
            mid_ends = min_ends + ((max_ends - min_ends) / 2).round().astype(int)

            text_tokens = [t for w in words for t in w.tokens if t < tokenizer.eot]
            word_tokens = [[t for t in w.tokens if t < tokenizer.eot] for w in words]
            orig_mel_segment = log_mel_spectrogram(audio_segment, padding=sample_padding)
            orig_mel_segment = pad_or_trim(orig_mel_segment, N_FRAMES).to(device=model.device)

            def get_prob():

                tokens = torch.tensor(
                    [
                        *tokenizer.sot_sequence,
                        tokenizer.no_timestamps,
                        *text_tokens,
                        tokenizer.eot,
                    ]
                ).to(model.device)

                with torch.no_grad():
                    curr_mel_segment = mel_segment if prob_indices else orig_mel_segment
                    if single_batch:
                        logits = torch.cat(
                            [model(_mel.unsqueeze(0), tokens.unsqueeze(0)) for _mel in curr_mel_segment]
                        )
                    else:
                        logits = model(curr_mel_segment, tokens.unsqueeze(0))

                sampled_logits = logits[:, len(tokenizer.sot_sequence):, : tokenizer.eot]
                token_probs = sampled_logits.softmax(dim=-1)

                text_token_probs = token_probs[:, np.arange(len(text_tokens)), text_tokens]
                token_positions = token_probs[:, np.arange(len(text_tokens))]
                if logits.shape[0] != 1 and prob_indices is not None:
                    indices1 = np.arange(len(prob_indices))
                    text_token_probs = text_token_probs[prob_indices, indices1]
                    token_positions = token_positions[prob_indices, indices1]
                else:
                    text_token_probs.squeeze_(0)

                text_token_probs = text_token_probs.tolist()
                token_positions = \
                    (
                            token_positions.sort().indices == tokens[len(tokenizer.sot_sequence) + 1:-1][:, None]
                    ).nonzero()[:, -1].tolist()

                word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens]), (1, 0))
                word_probabilities = np.array([
                    text_token_probs[j-1] if is_end_ts else text_token_probs[i]
                    for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
                ])
                token_positions = [
                    token_positions[j-1] if is_end_ts else token_positions[i]
                    for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
                ]

                return word_probabilities, token_positions

            def update_ts():
                if not is_finish[idx] or changes[idx, -1] == -1:
                    return
                new_ts = round(time_offset + (changes[idx, -1] / FRAMES_PER_SECOND), 3)
                if changes[idx, 0] and not changes[idx, 1]:
                    if is_end_ts:
                        if new_ts <= words[idx].end:
                            return
                    elif new_ts >= words[idx].start:
                        return
                if not verbose:
                    return
                curr_word = words[idx]
                word_info = (f'[Word="{curr_word.word}"] '
                             f'[Segment ID: {curr_word.segment_id}] '
                             f'[Word ID: {curr_word.id}]')
                if is_end_ts:
                    print(f'End: {words[idx].end} -> {new_ts}  {word_info}')
                    words[idx].end = new_ts
                else:
                    print(f'Start: {words[idx].start} -> {new_ts}  {word_info}')
                    words[idx].start = new_ts

            mel_segment = orig_mel_segment.clone().repeat_interleave(2, 0)
            is_end_ts = _step == 'e'

            prob_indices = []
            is_finish = np.less([w.probability for w in words], prob_threshold)
            is_finish = np.logical_or(is_finish, [w.duration == 0 for w in words])
            if not word_level:
                is_finish[edge_mask != (2 if is_end_ts else 1)] = True

            orig_probs, orig_tk_poss = get_prob()
            changes = np.zeros((orig_probs.shape[-1], 3), dtype=int)
            changes[:, -1] = -1

            frame_indices = (mid_ends, max_ends) if is_end_ts else (min_starts, mid_starts)
            for idx, (_s, _e) in enumerate(zip(*frame_indices)):
                row = idx % 2
                prob_indices.extend([row] * len(words[idx].tokens))
                if is_finish[idx]:
                    continue

                half_dur = words[idx].duration / 2
                if is_end_ts:
                    _p = mel_segment.shape[-1] if idx == len(words)-1 else round(_e+half_dur)
                    mel_segment[row, :, _s:_p] = 0
                else:
                    _p = 0 if idx == 0 else max(round(_s-half_dur), 0)
                    mel_segment[row, :, _p:_e] = 0

            new_probs = prev_probs = orig_probs
            while not np.all(is_finish):
                probs, tk_poss = get_prob()
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
                            abs_diff > abs_prob_decrease or
                            rel_diff > rel_prob_decrease or
                            rel_change_diff > rel_rel_prob_decrease or
                            prob < prob_threshold or
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

                    if (new_mid_change := round((curr_max - curr_min) / 2)) < frame_precision:
                        is_finish[idx] = True
                        update_ts()
                        continue

                    new_mid = curr_min + new_mid_change
                    if failed_requirements:
                        if is_end_ts:
                            mel_segment[row, :, curr_min:new_mid] = orig_mel_segment[0, :, curr_min:new_mid]
                        else:
                            mel_segment[row, :, new_mid:curr_max] = orig_mel_segment[0, :, new_mid:curr_max]

                    else:
                        if is_end_ts:
                            mel_segment[row, :, new_mid:curr_max] = 0
                        else:
                            mel_segment[row, :, curr_min:new_mid] = 0

                    if is_end_ts:
                        min_ends[idx], max_ends[idx], mid_ends[idx] = curr_min, curr_max, new_mid
                    else:
                        min_starts[idx], max_starts[idx], mid_starts[idx] = curr_min, curr_max, new_mid
                    if not failed_requirements:
                        changes[idx][-1] = new_mid
                    new_probs[idx] = prob

            update_pbar(words[-1].end)

    with tqdm(total=round(total_duration, 2), unit='sec', disable=verbose is not False, desc='Refine') as tqdm_pbar:

        def update_pbar(last_ts: float):
            nonlocal prev_ts
            tqdm_pbar.update(round(((last_ts - prev_ts) / len(steps)), 2))
            prev_ts = last_ts

        for step_count, step in enumerate(steps, 1):
            prev_ts = 0
            _refine(step)
            update_pbar(round(tqdm_pbar.total // len(step), 2))
        tqdm_pbar.update(tqdm_pbar.total - tqdm_pbar.n)

    result.update_all_segs_with_words()

    return result
