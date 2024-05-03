import copy
import re
import warnings

import torch
import numpy as np
from tqdm import tqdm
from typing import TYPE_CHECKING, Union, List, Callable, Optional, Tuple

from .result import WhisperResult, Segment
from .timing import add_word_timestamps_stable, split_word_tokens
from .audio import prep_audio, AudioLoader, audioloader_not_supported, convert_demucs_kwargs
from .utils import safe_print, format_timestamp
from .whisper_compatibility import warn_compatibility_issues, get_tokenizer
from .stabilization import NonSpeechPredictor
from .default import get_min_word_dur, get_prepend_punctuations, get_append_punctuations

from .whisper_compatibility import (
    SAMPLE_RATE, N_FRAMES, N_FFT, pad_or_trim, log_mel_spectrogram, FRAMES_PER_SECOND, CHUNK_LENGTH, N_SAMPLES,
    median_filter, DecodingTask, DecodingOptions, SuppressTokens, whisper, TOKENS_PER_SECOND
)

if TYPE_CHECKING:
    from .whisper_compatibility import Whisper
    from .whisper_compatibility import Tokenizer

__all__ = ['align', 'refine', 'locate']


def align(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
        text: Union[str, List[int], WhisperResult],
        language: str = None,
        *,
        verbose: Optional[bool] = False,
        regroup: bool = True,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        suppress_attention: bool = False,
        use_word_position: bool = True,
        min_word_dur: Optional[float] = None,
        min_silence_dur: Optional[float] = None,
        nonspeech_error: float = 0.1,
        q_levels: int = 20,
        k_size: int = 5,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: Union[bool, torch.nn.Module] = False,
        demucs_options: dict = None,
        only_voice_freq: bool = False,
        prepend_punctuations: Optional[str] = None,
        append_punctuations: Optional[str] = None,
        progress_callback: Callable = None,
        ignore_compatibility: bool = False,
        remove_instant_words: bool = False,
        token_step: int = 100,
        original_split: bool = False,
        word_dur_factor: Optional[float] = 2.0,
        max_word_dur: Optional[float] = 3.0,
        nonspeech_skip: Optional[float] = 5.0,
        fast_mode: bool = False,
        tokenizer: "Tokenizer" = None,
        stream: Optional[bool] = None,
        failure_threshold: Optional[float] = None,
        extra_models: Optional[List["Whisper"]] = None,
        presplit: Union[bool, List[str]] = True,
        gap_padding: str = ' ...'
) -> Union[WhisperResult, None]:
    """
    Align plain text or tokens with audio at word-level.

    Since this is significantly faster than transcribing, it is a more efficient method for testing various settings
    without re-transcribing. This is also useful for timing a more correct transcript than one that Whisper can produce.

    Parameters
    ----------
    model : "Whisper"
        The Whisper ASR model modified instance
    audio : str or numpy.ndarray or torch.Tensor or bytes or AudioLoader
        Path/URL to the audio file, the audio waveform, or bytes of audio file or
        instance of :class:`stable_whisper.audio.AudioLoader`.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    text : str or list of int or stable_whisper.result.WhisperResult
        String of plain-text, list of tokens, or instance of :class:`stable_whisper.result.WhisperResult`.
    language : str, default None, uses ``language`` in ``text`` if it is a :class:`stable_whisper.result.WhisperResult`
        Language of ``text``. Required if ``text`` does not contain ``language``.
    remove_instant_words : bool, default False
        Whether to truncate any words with zero duration.
    token_step : int, default 100
        Max number of tokens to align each pass. Use higher values to reduce chance of misalignment.
    original_split : bool, default False
        Whether to preserve the original segment groupings. Segments are spit by line break if ``text`` is plain-text.
    max_word_dur : float or None, default 3.0
        Global maximum word duration in seconds. Re-align words that exceed the global maximum word duration.
    word_dur_factor : float or None, default 2.0
        Factor to compute the Local maximum word duration, which is ``word_dur_factor`` * local medium word duration.
        Words that need re-alignment, are re-algined with duration <= local/global maximum word duration.
    nonspeech_skip : float or None, default 5.0
        Skip non-speech sections that are equal or longer than this duration in seconds. Disable skipping if ``None``.
    fast_mode : bool, default False
        Whether to speed up alignment by re-alignment with local/global maximum word duration.
        ``True`` tends produce better timestamps when ``text`` is accurate and there are no large speechless gaps.
    tokenizer : "Tokenizer", default None, meaning a new tokenizer is created according ``language`` and ``model``
        A tokenizer to used tokenizer text and detokenize tokens.
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
        Whether to adjust word timestamps based on the detected silence. Only enabled if ``suppress_silence = True``.
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
    vad : bool, default False
        Whether to use Silero VAD to generate timestamp suppression mask.
        Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
    vad_threshold : float, default 0.35
        Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
    vad_onnx : bool, default False
        Whether to use ONNX for Silero VAD.
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
    ignore_compatibility : bool, default False
        Whether to ignore warnings for compatibility issues with the detected Whisper version.
    extra_models : list of whisper.model.Whisper, optional
        List of additional Whisper model instances to use for computing word-timestamps along with ``model``.
    presplit : bool or list of str, default True meaning ['.', '。', '?', '？']
        List of ending punctuation used to split ``text`` into segments for applying ``gap_padding``,
        but segmentation of final output is unnaffected unless ``original_split=True``.
        If ``original_split=True``, the original split is used instead of split from ``presplit``.
        Ignored if ``model`` is a faster-whisper model.
    gap_padding : str, default ' ...'
        Only if ``presplit=True``, ``gap_padding`` is prepended to each segments for word timing alignment.
        Used to reduce the probability of model predicting timestamps earlier than the first utterance.
        Ignored if ``model`` is a faster-whisper model.

    Returns
    -------
    stable_whisper.result.WhisperResult or None
        All timestamps, words, probabilities, and other data from the alignment of ``audio``. Return None if alignment
        fails and ``remove_instant_words = True``.

    Notes
    -----
    If ``token_step`` is less than 1, ``token_step`` will be set to its maximum value, 442. This value is computed with
    ``whisper.model.Whisper.dims.n_text_ctx`` - 6.

    IF ``original_split = True`` and a line break is found in middle of a word in ``text``, the split will occur after
    that word.

    ``regroup`` is ignored if ``original_split = True``.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.align('helloworld.mp3', 'Hello, World!', 'English')
    >>> result.to_srt_vtt('helloword.srt')
    Saved 'helloworld.srt'
    """
    if suppress_attention:
        warnings.warn('``suppress_attention`` is deprecated and will be removed in future versions',
                      stacklevel=2)
    denoiser, denoiser_options = convert_demucs_kwargs(
        denoiser, denoiser_options, demucs=demucs, demucs_options=demucs_options
    )
    prepend_punctuations = get_prepend_punctuations(prepend_punctuations)
    append_punctuations = get_append_punctuations(append_punctuations)
    min_word_dur = get_min_word_dur(min_word_dur)
    if failure_threshold is not None and (failure_threshold < 0 or failure_threshold > 1):
        raise ValueError(f'``failure_threshold`` ({failure_threshold}) must be between 0 and 1.')
    is_faster_model = model.__module__.startswith('faster_whisper.')
    if not is_faster_model:
        warn_compatibility_issues(whisper, ignore_compatibility)
    max_token_step = (model.max_length if is_faster_model else model.dims.n_text_ctx) - 6
    if token_step < 1:
        token_step = max_token_step
    elif token_step > max_token_step:
        raise ValueError(f'The max value for [token_step] is {max_token_step} but got {token_step}.')

    split_indices_by_char = []
    if isinstance(text, WhisperResult):
        if language is None:
            language = text.language
        if original_split and len(text.segments) > 1 and text.has_words:
            split_indices_by_char = np.cumsum([sum(len(w.word) for w in seg.words) for seg in text.segments])
        text = text.all_tokens() if text.has_words and all(w.tokens for w in text.all_words()) else text.text
    elif isinstance(text, str):
        if original_split and '\n' in text:
            text_split = [
                ' '+norm_line
                for line in text.splitlines()
                if (norm_line := re.sub(r'\s', ' ', line).strip())
            ]
            split_indices_by_char = np.cumsum([len(seg) for seg in text_split])
            text = ''.join(seg for seg in text_split)
        else:
            text = re.sub(r'\s', ' ', text)
            if not text.startswith(' '):
                text = ' ' + text
    if language is None:
        raise TypeError('expected argument for language')
    if tokenizer is None:
        tokenizer = get_tokenizer(model, is_faster_model=is_faster_model, language=language, task='transcribe')
    tokens = tokenizer.encode(text) if isinstance(text, str) else text
    tokens = [t for t in tokens if t < tokenizer.eot]
    _, (words, word_tokens), _ = split_word_tokens([dict(tokens=tokens)], tokenizer)
    pad_mask = None
    if is_faster_model:
        presplit = False
    if presplit:
        if not isinstance(presplit, List):
            presplit = ['.', '。', '?', '？']
        if len(split_indices_by_char):
            pad_mask = []
            cumsums = split_indices_by_char.tolist()
            cumsum_len = 0
            for word in words:
                cumsum_len += len(word)
                if cumsums and cumsum_len >= cumsums[0]:
                    cumsums.pop(0)
                    pad_mask.append(True)
                else:
                    pad_mask.append(False)
        else:
            pad_mask = [any(map(w.endswith, presplit)) for w in words]

    if isinstance(audio, AudioLoader):
        audio.validate_external_args(
            sr=SAMPLE_RATE,
            vad=vad,
            stream=stream,
            denoiser=denoiser,
            denoiser_options=denoiser_options,
            only_voice_freq=only_voice_freq
        )
    else:
        audio = AudioLoader(
            audio,
            sr=SAMPLE_RATE,
            denoiser=denoiser,
            denoiser_options=denoiser_options,
            only_voice_freq=only_voice_freq,
            verbose=verbose,
            new_chunk_divisor=512,
            stream=stream
        )

    initial_duration = audio.get_duration(2)

    seek_sample = 0
    total_words = len(words)

    if is_faster_model:

        def timestamp_words():
            temp_segment = dict(
                seek=0,
                start=0.0,
                end=round(segment_samples / model.feature_extractor.sampling_rate, 3),
                tokens=[t for wt in curr_word_tokens for t in wt],
            )
            features = model.feature_extractor(audio_segment.cpu().numpy())
            encoder_output = model.encode(features[:, : model.feature_extractor.nb_max_frames])

            model.add_word_timestamps(
                segments=[temp_segment],
                tokenizer=tokenizer,
                encoder_output=encoder_output,
                num_frames=round(segment_samples / model.feature_extractor.hop_length),
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                last_speech_timestamp=temp_segment['start'],
            )

            cumsum_lens = np.cumsum([len(w) for w in curr_words]).tolist()
            final_cumsum_lens = np.cumsum([len(w['word']) for w in temp_segment['words']]).tolist()

            assert not (set(final_cumsum_lens) - set(cumsum_lens)), 'word mismatch'
            prev_l_idx = 0
            for w_idx, cs_len in enumerate(final_cumsum_lens):
                temp_segment['words'][w_idx]['start'] = round(temp_segment['words'][w_idx]['start'] + time_offset, 3)
                temp_segment['words'][w_idx]['end'] = round(temp_segment['words'][w_idx]['end'] + time_offset, 3)
                l_idx = cumsum_lens.index(cs_len)+1
                temp_segment['words'][w_idx]['tokens'] = [t for wt in curr_word_tokens[prev_l_idx:l_idx] for t in wt]
                prev_l_idx = l_idx

            return temp_segment

    else:
        def timestamp_words():
            if curr_split_indices:
                temp_split_indices = [0] + curr_split_indices
                if temp_split_indices[-1] < len(curr_words):
                    temp_split_indices.append(len(curr_words))
                temp_segments = [
                    dict(
                        seek=time_offset,
                        tokens=(curr_words[i:j], curr_word_tokens[i:j])
                    )
                    for i, j in zip(temp_split_indices[:-1], temp_split_indices[1:])
                ]
            else:
                temp_segments = [dict(seek=time_offset, tokens=(curr_words, curr_word_tokens))]
            sample_padding = max(N_SAMPLES - audio_segment.shape[-1], 0)
            mel_segment = log_mel_spectrogram(audio_segment, model.dims.n_mels, padding=sample_padding)
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device=model.device)

            add_word_timestamps_stable(
                segments=temp_segments,
                model=model,
                tokenizer=tokenizer,
                mel=mel_segment,
                num_samples=segment_samples,
                split_callback=(lambda x, _: x),
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                gap_padding=gap_padding if presplit else None,
                extra_models=extra_models,
            )
            if len(temp_segments) == 1:
                return temp_segments[0]
            return dict(words=[w for seg in temp_segments for w in seg['words']])

    def get_curr_words():
        nonlocal words, word_tokens, pad_mask
        curr_tk_count = 0
        w, wt, m = [], [], []
        for i in range(len(words)):
            tk_count = len(word_tokens[0])
            m_count = 1 if pad_mask and pad_mask[0] else 0
            if curr_tk_count + len(m) + tk_count + m_count > token_step and w:
                break
            w.append(words.pop(0))
            wt.append(word_tokens.pop(0))
            curr_tk_count += tk_count
            if pad_mask and pad_mask.pop(0):
                m.append(i+1)
        return w, wt, m
    result = []

    nonspeech_predictor = NonSpeechPredictor(
        vad=vad if suppress_silence else None,
        mask_pad_func=pad_or_trim,
        get_mask=True,
        min_word_dur=min_word_dur,
        q_levels=q_levels,
        k_size=k_size,
        vad_threshold=vad_threshold,
        vad_onnx=vad_onnx,
        vad_window=audio.new_chunk_divisor,
        sampling_rate=SAMPLE_RATE,
        verbose=None if audio.stream else verbose,
        store_timings=True,
        ignore_is_silent=True,
        min_silence_dur=min_silence_dur
    )
    audio.update_post_prep_callback(nonspeech_predictor.get_on_prep_callback(audio.stream))
    failure_count, max_fail = 0, total_words * (failure_threshold or 1)
    all_punctuations = prepend_punctuations + append_punctuations

    def fix_temp_words(target_word: dict, word_sources: List[dict], second_target: Optional[dict] = None):
        first_word_src = word_sources[0]
        assert target_word['word'].startswith(first_word_src['word'])
        if target_word['word'] != first_word_src['word']:
            if len(word_sources) < 2:
                return None, []
            first_word_src['probability'] = [first_word_src['probability']]
            if first_word_src['word'].strip() in all_punctuations:
                first_word_src['start'], first_word_src['end'] = word_sources[1]['start'], word_sources[1]['end']
            for _ in range(len(word_sources)-1):
                tw = word_sources.pop(1)
                fullword = first_word_src['word'] + tw['word']
                assert target_word['word'].startswith(fullword)
                first_word_src['word'] = fullword
                first_word_src['tokens'] += tw['tokens']
                first_word_src['probability'].append(tw['probability'])
                if tw['word'].strip() not in all_punctuations:
                    first_word_src['end'] = tw['end']
                if target_word['word'] == first_word_src['word']:
                    break
            if target_word['word'] != first_word_src['word']:
                return None, []
            if isinstance(first_word_src['probability'], list):
                first_word_src['probability'] = np.mean(first_word_src['probability']).item()
        elif second_target:
            if len(word_sources) == 1:
                return first_word_src, []
            second_word_src, word_sources = fix_temp_words(second_target, word_sources[1:])
            if second_word_src is not None:
                word_sources = [second_word_src] + word_sources
            return first_word_src, word_sources

        return first_word_src, word_sources[1:]

    def speech_percentage(_word: dict, _mask: torch.Tensor, _offset: float):
        s, e = _word['start'], _word['end']
        s = int((s - _offset) * TOKENS_PER_SECOND)
        e = int((e - _offset) * TOKENS_PER_SECOND)
        return 1 - _mask[s:e].float().mean().nan_to_num()

    def is_new_better(w0, m0, o0, w1, m1, o1):
        speech0 = speech_percentage(w0, m0, o0).round(decimals=1)
        speech1 = speech_percentage(w1, m1, o1).round(decimals=1)
        return speech0 >= speech1 or w0['probability'] >= w1['probability']

    with tqdm(total=initial_duration, unit='sec', disable=verbose is not False, desc='Align') as tqdm_pbar:

        def update_pbar(finish: bool = False):
            curr_total = audio.get_duration(2)
            if need_refresh := curr_total != tqdm_pbar.total:
                tqdm_pbar.total = curr_total
            tqdm_pbar.update((curr_total if finish else min(round(last_ts, 2), curr_total)) - tqdm_pbar.n)
            if need_refresh:
                tqdm_pbar.refresh()
            if progress_callback is not None:
                progress_callback(seek=tqdm_pbar.n, total=tqdm_pbar.total)

        def update_curr_words():
            if temp_data['temp_word'] is not None:
                temp_words = [temp_data['temp_word']] + temp_data['extra_words'][:len(curr_words) - 1]
                curr_words[:len(temp_words)] = temp_words
                temp_data['temp_word'] = None

        def redo_words(_idx: int = None):
            nonlocal seg_words, seg_tokens, seg_words, words, word_tokens, curr_words, temp_data
            if _idx is not None and curr_words and temp_data['temp_word'] is not None:
                temp_data['temp_word'], temp_data['extra_words'] = fix_temp_words(
                    curr_words[0],
                    [temp_data['temp_word']] + temp_data['extra_words'],
                    curr_words[1] if len(curr_words) > 1 else None
                )

                if temp_data['temp_word']:
                    use_new = is_new_better(
                        curr_words[0], nonspeech_preds['mask'], time_offset,
                        temp_data['temp_word'], temp_data['temp_mask'], temp_data['temp_offset']
                    )
                    new_extra_words = []
                    if use_new:
                        temp_data['temp_word'] = curr_words[0]
                    else:
                        for wi, (cw, tw) in enumerate(zip(curr_words[1:], temp_data['extra_words'])):
                            assert cw['word'].startswith(tw['word'])
                            use_new = is_new_better(
                                cw, nonspeech_preds['mask'], time_offset,
                                tw, temp_data['temp_mask'], temp_data['temp_offset']
                            )
                            if use_new or cw['word'] != tw['word'] or cw['end'] < tw['end']:
                                break
                            new_extra_words.append(tw)
                    temp_data['extra_words'] = new_extra_words

            if _idx is None:  # redo all
                words = seg_words + words
                word_tokens = seg_tokens + word_tokens
                curr_words = []
            elif _idx != len(seg_words):  # redo from _idx
                words = seg_words[_idx:] + words
                word_tokens = seg_tokens[_idx:] + word_tokens
                curr_words, new_extra_words = curr_words[:_idx], curr_words[_idx:]
                if curr_words:
                    update_curr_words()
                    words = seg_words[_idx-1:_idx] + words
                    word_tokens = seg_tokens[_idx-1:_idx] + word_tokens
                    temp_data['temp_word'] = curr_words.pop(-1)
                    temp_data['extra_words'] = new_extra_words
                    temp_data['temp_mask'] = nonspeech_preds['mask']
                    temp_data['temp_offset'] = time_offset
            else:
                update_curr_words()

        temp_data: dict = dict(temp_word=None)

        while words:

            time_offset = seek_sample / SAMPLE_RATE
            audio_segment = audio.next_chunk(seek_sample, N_SAMPLES)
            if audio_segment is None:
                break

            segment_samples = audio_segment.shape[-1]
            curr_total_samples = audio.get_total_samples()

            nonspeech_preds = nonspeech_predictor.predict(audio=audio_segment, offset=time_offset)

            if nonspeech_skip is not None:
                segment_nonspeech_timings = nonspeech_preds['timings']

                if segment_nonspeech_timings is not None and len(segment_nonspeech_timings[0]):

                    if (
                            (segment_nonspeech_timings[0][0] <= time_offset + min_word_dur) and
                            (segment_nonspeech_timings[1][0] >= time_offset + segment_samples - min_word_dur)
                    ):
                        seek_sample += segment_samples
                        continue

                    timing_indices = (segment_nonspeech_timings[1] - segment_nonspeech_timings[0]) >= nonspeech_skip
                    if timing_indices.any():
                        nonspeech_starts = segment_nonspeech_timings[0][timing_indices]
                        nonspeech_ends = segment_nonspeech_timings[1][timing_indices]

                        if nonspeech_ends[0] > round(time_offset, 3) >= nonspeech_starts[0]:
                            seek_sample = round(nonspeech_ends[0] * SAMPLE_RATE)
                            if seek_sample + (min_word_dur * SAMPLE_RATE) >= curr_total_samples:
                                seek_sample = curr_total_samples
                                continue
                            time_offset = seek_sample / SAMPLE_RATE

                            audio_segment = audio.next_chunk(seek_sample, N_SAMPLES)
                            if audio_segment is None:
                                break
                            nonspeech_preds = nonspeech_predictor.predict(audio=audio_segment, offset=time_offset)
                            if len(nonspeech_starts) > 1:
                                new_sample_count = round((nonspeech_starts[1] - nonspeech_ends[0]) * SAMPLE_RATE)
                            else:
                                new_sample_count = None
                            audio_segment = audio_segment[:new_sample_count]
                            segment_samples = audio_segment.shape[-1]

            curr_words, curr_word_tokens, curr_split_indices = get_curr_words()

            segment = timestamp_words()
            curr_words = segment['words']
            seg_words = [w['word'] for w in curr_words]
            seg_tokens = [w['tokens'] for w in curr_words]
            durations = np.array([w['end'] - w['start'] for w in curr_words]).round(3)
            nonzero_mask = durations > 0
            nonzero_indices = np.flatnonzero(nonzero_mask)
            if len(nonzero_indices):
                redo_index = nonzero_indices[-1] + 1
                if (
                        words and
                        len(nonzero_indices) > 1 and
                        curr_words[nonzero_indices[-1]]['end'] >= np.floor(time_offset + segment_samples / SAMPLE_RATE)
                ):
                    nonzero_mask[nonzero_indices[-1]] = False
                    nonzero_indices = nonzero_indices[:-1]
                    redo_index = nonzero_indices[-1] + 1
                med_dur = np.median(durations[:redo_index])

                if fast_mode:
                    new_start = None
                    global_max_dur = None
                else:
                    local_max_dur = round(med_dur * word_dur_factor, 3) if word_dur_factor else None
                    if max_word_dur:
                        local_max_dur = min(local_max_dur, max_word_dur) if local_max_dur else max_word_dur
                        global_max_dur = max_word_dur
                    else:
                        global_max_dur = local_max_dur or None
                    if global_max_dur and med_dur > global_max_dur:
                        med_dur = global_max_dur
                    if (
                            local_max_dur and durations[nonzero_indices[0]] > global_max_dur
                    ):
                        new_start = round(max(
                            curr_words[nonzero_indices[0]]['end'] - (med_dur * nonzero_indices[0] + local_max_dur),
                            curr_words[nonzero_indices[0]]['start']
                        ), 3)
                        if new_start <= time_offset:
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
                    last_ts = curr_words[redo_index - 1]['end']
                    redo_words(redo_index)
                else:
                    last_ts = new_start
                    redo_words()
                seek_sample = round(last_ts * SAMPLE_RATE)
            else:
                seek_sample += segment_samples
                last_ts = round(seek_sample / SAMPLE_RATE, 2)
                redo_words()

            update_pbar()

            result.extend(curr_words)

            if verbose:
                line = '\n'.join(
                    f"[{format_timestamp(word['start'])}] -> "
                    f"[{format_timestamp(word['end'])}] \"{word['word']}\""
                    for word in curr_words
                )
                safe_print(line)

            if failure_threshold is not None:
                failure_count += sum(1 for w in curr_words if w['end'] - w['start'] == 0)
                if failure_count > max_fail:
                    break

        update_pbar(failure_count <= max_fail)

    if temp_data.get('temp_word') is not None:
        result.append(temp_data['temp_word'])
    if not result:
        warnings.warn('Failed to align text.', stacklevel=2)
    if failure_count > max_fail:
        warnings.warn(f'Alignment aborted. Failed word percentage exceeded {failure_threshold * 100}% at '
                      f'{format_timestamp(seek_sample / SAMPLE_RATE)}.',
                      stacklevel=2)
    elif words:
        warnings.warn(f'Failed to align the last {len(words)}/{total_words} words after '
                      f'{format_timestamp(result[-1]["end"])}.', stacklevel=2)

    if words and not remove_instant_words:
        final_total_duration = audio.get_duration(3)
        result.extend(
            [
                dict(word=w, start=final_total_duration, end=final_total_duration, probability=0.0, tokens=wt)
                for w, wt in zip(words, word_tokens)
            ]
        )

    audio.terminate()
    nonspeech_predictor.finalize_timings()

    if not result:
        return

    if len(split_indices_by_char):
        word_lens = np.cumsum([[len(w['word']) for w in result]])
        split_indices = [np.flatnonzero(word_lens >= i)[0]+1 for i in split_indices_by_char]
        result = WhisperResult([result[i:j] for i, j in zip([0]+split_indices[:-1], split_indices) if i != j])
    else:
        result = WhisperResult([result])

    if suppress_silence and (nonspeech_timings := nonspeech_predictor.nonspeech_timings) is not None:
        result.suppress_silence(
            *nonspeech_timings,
            min_word_dur=min_word_dur,
            word_level=suppress_word_ts,
            nonspeech_error=nonspeech_error,
            use_word_position=use_word_position,
            verbose=verbose is not None
        )
        result.update_nonspeech_sections(*nonspeech_timings)
    if not original_split:
        result.regroup(regroup)

    if fail_segs := len([None for s in result.segments if s.end-s.start <= 0]):
        warnings.warn(f'{fail_segs}/{len(result.segments)} segments failed to align.', stacklevel=2)

    return result


def refine(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
        result: WhisperResult,
        *,
        steps: str = None,
        rel_prob_decrease: float = .03,
        abs_prob_decrease: float = .05,
        rel_rel_prob_decrease: Optional[float] = None,
        prob_threshold: float = .5,
        rel_dur_change: Optional[float] = .5,
        abs_dur_change: Optional[float] = None,
        word_level: bool = True,
        precision: float = None,
        single_batch: bool = False,
        inplace: bool = True,
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: Union[bool, torch.nn.Module] = False,
        demucs_options: dict = None,
        only_voice_freq: bool = False,
        verbose: Optional[bool] = False
) -> WhisperResult:
    """
    Improve existing timestamps.

    This function iteratively muting portions of the audio and monitoring token probabilities to find the most precise
    timestamps. This "most precise" in this case means the latest start and earliest end of a word that maintains an
    acceptable probability determined by the specified arguments.

    This is useful readjusting timestamps when they start too early or end too late.

    Parameters
    ----------
    model : "Whisper"
        The Whisper ASR model modified instance
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    result : stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the transcription of ``audio``.
    steps : str, default 'se'
        Instructions for refinement. A 's' means refine start-timestamps. An 'e' means refine end-timestamps.
    rel_prob_decrease : float, default 0.3
        Maximum percent decrease in probability relative to original probability which is the probability from muting
        according initial timestamps.
    abs_prob_decrease : float, default 0.05
        Maximum decrease in probability from original probability.
    rel_rel_prob_decrease : float, optional
        Maximum percent decrease in probability relative to previous probability which is the probability from previous
        iteration of muting.
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
    inplace : bool, default True, meaning return a deepcopy of ``result``
        Whether to alter timestamps in-place.
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

    Returns
    -------
    stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the refinement of ``text`` with ``audio``.

    Notes
    -----
    The lower the ``precision``, the longer the processing time.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3')
    >>> model.refine('audio.mp3', result)
    >>> result.to_srt_vtt('audio.srt')
    Saved 'audio.srt'
    """
    audioloader_not_supported(audio)
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

    audio = prep_audio(
        audio,
        denoiser=denoiser,
        denoiser_options=denoiser_options,
        demucs=demucs,
        demucs_options=demucs_options,
        only_voice_freq=only_voice_freq,
        verbose=verbose
    )
    max_inference_tokens = model.dims.n_text_ctx - 6
    sample_padding = int(N_FFT // 2) + 1
    frame_precision = max(round(precision * FRAMES_PER_SECOND), 2)
    total_duration = round(audio.shape[-1] / SAMPLE_RATE, 3)
    tokenizer = get_tokenizer(model, language=result.language, task='transcribe')

    def ts_to_frames(timestamps: Union[np.ndarray, list]) -> np.ndarray:
        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)
        return (timestamps * FRAMES_PER_SECOND).round().astype(np.int32)

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
        curr_token_count = 0

        for i, w in enumerate(all_words, 1):
            if (
                    (end_times[0] - start > 30) or
                    (curr_token_count + len(w.tokens) > max_inference_tokens)
            ):
                if curr_words:
                    yield curr_words, curr_starts, curr_ends, seg_edge_mask[prev_i:prev_i+len(curr_words)]
                    curr_words, curr_starts, curr_ends = [], [], []
                start = start_times[0]
                prev_i = i - 1
                curr_token_count = 0

            curr_words.append(w)
            curr_starts.append(start_times.pop(0))
            curr_ends.append(end_times.pop(0))
            curr_token_count += len(w.tokens)

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

            mid_starts = min_starts + ((max_starts - min_starts) / 2).round().astype(np.int32)
            mid_ends = min_ends + ((max_ends - min_ends) / 2).round().astype(np.int32)

            text_tokens = [t for w in words for t in w.tokens if t < tokenizer.eot]
            word_tokens = [[t for t in w.tokens if t < tokenizer.eot] for w in words]
            orig_mel_segment = log_mel_spectrogram(audio_segment, model.dims.n_mels, padding=sample_padding)
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
            for idx, _i in enumerate(max_starts if is_end_ts else min_ends):
                row = idx % 2
                prob_indices.extend([row] * len(words[idx].tokens))
                if is_finish[idx]:
                    continue
                if is_end_ts:
                    _p = mel_segment.shape[-1] if idx == len(words)-1 else mid_ends[idx+1]
                    mel_segment[row, :, _i:_p] = 0
                else:
                    _p = 0 if idx == 0 else mid_starts[idx-1]
                    mel_segment[row, :, _p:_i] = 0
            orig_probs, orig_tk_poss = get_prob()
            changes = np.zeros((orig_probs.shape[-1], 3), dtype=np.int32)
            changes[:, -1] = -1
            frame_indices = (mid_ends, max_starts) if is_end_ts else (min_ends, mid_starts)
            for idx, (_s, _e) in enumerate(zip(*frame_indices)):
                row = idx % 2
                if is_finish[idx]:
                    continue
                mel_segment[row, :, _s:_e] = 0

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
                            (rel_rel_prob_decrease is not None and rel_change_diff > rel_rel_prob_decrease) or
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
                    if not best_tks_changed:
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
            update_pbar(round(tqdm_pbar.total / len(step), 2))
        tqdm_pbar.update(tqdm_pbar.total - tqdm_pbar.n)

    result.reassign_ids()

    return result


def locate(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
        text: Union[str, List[int]],
        language: str,
        count: int = 1,
        duration_window: Union[float, Tuple[float, float]] = 3.0,
        *,
        mode: int = 0,
        start: float = None,
        end: float = None,
        probability_threshold: float = 0.5,
        eots: int = 1,
        max_token_per_seg: int = 20,
        exact_token: bool = False,
        case_sensitive: bool = False,
        verbose: bool = False,
        initial_prompt: str = None,
        suppress_tokens: Union[str, List[int]] = '-1',
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: Union[bool, torch.nn.Module] = False,
        demucs_options: dict = None,
        only_voice_freq: bool = False,
) -> Union[List[Segment], List[dict]]:
    """
    Locate when specific words are spoken in ``audio`` without fully transcribing.

    This is usefully for quickly finding at what time the specify words or phrases are spoken in an audio. Since it
    does not need to transcribe the audio to approximate the time, it is significantly faster transcribing then
    locating the word in the transcript.

    It can also transcribe few seconds around the approximated time to find out what was said around those words or
    confirm if the word was even spoken near that time.

    Parameters
    ----------
    model : whisper.model.Whisper
        An instance of Whisper ASR model.
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    text: str or list of int
        Words/phrase or list of tokens to search for in ``audio``.
    language : str
        Language of the ``text``.
    count : int, default 1, meaning stop search after 1 match
        Number of matches to find. Use 0 to look for all.
    duration_window : float or tuple of (float, float), default 3.0, same as (3.0, 3.0)
        Seconds before and after the end timestamp approximations to transcribe after mode 1.
        If tuple pair of values, then the 1st value will be seconds before the end and 2nd value will be seconds after.
    mode : int, default 0
        Mode of search.
        2, Approximates the end timestamp of ``text`` in the audio. This mode does not confirm whether ``text`` is
            spoken at the timestamp
        1, Completes mode 2 then transcribes audio within ``duration_window`` to confirm whether `text` is a match at
            the approximated timestamp by checking if ``text`` at that ``duration_window`` is within
            ``probability_threshold`` or matching the string content if ``text`` with the transcribed text at the
            ``duration_window``.
        0, Completes mode 1 then add word timestamps to the transcriptions of each match.
        Modes from fastest to slowest: 2, 1, 0
    start : float, optional, meaning it starts from 0s
        Seconds into the audio to start searching for ``text``.
    end : float, optional
        Seconds into the audio to stop searching for ``text``.
    probability_threshold : float, default 0.5
        Minimum probability of each token in ``text`` for it to be considered a match.
    eots : int, default 1
        Number of EOTs to reach before stopping transcription at mode 1. When transcription reach a EOT, it usually
        means the end of the segment or audio. Once ``text`` is found in the ``duration_window``, the transcription
        will stop immediately upon reaching a EOT.
    max_token_per_seg : int, default 20
        Maximum number of tokens to transcribe in the ``duration_window`` before stopping.
    exact_token : bool, default False
        Whether to find a match base on the exact tokens that make up ``text``.
    case_sensitive : bool, default False
        Whether to consider the case of ``text`` when matching in string content.
    verbose : bool or None, default False
        Whether to display the text being decoded to the console.
        Displays all the details if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.
    initial_prompt : str, optional
        Text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.
    suppress_tokens : str or list of int, default '-1', meaning suppress special characters except common punctuations
        List of tokens to suppress.
    denoiser : str, optional
        String of the denoiser to use for preprocessing ``audio``.
        See ``stable_whisper.audio.SUPPORTED_DENOISERS`` for supported denoisers.
    denoiser_options : dict, optional
        Options to use for ``denoiser``.
    only_voice_freq : bool, default False
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.

    Returns
    -------
    stable_whisper.result.Segment or list of dict or list of float
        Mode 0, list of instances of :class:`stable_whisper.result.Segment`.
        Mode 1, list of dictionaries with end timestamp approximation of matches and transcribed neighboring words.
        Mode 2, list of timestamps in seconds for each end timestamp approximation.

    Notes
    -----
    For ``text``, the case and spacing matters as 'on', ' on', ' On' are different tokens, therefore chose the one that
    best suits the context (e.g. ' On' to look for it at the beginning of a sentence).

    Use a sufficiently large first value of ``duration_window`` i.e. the value > time it is expected to speak ``text``.

    If ``exact_token = False`` and the string content matches, then ``probability_threshold`` is not used.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> matches = model.locate('audio.mp3', 'are', language='English', verbose=True)

    Some words can sound the same but have different spellings to increase of the chance of finding such words use
    ``initial_prompt``.

    >>> matches = model.locate('audio.mp3', ' Nickie', 'English', verbose=True, initial_prompt='Nickie')
    """
    audioloader_not_supported(audio)

    sample_padding = int(N_FFT // 2) + 1
    sec_per_emb = model.dims.n_audio_ctx / CHUNK_LENGTH
    CHUNK_SAMPLES = round(CHUNK_LENGTH * SAMPLE_RATE)
    if isinstance(duration_window, (float, int)):
        duration_window = [duration_window] * 2
    window_sum = sum(duration_window)
    assert CHUNK_SAMPLES > window_sum, \
        f'Sum of [duration_window] must be less than {CHUNK_SAMPLES}, got {window_sum}'
    adjusted_chunk_size = CHUNK_SAMPLES - round(duration_window[0]*SAMPLE_RATE)
    if initial_prompt:
        initial_prompt = ' ' + initial_prompt.strip()
    task = DecodingTask(model, DecodingOptions(
        language=language, prompt=initial_prompt, suppress_tokens=suppress_tokens, without_timestamps=True,
    ))
    tokenizer = task.tokenizer
    initial_tokens = list(task.initial_tokens)
    text_tokens, text = (tokenizer.encode(text), text) if isinstance(text, str) else (text, tokenizer.decode(text))
    if not exact_token and not case_sensitive:
        text = text.lower()

    tk_suppress_masks = [
        [i for i in fil.suppress_tokens if i < tokenizer.eot]
        for fil in task.logit_filters if isinstance(fil, SuppressTokens)
    ]

    audio = prep_audio(
        audio,
        denoiser=denoiser,
        denoiser_options=denoiser_options,
        demucs=demucs,
        demucs_options=demucs_options,
        only_voice_freq=only_voice_freq,
        verbose=verbose
    )
    prev_target_end = None
    found = 0
    if end:
        audio = audio[:round(end * SAMPLE_RATE)]
    seek_sample = round(start * SAMPLE_RATE) if start else 0
    total_samples = audio.shape[-1]

    def _locate():
        nonlocal seek_sample, found
        seek = round(seek_sample / SAMPLE_RATE, 3)
        audio_segment = audio[seek_sample: seek_sample + CHUNK_SAMPLES]
        mel_segment = log_mel_spectrogram(audio_segment, model.dims.n_mels, padding=sample_padding)
        mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device=model.device)

        QKs = [None] * model.dims.n_text_layer
        hooks = [
            block.cross_attn.register_forward_hook(
                lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
            )
            for i, block in enumerate(model.decoder.blocks)
        ]
        tokens = torch.tensor([initial_tokens + text_tokens]).to(model.device)
        with torch.no_grad():
            audio_features = model.encoder(mel_segment.unsqueeze(0))
            model.decoder(tokens, audio_features)

        for hook in hooks:
            hook.remove()

        weights = torch.cat([QKs[_l][:, _h] for _l, _h in model.alignment_heads.indices().T], dim=0)
        weights = weights.softmax(dim=-1)
        std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
        weights = (weights - mean) / std
        weights = median_filter(weights, 7)

        matrix = weights.mean(axis=0)
        target_end = round((matrix[-1].argmax()/sec_per_emb).item(), 3)
        found_msg = f'"{text}" ending at ~{format_timestamp(target_end+seek)}' if verbose else ''

        if mode == 2:
            if found_msg:
                safe_print('Unconfirmed:' + found_msg)
            nonlocal prev_target_end
            found += 1
            if (
                    (seek_sample + CHUNK_SAMPLES >= total_samples) or
                    (count and found >= count) or
                    (prev_target_end == target_end)
            ):
                seek_sample = total_samples
            else:
                seek_sample += round(target_end * SAMPLE_RATE)
            prev_target_end = target_end
            return dict(tokens=[], target_end=target_end+seek)

        curr_start = round(max(target_end - duration_window[0], 0.), 3)
        curr_end = round(target_end + duration_window[1], 3)
        start_frame = round(curr_start * FRAMES_PER_SECOND)
        end_frame = round(curr_end * FRAMES_PER_SECOND)
        mel_segment_section = pad_or_trim(mel_segment[..., start_frame:end_frame], N_FRAMES)
        temp_tokens = torch.tensor([initial_tokens]).to(model.device)

        predictions = []

        target_token_idx = 0
        not_end = True
        found_target = False
        curr_eots = 0
        temp_audio_features = model.encoder(mel_segment_section.unsqueeze(0))
        tokens_to_decode = []
        replace_found_tokens = []
        infer_tokens = [temp_tokens[0]]
        kv_cache, hooks = model.install_kv_cache_hooks()
        while not_end:
            with torch.no_grad():
                logits = model.decoder(temp_tokens, temp_audio_features, kv_cache=kv_cache)[0, -1, :tokenizer.eot+1]
            for tks in tk_suppress_masks:
                logits[tks] = -np.inf
            sorted_logits_idxs = logits.sort(dim=-1).indices[-2:]
            best_token = sorted_logits_idxs[-1]
            best_non_eot_token = sorted_logits_idxs[-2] if best_token == tokenizer.eot else best_token

            logits = logits[:tokenizer.eot].softmax(dim=-1)
            if found_target:
                target_word_prob = is_match = None
            else:
                if exact_token:
                    is_match = False
                else:
                    tokens_to_decode.append(best_non_eot_token)
                    temp_text = tokenizer.decode(tokens_to_decode)
                    if not case_sensitive:
                        temp_text = temp_text.lower()
                    if is_match := temp_text.endswith(text):
                        tokens_to_decode = []
                target_word_prob = logits[text_tokens[target_token_idx]].item()
            if (
                    target_word_prob is not None and
                    (
                            target_word_prob >= probability_threshold or
                            best_non_eot_token == text_tokens[target_token_idx] or
                            is_match
                    )
            ):
                if is_match:
                    best_token = best_non_eot_token
                    token_prob = logits[best_token].item()
                    found_target = True
                else:
                    best_token[None] = text_tokens[target_token_idx]
                    if len(replace_found_tokens) or best_non_eot_token != text_tokens[target_token_idx]:
                        replace_found_tokens.append(best_non_eot_token)
                    target_token_idx += 1
                    if target_token_idx == len(text_tokens):
                        found_target = True
                    token_prob = target_word_prob
                if found_target:
                    found += 1
                curr_eots = 0
            else:
                if not found_target:
                    if len(replace_found_tokens):
                        temp_tokens = torch.cat(infer_tokens)[None]
                        temp_tokens = torch.cat(
                            [temp_tokens[..., :-len(replace_found_tokens)],
                             torch.stack(replace_found_tokens)[None]]
                        )
                        replace_found_tokens = []
                        kv_cache.clear()
                    target_token_idx = 0
                if best_token == tokenizer.eot:
                    if curr_eots >= eots or found_target:
                        not_end = False
                    else:
                        curr_eots += 1
                        best_token = best_non_eot_token
                else:
                    curr_eots = 0
                token_prob = None if best_token == tokenizer.eot else logits[best_token].item()

            predictions.append(dict(token=best_token.item(), prob=token_prob))
            if len(predictions) > max_token_per_seg:
                not_end = False
            if not_end:
                infer_tokens.append(best_token[None])
                temp_tokens = best_token[None, None]
        kv_cache.clear()
        for hook in hooks:
            hook.remove()
        segment = None

        if found_target:
            if found_msg:
                safe_print('Confirmed: ' + found_msg, tqdm_pbar.write)
            final_tokens = [p['token'] for p in predictions]
            if mode == 1:
                _, (ws, wts), _ = split_word_tokens([dict(tokens=final_tokens)], tokenizer)
                final_token_probs = [p['prob'] for p in predictions]
                wps = [float(np.mean([final_token_probs.pop(0) for _ in wt])) for wt in wts]
                words = [dict(word=w, tokens=wt, probability=wp) for w, wt, wp in zip(ws, wts, wps)]
                final_end = target_end+seek
                near_text = "".join(ws)
                segment = dict(end=final_end, text=text, duration_window_text=near_text, duration_window_word=words)
                if verbose:
                    safe_print(f'Duration Window: "{near_text}"\n', tqdm_pbar.write)
                seek_sample += round(curr_end * SAMPLE_RATE)
            else:

                segment = dict(
                    seek=0,
                    tokens=final_tokens
                )

                add_word_timestamps_stable(
                    segments=[segment],
                    model=model,
                    tokenizer=tokenizer,
                    mel=mel_segment,
                    num_samples=round(curr_end*SAMPLE_RATE),
                    gap_padding=None
                )
                segment = Segment(words=segment['words'])
                seek_sample += round(segment.words[-1].end * SAMPLE_RATE)
                segment.offset_time(seek)
                segment.seek = curr_start
                if verbose:
                    safe_print(segment.to_display_str(), tqdm_pbar.write)

        else:
            seek_sample += adjusted_chunk_size if audio_segment.shape[-1] == CHUNK_SAMPLES else audio_segment.shape[-1]

        return segment

    total_duration = round(total_samples / SAMPLE_RATE, 2)
    matches = []
    with tqdm(total=total_duration, unit='sec', disable=verbose is None, desc='Locate') as tqdm_pbar:
        while seek_sample < total_samples and (not count or found < count):
            if match := _locate():
                matches.append(match)
            tqdm_pbar.update(round(seek_sample/SAMPLE_RATE, 2) - tqdm_pbar.n)
        tqdm_pbar.update(tqdm_pbar.total - tqdm_pbar.n)
    if verbose and not matches:
        safe_print(f'Failed to locate "{text}".')
    return matches
