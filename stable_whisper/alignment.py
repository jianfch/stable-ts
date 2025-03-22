import torch
import numpy as np
from tqdm import tqdm
from typing import TYPE_CHECKING, Union, List, Optional, Tuple

from .result import WhisperResult, Segment
from .timing import add_word_timestamps_stable, split_word_tokens
from .audio import prep_audio, AudioLoader, audioloader_not_supported
from .utils import safe_print, format_timestamp
from .whisper_compatibility import warn_compatibility_issues, get_tokenizer, disable_sdpa
from .non_whisper.alignment import Aligner, WordToken
from .non_whisper.refinement import Refiner
from .options import AllOptions

from .whisper_compatibility import (
    SAMPLE_RATE, N_FRAMES, N_FFT, pad_or_trim, log_mel_spectrogram, FRAMES_PER_SECOND, CHUNK_LENGTH, N_SAMPLES,
    median_filter, DecodingTask, DecodingOptions, SuppressTokens, whisper, TOKENS_PER_SECOND, as_vanilla
)

if TYPE_CHECKING:
    from .whisper_compatibility import Whisper
    from .whisper_compatibility import Tokenizer

__all__ = ['align', 'refine', 'locate', 'align_words']


def align(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
        text: Union[str, List[int], WhisperResult],
        language: str = None,
        *,
        tokenizer: "Tokenizer" = None,
        ignore_compatibility: bool = False,
        remove_instant_words: bool = False,
        token_step: int = 100,
        original_split: bool = False,
        word_dur_factor: Optional[float] = 2.0,
        max_word_dur: Optional[float] = 3.0,
        nonspeech_skip: Optional[float] = 5.0,
        fast_mode: bool = False,
        failure_threshold: Optional[float] = None,
        **options
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
        Whether to preserve the original segment groupings. Segments are split by line breaks if ``text`` is plain-text.
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
    ignore_compatibility : bool, default False
        Whether to ignore warnings for compatibility issues with the detected Whisper version.
    extra_models : list of whisper.model.Whisper, optional
        List of additional Whisper model instances to use for computing word-timestamps along with ``model``.
    presplit : bool or list of str, default True meaning same as ``append_punctuations``
        List of ending punctuation used to split ``text`` into segments for applying ``gap_padding``,
        but segmentation of final output is unnaffected unless ``original_split=True``.
        If ``original_split=True``, the original split is used instead of split from ``presplit``.
        Ignored if ``model`` is a faster-whisper model.
    gap_padding : str, default ' ...'
        Only if ``presplit=True``, ``gap_padding`` is prepended to each segments for word timing alignment.
        Used to reduce the probability of model predicting timestamps earlier than the first utterance.
        Ignored if ``model`` is a faster-whisper model.
    dynamic_heads : bool or int or str, optional
        Whether to find optimal cross-attention heads during runtime instead of using the predefined heads for
        word-timestamp extraction. Specify the number of heads or `True` for default of 6 heads.
        To specify number of iterations for finding the optimal heads,
        use string with "," to separate heads and iterations (e.g. "8,3" for 8 heads and 3 iterations).

    Returns
    -------
    stable_whisper.result.WhisperResult or None
        All timestamps, words, probabilities, and other data from the alignment of ``audio``. Return None if alignment
        fails and ``remove_instant_words = True``.

    Notes
    -----
    If ``token_step`` is less than 1, ``token_step`` will be set to its maximum value, 442. This value is computed with
    ``whisper.model.Whisper.dims.n_text_ctx`` - 6.

    ``regroup`` is ignored if ``original_split = True``.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.align('helloworld.mp3', 'Hello, World!', 'English')
    >>> result.to_srt_vtt('helloword.srt')
    Saved 'helloworld.srt'
    """
    model = as_vanilla(model)
    is_faster_model = model.__module__.startswith('faster_whisper.')
    if not is_faster_model:
        warn_compatibility_issues(whisper, ignore_compatibility)
    max_token_step = (model.max_length if is_faster_model else model.dims.n_text_ctx) - 6
    if token_step < 1:
        token_step = max_token_step
    elif token_step > max_token_step:
        raise ValueError(f'The max value for [token_step] is {max_token_step} but got {token_step}.')

    tokenizer, supported_languages = get_alignment_tokenizer(model, is_faster_model, text, language, tokenizer)

    options = AllOptions(options, vanilla_align=not is_faster_model)
    split_words_by_space = getattr(tokenizer, 'language_code', tokenizer.language) not in {"zh", "ja", "th", "lo", "my"}
    model_type = 'fw' if is_faster_model else None
    inference_func = get_whisper_alignment_func(model, tokenizer, model_type, options)

    aligner = Aligner(
        inference_func=inference_func,
        decode=tokenizer.decode,
        encode=tokenizer.encode,
        split_words_by_space=split_words_by_space,
        sample_rate=SAMPLE_RATE,
        tokens_per_sec=TOKENS_PER_SECOND,
        max_segment_length=N_SAMPLES,
        remove_instant_words=remove_instant_words,
        token_step=token_step,
        original_split=original_split,
        word_dur_factor=word_dur_factor,
        max_word_dur=max_word_dur,
        nonspeech_skip=nonspeech_skip,
        fast_mode=fast_mode,
        failure_threshold=failure_threshold,
        all_options=options
    )

    result = aligner.align(audio, text)
    set_result_language(result, tokenizer, language, supported_languages)

    return result


def align_words(
       model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
        result: Union[WhisperResult, List[dict]],
        language: str = None,
        *,
        ignore_compatibility: bool = False,
        tokenizer: "Tokenizer" = None,
        normalize_text: bool = True,
        inplace: bool = True,
        **options
) -> WhisperResult:
    """
    Align segments of plain text or tokens with audio at word-level at specified start and end of each segment.

    This is a version of ``align()`` that confines each segment to a range of timestamps which eliminates the need
    for the fallback mechanisms used in ``align()``. This makes this method is drastically faster than ``align()`` and
    reduces word-timstamp errors if the provided start and end timestamps of each segment is accurate.

    Parameters
    ----------
    model : "Whisper"
        The Whisper ASR model modified instance
    audio : str or numpy.ndarray or torch.Tensor or bytes or AudioLoader
        Path/URL to the audio file, the audio waveform, or bytes of audio file or
        instance of :class:`stable_whisper.audio.AudioLoader`.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    result : stable_whisper.result.WhisperResult or list of dict
        Instance of :class:`stable_whisper.result.WhisperResult` or List of dictionaries with start, end, and text.
    language : str, default None, uses ``language`` in ``text`` if it is a :class:`stable_whisper.result.WhisperResult`
        Language of ``text``. Required if ``text`` does not contain ``language``.
    tokenizer : "Tokenizer", default None, meaning a new tokenizer is created according ``language`` and ``model``
        A tokenizer to used tokenizer text and detokenize tokens.
    stream : bool or None, default None
        Whether to loading ``audio`` in chunks of 30 seconds until the end of file/stream.
        If ``None`` and ``audio`` is a string then set to ``True`` else ``False``.
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
    ignore_compatibility : bool, default False
        Whether to ignore warnings for compatibility issues with the detected Whisper version.
    extra_models : list of whisper.model.Whisper, optional
        List of additional Whisper model instances to use for computing word-timestamps along with ``model``.
    presplit : bool or list of str, default True meaning same as ``append_punctuations``
        List of ending punctuation used to split ``text`` into segments for applying ``gap_padding``,
        but segmentation of final output is unnaffected unless ``original_split=True``.
        If ``original_split=True``, the original split is used instead of split from ``presplit``.
        Ignored if ``model`` is a faster-whisper model.
    gap_padding : str, default ' ...'
        Only if ``presplit=True``, ``gap_padding`` is prepended to each segments for word timing alignment.
        Used to reduce the probability of model predicting timestamps earlier than the first utterance.
        Ignored if ``model`` is a faster-whisper model.
    dynamic_heads : bool or int or str, optional
        Whether to find optimal cross-attention heads during runtime instead of using the predefined heads for
        word-timestamp extraction. Specify the number of heads or `True` for default of 6 heads.
        To specify number of iterations for finding the optimal heads,
        use string with "," to separate heads and iterations (e.g. "8,3" for 8 heads and 3 iterations).
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

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = [dict(start=0.0, end=0.5, text='hello world 1'), dict(start=0.5, end=1.0, text='hello world 2')]
    >>> result = model.align_words('audio.mp3', result, 'English')
    """
    model = as_vanilla(model)
    is_faster_model = model.__module__.startswith('faster_whisper.')
    if not is_faster_model:
        warn_compatibility_issues(whisper, ignore_compatibility)
    tokenizer, supported_languages = get_alignment_tokenizer(model, is_faster_model, result, language, tokenizer)

    options = AllOptions(options)
    split_words_by_space = getattr(tokenizer, 'language_code', tokenizer.language) not in {"zh", "ja", "th", "lo", "my"}
    max_segment_tokens = model.max_length if is_faster_model else model.dims.n_text_ctx
    inference_func = get_whisper_alignment_func(model, tokenizer, 'fw' if is_faster_model else None, options)

    aligner = Aligner(
        inference_func=inference_func,
        decode=tokenizer.decode,
        encode=tokenizer.encode,
        split_words_by_space=split_words_by_space,
        sample_rate=SAMPLE_RATE,
        max_segment_length=N_SAMPLES,
        time_precision=1 / TOKENS_PER_SECOND,
        token_step=max_segment_tokens,
        all_options=options
    )
    result = aligner.align_words(audio, result, normalize_text, inplace)
    set_result_language(result, tokenizer, language, supported_languages)

    return result


def get_alignment_tokenizer(model, is_faster_model: bool, text, language=None, tokenizer=None):
    if is_faster_model:
        supported_languages = model.supported_languages
    else:
        supported_languages = None if model.is_multilingual else ['en']

    if tokenizer is None:
        if (
                not language and
                (supported_languages is None or len(supported_languages) > 1) and
                (language := getattr(text, 'language', None)) is None
        ):
            raise TypeError('expected argument for language')
        tokenizer = get_tokenizer(model, is_faster_model=is_faster_model, language=language, task='transcribe')

    return tokenizer, supported_languages


def set_result_language(result: WhisperResult, tokenizer, language, supported_languages):
    result.language = \
        tokenizer.language_code if hasattr(tokenizer, 'language_code') else getattr(tokenizer, 'language', language)
    if not result.language and supported_languages and len(supported_languages) == 1:
        result.language = supported_languages[0]


def get_whisper_alignment_func(
        model,
        tokenizer,
        model_type: Optional[str] = None,
        options: Optional[AllOptions] = None
):
    assert model_type in (None, 'fw')

    if model_type is None:
        def compute_timestamps(audio_segment: torch.Tensor, word_tokens: List[WordToken]) -> List[dict]:
            curr_words = [wt.word for wt in word_tokens]
            curr_word_tokens = [wt.tokens for wt in word_tokens]
            temp_segments = [dict(seek=0, tokens=(curr_words, curr_word_tokens))]

            segment_samples = int(audio_segment.size(-1))
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
                prepend_punctuations='',
                append_punctuations='',
                gap_padding=None,
                extra_models=options.align.extra_models,
                dynamic_heads=options.align.dynamic_heads
            )
            return [w for seg in temp_segments for w in seg['words']]

    else:
        from .whisper_compatibility import is_faster_whisper_on_pt
        from faster_whisper.version import __version__ as fw_ver

        def compute_timestamps(audio_segment: torch.Tensor, word_tokens: List[WordToken]) -> List[dict]:
            segment_samples = int(audio_segment.size(-1))
            temp_segment = dict(
                seek=0,
                start=0.0,
                end=round(segment_samples / model.feature_extractor.sampling_rate, 3),
                tokens=[t for wt in word_tokens for t in wt.tokens],
            )
            is_on_pt = is_faster_whisper_on_pt()
            if is_on_pt:
                features = model.feature_extractor(audio_segment)
            else:
                features = model.feature_extractor(audio_segment.cpu().numpy())
            encoder_output = model.encode(features[:, : model.feature_extractor.nb_max_frames])

            model.add_word_timestamps(
                segments=[[temp_segment]] if is_on_pt or not fw_ver.startswith('1.0') else [temp_segment],
                tokenizer=tokenizer,
                encoder_output=encoder_output,
                num_frames=round(segment_samples / model.feature_extractor.hop_length),
                prepend_punctuations='',
                append_punctuations='',
                last_speech_timestamp=temp_segment['start'],
            )

            return temp_segment['words']

    return compute_timestamps


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
        **options
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

    Returns
    -------
    stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the refinement of ``text`` with ``audio``.

    Notes
    -----
    The lower the ``precision``, the longer the processing time.

    Faster-Whisper models are significantly slower than vanilla models with this function.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3')
    >>> model.refine('audio.mp3', result)
    >>> result.to_srt_vtt('audio.srt')
    Saved 'audio.srt'
    """
    model = as_vanilla(model)
    if result:
        if not result.has_words:
            if not result.language:
                raise RuntimeError(f'cannot add word-timestamps to result with missing language')
            align_words(model, audio, result)
        elif not all(word.tokens for word in result.all_words()):
            tokenizer = get_tokenizer(model)
            for word in result.all_words():
                word.tokens = tokenizer.encode(word.word)
    tokenizer = get_tokenizer(model, language=result.language, task='transcribe')

    options = AllOptions(options, post=False, silence=False, align=False)
    model_type = 'fw' if (is_faster_model := model.__module__.startswith('faster_whisper.')) else None
    inference_func = get_whisper_refinement_func(model, tokenizer, model_type, single_batch)
    max_inference_tokens = (model.max_length if is_faster_model else model.dims.n_text_ctx) - 6

    refiner: Refiner = Refiner(
        inference_func=inference_func,
        sample_rate=SAMPLE_RATE,
        steps=steps,
        rel_prob_decrease=rel_prob_decrease,
        abs_prob_decrease=abs_prob_decrease,
        rel_rel_prob_decrease=rel_rel_prob_decrease,
        prob_threshold=prob_threshold,
        rel_dur_change=rel_dur_change,
        abs_dur_change=abs_dur_change,
        word_level=word_level,
        precision=precision,
        max_inference_tokens=max_inference_tokens,
        all_options=options
    )
    result = refiner.refine(audio, result, inplace)

    return result


def get_whisper_refinement_func(
        model,
        tokenizer,
        model_type: Optional[str] = None,
        single_batch: bool = False
):
    assert model_type in (None, 'fw')

    if model_type is None:
        def inference_func(audio_segment: torch, tokens: List[int]) -> torch.Tensor:
            input_tokens = torch.tensor(
                [
                    *tokenizer.sot_sequence,
                    tokenizer.no_timestamps,
                    *tokens,
                    tokenizer.eot,
                ]
            ).to(model.device)

            with torch.no_grad():
                mel_segments = log_mel_spectrogram(audio_segment.to(device=model.device), model.dims.n_mels)
                mel_segments = pad_or_trim(mel_segments, N_FRAMES)
                if single_batch:
                    logits = torch.cat(
                        [model(single_mel.unsqueeze(0), input_tokens.unsqueeze(0)) for single_mel in mel_segments]
                    )
                else:
                    logits = model(mel_segments, input_tokens.unsqueeze(0))

            sot_len = len(tokenizer.sot_sequence)
            sampled_logits = logits[:, sot_len:sot_len+len(tokens), : tokenizer.eot]
            token_probs = sampled_logits.softmax(dim=-1)
            return token_probs

    else:
        from .whisper_compatibility import is_faster_whisper_on_pt
        from faster_whisper.version import __version__ as fw_ver

        def _inference_func(audio_segment: torch, tokens: List[int]) -> List[float]:
            segment_samples = int(audio_segment.size(-1))

            is_on_pt = is_faster_whisper_on_pt()
            if is_on_pt:
                features = model.feature_extractor(audio_segment)
            else:
                features = model.feature_extractor(audio_segment.cpu().numpy())
            encoder_output = model.encode(features[:, : model.feature_extractor.nb_max_frames])
            num_frames = round(segment_samples / model.feature_extractor.hop_length)

            text_token_probs = model.model.align(
                encoder_output,
                tokenizer.sot_sequence,
                [tokens],
                num_frames,
                median_filter_width=1
            )[0].text_token_probs

            return text_token_probs

        def inference_func(audio_segment: torch, tokens: List[int]) -> torch.Tensor:
            text_token_probs = [_inference_func(audio, tokens) for audio in audio_segment]
            return torch.tensor(text_token_probs)

    return inference_func


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
        with torch.no_grad(), disable_sdpa():
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
