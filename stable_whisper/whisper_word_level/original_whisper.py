import warnings
from types import MethodType
from typing import TYPE_CHECKING, Union, Optional, Tuple, Callable, List

import torch
import numpy as np
from tqdm import tqdm

from ..whisper_compatibility import (
    whisper, log_mel_spectrogram, pad_or_trim, DecodingResult, DecodingOptions,
    SAMPLE_RATE, N_FRAMES, HOP_LENGTH, N_SAMPLES_PER_TOKEN, N_SAMPLES, LANGUAGES
)

from ..result import WhisperResult, Segment
from ..audio import AudioLoader, audioloader_not_supported, convert_demucs_kwargs
from ..decode import decode_stable
from ..stabilization import NonSpeechPredictor
from ..timing import add_word_timestamps_stable
from ..utils import safe_print, isolate_useful_options, update_options, exact_div
from ..whisper_compatibility import warn_compatibility_issues, get_tokenizer
from ..default import get_min_word_dur, get_prepend_punctuations, get_append_punctuations

if TYPE_CHECKING:
    from whisper.model import Whisper


def transcribe_stable(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
        *,
        verbose: Optional[bool] = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,
        regroup: Union[bool, str] = True,
        ts_num: int = 0,
        ts_noise: float = None,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        suppress_attention: bool = False,
        use_word_position: bool = True,
        q_levels: int = 20,
        k_size: int = 5,
        time_scale: float = None,
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: Union[bool, torch.nn.Module] = False,
        demucs_options: dict = None,
        vad: Union[bool, dict] = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: Optional[float] = None,
        min_silence_dur: Optional[float] = None,
        nonspeech_error: float = 0.1,
        only_voice_freq: bool = False,
        prepend_punctuations: Optional[str] = None,
        append_punctuations: Optional[str] = None,
        stream: Optional[bool] = None,
        mel_first: Optional[bool] = None,
        split_callback: Callable = None,
        suppress_ts_tokens: bool = False,
        gap_padding: str = ' ...',
        only_ffmpeg: bool = False,
        max_instant_words: float = 0.5,
        avg_prob_threshold: Optional[float] = None,
        nonspeech_skip: Optional[float] = None,
        progress_callback: Callable = None,
        ignore_compatibility: bool = False,
        extra_models: Optional[List["Whisper"]] = None,
        dynamic_heads: Optional[Union[bool, int, str]] = None,
        clip_timestamps: Optional[Union[str, List[float]]] = None,
        **decode_options) \
        -> WhisperResult:
    """
    Transcribe audio using Whisper.

    This is a modified version of :func:`whisper.transcribe.transcribe` with slightly different decoding logic while
    allowing additional preprocessing and postprocessing. The preprocessing performed on the audio includes:
    voice isolation / noise removal and low/high-pass filter. The postprocessing performed on the transcription
    result includes: adjusting timestamps with VAD and custom regrouping segments based punctuation and speech gaps.

    Parameters
    ----------
    model : whisper.model.Whisper
        An instance of Whisper ASR model.
    audio : str or numpy.ndarray or torch.Tensor or bytes or AudioLoader
        Path/URL to the audio file, the audio waveform, or bytes of audio file or
        instance of :class:`stable_whisper.audio.AudioLoader`.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    verbose : bool or None, default False
        Whether to display the text being decoded to the console.
        Displays all the details if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.
    temperature : float or iterable of float, default (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either ``compression_ratio_threshold`` or ``logprob_threshold``.
    compression_ratio_threshold : float, default 2.4
        If the gzip compression ratio is above this value, treat as failed.
    logprob_threshold : float, default -1
        If the average log probability over sampled tokens is below this value, treat as failed
    no_speech_threshold : float, default 0.6
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below ``logprob_threshold``, consider the segment as silent
    condition_on_previous_text : bool, default True
        If ``True``, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
    initial_prompt : str, optional
        Text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.
    word_timestamps : bool, default True
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.
        Disabling this will prevent segments from splitting/merging properly.
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
    stream : bool or None, default None
        Whether to loading ``audio`` in chunks of 30 seconds until the end of file/stream.
        If ``None`` and ``audio`` is a string then set to ``True`` else ``False``.
    mel_first : bool, optional
        Process entire audio track into log-Mel spectrogram first instead in chunks.
        Used if odd behavior seen in stable-ts but not in whisper, but use significantly more memory for long audio.
    split_callback : Callable, optional
        Custom callback for grouping tokens up with their corresponding words.
        The callback must take two arguments, list of tokens and tokenizer.
        The callback returns a tuple with a list of words and a corresponding nested list of tokens.
    suppress_ts_tokens : bool, default False
        Whether to suppress timestamp tokens during inference for timestamps are detected at silent.
        Reduces hallucinations in some cases, but also prone to ignore disfluencies and repetitions.
        This option is ignored if ``suppress_silence = False``.
    gap_padding : str, default ' ...'
        Padding prepend to each segments for word timing alignment.
        Used to reduce the probability of model predicting timestamps earlier than the first utterance.
    only_ffmpeg : bool, default False
        Whether to use only FFmpeg (instead of not yt-dlp) for URls
    max_instant_words : float, default 0.5
        If percentage of instantaneous words in a segment exceed this amount, the segment is removed.
    avg_prob_threshold: float or None, default None
        Transcribe the gap after the previous word and if the average word proababiliy of a segment falls below this
        value, discard the segment. If ``None``, skip transcribing the gap to reduce chance of timestamps starting
        before the next utterance.
    nonspeech_skip : float or None, default None
        Skip non-speech sections that are equal or longer than this duration in seconds. Disable skipping if ``None``.
        Reduce text and timing hallucinations in non-speech sections but may increase processing time.
    progress_callback : Callable, optional
        A function that will be called when transcription progress is updated.
        The callback need two parameters.
        The first parameter is a float for seconds of the audio that has been transcribed.
        The second parameter is a float for total duration of audio in seconds.
    ignore_compatibility : bool, default False
        Whether to ignore warnings for compatibility issues with the detected Whisper version.
    extra_models : list of whisper.model.Whisper, optional
        List of additional Whisper model instances to use for computing word-timestamps along with ``model``.
    dynamic_heads : bool or int or str, optional
        Whether to find optimal cross-attention heads during runtime instead of using the predefined heads for
        word-timestamp extraction. Specify the number of heads or `True` for default of 6 heads.
        To specify number of iterations for finding the optimal heads,
        use string with "," to separate heads and iterations (e.g. "8,3" for 8 heads and 3 iterations).
    clip_timestamps : str or list of float
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.
    decode_options
        Keyword arguments to construct class:`whisper.decode.DecodingOptions` instances.

    Returns
    -------
    stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the transcription of ``audio``.

    See Also
    --------
    stable_whisper.non_whisper.transcribe_any : Return :class:`stable_whisper.result.WhisperResult` containing all the
        data from transcribing audio with unmodified :func:`whisper.transcribe.transcribe` with preprocessing and
        postprocessing.
    stable_whisper.whisper_word_level.faster_whisper.faster_transcribe : Return
        :class:`stable_whisper.result.WhisperResult` containing all the data from transcribing audio with
        :meth:`faster_whisper.WhisperModel.transcribe` with preprocessing and postprocessing.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe('audio.mp3', vad=True)
    >>> result.to_srt_vtt('audio.srt')
    Saved: audio.srt
    """
    if mel_first is not None:
        warnings.warn('``mel_first`` is deprecated and will be removed in future versions.'
                      'Use ``stream`` (e.g. replace ``mel_first=True`` with ``stream=False``).',
                      stacklevel=2)
        stream = not mel_first

    if suppress_attention:
        warnings.warn('``suppress_attention`` is deprecated and will be removed in future versions',
                      stacklevel=2)

    prepend_punctuations = get_prepend_punctuations(prepend_punctuations)
    append_punctuations = get_append_punctuations(append_punctuations)
    min_word_dur = get_min_word_dur(min_word_dur)

    warn_compatibility_issues(whisper, ignore_compatibility, 'Or use transcribe_minimal().')
    dtype = torch.float16 if decode_options.get("fp16", True) and not getattr(model, 'dq', False) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if 'max_initial_timestamp' not in decode_options:
        decode_options['max_initial_timestamp'] = None

    device = model.device

    if time_scale:
        warnings.warn('``time_scale`` is deprecated and will be removed in future versions. '
                      'It currently does not affect results.',
                      stacklevel=2)
    if decode_options.pop('input_sr', None):
        warnings.warn('``input_sr`` is deprecated and will be removed in future versions. '
                      '``audio`` of types numpy.ndarray and torch.Tensor inputs must be already at 16kHz. '
                      'To higher sample rates for ``audio`` use str or bytes.',
                      stacklevel=2)
    denoiser, denoiser_options = convert_demucs_kwargs(
        denoiser, denoiser_options, demucs=demucs, demucs_options=demucs_options
    )

    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    if clip_timestamps:
        clip_timestamps = [clip_timestamps[i:i+2] for i in range(0, len(clip_timestamps), 2)]
        if len(clip_timestamps[-1]) == 1:
            clip_timestamps[-1] = [clip_timestamps[-1][0], None]

    if isinstance(audio, AudioLoader):
        audio.validate_external_args(
            sr=SAMPLE_RATE,
            vad=vad,
            stream=stream,
            denoiser=denoiser,
            denoiser_options=denoiser_options,
            only_voice_freq=only_voice_freq
        )
        audio.load_sections = clip_timestamps
    else:
        denoiser_options = update_options(denoiser_options, device=device)
        audio = AudioLoader(
            audio,
            stream=stream,
            denoiser=denoiser,
            denoiser_options=denoiser_options,
            only_voice_freq=only_voice_freq,
            only_ffmpeg=only_ffmpeg,
            verbose=verbose,
            new_chunk_divisor=512 if vad else None,
            load_sections=clip_timestamps
        )
    tokenizer = None
    language = None
    initial_prompt_tokens = []
    task = decode_options.get("task", "transcribe")
    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

    def detect_language():
        nonlocal tokenizer
        if tokenizer is None:
            if not decode_options.get("language"):
                if not model.is_multilingual:
                    decode_options["language"] = "en"
                else:
                    if verbose:
                        print("Detecting language using up to 30 seconds following first non-silent sample. "
                              "Use `--language` to specify the language")
                    _, probs = model.detect_language(mel_segment)
                    decode_options["language"] = max(probs, key=probs.get)
                    if verbose is not None:
                        detected_msg = f"Detected language: {LANGUAGES[decode_options['language']]}"
                        if tqdm_pbar.disable:
                            print(detected_msg)
                        else:
                            tqdm_pbar.write(detected_msg)

            nonlocal language
            language = decode_options["language"]
            tokenizer = get_tokenizer(model, language=language, task=task)

            if initial_prompt is not None:
                nonlocal initial_prompt_tokens
                initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
                all_tokens.extend(initial_prompt_tokens)

    audio_features = None

    def decode_with_fallback(seg: torch.Tensor,
                             ts_token_mask: torch.Tensor = None) \
            -> DecodingResult:
        nonlocal audio_features
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result, audio_features = decode_stable(model,
                                                          seg,
                                                          options,
                                                          ts_token_mask=ts_token_mask if suppress_ts_tokens else None,
                                                          audio_features=audio_features)

            needs_fallback = False
            if (
                    compression_ratio_threshold is not None
                    and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                    logprob_threshold is not None
                    and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence

            if not needs_fallback:
                break

        return decode_result

    seek_sample = 0  # samples
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    def new_segment(
            *, start: float, end: float, tokens: torch.Tensor, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": round(seek_sample / SAMPLE_RATE, 3),  # units in seconds
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    punctuations = prepend_punctuations + append_punctuations

    initial_duration = audio.get_duration(2)

    nonspeech_predictor = NonSpeechPredictor(
        vad=vad if suppress_silence else None,
        mask_pad_func=pad_or_trim,
        get_mask=suppress_ts_tokens,
        min_word_dur=min_word_dur,
        q_levels=q_levels,
        k_size=k_size,
        vad_threshold=vad_threshold,
        vad_onnx=vad_onnx,
        vad_window=512,
        sampling_rate=SAMPLE_RATE,
        verbose=None if audio.stream else verbose,
        store_timings=True,
        min_silence_dur=min_silence_dur
    )
    audio.update_post_prep_callback(nonspeech_predictor.get_on_prep_callback(audio.stream))

    with tqdm(total=initial_duration, unit='sec', disable=verbose is not False, desc=task.title()) as tqdm_pbar:

        def update_pbar(new_total=None):
            nonlocal audio_features
            audio_features = None
            curr_total_duration = audio.get_duration(2) if new_total is None else new_total
            if curr_total_duration != tqdm_pbar.total:
                tqdm_pbar.total = curr_total_duration
                tqdm_pbar.refresh()
            seek_duration = min(curr_total_duration, round(seek_sample / SAMPLE_RATE, 2))
            if not tqdm_pbar.disable:
                tqdm_pbar.update(seek_duration - tqdm_pbar.n)
            if progress_callback is not None:
                progress_callback(seek_duration, curr_total_duration)

        def update_seek():
            nonlocal seek_sample
            seek_sample += segment_samples

        def fast_forward():
            # fast-forward to the next segment boundary
            update_seek()
            update_pbar()

        while True:
            audio_segment, new_seek = audio.next_valid_chunk(seek_sample, N_SAMPLES)
            if audio_segment is None:
                break
            if new_seek != seek_sample:
                seek_sample = new_seek
                update_pbar()
            time_offset = seek_sample / SAMPLE_RATE
            segment_samples = audio_segment.shape[-1]
            segment_duration = segment_samples / SAMPLE_RATE

            silence_preds = nonspeech_predictor.predict(audio_segment, offset=time_offset)
            segment_silence_timing = silence_preds['timings'] if suppress_silence else None
            ts_token_mask = silence_preds['mask'] if suppress_ts_tokens else None
            is_silent_segment = silence_preds['is_silent']

            if is_silent_segment:
                fast_forward()
                continue

            if nonspeech_skip and silence_preds['timings'] is not None:
                silence_starts = silence_preds['timings'][0] - time_offset
                silence_ends = silence_preds['timings'][1] - time_offset
                silence_durations = silence_ends - silence_starts
                skip_silence_indices = np.flatnonzero(silence_durations >= nonspeech_skip)
                if len(skip_silence_indices):
                    skip_idx = skip_silence_indices[0]
                    if silence_starts[skip_idx] < min_word_dur or int(silence_starts[skip_idx] * SAMPLE_RATE) == 0:
                        segment_samples = round(silence_ends[skip_idx] * SAMPLE_RATE)
                        fast_forward()
                        continue
                    audio_segment = audio_segment[..., :int(silence_starts[skip_idx] * SAMPLE_RATE)]
                    segment_samples = audio_segment.shape[-1]
                    segment_duration = segment_samples / SAMPLE_RATE

            sample_padding = max(N_SAMPLES - segment_samples, 0)
            mel_segment = log_mel_spectrogram(audio_segment, model.dims.n_mels, padding=sample_padding)
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device=model.device, dtype=dtype)

            detect_language()
            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(mel_segment, ts_token_mask=ts_token_mask)
            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    fast_forward()
                    continue

            current_segments = []

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                            sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=round(time_offset + start_timestamp_pos * time_precision, 3),
                            end=round(time_offset + min(end_timestamp_pos * time_precision, segment_duration), 3),
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (
                        len(timestamps) > 0
                        and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    end_timestamp_pos = (
                            timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = min(end_timestamp_pos * time_precision, segment_duration)
                else:
                    end_timestamp_pos = 0

                current_segments.append(
                    new_segment(
                        start=round(time_offset, 3),
                        end=round(time_offset + duration, 3),
                        tokens=tokens,
                        result=result,
                    )
                )

            for i in reversed(range(len(current_segments))):
                seg = current_segments[i]
                if seg["text"].strip() in punctuations:
                    del current_segments[i]
                else:
                    if word_timestamps:
                        if seg["start"] == seg["end"]:
                            del current_segments[i]
                    else:
                        prev_i = i+1
                        if prev_i >= len(current_segments):
                            max_end = seg['end']
                        else:
                            max_end = current_segments[prev_i]['start']
                        if seg["start"] > seg["end"]:
                            if (
                                    i != 0 and
                                    current_segments[i - 1]['end'] != current_segments[i - 1]['start'] and
                                    current_segments[i - 1]['end'] < max_end
                            ):
                                new_start = current_segments[i-1]['end']
                            else:
                                new_start = max_end
                            seg['start'] = new_start

            num_samples = (
                min(round(end_timestamp_pos * N_SAMPLES_PER_TOKEN), segment_samples)
                if end_timestamp_pos > 0 else
                segment_samples
            )

            if word_timestamps:
                add_word_timestamps_stable(
                    segments=current_segments,
                    model=model,
                    tokenizer=tokenizer,
                    mel=mel_segment,
                    num_samples=num_samples,
                    prepend_punctuations=prepend_punctuations,
                    append_punctuations=append_punctuations,
                    audio_features=audio_features,
                    ts_num=ts_num,
                    ts_noise=ts_noise,
                    split_callback=split_callback,
                    gap_padding=gap_padding,
                    extra_models=extra_models,
                    dynamic_heads=dynamic_heads
                )

                for i in reversed(range(len(current_segments))):
                    zero_duration_percent = (
                        np.array(
                            [w['start'] == w['end'] for w in current_segments[i]['words']]
                        )
                        .astype(np.float16)
                        .mean()
                    )
                    if zero_duration_percent > max_instant_words:
                        del current_segments[i]

                if avg_prob_threshold and current_segments:
                    if (
                            single_timestamp_ending and
                            (np.mean([w['probability'] for s in current_segments for w in s['words']]) <
                             avg_prob_threshold)
                    ):
                        num_samples = segment_samples
                        current_segments = []
                    else:
                        num_samples = round((current_segments[-1]['words'][-1]['end']-time_offset) * SAMPLE_RATE)

            if len(current_segments) == 0:
                fast_forward()
                continue

            if segment_silence_timing is not None:
                for seg_i, segment in enumerate(current_segments):
                    segment = Segment(**segment, ignore_unused_args=True).suppress_silence(
                            *segment_silence_timing,
                            min_word_dur=min_word_dur,
                            word_level=suppress_word_ts,
                            nonspeech_error=nonspeech_error,
                            use_word_position=use_word_position,
                        )
                    if verbose:
                        safe_print(segment.to_display_str())
                    current_segments[seg_i] = segment.to_dict()

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(current_segments, start=len(all_segments))
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )
            if not single_timestamp_ending or avg_prob_threshold:
                segment_samples = num_samples

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            fast_forward()

        # final update
        update_pbar(seek_sample / SAMPLE_RATE)

    if model.device != torch.device('cpu'):
        torch.cuda.empty_cache()

    audio.terminate()
    nonspeech_predictor.finalize_timings()

    text = '' if tokenizer is None else tokenizer.decode(all_tokens[len(initial_prompt_tokens):])
    final_result = WhisperResult(
        dict(
            text=text,
            segments=all_segments,
            language=language,
            time_scale=time_scale
        ),
        force_order=not word_timestamps
    )
    if word_timestamps and regroup:
        final_result.regroup(regroup)

    if time_scale is not None:
        final_result.rescale_time(1 / time_scale)

    if len(final_result.text) == 0:
        warnings.warn(f'Failed to {task} audio. Result contains no text. ')

    if suppress_silence and (final_nonspeech_timings := nonspeech_predictor.nonspeech_timings):
        final_result.update_nonspeech_sections(*final_nonspeech_timings)

    return final_result


def transcribe_minimal(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
        *,
        verbose: Optional[bool] = False,
        word_timestamps: bool = True,
        regroup: Union[bool, str] = True,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        use_word_position: bool = True,
        q_levels: int = 20,
        k_size: int = 5,
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: bool = False,
        demucs_options: dict = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: float = 0.1,
        nonspeech_error: float = 0.1,
        only_voice_freq: bool = False,
        only_ffmpeg: bool = False,
        **options) \
        -> WhisperResult:
    """
    Transcribe audio using Whisper.

    This is uses the original whisper transcribe function, :func:`whisper.transcribe.transcribe`, while still allowing
    additional preprocessing and postprocessing. The preprocessing performed on the audio includes: voice isolation /
    noise removal and low/high-pass filter. The postprocessing performed on the transcription result includes:
    adjusting timestamps with VAD and custom regrouping segments based punctuation and speech gaps.

    Parameters
    ----------
    model : whisper.model.Whisper
        An instance of Whisper ASR model.
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
        If audio is ``numpy.ndarray`` or ``torch.Tensor``, the audio must be already at sampled to 16kHz.
    verbose : bool or None, default False
        Whether to display the text being decoded to the console.
        Displays all the details if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.
    word_timestamps : bool, default True
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.
        Disabling this will prevent segments from splitting/merging properly.
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
    min_word_dur : float, default 0.1
        Shortest duration each word is allowed to reach for silence suppression.
    nonspeech_error : float, default 0.1
        Relative error of non-speech sections that appear in between a word for silence suppression.
    only_voice_freq : bool, default False
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
    only_ffmpeg : bool, default False
        Whether to use only FFmpeg (instead of not yt-dlp) for URls
    options
        Additional options used for :func:`whisper.transcribe.transcribe` and
        :func:`stable_whisper.non_whisper.transcribe_any`.
    Returns
    -------
    stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the transcription of ``audio``.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_model('base')
    >>> result = model.transcribe_minimal('audio.mp3', vad=True)
    >>> result.to_srt_vtt('audio.srt')
    Saved: audio.srt
    """
    audioloader_not_supported(audio)
    from ..non_whisper import transcribe_any
    inference_kwargs = dict(
        model=model,
        audio=audio,
        word_timestamps=word_timestamps,
        verbose=verbose
    )
    extra_options = isolate_useful_options(options, transcribe_any, True)
    denoiser, denoiser_options = convert_demucs_kwargs(
        denoiser, denoiser_options, demucs=demucs, demucs_options=demucs_options
    )
    if not isinstance(audio, (str, bytes)):
        if 'input_sr' not in extra_options:
            extra_options['input_sr'] = SAMPLE_RATE

    if denoiser or only_voice_freq:
        if 'audio_type' not in extra_options:
            extra_options['audio_type'] = 'torch'
        if 'model_sr' not in extra_options:
            extra_options['model_sr'] = SAMPLE_RATE
    inference_kwargs.update(options)
    return transcribe_any(
        inference_func=whisper.transcribe,
        audio=audio,
        inference_kwargs=inference_kwargs,
        verbose=verbose,
        regroup=regroup,
        suppress_silence=suppress_silence,
        suppress_word_ts=suppress_word_ts,
        q_levels=q_levels,
        k_size=k_size,
        denoiser=denoiser,
        denoiser_options=denoiser_options,
        vad=vad,
        vad_threshold=vad_threshold,
        vad_onnx=vad_onnx,
        min_word_dur=min_word_dur,
        nonspeech_error=nonspeech_error,
        use_word_position=use_word_position,
        only_voice_freq=only_voice_freq,
        only_ffmpeg=only_ffmpeg,
        force_order=True,
        **extra_options
    )


def modify_model(model: "Whisper"):
    """
    Modify an instance if :class:`whisper.model.Whisper`.

    The following are performed:
    -replace :meth:`whisper.model.Whisper.transcribe` with :func:`stable_whisper.whisper_word_level.transcribe_stable`
    -assign :meth:`whisper.model.transcribe_minimal` to :func:`stable_whisper.whisper_word_level.transcribe_minimal`
    -assign :meth:`whisper.model.Whisper.transcribe_original` to :meth:`whisper.model.Whisper.transcribe`
    -assign :meth:`whisper.model.Whisper.align` to :func:`stable_whisper.alignment.align`
    -assign :meth:`whisper.model.Whisper.locate` to :func:`stable_whisper.alignment.locate`
    """
    model.transcribe = MethodType(transcribe_stable, model)
    model.transcribe_minimal = MethodType(transcribe_minimal, model)
    model.transcribe_original = MethodType(whisper.transcribe, model)
    from ..alignment import align, refine, locate
    model.align = MethodType(align, model)
    model.refine = MethodType(refine, model)
    model.locate = MethodType(locate, model)


# modified version of whisper.load_model
def load_model(name: str, device: Optional[Union[str, torch.device]] = None,
               download_root: str = None, in_memory: bool = False,
               cpu_preload: bool = True, dq: bool = False, engine: Optional[str] = None) -> "Whisper":
    """
    Load an instance if :class:`whisper.model.Whisper`.

    Parameters
    ----------
    name : {'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1',
        'large-v2', 'large-v3', or 'large'}
        One of the official model names listed by :func:`whisper.available_models`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : str or torch.device, optional
        PyTorch device to put the model into.
    download_root : str, optional
        Path to download the model files; by default, it uses "~/.cache/whisper".
    in_memory : bool, default False
        Whether to preload the model weights into host memory.
    cpu_preload : bool, default True
        Load model into CPU memory first then move model to specified device
        to reduce GPU memory usage when loading model
    dq : bool, default False
        Whether to apply Dynamic Quantization to model to reduced memory usage and increase inference speed
        but at the cost of a slight decrease in accuracy. Only for CPU.
    engine : str, optional
        Engine for Dynamic Quantization.

    Returns
    -------
    model : "Whisper"
        The Whisper ASR model instance.

    Notes
    -----
    The overhead from ``dq = True`` might make inference slower for models smaller than 'large'.
    """
    if whisper is None:
        from ..whisper_compatibility import whisper_not_available
        whisper_not_available()
    if device is None or dq:
        device = "cuda" if torch.cuda.is_available() and not dq else "cpu"
    if cpu_preload:
        model = whisper.load_model(name, device='cpu', download_root=download_root, in_memory=in_memory)
        cuda_index = None
        if isinstance(device, str) and device.startswith('cuda'):
            try:
                cuda_index = [] if device == 'cuda' else [int(device.split(':')[-1])]
            except ValueError:
                pass
        model = model.to(device=device) if cuda_index is None else model.cuda(*cuda_index)
    else:
        model = whisper.load_model(name, device=device, download_root=download_root, in_memory=in_memory)
    modify_model(model)
    if dq:
        from ..quantization import ptdq_linear
        ptdq_linear(model, engine=engine)
    return model
