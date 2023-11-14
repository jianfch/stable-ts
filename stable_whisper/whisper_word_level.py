import warnings
import torch
import numpy as np
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Callable
from types import MethodType
from tqdm import tqdm

import whisper
from whisper.audio import (
    SAMPLE_RATE, N_FRAMES, HOP_LENGTH, N_SAMPLES, N_SAMPLES_PER_TOKEN, TOKENS_PER_SECOND, FRAMES_PER_SECOND, N_FFT,
    pad_or_trim, log_mel_spectrogram
)
from whisper.utils import exact_div
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.decoding import DecodingOptions, DecodingResult

from .audio import prep_audio
from .decode import decode_stable
from .result import WhisperResult, Segment
from .timing import add_word_timestamps_stable
from .stabilization import get_vad_silence_func, wav2mask, mask2timing, timing2mask
from .non_whisper import transcribe_any
from .utils import isolate_useful_options, safe_print
from .whisper_compatibility import warn_compatibility_issues, get_tokenizer

if TYPE_CHECKING:
    from whisper.model import Whisper

__all__ = ['modify_model', 'load_model', 'load_faster_whisper']

warnings.filterwarnings('ignore', module='whisper', message='.*Triton.*', category=UserWarning)


# modified version of whisper.transcribe.transcribe
def transcribe_stable(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor, bytes],
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
        ts_noise: float = 0.1,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        q_levels: int = 20,
        k_size: int = 5,
        time_scale: float = None,
        demucs: Union[bool, torch.nn.Module] = False,
        demucs_output: str = None,
        demucs_options: dict = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: float = 0.1,
        only_voice_freq: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        mel_first: bool = False,
        split_callback: Callable = None,
        suppress_ts_tokens: bool = False,
        gap_padding: str = ' ...',
        only_ffmpeg: bool = False,
        max_instant_words: float = 0.5,
        avg_prob_threshold: Optional[float] = None,
        progress_callback: Callable = None,
        ignore_compatibility: bool = False,
        **decode_options) \
        -> WhisperResult:
    """
    Transcribe audio using Whisper.

    This is a modified version of :func:`whisper.transcribe.transcribe` with slightly different decoding logic while
    allowing additional preprocessing and postprocessing. The preprocessing performed on the audio includes: isolating
    voice / removing noise with Demucs and low/high-pass filter. The postprocessing performed on the transcription
    result includes: adjusting timestamps with VAD and custom regrouping segments based punctuation and speech gaps.

    Parameters
    ----------
    model : whisper.model.Whisper
        An instance of Whisper ASR model.
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
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
    ts_num : int, default 0, meaning disable this option
        Number of extra timestamp inferences to perform then use average of these extra timestamps.
        An experimental option that might hurt performance.
    ts_noise : float, default 0.1
        Percentage of noise to add to audio_features to perform inferences for ``ts_num``.
    suppress_silence : bool, default True
        Whether to enable timestamps adjustments based on the detected silence.
    suppress_word_ts : bool, default True
        Whether to adjust word timestamps based on the detected silence. Only enabled if ``suppress_silence = True``.
    q_levels : int, default 20
        Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
        Acts as a threshold to marking sound as silent.
        Fewer levels will increase the threshold of volume at which to mark a sound as silent.
    k_size : int, default 5
        Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
        Recommend 5 or 3; higher sizes will reduce detection of silence.
    time_scale : float, optional
        Factor for scaling audio duration for inference.
        Greater than 1.0 'slows down' the audio, and less than 1.0 'speeds up' the audio. None is same as 1.0.
        A factor of 1.5 will stretch 10s audio to 15s for inference. This increases the effective resolution
        of the model but can increase word error rate.
    demucs : bool or torch.nn.Module, default False
        Whether to preprocess ``audio`` with Demucs to isolate vocals / remove noise. Set ``demucs`` to an instance of
        a Demucs model to avoid reloading the model for each run.
        Demucs must be installed to use. Official repo. https://github.com/facebookresearch/demucs.
    demucs_output : str, optional
        Path to save the vocals isolated by Demucs as WAV file. Ignored if ``demucs = False``.
        Demucs must be installed to use. Official repo. https://github.com/facebookresearch/demucs.
    demucs_options : dict, optional
        Options to use for :func:`stable_whisper.audio.demucs_audio`.
    vad : bool, default False
        Whether to use Silero VAD to generate timestamp suppression mask.
        Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
    vad_threshold : float, default 0.35
        Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
    vad_onnx : bool, default False
        Whether to use ONNX for Silero VAD.
    min_word_dur : float, default 0.1
        Only allow suppressing timestamps that result in word durations greater than this value.
    only_voice_freq : bool, default False
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
    prepend_punctuations : str, default '"\'“¿([{-)'
        Punctuations to prepend to next word.
    append_punctuations : str, default '.。,，!！?？:：”)]}、)'
        Punctuations to append to previous word.
    mel_first : bool, default False
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
    progress_callback : Callable, optional
        A function that will be called when transcription progress is updated.
        The callback need two parameters.
        The first parameter is a float for seconds of the audio that has been transcribed.
        The second parameter is a float for total duration of audio in seconds.
    ignore_compatibility : bool, default False
        Whether to ignore warnings for compatibility issues with the detected Whisper version.
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
    stable_whisper.whisper_word_level.load_faster_whisper.faster_transcribe : Return
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
        warnings.warn('``time_scale`` is deprecated. It will not affect results.',
                      DeprecationWarning, stacklevel=2)
    if decode_options.pop('input_sr', None):
        warnings.warn('``input_sr`` is deprecated. '
                      '``audio`` of types numpy.ndarray and torch.Tensor inputs must be already at 16kHz. '
                      'To higher sample rates for ``audio`` use str or bytes.',
                      DeprecationWarning, stacklevel=2)
    if not demucs_options:
        demucs_options = {}
    if demucs_output:
        if 'save_path' not in demucs_options:
            demucs_options['save_path'] = demucs_output
        warnings.warn('``demucs_output`` is deprecated. Use ``demucs_options`` with ``save_path`` instead. '
                      'E.g. demucs_options=dict(save_path="demucs_output.mp3")',
                      DeprecationWarning, stacklevel=2)
    if 'device' not in demucs_options:
        demucs_options['device'] = device
    audio = prep_audio(
        audio,
        demucs=demucs,
        demucs_options=demucs_options,
        only_voice_freq=only_voice_freq,
        only_ffmpeg=only_ffmpeg,
        verbose=verbose
    )
    sample_padding = int(N_FFT // 2) + 1
    whole_mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=sample_padding) if mel_first else None
    tokenizer = None
    language = None
    initial_prompt_tokens = []
    task = decode_options.get("task", "transcribe")

    def detect_language():
        nonlocal tokenizer
        if tokenizer is None:
            if decode_options.get("language", None) is None and model:
                if not model.is_multilingual:
                    decode_options["language"] = "en"
                else:
                    if verbose:
                        print("Detecting language using up to 30 seconds following first non-silent sample. "
                              "Use `--language` to specify the language")
                    timing_mask = None
                    if segment_silence_timing is not None:
                        timing_mask = np.logical_and(
                            segment_silence_timing[0] <= time_offset,
                            segment_silence_timing[1] >= time_offset
                        )
                    start_sample = (
                        None
                        if segment_silence_timing is None or not timing_mask.any() else
                        round(segment_silence_timing[1][timing_mask.nonzero()[0]][0] * SAMPLE_RATE)
                    )
                    if start_sample is None:
                        nonlocal mel_segment
                        curr_mel_segment = mel_segment
                    else:
                        if whole_mel is None:
                            curr_mel_segment = log_mel_spectrogram(
                                audio[..., start_sample:start_sample+N_SAMPLES],
                                model.dims.n_mels,
                                padding=sample_padding
                            )
                        else:
                            start_frame = int(start_sample/HOP_LENGTH)
                            curr_mel_segment = whole_mel[..., start_frame:start_frame+N_FRAMES]
                        curr_mel_segment = pad_or_trim(curr_mel_segment, N_FRAMES).to(device=device, dtype=dtype)
                    _, probs = model.detect_language(curr_mel_segment)
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

            if word_timestamps and task == "translate":
                warnings.warn("Word-level timestamps on translations may not be reliable.")

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

    total_samples = audio.shape[-1]
    total_duration = round(total_samples / SAMPLE_RATE, 2)
    n_samples_per_frame = exact_div(N_SAMPLES_PER_TOKEN * TOKENS_PER_SECOND, FRAMES_PER_SECOND)

    silence_timing = None
    if suppress_silence and vad:
        silence_timing = get_vad_silence_func(onnx=vad_onnx, verbose=verbose)(audio, speech_threshold=vad_threshold)

    with tqdm(total=total_duration, unit='sec', disable=verbose is not False, desc=task.title()) as tqdm_pbar:

        def update_pbar():
            nonlocal audio_features
            audio_features = None
            seek_duration = min(total_duration, round(seek_sample / SAMPLE_RATE, 2))
            if not tqdm_pbar.disable:
                tqdm_pbar.update(seek_duration - tqdm_pbar.n)
            if progress_callback is not None:
                progress_callback(seek=seek_duration, total=total_duration)

        def update_seek():
            nonlocal seek_sample
            seek_sample += segment_samples

        def fast_forward():
            # fast-forward to the next segment boundary
            update_seek()
            update_pbar()

        while seek_sample < audio.shape[-1]:
            seek_sample_end = seek_sample + N_SAMPLES
            audio_segment = audio[seek_sample:seek_sample_end]
            time_offset = seek_sample / SAMPLE_RATE
            segment_samples = audio_segment.shape[-1]
            segment_duration = segment_samples / SAMPLE_RATE

            mel_segment = (
                log_mel_spectrogram(audio_segment, model.dims.n_mels, padding=sample_padding)
                if whole_mel is None else
                whole_mel[..., round(seek_sample / n_samples_per_frame): round(seek_sample_end / n_samples_per_frame)]
            )

            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device=model.device, dtype=dtype)

            segment_silence_timing = None
            ts_token_mask = None
            if suppress_silence:
                if silence_timing is None:
                    ts_token_mask = wav2mask(audio_segment, q_levels=q_levels, k_size=k_size)
                    segment_silence_timing = mask2timing(ts_token_mask, time_offset=time_offset)
                else:
                    timing_indices = np.logical_and(
                        silence_timing[1] > time_offset,
                        silence_timing[0] < time_offset + segment_duration
                    )
                    segment_silence_timing = (silence_timing[0][timing_indices], silence_timing[1][timing_indices])

                    ts_token_mask = timing2mask(*segment_silence_timing, size=1501, time_offset=time_offset)

                    if mn := timing_indices.argmax():
                        silence_timing = (silence_timing[0][mn:], silence_timing[1][mn:])

                if ts_token_mask is not None:
                    if ts_token_mask.all():  # segment is silent
                        fast_forward()
                        continue
                    ts_token_mask = pad_or_trim(ts_token_mask, 1501)

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

            # if a segment is instantaneous or does not contain text, remove it
            for i in reversed(range(len(current_segments))):
                seg = current_segments[i]
                if seg["start"] == seg["end"] or seg["text"].strip() in punctuations:
                    del current_segments[i]

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
                    gap_padding=gap_padding
                )

                # if [max_instant_words] of the words in a segment are instantaneous, remove it
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
                    segment = Segment(**segment).suppress_silence(
                            *segment_silence_timing,
                            min_word_dur=min_word_dur,
                            word_level=suppress_word_ts
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
        update_pbar()

    if model.device != torch.device('cpu'):
        torch.cuda.empty_cache()

    text = '' if tokenizer is None else tokenizer.decode(all_tokens[len(initial_prompt_tokens):])
    final_result = WhisperResult(dict(text=text,
                                      segments=all_segments,
                                      language=language,
                                      time_scale=time_scale))
    if word_timestamps and regroup:
        final_result.regroup(regroup)

    if time_scale is not None:
        final_result.rescale_time(1 / time_scale)

    if len(final_result.text) == 0:
        warnings.warn(f'Failed to {task} audio. Result contains no text. ')

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
        q_levels: int = 20,
        k_size: int = 5,
        demucs: bool = False,
        demucs_output: str = None,
        demucs_options: dict = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: float = 0.1,
        only_voice_freq: bool = False,
        only_ffmpeg: bool = False,
        **options) \
        -> WhisperResult:
    """
    Transcribe audio using Whisper.

    This is uses the original whisper transcribe function, :func:`whisper.transcribe.transcribe`, while still allowing
    additional preprocessing and postprocessing. The preprocessing performed on the audio includes: isolating voice /
    removing noise with Demucs and low/high-pass filter. The postprocessing performed on the transcription
    result includes: adjusting timestamps with VAD and custom regrouping segments based punctuation and speech gaps.

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
    q_levels : int, default 20
        Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
        Acts as a threshold to marking sound as silent.
        Fewer levels will increase the threshold of volume at which to mark a sound as silent.
    k_size : int, default 5
        Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
        Recommend 5 or 3; higher sizes will reduce detection of silence.
    demucs : bool or torch.nn.Module, default False
        Whether to preprocess ``audio`` with Demucs to isolate vocals / remove noise. Set ``demucs`` to an instance of
        a Demucs model to avoid reloading the model for each run.
        Demucs must be installed to use. Official repo, https://github.com/facebookresearch/demucs.
    demucs_output : str, optional
        Path to save the vocals isolated by Demucs as WAV file. Ignored if ``demucs = False``.
        Demucs must be installed to use. Official repo, https://github.com/facebookresearch/demucs.
    demucs_options : dict, optional
        Options to use for :func:`stable_whisper.audio.demucs_audio`.
    vad : bool, default False
        Whether to use Silero VAD to generate timestamp suppression mask.
        Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
    vad_threshold : float, default 0.35
        Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
    vad_onnx : bool, default False
        Whether to use ONNX for Silero VAD.
    min_word_dur : float, default 0.1
        Only allow suppressing timestamps that result in word durations greater than this value.
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
    inference_kwargs = dict(
        model=model,
        audio=audio,
        word_timestamps=word_timestamps,
        verbose=verbose
    )
    extra_options = isolate_useful_options(options, transcribe_any, True)
    if demucs or only_voice_freq:
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
        demucs=demucs,
        demucs_output=demucs_output,
        demucs_options=demucs_options,
        vad=vad,
        vad_threshold=vad_threshold,
        vad_onnx=vad_onnx,
        min_word_dur=min_word_dur,
        only_voice_freq=only_voice_freq,
        only_ffmpeg=only_ffmpeg,
        force_order=True,
        **extra_options
    )


def load_faster_whisper(model_size_or_path: str, **model_init_options):
    """
    Load an instance of :class:`faster_whisper.WhisperModel`.

    Parameters
    ----------
    model_size_or_path : {'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1',
        'large-v2', or 'large'}
        Size of the model.

    model_init_options
        Additional options to use for initialization of :class:`faster_whisper.WhisperModel`.

    Returns
    -------
    faster_whisper.WhisperModel
        A modified instance with :func:`stable_whisper.whisper_word_level.load_faster_whisper.faster_transcribe`
        assigned to :meth:`faster_whisper.WhisperModel.transcribe_stable`.
    """
    from faster_whisper import WhisperModel
    faster_model = WhisperModel(model_size_or_path, **model_init_options)

    def _inner_transcribe(model, audio, verbose, **faster_transcribe_options):
        if isinstance(audio, bytes):
            import io
            audio = io.BytesIO(audio)
        progress_callback = faster_transcribe_options.pop('progress_callback', None)
        segments, info = model.transcribe(audio, **faster_transcribe_options)
        language = LANGUAGES.get(info.language, info.language)
        if verbose is not None:
            print(f'Detected Language: {language}')
            print(f'Transcribing with faster-whisper ({model_size_or_path})...\r', end='')

        final_segments = []
        task = faster_transcribe_options.get('task', 'transcribe').title()
        total_duration = round(info.duration, 2)

        with tqdm(total=total_duration, unit='sec', disable=verbose is not False, desc=task) as tqdm_pbar:

            def update_pbar(seek):
                tqdm_pbar.update(seek - tqdm_pbar.n)
                if progress_callback is not None:
                    progress_callback(seek, total_duration)

            for segment in segments:
                segment = segment._asdict()
                if (words := segment.get('words')) is not None:
                    segment['words'] = [w._asdict() for w in words]
                else:
                    del segment['words']
                if verbose:
                    safe_print(Segment(**segment).to_display_str())
                final_segments.append(segment)
                update_pbar(segment["end"])
            update_pbar(tqdm_pbar.total)

        if verbose:
            print(f'Completed transcription with faster-whisper ({model_size_or_path}).')

        return dict(language=language, segments=final_segments)

    def faster_transcribe(
            model: WhisperModel,
            audio: Union[str, bytes, np.ndarray],
            *,
            word_timestamps: bool = True,
            verbose: Optional[bool] = False,
            regroup: Union[bool, str] = True,
            suppress_silence: bool = True,
            suppress_word_ts: bool = True,
            q_levels: int = 20,
            k_size: int = 5,
            demucs: bool = False,
            demucs_output: str = None,
            demucs_options: dict = None,
            vad: bool = False,
            vad_threshold: float = 0.35,
            vad_onnx: bool = False,
            min_word_dur: float = 0.1,
            only_voice_freq: bool = False,
            only_ffmpeg: bool = False,
            check_sorted: bool = True,
            progress_callback: Callable = None,
            **options
    ) -> WhisperResult:
        """
        Transcribe audio using faster-whisper (https://github.com/guillaumekln/faster-whisper).

        This is uses the transcribe method from faster-whisper, :meth:`faster_whisper.WhisperModel.transcribe`, while
        still allowing additional preprocessing and postprocessing. The preprocessing performed on the audio includes:
        isolating voice / removing noise with Demucs and low/high-pass filter. The postprocessing performed on the
        transcription result includes: adjusting timestamps with VAD and custom regrouping segments based punctuation
        and speech gaps.

        Parameters
        ----------
        model : faster_whisper.WhisperModel
            The faster-whisper ASR model instance.
        audio : str or numpy.ndarray or torch.Tensor or bytes
            Path/URL to the audio file, the audio waveform, or bytes of audio file.
            If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
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
        q_levels : int, default 20
            Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
            Acts as a threshold to marking sound as silent.
            Fewer levels will increase the threshold of volume at which to mark a sound as silent.
        k_size : int, default 5
            Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
            Recommend 5 or 3; higher sizes will reduce detection of silence.
        demucs : bool or torch.nn.Module, default False
            Whether to preprocess ``audio`` with Demucs to isolate vocals / remove noise. Set ``demucs`` to an instance
            of a Demucs model to avoid reloading the model for each run.
            Demucs must be installed to use. Official repo, https://github.com/facebookresearch/demucs.
        demucs_output : str, optional
            Path to save the vocals isolated by Demucs as WAV file. Ignored if ``demucs = False``.
            Demucs must be installed to use. Official repo, https://github.com/facebookresearch/demucs.
        demucs_options : dict, optional
            Options to use for :func:`stable_whisper.audio.demucs_audio`.
        vad : bool, default False
            Whether to use Silero VAD to generate timestamp suppression mask.
            Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
        vad_threshold : float, default 0.35
            Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
        vad_onnx : bool, default False
            Whether to use ONNX for Silero VAD.
        min_word_dur : float, default 0.1
            Only allow suppressing timestamps that result in word durations greater than this value.
        only_voice_freq : bool, default False
            Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
        only_ffmpeg : bool, default False
            Whether to use only FFmpeg (instead of not yt-dlp) for URls
        check_sorted : bool, default True
            Whether to raise an error when timestamps returned by faster-whipser are not in ascending order.
        progress_callback : Callable, optional
            A function that will be called when transcription progress is updated.
            The callback need two parameters.
            The first parameter is a float for seconds of the audio that has been transcribed.
            The second parameter is a float for total duration of audio in seconds.
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
        >>> model = stable_whisper.load_faster_whisper('base')
        >>> result = model.transcribe_stable('audio.mp3', vad=True)
        >>> result.to_srt_vtt('audio.srt')
        Saved: audio.srt
        """
        extra_options = isolate_useful_options(options, transcribe_any, pop=True)
        if demucs or only_voice_freq:
            if 'audio_type' not in extra_options:
                extra_options['audio_type'] = 'numpy'
            if 'model_sr' not in extra_options:
                extra_options['model_sr'] = SAMPLE_RATE
        faster_whisper_options = options
        faster_whisper_options['model'] = model
        faster_whisper_options['audio'] = audio
        faster_whisper_options['word_timestamps'] = word_timestamps
        faster_whisper_options['verbose'] = verbose
        faster_whisper_options['progress_callback'] = progress_callback
        if not demucs_options:
            demucs_options = {}
        if demucs_output:
            if 'save_path' not in demucs_options:
                demucs_options['save_path'] = demucs_output
            warnings.warn('``demucs_output`` is deprecated. Use ``demucs_options`` with ``save_path`` instead. '
                          'E.g. demucs_options=dict(save_path="demucs_output.mp3")',
                          DeprecationWarning, stacklevel=2)

        return transcribe_any(
            inference_func=_inner_transcribe,
            audio=audio,
            inference_kwargs=faster_whisper_options,
            verbose=verbose,
            regroup=regroup,
            suppress_silence=suppress_silence,
            suppress_word_ts=suppress_word_ts,
            q_levels=q_levels,
            k_size=k_size,
            demucs=demucs,
            demucs_options=demucs_options,
            vad=vad,
            vad_threshold=vad_threshold,
            vad_onnx=vad_onnx,
            min_word_dur=min_word_dur,
            only_voice_freq=only_voice_freq,
            only_ffmpeg=only_ffmpeg,
            force_order=True,
            check_sorted=check_sorted,
            **extra_options
        )

    faster_model.transcribe_stable = MethodType(faster_transcribe, faster_model)
    from .alignment import align
    faster_model.align = MethodType(align, faster_model)

    return faster_model


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
    from .alignment import align, refine, locate
    model.align = MethodType(align, model)
    model.refine = MethodType(refine, model)
    model.locate = MethodType(locate, model)


# modified version of whisper.load_model
def load_model(name: str, device: Optional[Union[str, torch.device]] = None,
               download_root: str = None, in_memory: bool = False,
               cpu_preload: bool = True, dq: bool = False) -> "Whisper":
    """
    Load an instance if :class:`whisper.model.Whisper`.

    Parameters
    ----------
    name : {'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1',
        'large-v2', or 'large'}
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

    Returns
    -------
    model : "Whisper"
        The Whisper ASR model instance.

    Notes
    -----
    The overhead from ``dq = True`` might make inference slower for models smaller than 'large'.
    """
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
        from .quantization import ptdq_linear
        ptdq_linear(model)
    return model


# modified version of whisper.transcribe.cli
def cli():
    import argparse
    import os
    from os.path import splitext, split, isfile, join
    from whisper import available_models
    from whisper.utils import optional_int, optional_float
    from .utils import str_to_valid_type, get_func_parameters

    str2val = {"true": True, "false": False, "1": True, "0": False}

    def str2bool(string: str) -> bool:
        string = string.lower()
        if string in str2val:
            return str2val[string]
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    def valid_model_name(name):
        if name in available_models() or os.path.exists(name):
            return name
        raise ValueError(
            f"model should be one of {available_models()} or path to a model checkpoint"
        )

    def update_options_with_args(arg_key: str, options: Optional[dict] = None, pop: bool = False):
        extra_options = args.pop(arg_key) if pop else args.get(arg_key)
        if not extra_options:
            return
        extra_options = [kv.split('=', maxsplit=1) for kv in extra_options]
        missing_val = [kv[0] for kv in extra_options if len(kv) == 1]
        if missing_val:
            raise ValueError(f'Following expected values for the following custom options: {missing_val}')
        extra_options = dict((k, str_to_valid_type(v)) for k, v in extra_options)
        if options is None:
            return extra_options
        options.update(extra_options)

    OUTPUT_FORMATS_METHODS = {
        "srt": "to_srt_vtt",
        "ass": "to_ass",
        "json": "save_as_json",
        "vtt": "to_srt_vtt",
        "tsv": "to_tsv"}

    OUTPUT_FORMATS = set(OUTPUT_FORMATS_METHODS.keys())

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=str,
                        help="audio/video filepath/URL(s) to transcribe "
                             "or json file(s) to process into [output_format]")
    parser.add_argument("--output", "-o", action="extend", nargs="+", type=str,
                        help="output filepaths(s);"
                             "if not specified, auto-named output file(s) will be saved to "
                             "[output_dir] or current dir if not specified.")
    parser.add_argument("--model", '-m', default="base", type=valid_model_name,
                        help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use for PyTorch inference")
    parser.add_argument("--cpu_preload", type=str2bool, default=True,
                        help="load model into CPU memory first then move model to specified device; "
                             "this reduces GPU memory usage when loading model.")
    parser.add_argument("--output_dir", "-d", type=str,
                        help="directory to save the outputs;"
                             "if a path in [output] does not have parent, that output will be save to this directory")
    parser.add_argument("--output_format", "-f", type=str,
                        help="format of the output file(s); "
                             f"Supported Formats: {OUTPUT_FORMATS}; "
                             "use ',' to separate multiple formats")
    parser.add_argument("--verbose", '-v', type=int, default=1, choices=(0, 1, 2),
                        help="whether to display the text being decoded to the console; "
                             "if 2, display all the details; "
                             "if 1, display progressbar; "
                             "if 0, display nothing")

    parser.add_argument("--dynamic_quantization", "-dq", action='store_true',
                        help="whether to apply Dynamic Quantization to model "
                             "to reduced memory usage (~half less) and increase inference speed "
                             "at cost of slight decrease in accuracy; Only for CPU; "
                             "NOTE: overhead might make inference slower for models smaller than 'large'")

    parser.add_argument("--task", type=str, default="transcribe",
                        choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') "
                             "or X->English translation ('translate')")
    parser.add_argument("--language", '-l', type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--prepend_punctuations", '-pp', type=str, default="\"'“¿([{-",
                        help="Punctuations to prepend to next word")
    parser.add_argument("--append_punctuations", '-ap', type=str, default="\"'.。,，!！?？:：”)]}、",
                        help="Punctuations to append to previous word")

    parser.add_argument("--gap_padding", type=str, default=" ...",
                        help="padding prepend to each segments for word timing alignment;"
                             "used to reduce the probability of model predicting timestamps "
                             "earlier than the first utterance")

    parser.add_argument("--word_timestamps", type=str2bool, default=True,
                        help="extract word-level timestamps using the cross-attention pattern and dynamic time warping,"
                             "and include the timestamps for each word in each segment;"
                             "disabling this will prevent segments from splitting/merging properly.")

    parser.add_argument("--regroup", type=str, default="True",
                        help="whether to regroup all words into segments with more natural boundaries;"
                             "specify string for customizing the regrouping algorithm"
                             "ignored if [word_timestamps]=False.")

    parser.add_argument('--ts_num', type=int, default=0,
                        help="number of extra inferences to perform to find the mean timestamps")
    parser.add_argument('--ts_noise', type=float, default=0.1,
                        help="percentage of noise to add to audio_features to perform inferences for [ts_num]")

    parser.add_argument('--suppress_silence', type=str2bool, default=True,
                        help="whether to suppress timestamp where audio is silent at segment-level"
                             "and word-level if [suppress_word_ts]=True")
    parser.add_argument('--suppress_word_ts', type=str2bool, default=True,
                        help="whether to suppress timestamps where audio is silent at word-level; "
                             "ignored if [suppress_silence]=False")

    parser.add_argument('--suppress_ts_tokens', type=str2bool, default=False,
                        help="whether to use silence mask to suppress silent timestamp tokens during inference; "
                             "increases word accuracy in some cases, but tends reduce 'verbatimness' of the transcript"
                             "ignored if [suppress_silence]=False")

    parser.add_argument("--q_levels", type=int, default=20,
                        help="quantization levels for generating timestamp suppression mask; "
                             "acts as a threshold to marking sound as silent;"
                             "fewer levels will increase the threshold of volume at which to mark a sound as silent")

    parser.add_argument("--k_size", type=int, default=5,
                        help="Kernel size for average pooling waveform to generate suppression mask; "
                             "recommend 5 or 3; higher sizes will reduce detection of silence")

    parser.add_argument('--time_scale', type=float,
                        help="factor for scaling audio duration for inference;"
                             "greater than 1.0 'slows down' the audio; "
                             "less than 1.0 'speeds up' the audio; "
                             "1.0 is no scaling")

    parser.add_argument('--vad', type=str2bool, default=False,
                        help='whether to use Silero VAD to generate timestamp suppression mask; '
                             'Silero VAD requires PyTorch 1.12.0+;'
                             'Official repo: https://github.com/snakers4/silero-vad')
    parser.add_argument('--vad_threshold', type=float, default=0.35,
                        help='threshold for detecting speech with Silero VAD. (Default: 0.35); '
                             'low threshold reduces false positives for silence detection')
    parser.add_argument('--vad_onnx', type=str2bool, default=False,
                        help='whether to use ONNX for Silero VAD')

    parser.add_argument('--min_word_dur', type=float, default=0.1,
                        help="only allow suppressing timestamps that result in word durations greater than this value")

    parser.add_argument('--max_chars', type=int,
                        help="maximum number of character allowed in each segment")
    parser.add_argument('--max_words', type=int,
                        help="maximum number of words allowed in each segment")

    parser.add_argument('--demucs', type=str2bool, default=False,
                        help='whether to reprocess the audio track with Demucs to isolate vocals/remove noise; '
                             'Demucs official repo: https://github.com/facebookresearch/demucs')
    parser.add_argument('--demucs_output', action="extend", nargs="+", type=str,
                        help='path(s) to save the vocals isolated by Demucs as WAV file(s); '
                             'ignored if [demucs]=False')
    parser.add_argument('--only_voice_freq', '-ovf', action='store_true',
                        help='whether to only use sound between 200 - 5000 Hz, where majority of human speech are.')

    parser.add_argument('--strip', type=str2bool, default=True,
                        help="whether to remove spaces before and after text on each segment for output")

    parser.add_argument('--tag', type=str, action="extend", nargs="+",
                        help="a pair tags used to change the properties a word at its predicted time"
                             "SRT Default: '<font color=\"#00ff00\">', '</font>'"
                             "VTT Default: '<u>', '</u>'"
                             "ASS Default: '{\\1c&HFF00&}', '{\\r}'")
    parser.add_argument('--segment_level', type=str2bool, default=True,
                        help="whether to use segment-level timestamps in output")
    parser.add_argument('--word_level', type=str2bool, default=True,
                        help="whether to use word-level timestamps in output")

    parser.add_argument('--reverse_text', type=str2bool, default=False,
                        help="whether to reverse the order of words for each segment of text output")

    # ass output
    parser.add_argument('--font', type=str, default='Arial',
                        help="word font for ASS output(s)")
    parser.add_argument('--font_size', type=int, default=48,
                        help="word font size for ASS output(s)")
    parser.add_argument('--karaoke', type=str2bool, default=False,
                        help="whether to use progressive filling highlights for karaoke effect (only for ASS outputs)")

    parser.add_argument("--temperature", type=float, default=0,
                        help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int,
                        help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int,
                        help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None,
                        help="optional patience value to use in beam decoding, "
                             "as in https://arxiv.org/abs/2204.05424, "
                             "the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None,
                        help="optional token length penalty coefficient (alpha) "
                             "as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1",
                        help="comma-separated list of token ids to suppress during sampling; "
                             "'-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None,
                        help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True,
                        help="if True, provide the previous output of the model as a prompt for the next window; "
                             "disabling may make the text inconsistent across windows, "
                             "but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2,
                        help="temperature to increase when falling back when the decoding fails to meet either of "
                             "the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4,
                        help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0,
                        help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6,
                        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding "
                             "has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0,
                        help="number of threads used by torch for CPU inference; "
                             "supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument('--mel_first', action='store_true',
                        help='process entire audio track into log-Mel spectrogram first instead in chunks')

    parser.add_argument('--only_ffmpeg', action='store_true',
                        help='whether to use only FFmpeg (and not yt-dlp) for URls')

    parser.add_argument('--overwrite', '-y', action='store_true',
                        help='overwrite all output files')

    parser.add_argument('--debug', action='store_true',
                        help='print all input/output pair(s) and all arguments used for transcribing/translating')

    parser.add_argument('--transcribe_method', '-tm', type=str, default='transcribe',
                        choices=('transcribe', 'transcribe_minimal'))

    parser.add_argument('--align', '-a', action="extend", nargs='+', type=str,
                        help='path(s) to TXT file(s) or JSON previous result(s)')

    parser.add_argument('--refine', '-r', action='store_true',
                        help='Refine timestamps to increase precision of timestamps')

    parser.add_argument('--locate', '-lc', action="extend", nargs='+', type=str,
                        help='words to locate in the audio(s); skips transcription and output')

    parser.add_argument('--refine_option', '-ro', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for refining timestamps; Replace True/False with 1/0; '
                             'E.g. --refine_option "steps=sese" --refine_options "rel_prob_decrease=0.05"')
    parser.add_argument('--demucs_option', '-do', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for demucs; Replace True/False with 1/0; '
                             'E.g. --demucs_option "shifts=3" --demucs_options "overlap=0.5"')
    parser.add_argument('--model_option', '-mo', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for loading model; Replace True/False with 1/0; '
                             'E.g. --model_option "download_root=./downloads"')
    parser.add_argument('--transcribe_option', '-to', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for transcribing/alignment/locating; Replace True/False with 1/0; '
                             'E.g. --transcribe_option "ignore_compatibility=1"')
    parser.add_argument('--save_option', '-so', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for text outputs; Replace True/False with 1/0; '
                             'E.g. --save_option "highlight_color=ffffff"')

    parser.add_argument('--faster_whisper', '-fw', action='store_true',
                        help='whether to use faster-whisper (https://github.com/guillaumekln/faster-whisper); '
                             'note: some features may not be available')

    args = parser.parse_args().__dict__
    debug = args.pop('debug')
    if not args['language'] and (args['align'] or args['locate']):
        raise ValueError('langauge is required for --align / --locate')

    is_faster_whisper = args.pop('faster_whisper')
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    inputs: List[Union[str, torch.Tensor]] = args.pop("inputs")
    outputs: List[str] = args.pop("output")
    output_dir: str = args.pop("output_dir")
    output_format = args.pop("output_format")
    overwrite: bool = args.pop("overwrite")
    use_demucs = args['demucs'] or False
    demucs_outputs: List[Optional[str]] = args.pop("demucs_output")
    args['demucs_options'] = update_options_with_args('demucs_option', pop=True)
    regroup = args.pop('regroup')
    max_chars = args.pop('max_chars')
    max_words = args.pop('max_words')
    args['verbose'] = False if args['verbose'] == 1 else (True if args['verbose'] == 2 else None)
    show_curr_task = args['verbose'] is not None
    strings_to_locate = args.pop('locate')
    if dq := args.pop('dynamic_quantization', False):
        args['device'] = 'cpu'
    if args['reverse_text']:
        args['reverse_text'] = (args.get('prepend_punctuations'), args.get('append_punctuations'))

    if regroup:
        try:
            regroup = str2bool(regroup)
        except ValueError:
            pass
    curr_output_formats: List[str] = output_format.split(',') if output_format else []
    unsupported_formats = list(set(map(str.lower, curr_output_formats)) - OUTPUT_FORMATS)
    if outputs:
        unsupported_formats.extend(list(set(splitext(o)[-1].lower().strip('.') for o in outputs) - OUTPUT_FORMATS))
    if len(unsupported_formats) != 0:
        raise NotImplementedError(f'{unsupported_formats} are not supported. Supported formats: {OUTPUT_FORMATS}.')

    has_demucs_output = bool(demucs_outputs)
    if use_demucs and has_demucs_output and len(demucs_outputs) != len(inputs):
        raise NotImplementedError(f'[demucs_output] and [inputs] do not match in count. '
                                  f'Got {len(demucs_outputs)} and {len(inputs)}')

    if tag := args.get('tag'):
        assert tag == ['-1'] or len(tag) == 2, f'[tag] must be a pair of str but got {tag}'

    def make_parent(filepath: str):
        if parent := split(filepath)[0]:
            os.makedirs(parent, exist_ok=True)

    def is_json(file: str):
        return file.endswith(".json")

    def call_method_with_options(method, options: dict, include_first: bool = True):
        def val_to_str(val) -> str:
            if isinstance(val, (np.ndarray, torch.Tensor)):
                return f'{val.__class__}(shape:{list(val.shape)})'
            elif isinstance(val, str):
                return f'"{val}"'
            elif isinstance(val, bytes):
                return f'{type(val)}(len:{len(val)})'
            elif isinstance(val, torch.nn.Module):
                return str(type(val))
            return str(val)

        params = tuple(get_func_parameters(method))
        if debug:
            temp_options = {k: options.pop(k) for k in params if k in options}
            temp_options.update(options)
            options = temp_options
            options_str = ',\n'.join(
                f'    {k}={val_to_str(v)}'
                for k, v in options.items()
                if include_first or k != params[0]
            )
            if options_str:
                options_str = f'\n{options_str}\n'
            else:
                print(options, params)
            print(f'{method.__qualname__}({options_str})')
        return method(**options)

    if alignments := args['align']:
        if unsupported_align_fmts := \
                [_ext for p in alignments if (_ext := splitext(p)[-1].lower()) not in ('.json', '.txt')]:
            raise NotImplementedError(
                f'Unsupported format(s) for alignment: {unsupported_align_fmts}'
            )
        if len(inputs) != len(alignments):
            raise NotImplementedError(
                f'Got {len(inputs)} audio file(s) but specified {len(alignments)} file(s) to align.'
            )
    else:
        alignments = ['']*len(inputs)

    def finalize_outputs(input_file: str, _output: str = None, _alignment: str = None) -> List[str]:
        _curr_output_formats = curr_output_formats.copy()
        basename, ext = splitext(_output or input_file)
        ext = ext[1:]
        if _output:
            if ext.lower() in OUTPUT_FORMATS:
                _curr_output_formats.append(ext)
            else:
                basename = _output
        if not _curr_output_formats:
            _curr_output_formats = ["srt" if is_json(input_file) or is_json(_alignment) else "json"]
        _outputs = [f'{basename}.{ext}' for ext in set(_curr_output_formats)]
        if output_dir:
            _outputs = [join(output_dir, o) for o in _outputs]

        return _outputs

    if outputs:
        if len(outputs) != len(inputs):
            raise NotImplementedError(f'Got {len(inputs)} audio file(s) but specified {len(outputs)} output file(s).')
        final_outputs = [finalize_outputs(i, o, a) for i, o, a in zip(inputs, outputs, alignments)]
    else:
        if not output_dir:
            output_dir = '.'
        final_outputs = [finalize_outputs(i, _alignment=a) for i, a in zip(inputs, alignments)]

    if not overwrite:

        def cancel_overwrite():
            resp = input(f'{path} already exist, overwrite (y/n)? ').lower()
            if resp in ('y', 'n'):
                return resp == 'n'
            print(f'Expected "y" or "n", but got {resp}.')
            return True

        for paths in final_outputs:
            for path in paths:
                if isfile(path) and cancel_overwrite():
                    return

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(f"{model_name} is an English-only model but receipted "
                          f"'{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    increment = args.pop("temperature_increment_on_fallback")
    if increment is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    args['temperature'] = temperature

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    if debug:
        print('Input(s)  ->  Outputs(s)')
        for i, (input_audio, output_paths, alignment) in enumerate(zip(inputs, final_outputs, alignments)):
            dm_output = f' {demucs_outputs[i]} ->' if demucs_outputs else ''
            alignment = f' + "{alignment}"' if alignment else ''
            print(f'"{input_audio}"{alignment}  ->{dm_output}  {output_paths}')
        print('')

    if show_curr_task:
        model_from_str = '' if model_dir is None else f' from {model_dir}'
        model_loading_str = f'{"Faster-Whisper" if is_faster_whisper else "Whisper"} {model_name} model {model_from_str}'
        print(f'Loading {model_loading_str}\r', end='\n' if debug else '')
    else:
        model_loading_str = ''

    alignments = args['align']
    model = None

    def _load_model():
        nonlocal model
        if model is None:
            model_options = dict(
                name=model_name,
                model_size_or_path=model_name,
                device=args.get('device'),
                download_root=model_dir,
                dq=dq,
            )
            load_model_func = load_faster_whisper if is_faster_whisper else load_model
            model_options = isolate_useful_options(model_options, load_model_func)
            update_options_with_args('model_option', model_options)
            model = call_method_with_options(load_model_func, model_options)
            if model_loading_str:
                print(f'Loaded {model_loading_str}  ')
        return model

    for i, (input_audio, output_paths) in enumerate(zip(inputs, final_outputs)):
        skip_output = False
        if isinstance(input_audio, str) and is_json(input_audio):
            result = WhisperResult(input_audio)
        else:
            model = _load_model()
            args['regroup'] = False
            args['audio'] = input_audio
            if has_demucs_output:
                args['demucs_output'] = demucs_outputs[i]
            transcribe_method = args.get('transcribe_method')
            text = None
            if alignments and (text := alignments[i]):
                if text.endswith('.json'):
                    text = WhisperResult(text)
                else:
                    with open(text, 'r', encoding='utf-8') as f:
                        text = f.read()
                args['text'] = text
                transcribe_method = 'align'
            if is_faster_whisper and transcribe_method == 'transcribe':
                transcribe_method = 'transcribe_stable'
            if strings_to_locate and (text := strings_to_locate[i]):
                args['text'] = text
                transcribe_method = 'locate'
                skip_output = args['verbose'] = True
            transcribe_method = getattr(model, transcribe_method)
            transcribe_options = isolate_useful_options(args, transcribe_method)
            if not text:
                decoding_options = (
                    isolate_useful_options(args, model.transcribe if is_faster_whisper else DecodingOptions)
                )
                if is_faster_whisper:
                    if decoding_options['suppress_tokens']:
                        decoding_options['suppress_tokens'] = (
                            list(map(int, decoding_options['suppress_tokens'].split(',')))
                        )
                    for k in list(decoding_options.keys()):
                        if decoding_options[k] is None:
                            del decoding_options[k]
                transcribe_options.update(decoding_options)
            update_options_with_args('transcribe_option', transcribe_options)
            result: WhisperResult = call_method_with_options(transcribe_method, transcribe_options)

        if skip_output:
            continue

        if args['refine']:
            model = _load_model()
            refine_options = isolate_useful_options(args, model.refine)
            refine_options['result'] = result
            update_options_with_args('refine_option', refine_options)
            call_method_with_options(model.refine, refine_options)

        if args.get('word_timestamps'):
            if regroup:
                result.regroup(regroup, verbose=args['verbose'] or debug)
            if max_chars or max_words:
                result.split_by_length(max_chars=max_chars, max_words=max_words)

        for path in output_paths:
            make_parent(path)
            save_method = getattr(result, OUTPUT_FORMATS_METHODS[splitext(path)[-1][1:]])
            args['filepath'] = path
            args['path'] = path
            save_options = isolate_useful_options(args, save_method)
            update_options_with_args('save_option', save_options)
            call_method_with_options(save_method, save_options)


if __name__ == '__main__':
    cli()
