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
from whisper.utils import exact_div, format_timestamp, make_safe
from whisper.tokenizer import get_tokenizer, LANGUAGES, TO_LANGUAGE_CODE
from whisper.decoding import DecodingOptions, DecodingResult

from .audio import load_audio
from .decode import decode_stable
from .result import WhisperResult, Segment
from .timing import add_word_timestamps_stable
from .stabilization import get_vad_silence_func, wav2mask, mask2timing, timing2mask

if TYPE_CHECKING:
    from whisper.model import Whisper

__all__ = ['modify_model', 'load_model']

_processes = {}

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
        regroup: bool = True,
        ts_num: int = 0,
        ts_noise: float = 0.1,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        q_levels: int = 20,
        k_size: int = 5,
        time_scale: float = None,
        demucs: bool = False,
        demucs_output: str = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: float = 0.1,
        only_voice_freq: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        mel_first: bool = False,
        split_callback: Callable = None,
        suppress_ts_tokens: bool = True,
        gap_padding: str = ' ...',
        only_ffmpeg: bool = False,
        **decode_options) \
        -> WhisperResult:
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    audio: Union[str, np.ndarray, torch.Tensor, bytes]
        The path/URL to the audio file, the audio waveform, or bytes of audio file.

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays progressbar. If None, does not display anything (Default: False)

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment. (Default: True)
        Disabling this will prevent segments from splitting/merging properly.

    regroup: bool
        Regroup all words into segments with more natural boundaries.(Default: True)
        Ignored if [word_timestamps]=False.

    ts_num: int
        Number of extra timestamp inferences to perform then use average of these extra timestamps. (Default: 0).

    ts_noise: float
        Percentage of noise to add to audio_features to perform inferences for [ts_num]. (Default: 0.1)

    suppress_silence: bool
        Whether to suppress timestamp where audio is silent at segment-level
        and word-level if [suppress_word_ts]=True. (Default: True)

    suppress_word_ts: bool
        Whether to suppress timestamps, if [suppress_silence]=True, where audio is silent at word-level. (Default: True)

    q_levels: int
        Quantization levels for generating timestamp suppression mask; ignored if [vad]=true. (Default: 20)
        Acts as a threshold to marking sound as silent.
        Fewer levels will increase the threshold of volume at which to mark a sound as silent.

    k_size: int
        Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if [vad]=true. (Default: 5)
        Recommend 5 or 3; higher sizes will reduce detection of silence.

    time_scale: float
        Factor for scaling audio duration for inference. (Default: None)
        Greater than 1.0 'slows down' the audio, and less than 1.0 'speeds up' the audio. None is same as 1.0.
        A factor of 1.5 will stretch 10s audio to 15s for inference. This increases the effective resolution
        of the model but can increase word error rate.

    demucs: bool
        Whether to preprocess the audio track with Demucs to isolate vocals/remove noise. (Default: False)
        Demucs must be installed to use. Official repo: https://github.com/facebookresearch/demucs

    demucs_output: str
        Path to save the vocals isolated by Demucs as WAV file. Ignored if [demucs]=False.
        Demucs must be installed to use. Official repo: https://github.com/facebookresearch/demucs

    vad: bool
        Whether to use Silero VAD to generate timestamp suppression mask. (Default: False)
        Silero VAD requires PyTorch 1.12.0+. Official repo: https://github.com/snakers4/silero-vad

    vad_threshold: float
        Threshold for detecting speech with Silero VAD. (Default: 0.35)
        Low threshold reduces false positives for silence detection.

    vad_onnx: bool
        Whether to use ONNX for Silero VAD. (Default: False)

    min_word_dur: float
        Only allow suppressing timestamps that result in word durations greater than this value. (default: 0.1)

    only_voice_freq: bool
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are. (Default: False)

    prepend_punctuations: str
        Punctuations to prepend to next word (Default: "'“¿([{-)

    append_punctuations: str
        Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)

    mel_first: bool
        Process entire audio track into log-Mel spectrogram first instead in chunks. (Default: False)
        Used if odd behavior seen in stable-ts but not in whisper, but use significantly more memory for long audio.

    split_callback: Callable
        Custom callback for grouping tokens up with their corresponding words.
        Takes argument: list of tokens; default tokenizer
        Returns a tuple pair containing: list of words; list of token groups (i.e. each group is list of token(s))

    suppress_ts_tokens: bool
        Whether to use silence mask to suppress silent timestamp tokens during inference. (Default: True)
        Reduces hallucinations in some cases,
            but also can reduce 'verbatimness' (i.e. ignores disfluencies and repetitions).

    gap_padding: str
        Padding prepend to each segments for word timing alignment. (Default: ' ...')
        Used to reduce the probability of model predicting timestamps earlier than the first utterance.

    only_ffmpeg: bool
        Whether to use only FFmpeg (and not yt-dlp) for URls. (Default: False)

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

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

    if time_scale == 1:
        time_scale = None

    curr_sr = SAMPLE_RATE if time_scale is None else SAMPLE_RATE * time_scale
    if isinstance(audio, (str, bytes)):
        if demucs:
            from .audio import demucs_audio
            audio = demucs_audio(audio,
                                 output_sr=curr_sr,
                                 device=device,
                                 verbose=verbose,
                                 save_path=demucs_output)
        else:
            audio = torch.from_numpy(load_audio(audio, sr=curr_sr, verbose=verbose, only_ffmpeg=only_ffmpeg))
    else:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        input_sr = decode_options.pop('input_sr', SAMPLE_RATE)
        if demucs:
            from .audio import demucs_audio
            audio = demucs_audio(audio,
                                 input_sr=input_sr,
                                 output_sr=curr_sr,
                                 device=device,
                                 verbose=verbose,
                                 save_path=demucs_output)
        elif input_sr != curr_sr:
            from torchaudio.functional import resample
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            audio = resample(audio, input_sr, curr_sr, resampling_method="kaiser_window")
    if only_voice_freq:
        from .audio import voice_freq_filter
        audio = voice_freq_filter(audio, curr_sr)
    sample_padding = int(N_FFT // 2) + 1
    whole_mel = log_mel_spectrogram(audio, padding=sample_padding) if mel_first else None

    if decode_options.get("language", None) is None and model:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            mel_segment = log_mel_spectrogram(audio[..., :N_SAMPLES], padding=sample_padding) \
                if whole_mel is None else whole_mel[..., :N_FRAMES]
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device=model.device, dtype=dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(f"Detected language: {LANGUAGES[decode_options['language']]}")

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

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
            decode_result, audio_features = model.decode(seg,
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

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

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

    total_samples = audio.shape[-1]
    total_duration = round(total_samples / curr_sr, 2)
    n_samples_per_frame = exact_div(N_SAMPLES_PER_TOKEN * TOKENS_PER_SECOND, FRAMES_PER_SECOND)

    silence_timing = None
    if suppress_silence and vad:
        silence_timing = get_vad_silence_func(onnx=vad_onnx, verbose=verbose)(audio, speech_threshold=vad_threshold)

    with tqdm(total=total_duration, unit='sec', disable=verbose is not False) as tqdm_pbar:

        def update_pbar():
            nonlocal audio_features
            audio_features = None
            if not tqdm_pbar.disable:
                tqdm_pbar.update(min(total_duration, round(seek_sample / curr_sr, 2)) - tqdm_pbar.n)

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
            segment_samples = min(N_SAMPLES, total_samples - seek_sample)
            segment_duration = segment_samples / SAMPLE_RATE

            mel_segment = (
                log_mel_spectrogram(audio_segment, padding=sample_padding)
                if whole_mel is None else
                whole_mel[..., round(seek_sample / n_samples_per_frame): round(seek_sample_end / n_samples_per_frame)]
            )

            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(device=model.device, dtype=dtype)

            segment_silence_timing = None
            ts_token_mask = None
            if suppress_silence:
                if silence_timing is None:
                    ts_token_mask = wav2mask(audio_segment, q_levels=q_levels, k_size=k_size)
                    segment_silence_timing = mask2timing(ts_token_mask)
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
                            end=round(time_offset + end_timestamp_pos * time_precision, 3),
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if not single_timestamp_ending:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                            tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    segment_samples = min(segment_samples, round(last_timestamp_pos * N_SAMPLES_PER_TOKEN))
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (
                        len(timestamps) > 0
                        and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = (
                            timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

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
                if seg["start"] == seg["end"] or seg["text"].strip() == "":
                    del current_segments[i]

            if len(current_segments) == 0:
                fast_forward()
                continue

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            if word_timestamps:
                add_word_timestamps_stable(
                    segments=current_segments,
                    model=model,
                    tokenizer=tokenizer,
                    mel=mel_segment,
                    num_samples=segment_samples,
                    prepend_punctuations=prepend_punctuations,
                    append_punctuations=append_punctuations,
                    audio_features=audio_features,
                    ts_num=ts_num,
                    ts_noise=ts_noise,
                    split_callback=split_callback,
                    gap_padding=gap_padding
                )

            if segment_silence_timing is not None:
                for seg_i, segment in enumerate(current_segments):
                    current_segments[seg_i] = (
                        Segment(**segment)
                        .suppress_silence(
                            *segment_silence_timing,
                            min_word_dur=min_word_dur,
                            word_level=suppress_word_ts
                        )
                        .to_dict()
                    )

            if verbose:
                for segment in current_segments:
                    line = f"[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] " \
                           f"{segment['text']}"
                    if word_timestamps:
                        words_str = '\n'.join(f"-[{format_timestamp(w['start'])}] -> "
                                              f"[{format_timestamp(w['end'])}] \"{w['word']}\""
                                              for w in segment.get('words', []))
                        if words_str:
                            line += f'\n{words_str}\n'
                        else:
                            line = ''
                    if line:
                        print(make_safe(line))

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(current_segments, start=len(all_segments))
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )
            if not single_timestamp_ending and len(consecutive) > 0 and \
                    (alt_segment_duration := (current_segments[-1]['end'] - time_offset)) > 0:
                segment_samples = min(round(alt_segment_duration * SAMPLE_RATE), segment_samples)
            fast_forward()

        # final update
        update_pbar()

    if model.device != torch.device('cpu'):
        torch.cuda.empty_cache()

    final_result = WhisperResult(dict(text=tokenizer.decode(all_tokens[len(initial_prompt_tokens):]),
                                      segments=all_segments,
                                      language=language,
                                      time_scale=time_scale))
    if word_timestamps and regroup:
        final_result.regroup()

    if time_scale is not None:
        final_result.rescale_time(1 / time_scale)

    return final_result


def modify_model(model: "Whisper"):
    """
    Modifies model instance by:
        -replacing model.decode with decode_word_level
        -replacing model.transcribe with transcribe_word_level
    """
    model.decode = MethodType(decode_stable, model)
    model.transcribe = MethodType(transcribe_stable, model)


# modified version of whisper.load_model
def load_model(name: str, device: Optional[Union[str, torch.device]] = None,
               download_root: str = None, in_memory: bool = False,
               cpu_preload: bool = True, dq: bool = False) -> "Whisper":
    """
     Load a modified Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory
    cpu_preload: bool
        load model into CPU memory first then move model to specified device;
        this reduces GPU memory usage when loading model
    dq: bool
        whether to apply Dynamic Quantization to model to reduced memory usage and increase inference speed
        but at the cost of a slight decrease in accuracy. Only for CPU.
        Note: The overhead might make inference slower for models smaller than 'large'
    Returns
    -------
    model : "Whisper"
        The Whisper ASR model instance
    """
    if device is None or dq:
        device = "cuda" if torch.cuda.is_available() and not dq else "cpu"
    if cpu_preload:
        model = whisper.load_model(name, device='cpu', download_root=download_root, in_memory=in_memory).to(
            device=device)
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

    str2val = {"true": True, "false": False, "1": True, "0": False}

    def str2bool(string: str) -> bool:
        string = string.lower()
        if string in str2val:
            return str2val[string]
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    output_formats = {"srt", "ass", "json", "vtt"}

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=str,
                        help="audio/video filepath/URL(s) to transcribe "
                             "or json file(s) to process into [output_format]")
    parser.add_argument("--output", "-o", nargs="+", type=str,
                        help="output filepaths(s);"
                             "if not specified, auto-named output file(s) will be saved to "
                             "[output_dir] or current dir if not specified.")
    parser.add_argument("--model", '-m', default="base", choices=available_models(),
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
    parser.add_argument("--output_format", "-f", type=str, default='json', choices=list(output_formats),
                        help="format of the output file(s)")
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

    parser.add_argument("--regroup", type=str2bool, default=True,
                        help="regroup all words into segments with more natural boundaries;"
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

    parser.add_argument('--suppress_ts_tokens', type=str2bool, default=True,
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
    parser.add_argument('--demucs_output', nargs="+", type=str,
                        help='path(s) to save the vocals isolated by Demucs as WAV file(s); '
                             'ignored if [demucs]=False')
    parser.add_argument('--only_voice_freq', '-ovf', action='store_true',
                        help='whether to only use sound between 200 - 5000 Hz, where majority of human speech are.')

    parser.add_argument('--strip', type=str2bool, default=True,
                        help="whether to remove spaces before and after text on each segment for output")

    parser.add_argument('--tag', type=str, nargs="+",
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

    parser.add_argument('--only_ffmpeg', action='store_true',
                        help='whether to use only FFmpeg (and not yt-dlp) for URls')

    parser.add_argument('--overwrite', '-y', action='store_true',
                        help='overwrite all output files')

    parser.add_argument('--debug', action='store_true',
                        help='print all input/output pair(s) and all arguments used for transcribing/translating')

    args = parser.parse_args().__dict__
    debug = args.pop('debug')
    cpu_preload = args.pop('cpu_preload')

    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    inputs: List[Union[str, torch.Tensor]] = args.pop("inputs")
    outputs: List[str] = args.pop("output")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    overwrite: bool = args.pop("overwrite")
    use_demucs = args.pop('demucs')
    demucs_outputs: List[Optional[str]] = args.pop("demucs_output")
    regroup = args.pop('regroup')
    max_chars = args.pop('max_chars')
    max_words = args.pop('max_words')
    reverse_text = args.pop('reverse_text')

    if outputs:
        unsupported_formats = set(splitext(o)[-1].lower().strip('.') for o in outputs) - output_formats
        if len(unsupported_formats) != 0:
            raise NotImplementedError(f'{unsupported_formats} are not supported. Supported formats: {outputs}.')

    has_demucs_output = bool(demucs_outputs)
    if use_demucs and has_demucs_output and len(demucs_outputs) != len(inputs):
        raise NotImplementedError(f'[demucs_output] and [inputs] do not match in count. '
                                  f'Got {len(demucs_outputs)} and {len(inputs)}')

    strip: bool = args.pop('strip')

    segment_level: bool = args.pop('segment_level')
    word_level: bool = args.pop('word_level')
    tag: List[str] = args.pop('tag')
    if tag:
        assert len(tag) == 2, f'[tag] must be a pair of str but got {tag}'

    font: str = args.pop('font')
    font_size: int = args.pop('font_size')

    def make_parent(filepath: str):
        if parent := split(filepath)[0]:
            os.makedirs(parent, exist_ok=True)

    def is_json(file: str):
        return file.endswith(".json")

    def get_output_ext(input_file: str) -> str:
        if not output_format:
            return f'.{"srt" if is_json(input_file) else "json"}'
        return f'.{output_format}'

    if outputs:
        if len(outputs) != len(inputs):
            raise NotImplementedError(f'Got {len(inputs)} audio file(s) but specified {len(outputs)} output file(s).')
        if output_dir:
            for i in range(len(outputs)):
                if splitext(outputs[i])[1].strip('.') not in output_formats:
                    outputs[i] += get_output_ext(inputs[i])
                outputs[i] = join(output_dir, outputs[i])
    else:
        if not output_dir:
            output_dir = '.'
        outputs = [
            join(
                output_dir,
                f'{splitext(split(i)[1])[0]}{get_output_ext(i)}'
            )
            for i in inputs
        ]

    if not overwrite:

        def cancel_overwrite():
            resp = input(f'{path} already exist, overwrite (y/n)? ').lower()
            if resp in ('y', 'n'):
                return resp == 'n'
            print(f'Expected "y" or "n", but got {resp}.')
            return True

        for path in outputs:
            if isfile(path) and cancel_overwrite():
                return

    device: str = args.pop("device")
    dq = args.pop('dynamic_quantization', False)
    if dq:
        device = 'cpu'

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
        print(f'\nModel Arguments',
              f'\nModel: {model_name}\n'
              f'device: {device}\n'
              f'download_root: {model_dir}\n'
              f'dynamic_quantization: {dq}\n')
        print(f'Arguments for {args.get("task")}')
        for k, v in args.items():
            print(f'{k}: {v}')
        print(f'use_demucs: {use_demucs}')
        print(f'\nArguments for Output(s)',
              f'\noverwrite: {overwrite}\n'
              f'segment_level: {segment_level}\n'
              f'word_level: {word_level}\n'
              f'tag: {tag}\n'
              f'strip: {strip}\n'
              f'regroup: {regroup}\n'
              f'max_chars: {max_chars}\n'
              f'max_words: {max_words}\n'
              f'reverse_text: {reverse_text}\n'
              f'\nArguments for ASS Output',
              f'\nfont: {font}\n'
              f'font_size: {font_size}\n')

        print('Input(s)  ->  Outputs(s)')
        for i, (input_audio, output_path) in enumerate(zip(inputs, outputs)):
            dm_output = f' {demucs_outputs[i]} ->' if demucs_outputs else ''
            print(f'{input_audio}  ->{dm_output}  {output_path}')
        print('\n')

    args['verbose'] = False if args['verbose'] == 1 else (True if args['verbose'] == 2 else None)
    show_curr_task = args['verbose'] is not None

    if use_demucs:
        from .audio import demucs_audio, load_demucs_model
        demucs_model = load_demucs_model()
        audio_inputs = []
        for i, input_audio in enumerate(inputs):
            demucs_path = demucs_outputs[i] if has_demucs_output else None
            audio_inputs.append(
                demucs_audio(
                    input_audio,
                    model=demucs_model,
                    device=device,
                    verbose=show_curr_task,
                    save_path=demucs_path
                )
            )
        args['input_sr'] = demucs_model.samplerate
        inputs = audio_inputs
        del demucs_model
        if device != 'cpu':
            torch.cuda.empty_cache()

    if show_curr_task:
        model_from_str = '' if model_dir is None else f' from {model_dir}'
        model_loading_str = f'Whisper {model_name} model {model_from_str}'
        print(f'Loading {model_loading_str}\r', end='')
    else:
        model_loading_str = ''

    model = None

    for input_audio, output_path in zip(inputs, outputs):
        if isinstance(input_audio, str) and is_json(input_audio):
            result = WhisperResult(input_audio)
        else:
            if model is None:
                model = load_model(
                    model_name,
                    device=device,
                    download_root=model_dir,
                    cpu_preload=cpu_preload,
                    dq=dq
                )

                if model_loading_str:
                    print(f'Loaded {model_loading_str}  ')
            args['regroup'] = False
            result: WhisperResult = model.transcribe(input_audio, **args)

        if args.get('word_timestamps'):
            if regroup:
                result.regroup()
            if max_chars or max_words:
                result.split_by_length(max_chars=max_chars, max_words=max_words)

        if reverse_text:
            reverse_text = (args.get('prepend_punctuations'), args.get('append_punctuations'))
        make_parent(output_path)
        if is_json(output_path):
            result.save_as_json(output_path)
        elif output_path.endswith('.srt') or output_path.endswith('.vtt'):
            result.to_srt_vtt(
                filepath=output_path,
                segment_level=segment_level,
                word_level=word_level,
                tag=tag,
                strip=strip,
                reverse_text=reverse_text
            )
        else:
            result.to_ass(
                filepath=output_path,
                segment_level=segment_level,
                word_level=word_level,
                tag=tag,
                font=font,
                font_size=font_size,
                strip=strip,
                reverse_text=reverse_text
            )


if __name__ == '__main__':
    cli()
