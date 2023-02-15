import warnings
import torch
import whisper
import numpy as np
from typing import List, Optional, Tuple, Union
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim
from whisper.utils import exact_div, format_timestamp
from whisper.tokenizer import get_tokenizer, LANGUAGES, TO_LANGUAGE_CODE
from whisper import DecodingOptions, DecodingResult, Whisper, log_mel_spectrogram
from types import MethodType
from itertools import repeat
from tqdm import tqdm
from .audio import prep_wf_mask, remove_lower_quantile, finalize_mask
from .stabilization import stabilize_timestamps, add_whole_word_ts
from .decode_word_level import decode_word_level

__all__ = ['transcribe_word_level', 'modify_model', 'load_model']


# modified version of whisper.transcribe.transcribe
def transcribe_word_level(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor],
        *,
        verbose: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        stab=True, top_focus=False, ts_num: int = 10,
        alpha: float = None, print_stab=False, pbar=True,
        suppress_silence: bool = True,
        suppress_middle: bool = True,
        suppress_word_ts: bool = True,
        remove_background: bool = True,
        silence_threshold: float = 0.1,
        refine_ts_num: int = None,
        avg_refine: bool = False,
        time_scale: float = None,
        prepend_punctuations: Union[List[str], Tuple[str]] = None,
        append_punctuations: Union[List[str], Tuple[str]] = None,
        **decode_options):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the progress of decoded segments (not yet stabilized) to the console (Default: False)

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

    stab: bool
        Whether to stabilizing timestamps by cross compare timestamps and using additional top timestamp predictions
        to fill in when appropriate to ensure timestamps are chronological. (Default: True)

    top_focus: bool
        Whether to adhere closely to the top predictions for token timestamps stabilization. (Default: False)

    ts_num: int
        Number of top timestamp predictions to save for each word for postprocessing stabilization (Default: 10).

    alpha: float
        This is purely experimental. Will be deprecated.
        Amount of noise to add to audio to produce slightly difference results. (Default: None)
        audio_features *= torch.rand_like(audio_features) * alpha + 1

    print_stab: bool
        Whether to display the decoded text (with stabilized word-level timestamps) to the console. (Default: False)

    pbar: bool
        Whether to enable progress bar for the decoding process (Default: True). Ignored if verbose=True

    suppress_silence: bool
        Whether to suppress timestamp where audio is silent at segment-level. (Default: True)

    suppress_middle: bool
        Whether to suppress silence only for beginning and ending of segments. (Default: True)

    suppress_word_ts: bool
        Whether to suppress timestamp where audio is silent at word-level. (Default: True)

    remove_background: bool
        Whether to zero sections with background noise from waveform, so it will be marked as silent.
        Determined by parameters part of decode_options (i.e. specify like other options here):
            upper_quantile: float
                The upper quantile of amplitude to determine a max amplitude, mx (Default: 0.85)
            lower_quantile: float
                The lower quantile of amplitude to determine a min amplitude, mn (Default: 0.15)
            lower_threshold: float
                Zero sections of waveform where amplitude < lower_threshold*(mx-mn) + mn. (Default: 0.15)

    silence_threshold: float
        If an audio segment's percentage of silence >= silence_threshold
        then that segment will not have background removed even if remove_background=True. (Default: 0.1)
        e.g. 0.5 means if less than half of the audio segment is silent then background will be removed accordingly

    refine_ts_num: int
        Same as ts_num for word timestamp refinement. (Default: 10 times of ts_num)
        Must be either: 0 to disable refinement or greater than ts_num but maximum 1501.

    avg_refine: bool
        Whether to average the original word timestamps with the refined timestamps. (Default: False)

    time_scale: float
        Factor for scaling audio duration for inference. (Default: None)
        Greater than 1.0 'slows down' the audio, and less than 1.0 'speeds up' the audio. None is same as 1.0.
        A factor of 1.5 will stretch 10s audio to 15s for inference. This increases the effective resolution
        of the model but can increase word error rate.

    prepend_punctuations: Union[List[str], Tuple[str]]
        Punctuations to prepend to next word (Default: “¿([{)

    append_punctuations: Union[List[str], Tuple[str]]
        Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
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

    if time_scale == 1:
        time_scale = None
    if refine_ts_num is None:
        if ts_num is not None:
            refine_ts_num = min(1501, ts_num * 10)
    else:
        refine_ts_num = min(1501, refine_ts_num)

    decode_ts_num = max(ts_num, refine_ts_num)

    curr_sr = SAMPLE_RATE if time_scale is None else SAMPLE_RATE * time_scale
    if isinstance(audio, str):
        audio = whisper.load_audio(audio, sr=curr_sr)
    else:
        from torchaudio.functional import resample
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if time_scale is not None:
            audio = resample(audio, SAMPLE_RATE, curr_sr, resampling_method="kaiser_window")
    mel = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if verbose:
            print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
        _, probs = model.detect_language(segment)
        decode_options["language"] = max(probs, key=probs.get)
        print(f"Detected language: {LANGUAGES[decode_options['language']]}")

    mel = mel.unsqueeze(0)
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: torch.Tensor, suppress_ts_mask: torch.Tensor = None) \
            -> Union[List[DecodingResult], tuple]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        kwargs = {**decode_options}
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results, ts_tokens, ts_logits_ = model.decode(segment, options, ts_num=decode_ts_num, alpha=alpha,
                                                      suppress_ts_mask=suppress_ts_mask,
                                                      suppress_word_ts=suppress_word_ts)

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            needs_fallback = [
                compression_ratio_threshold is not None
                and result.compression_ratio > compression_ratio_threshold
                or logprob_threshold is not None
                and result.avg_logprob < logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                options = DecodingOptions(**kwargs, temperature=t)
                retries, r_ts_tokens, r_ts_logits = model.decode(segment[needs_fallback], options,
                                                                 ts_num=decode_ts_num, alpha=alpha,
                                                                 suppress_ts_mask=suppress_ts_mask,
                                                                 suppress_word_ts=suppress_word_ts)
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]
                    ts_tokens[original_index] = r_ts_tokens[retry_index]
                    ts_logits_[original_index] = r_ts_logits[retry_index]

        return results, ts_tokens, ts_logits_

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def _to_list(x: (torch.Tensor, None)):
        if x is None:
            return x
        return x.tolist()

    def add_segment(
            *, offset: float, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult,
            start_timestamps: torch.Tensor = None, end_timestamps: torch.Tensor = None,
            word_timestamps: torch.Tensor = None,
            start_ts_logits: list = None, end_ts_logits: list = None, word_ts_logits: torch.Tensor = None
    ):

        no_eot_mask = text_tokens < tokenizer.eot
        text_tokens_no_eot = text_tokens[no_eot_mask]
        text = tokenizer.decode(text_tokens_no_eot)

        if len(text.strip()) == 0:  # skip empty text output
            return

        if start_timestamps is not None:
            start_timestamps = start_timestamps[:ts_num]
        if end_timestamps is not None:
            end_timestamps = end_timestamps[:ts_num]
        if start_ts_logits is not None:
            start_ts_logits = start_ts_logits[:ts_num]
        if end_ts_logits is not None:
            end_ts_logits = end_ts_logits[:ts_num]

        if time_scale is None:
            if start_timestamps is not None:
                start_timestamps = start_timestamps.tolist()
            if end_timestamps is not None:
                end_timestamps = end_timestamps.tolist()
        else:
            offset = offset / time_scale
            start = start / time_scale
            end = end / time_scale
            if start_timestamps is not None:
                start_timestamps = (start_timestamps / time_scale).tolist()
            if end_timestamps is not None:
                end_timestamps = (end_timestamps / time_scale).tolist()
            if word_timestamps is not None:
                word_timestamps = word_timestamps / time_scale

        more_word_timestamps, more_word_ts_logits = None, None
        if word_timestamps is not None:
            more_word_timestamps = word_timestamps[no_eot_mask]
            word_timestamps = more_word_timestamps[..., :ts_num]
            assert word_timestamps.shape[0] == text_tokens_no_eot.shape[0]
            if word_ts_logits is None:
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps, repeat(None))
            else:
                more_word_ts_logits = word_ts_logits[no_eot_mask]
                word_ts_logits = more_word_ts_logits[:, :ts_num]
                assert word_ts_logits.shape[0] == text_tokens_no_eot.shape[0]
                non_inf_masks = [wtsl != -np.inf for wtsl in more_word_ts_logits]
                non_inf_masks = [nim if nim.nonzero().shape[0] > ts_num else None for nim in non_inf_masks]
                more_word_ts_logits = [word_ts_logits[m_i].tolist()
                                       if non_inf_masks[m_i] is None else
                                       mwtsl[non_inf_masks[m_i]].tolist()
                                       for m_i, mwtsl in enumerate(more_word_ts_logits)]
                more_word_timestamps = [word_timestamps[m_i].tolist()
                                        if non_inf_masks[m_i] is None else
                                        mwts[non_inf_masks[m_i]].tolist()
                                        for m_i, mwts in enumerate(more_word_timestamps)]
                word_ts_fields = zip(text_tokens_no_eot, word_timestamps, word_ts_logits)

            word_timestamps = [dict(word=tokenizer.decode([token]),
                                    token=token.item(),
                                    timestamps=timestamps_.tolist(),
                                    timestamp_logits=_to_list(ts_logits_))
                               for token, timestamps_, ts_logits_ in word_ts_fields]

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                'offset': offset,  # offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
                "alt_start_timestamps": start_timestamps,
                "start_ts_logits": start_ts_logits,
                "alt_end_timestamps": end_timestamps,
                "end_ts_logits": end_ts_logits,
                "unstable_word_timestamps": word_timestamps,
                "more_word_timestamps": more_word_timestamps,
                "more_word_ts_logits": more_word_ts_logits,
                'anchor_point': False
            }
        )
        if verbose:
            print(f'[{format_timestamp(start)} --> {format_timestamp(end)}] "{text}"')

    mel_scale = HOP_LENGTH / SAMPLE_RATE
    if suppress_silence:
        ts_scale = mel_scale / time_precision
        wfw = int(mel.shape[-1] * ts_scale)
        if wfw == 0:
            warnings.warn(f'[suppress_silence] will be set to False because '
                          f'audio duration shorter than the model\'s time precision ({time_precision} s).',
                          stacklevel=2)
            suppress_silence = False
        else:
            wf_mask = prep_wf_mask(audio, curr_sr, wfw)
            if not wf_mask.any():
                warnings.warn('The audio appears to be entirely silent. [suppress_silence] will be set to False',
                              stacklevel=2)
                suppress_silence = False

    upper_quantile = decode_options.pop('upper_quantile', 0.85)
    lower_quantile = decode_options.pop('lower_quantile', 0.15)
    lower_threshold = decode_options.pop('lower_threshold', 0.15)

    num_frames = mel.shape[-1]

    seek_mask_cache = {}

    with tqdm(total=num_frames, unit='frames', disable=(verbose or not pbar)) as tqdm_pbar:

        def update_pbar():
            if not tqdm_pbar.disable:
                tqdm_pbar.update(min(num_frames, seek) - tqdm_pbar.n)

        while seek < mel.shape[-1]:
            timestamp_offset = float(seek * mel_scale)
            remaining_duration = float((mel.shape[-1] - seek) * mel_scale)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES).to(device=model.device, dtype=dtype)
            segment_duration = min(float(segment.shape[-1] * mel_scale), remaining_duration)
            segment_max_ts = segment_duration / time_precision

            if suppress_silence:
                wf_seek = int(seek * ts_scale)
                suppress_ts_mask = wf_mask[..., wf_seek:wf_seek + 1501]
                if remove_background and \
                        (1 - suppress_ts_mask.clamp(max=1).mean()) < silence_threshold:
                    suppress_ts_mask = remove_lower_quantile(suppress_ts_mask,
                                                             upper_quantile=upper_quantile,
                                                             lower_quantile=lower_quantile,
                                                             lower_threshold=lower_threshold)

                suppress_ts_mask = pad_or_trim(suppress_ts_mask, 1501)
                suppress_ts_mask = finalize_mask(suppress_ts_mask,
                                                 suppress_middle=suppress_middle,
                                                 max_index=int(segment_max_ts))

                if suppress_ts_mask.all():  # segment is silent
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    update_pbar()
                    continue
            else:
                suppress_ts_mask = None

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result, finalized_ts_tokens, ts_logits = decode_with_fallback(segment,
                                                                          suppress_ts_mask=suppress_ts_mask)

            result = result[0]
            tokens = torch.tensor(result.tokens)
            finalized_ts_tokens = torch.tensor(finalized_ts_tokens[0])
            ts_logits = torch.tensor(ts_logits[0])

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            if refine_ts_num:
                seek_mask_cache[seek] = suppress_ts_mask

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    sliced_ts_tokens = finalized_ts_tokens[last_slice:current_slice]
                    sliced_ts_logits = ts_logits[last_slice:current_slice]
                    start_timestamp_position = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                            sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )

                    word_ts = timestamp_offset + sliced_ts_tokens * time_precision

                    add_segment(
                        offset=timestamp_offset,
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=min(timestamp_offset + end_timestamp_position * time_precision,
                                timestamp_offset + segment_duration),
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                        start_timestamps=word_ts[0],
                        end_timestamps=word_ts[-1],
                        word_timestamps=word_ts[1:-1],
                        start_ts_logits=sliced_ts_logits[0].tolist(),
                        end_ts_logits=sliced_ts_logits[-1].tolist(),
                        word_ts_logits=sliced_ts_logits[1:-1]
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                    min(tokens[last_slice - 1].item() - tokenizer.timestamp_begin, segment_max_ts)
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = min(timestamps[-1].item() - tokenizer.timestamp_begin, segment_max_ts)
                    duration = last_timestamp_position * time_precision

                word_ts = timestamp_offset + finalized_ts_tokens * time_precision

                add_segment(
                    offset=timestamp_offset,
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                    word_timestamps=word_ts,
                    word_ts_logits=ts_logits
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if all_segments:
                all_segments[-1]['anchor_point'] = True
                all_segments[-1]['next_offset'] = float(seek * HOP_LENGTH / SAMPLE_RATE)
            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            update_pbar()

    if len(all_segments) > 1 and all_segments[-1]['alt_start_timestamps'] is None:
        all_segments[-1]['alt_start_timestamps'] = all_segments[-2]['alt_end_timestamps']

    if stab:
        all_segments = stabilize_timestamps(all_segments, top_focus=top_focus)
        # from .refinement import merge_no_dur_segments

        if refine_ts_num:
            from .refinement import refine_word_level_ts
            all_segments = refine_word_level_ts(all_segments,
                                                tokenizer=tokenizer, ts_num=ts_num,
                                                stab_options=dict(top_focus=top_focus),
                                                average=avg_refine, )
        add_whole_word_ts(tokenizer, all_segments,
                          prepend_punctuations=prepend_punctuations,
                          append_punctuations=append_punctuations)
        # merge_no_dur_segments(all_segments)
        if print_stab:
            print('\nSTABILIZED:')
            for seg_ in all_segments:
                print(f'[{format_timestamp(seg_["start"])} -->'
                      f' {format_timestamp(seg_["end"])}] "{seg_["text"]}"')
                ts_str = (f' ->[{format_timestamp(ts_["timestamp"])}] "{ts_["word"].strip()}"'
                          for ts_ in seg_['whole_word_timestamps'])
                print('\n'.join(ts_str), end='\n\n')

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]),
                segments=all_segments,
                language=language,
                time_scale=time_scale)


def modify_model(model: Whisper):
    """
    Modifies model instance by:
        -replacing model.decode with decode_word_level
        -replacing model.transcribe with transcribe_word_level
    """
    model.decode = MethodType(decode_word_level, model)
    model.transcribe = MethodType(transcribe_word_level, model)


# modified version of whisper.load_model
def load_model(name: str, device: Optional[Union[str, torch.device]] = None,
               download_root: str = None, in_memory: bool = False) -> Whisper:
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

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """
    model = whisper.load_model(name, device=device, download_root=download_root, in_memory=in_memory)
    modify_model(model)
    return model


# modified version of whisper.transcribe.cli
def cli():
    import argparse
    import os
    from os.path import splitext, split
    from whisper import available_models
    from whisper.utils import optional_int, optional_float
    from .text_output import results_to_sentence_word_ass, results_to_sentence_srt, results_to_word_srt, \
        save_as_json, load_results

    str2val = {"true": True, "false": False, "1": True, "0": False}

    def str2bool(string: str) -> bool:
        if string in str2val:
            return str2val[string.lower()]
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    output_formats = ("srt", "ass", "json")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=str,
                        help="audio/video file(s) to transcribe or json file(s) to process into [output_format]")
    parser.add_argument("--output", "-o", nargs="+", type=str,
                        help="output filepaths(s);"
                             "if not specified, auto-named output file(s) will be saved to "
                             "[output_dir] or current dir if not specified.")
    parser.add_argument("--model", default="base", choices=available_models(),
                        help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-d", type=str,
                        help="directory to save the outputs;"
                             "if a path in [output] does not have parent, that output will be save to this directory")
    parser.add_argument("--output_format", "-f", type=str,
                        choices=output_formats,
                        help="format of the output file; "
                             "if not specified, it will parse format from [output] and "
                             "if parsed format not supported [output] or [output] not provided, "
                             "json will be used non-json input(s) and ass for json input(s)")
    parser.add_argument("--word_srt", action='store_true',
                        help="Force all srt outputs to be only be word-level.")
    parser.add_argument("--pbar", type=str2bool, default=True,
                        help="whether to show progress with a progressbar")
    parser.add_argument("--verbose", '-v', action='store_true',
                        help="whether to display the progress of decoded segments (not yet stabilized) to the console")
    parser.add_argument("--print_stab", '-ps', action='store_true',
                        help="whether to display the decoded text "
                             "(with stabilized word-level timestamps) to the console.")

    parser.add_argument("--task", type=str, default="transcribe",
                        choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') "
                             "or X->English translation ('translate')")
    parser.add_argument("--language", '-l', type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--prepend_punctuations", '-pp', nargs="+", type=str,
                        help="Punctuations to prepend to next word. (Default: “¿([{)")
    parser.add_argument("--append_punctuations", '-ap', nargs="+", type=str,
                        help="Punctuations to append to previous word (Default: .。,，!！?？:：”)]}、)")

    parser.add_argument('--stab', type=str2bool, default=True,
                        help="whether to stabilizing timestamps")
    parser.add_argument('--top_focus', type=str2bool, default=False,
                        help="whether to adhere closely to the top predictions while stabilizing timestamps")
    parser.add_argument('--ts_num', type=int, default=10,
                        help="number of top word-level timestamp predictions to use for stabilization")
    parser.add_argument('--suppress_silence', type=str2bool, default=True,
                        help="whether to suppress timestamp where audio is silent at segment-level")
    parser.add_argument('--suppress_middle', type=str2bool, default=True,
                        help="whether to suppress silence only for beginning and ending of segments")
    parser.add_argument('--suppress_word_ts', type=str2bool, default=True,
                        help="whether to suppress timestamp where audio is silent at word-level")

    parser.add_argument('--remove_background', type=str2bool, default=True,
                        help="whether to zero sections with background noise from waveform, "
                             "so it will be marked as silent")
    parser.add_argument('--upper_quantile', type=float,
                        help="upper quantile of amplitude to determine a max amplitude (Default: 0.85)")
    parser.add_argument('--lower_quantile', type=float,
                        help="lower quantile of amplitude to determine a min amplitude (Default: 0.15)")
    parser.add_argument('--lower_threshold', type=float,
                        help="zero sections of waveform to marked at silent where: "
                             "amplitude < [lower_threshold] * ([upper_quantile]-[lower_quantile]) + [lower_quantile]. "
                             "(Default: 0.15)")
    parser.add_argument('--silence_threshold', type=float, default=0.1,
                        help="If an audio segment's percentage of silence >= [silence_threshold], "
                             "then that segment will not have background removed even if [remove_background]=True.")

    parser.add_argument('--refine_ts_num', type=int,
                        help="Same as [ts_num] for word timestamp refinement (Default: 10 times of [ts_num])")
    parser.add_argument('--avg_refine', type=str2bool, default=False,
                        help="whether to average the original word timestamps with the refined timestamps")

    parser.add_argument('--time_scale', type=float, default=1.0,
                        help="Factor for scaling audio duration for inference."
                             "Greater than 1.0 'slows down' the audio. "
                             "Less than 1.0 'speeds up' the audio. "
                             "1.0 is no scaling.")

    # word/segment-level srt output
    parser.add_argument('--combine_compound', type=str2bool, default=False,
                        help="concatenate words without inbetween spacing")
    parser.add_argument('--strip', type=str2bool, default=True,
                        help="perform strip() on each word/segment")
    parser.add_argument('--min_dur', type=float, default=0.02,
                        help="minimum duration for each word")

    # segment-level srt/ass output
    parser.add_argument('--end_at_last_word', type=str2bool, default=False,
                        help="set end of segment to timestamp of last token")
    parser.add_argument('--end_before_period', type=str2bool, default=False,
                        help="set end of segment to timestamp of last non-period token")
    parser.add_argument('--start_at_first_word', type=str2bool, default=False,
                        help="set start of segment to timestamp of first-token")
    parser.add_argument('--force_max_len', type=int,
                        help="limit a max number of characters per segment. Ignored if None. "
                             "Note: character count is still allow to go under this number for stability reasons.")

    # ass output
    parser.add_argument('--underline', type=str2bool, default=True,
                        help="whether to underline a word at its corresponding timestamp for ASS output(s)")
    parser.add_argument('--color', type=str, default='00FF00',
                        help="color code for a word at its corresponding timestampASS output(s). "
                             "<bbggrr> reverse order hexadecimal RGB value "
                             "(e.g. FF0000 is full intensity blue).")
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

    parser.add_argument('--overwrite', '-y', action='store_true',
                        help='overwrite all output files')

    parser.add_argument('--debug', action='store_true',
                        help='print all input/output pair(s) and all arguments used for transcribing/translating')

    args = parser.parse_args().__dict__
    debug = args.pop('debug')

    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    inputs: List[str] = args.pop("inputs")
    outputs: List[str] = args.pop("output")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    overwrite: bool = args.pop("overwrite")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    combine_compound: bool = args.pop('combine_compound')
    strip: bool = args.pop('strip')
    min_dur: float = args.pop('min_dur')

    end_at_last_word: bool = args.pop('end_at_last_word')
    end_before_period: bool = args.pop('end_before_period')
    start_at_first_word: bool = args.pop('start_at_first_word')
    force_max_len: int = args.pop('force_max_len')

    underline: bool = args.pop('underline')
    color: str = args.pop('color')
    font: str = args.pop('font')
    font_size: int = args.pop('font_size')

    only_word_srt: bool = args.pop('word_srt')

    def is_json(file: str):
        return file.endswith(".json")

    def get_output_ext(file: str) -> str:
        if not output_format:
            return f'.{"srt" if is_json(file) else "json"}'
        return f'.{output_format}'

    if outputs:
        if len(outputs) != len(inputs):
            raise NotImplementedError(f'Got {len(inputs)} audio file(s) but specified {len(outputs)} output file(s).')
        if output_dir:
            for i in range(len(outputs)):
                if splitext(outputs[i])[1].strip('.') not in output_formats:
                    outputs[i] += get_output_ext(inputs[i])
                if not ('/' in outputs[i] or '\\' in outputs[i]):
                    outputs[i] = os.path.join(output_dir, outputs[i])
    else:
        if not output_dir:
            output_dir = '.'
        outputs = [os.path.join(output_dir, f'{splitext(split(i)[1])[0]}{get_output_ext(i)}') for i in inputs]

    if not overwrite:

        def cancel_overwrite():
            resp = input(f'{path} already exist, overwrite (y/n)? ').lower()
            if resp in ('y', 'n'):
                return resp == 'n'
            print(f'Expected "y" or "n", but got {resp}.')
            return True

        for path in outputs:
            if os.path.isfile(path) and cancel_overwrite():
                return

    device: str = args.pop("device")

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(f"{model_name} is an English-only model but receipted "
                          f"'{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    args['temperature'] = temperature

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    if debug:
        print(f'\nModel Arguments',
              f'\nModel: {model_name}\n'
              f'device: {device}\n'
              f'download_root: {model_dir}\n')
        print(f'Arguments for {args.get("task")}')
        for k, v in args.items():
            print(f'{k}: {v}')
        print(f'\nArguments for Word/Segment-Level SRT/ASS Output',
              f'\ncombine_compound: {combine_compound}\n'
              f'strip: {strip}\n'
              f'min_dur: {min_dur}\n'
              f'overwrite: {overwrite}\n'
              f'\nArguments for Segment-Level SRT/ASS Output',
              f'\nend_at_last_word: {end_at_last_word}\n'
              f'end_before_period: {end_before_period}\n'
              f'start_at_first_word: {start_at_first_word}\n'
              f'force_max_len: {force_max_len}\n'
              f'\nArguments for ASS Output',
              f'\nunderline: {underline}\n'
              f'color: {color}\n'
              f'font: {font}\n'
              f'font_size: {font_size}\n')

        print('Input(s)  ->  Outputs(s)')
        for input_path, output_path in zip(inputs, outputs):
            print(f'{input_path}  ->  {output_path}')
        print('\n')

    model = load_model(model_name, device=device, download_root=model_dir)

    for input_path, output_path in zip(inputs, outputs):
        if is_json(input_path):
            result = load_results(input_path)
            if args.get('prepend_punctuations') or args.get('append_punctuations'):
                add_whole_word_ts(get_tokenizer(model.is_multilingual, task=args.get('task')),
                                  result,
                                  prepend_punctuations=args.get('prepend_punctuations'),
                                  append_punctuations=args.get('append_punctuations'))
        else:
            result = model.transcribe(input_path, **args)

        if is_json(output_path):
            save_as_json(result, output_path)
        elif output_path.endswith('.srt'):
            if only_word_srt:
                results_to_word_srt(result, output_path,
                                    combine_compound=combine_compound,
                                    strip=strip,
                                    min_dur=min_dur)
            else:
                results_to_sentence_srt(result, output_path,
                                        end_at_last_word=end_at_last_word,
                                        end_before_period=end_before_period,
                                        start_at_first_word=start_at_first_word,
                                        force_max_len=force_max_len,
                                        strip=strip)
        elif output_path.endswith('.ass'):
            results_to_sentence_word_ass(result, output_path,
                                         color=color,
                                         underline=underline,
                                         font=font,
                                         font_size=font_size,
                                         end_at_last_word=end_at_last_word,
                                         end_before_period=end_before_period,
                                         start_at_first_word=start_at_first_word,
                                         combine_compound=combine_compound,
                                         min_dur=min_dur,
                                         force_max_len=force_max_len,
                                         strip=strip)


if __name__ == '__main__':
    cli()
