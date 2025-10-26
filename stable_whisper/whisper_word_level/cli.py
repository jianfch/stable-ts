import os
import warnings
import argparse
from typing import Optional, List, Union, Dict
from os.path import splitext, split, join
import shlex

import numpy as np
import torch

from ..result import WhisperResult
from ..utils import isolate_useful_options, str_to_valid_type, get_func_parameters
from ..audio import SUPPORTED_DENOISERS
from ..default import *

from ..whisper_compatibility import DecodingOptions, LANGUAGES, TO_LANGUAGE_CODE


def _split_input_args(args: str) -> List[str]:
    args = shlex.split(args)
    if args and args[0] == 'stable-ts':
        del args[0]
    return args


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


# modified version of whisper.transcribe.cli
def _cli(cmd: str = None, _cache: Dict[str, Union[bool, dict]] = None):

    supported_denoisers = tuple(SUPPORTED_DENOISERS.keys())

    str2val = {"true": True, "false": False, "1": True, "0": False}

    default_langs_choices = sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()])

    def str2bool(string: str) -> bool:
        string = string.lower()
        if string in str2val:
            return str2val[string]
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    def valid_model_name(name: str) -> str:
        available_models = {}
        if is_original_whisper:
            from whisper import available_models
        elif is_faster_whisper:
            from faster_whisper.utils import available_models
        _models = None if is_hf_whisper or is_mlx_whisper or available_models is None else available_models()

        if not _models or name in _models or os.path.exists(name):
            return name
        raise ValueError(
            f"model should be one of {_models} or path to a model checkpoint"
        )

    def valid_language(lang: Optional[str]) -> Union[str, None]:
        if lang is None:
            return
        lang_code = lang
        if is_faster_whisper:
            from faster_whisper.tokenizer import _LANGUAGE_CODES
            lang_code = lang
            if lang not in _LANGUAGE_CODES and lang.lower() in TO_LANGUAGE_CODE:
                lang_code = TO_LANGUAGE_CODE[lang]
            if lang_code not in _LANGUAGE_CODES:
                raise ValueError(
                    f'"{lang}" is not one of the available languages: {default_langs_choices}'
                )
        return lang_code

    def use_deprecated_args(
            key: str, old_key: str, pop: bool = False, expected_default=None, new_default=None, eg: str = None,
    ):
        new_val = args.pop(key) if pop else args[key]
        old_val = args.pop(old_key) if pop else args[old_key]

        if old_val != expected_default:
            eg_str = eg if eg is None else f' (e.g. {eg})'
            warnings.warn(f'{old_key} is deprecated and will be removed in future versions. '
                          f'Use {key}{eg_str}.', stacklevel=2)
            if new_val == expected_default:
                new_val = old_val
        elif new_val == expected_default:
            new_val = new_default
        return new_val

    def update_options_with_args(arg_key: Union[str, list], options: Optional[dict] = None, pop: bool = False):
        extra_options = arg_key if isinstance(arg_key, list) else (args.pop(arg_key) if pop else args.get(arg_key))
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

    def url_to_path(url: str):
        if '://' in url:
            from urllib.parse import urlparse
            return urlparse(url).path.strip('/')
        return url

    OUTPUT_FORMATS_METHODS = {
        "srt": "to_srt_vtt",
        "ass": "to_ass",
        "json": "save_as_json",
        "vtt": "to_srt_vtt",
        "tsv": "to_tsv",
        "txt": "to_txt",
    }

    OUTPUT_FORMATS = set(OUTPUT_FORMATS_METHODS.keys())

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=str,
                        help="audio/video filepath/URL(s) to transcribe "
                             "or json file(s) to process into [output_format]")
    parser.add_argument("--output", "-o", action="extend", nargs="+", type=str,
                        help="output filepaths(s);"
                             "if not specified, auto-named output file(s) will be saved to "
                             "[output_dir] or current dir if not specified.")
    parser.add_argument("--save_unfinished", "-su", action='store_true',
                        help="whether to save unfinished outputs caused by KeyboardInterrupt; "
                             "outputs are saved as JSON with suffix '-UNFINISHED.json'")
    parser.add_argument("--resume_input", "-ri", nargs="+", type=str,
                        help="JSON of unfinished output filepaths(s) to continue transcription from end of last word; "
                             "use '+' as suffix to redo the last segment (e.g 'output-UNFINISHED.json+')")
    parser.add_argument("--delete_resume", "-dr", action='store_true',
                        help="whether to delete file(s) from '--resume_input'/'-ri' when transcription finishes")
    parser.add_argument("--model", '-m', default="base", type=str,
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
                        choices=default_langs_choices,
                        help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--prepend_punctuations", '-pp', type=str, default=get_prepend_punctuations(),
                        help="Punctuations to prepend to next word")
    parser.add_argument("--append_punctuations", '-ap', type=str, default=get_append_punctuations(),
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
    parser.add_argument('--ts_noise', type=float,
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

    parser.add_argument('--min_word_dur', type=float, default=get_min_word_dur(),
                        help="shortest duration each word is allowed to reach for silence suppression")
    parser.add_argument('--nonspeech_error', type=float, default=0.1,
                        help="relative error of non-speech sections that appear in between a word for "
                             "silence suppression.")

    parser.add_argument('--max_chars', type=int,
                        help="maximum number of character allowed in each segment")
    parser.add_argument('--max_words', type=int,
                        help="maximum number of words allowed in each segment")

    parser.add_argument('--demucs', type=str2bool,
                        help='whether to reprocess the audio track with Demucs to isolate vocals/remove noise; '
                             'Demucs official repo: https://github.com/facebookresearch/demucs;'
                             'DEPRECATED and replace with --denoiser "demucs"')
    parser.add_argument('--demucs_output', action="extend", nargs="+", type=str,
                        help='path(s) to save the vocals isolated by Demucs as WAV file(s); '
                             'ignored if --demucs False; DEPRECATED and replace with --denoiser_output')
    parser.add_argument('--denoiser', type=str, choices=supported_denoisers,
                        help='name of denoiser to reprocess the audio track to isolate vocals/remove noise')
    parser.add_argument('--denoiser_output', action="extend", nargs="+", type=str,
                        help='path(s) to save the denoised audio as WAV file(s); '
                             'ignored if --denoiser is unspecified')
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
                        help='process entire audio track into log-Mel spectrogram first instead in chunks; '
                             'DEPRECATED and replaced with --no_stream')
    parser.add_argument('--no_stream', action='store_true',
                        help='whether to always load the entire audio track into memory')

    parser.add_argument('--only_ffmpeg', action='store_true',
                        help='whether to use only FFmpeg (and not yt-dlp) for URls')

    parser.add_argument('--overwrite', '-y', action='store_true',
                        help='overwrite all output files')

    parser.add_argument('--debug', action='store_true',
                        help='print all input/output pair(s) and all arguments used for transcribing/translating')

    parser.add_argument('--transcribe_method', '-tm', type=str, default='transcribe',
                        choices=('transcribe', 'transcribe_minimal'))

    parser.add_argument('--align', '-a', action="extend", nargs='+', type=str,
                        help='path(s) to TXT file(s) or JSON file previous result(s); '
                             'plain-text must begin with a "text=" (e.g. --align "text=plain-text")')

    parser.add_argument('--refine', '-r', action='store_true',
                        help='Refine timestamps to increase precision of timestamps')

    parser.add_argument('--locate', '-lc', action="extend", nargs='+', type=str,
                        help='words to locate in the audio(s); skips transcription and output')

    parser.add_argument('--refine_option', '-ro', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for refining timestamps; Replace True/False with 1/0; '
                             'E.g. --refine_option "steps=sese" --refine_options "rel_prob_decrease=0.05"')
    parser.add_argument('--demucs_option', '-do', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for demucs; Replace True/False with 1/0; '
                             'E.g. --demucs_option "shifts=3" --demucs_options "overlap=0.5"; '
                             'DEPRECATED and replaced with --denoiser_option')
    parser.add_argument('--denoiser_option', '-dno', action="extend", nargs='+', type=str,
                        help='Extra option(s) to use for denoiser; Replace True/False with 1/0; '
                             'E.g. --denoiser_option "shifts=3" --denoiser_option "overlap=0.5"')
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
    parser.add_argument('--huggingface_whisper', '-hw', action='store_true',
                        help='whether to run Whisper on Hugging Face Transformers for more speed than faster-whisper'
                             ' and even more speed with Flash Attention enabled on supported GPUs'
                             '(https://huggingface.co/openai/whisper-large-v3); '
                             'note: some features may not be available')
    parser.add_argument('--mlx_whisper', '-mlx', action='store_true',
                        help='whether to use mlx-whisper '
                             '(https://github.com/ml-explore/mlx-examples/tree/main/whisper); '
                             'note: some features may not be available')

    parser.add_argument('--persist', '-p', action='store_true',
                        help='Keep previous model loaded for the future sets of commands in the same CLI instance')

    args = _split_input_args(cmd) if cmd else []
    if _cache is None:
        _cache = {}
        if args:
            args = [args]
    elif _cache or args:
        if _cache and not args:
            curr_model_name = _cache['model']['fullname'] if 'model' in _cache else ''
            try:
                cmd = input(f"{curr_model_name}> ")
            except KeyboardInterrupt:
                args = []
            else:
                args = _split_input_args(cmd)

        if _cache and not args:
            _cache['persist'] = False
            return

        if _cache.get('persist') and not ('--persist' in args and '-p' in args):
            args.append('-p')

        if 'model' in _cache:
            if '--model' not in args and '-m' not in args:
                args.extend(['-m', _cache['model']['name']])
            model_type = _cache['model']['type']
            type_arg = '--faster_whisper' in args or '-fw' in args or '--huggingface_whisper' in args or '-hw' in args or '--mlx_whisper' in args or '-mlx' in args
            if not type_arg:
                if model_type == 'Faster-Whisper':
                    args.append('-fw')
                elif model_type == 'Hugging Face Whisper':
                    args.append('-hw')
                elif model_type == 'MLX Whisper':
                    args.append('-mlx')

        _, invalid_args = parser.parse_known_args(args)
        if invalid_args:
            print(f'Got invalid argument(s): {invalid_args}')
            return
        args = [args]

    args = parser.parse_args(*args).__dict__
    _cache['persist'] = args['persist']
    debug = args.pop('debug')
    if not args['language'] and (args['align'] or args['locate']):
        raise ValueError('langauge is required for --align / --locate')

    is_faster_whisper = args.pop('faster_whisper')
    is_mlx_whisper = args.pop('mlx_whisper')
    is_hf_whisper = args.pop('huggingface_whisper')
    assert not (is_faster_whisper and is_hf_whisper), f'--huggingface_whisper cannot be used with --faster_whisper'
    assert not (is_faster_whisper and is_mlx_whisper), f'--mlx_whisper cannot be used with --faster_whisper'
    assert not (is_hf_whisper and is_mlx_whisper), f'--mlx_whisper cannot be used with --huggingface_whisper'
    is_original_whisper = not (is_faster_whisper or is_hf_whisper or is_mlx_whisper)
    args['language'] = valid_language(args['language'])
    model_name: str = valid_model_name(args.pop("model"))
    model_dir: str = args.pop("model_dir")
    inputs: List[Union[str, torch.Tensor]] = args.pop("inputs")
    resume_files: List[str] = args.pop("resume_input")
    outputs: List[str] = args.pop("output")
    output_dir: str = args.pop("output_dir")
    output_format = args.pop("output_format")
    overwrite: bool = args.pop("overwrite")
    save_unfinished: bool = args.pop("save_unfinished")
    delete_resume: bool = args.pop("delete_resume")
    no_stream = use_deprecated_args('no_stream', 'mel_first', pop=True, expected_default=False)
    args['stream'] = None if not no_stream else False
    if overwrite:
        set_global_overwrite_permission(True)
    denoiser = use_deprecated_args('denoiser', 'demucs', pop=True, eg='--denoiser "demucs"')
    args['denoiser'] = 'demucs' if denoiser is True else (denoiser or None)
    denoiser_outputs = use_deprecated_args('denoiser_output', 'demucs_output', pop=True)
    denoiser_options = use_deprecated_args('denoiser_option', 'demucs_option', pop=True)
    args['denoiser_options'] = update_options_with_args(denoiser_options or '') or {}
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

    if is_original_whisper:
        model_type_name = 'Whisper'
        from .original_whisper import load_model as load_model_func
        model_name_kwarg = dict(name=model_name)
    else:
        if save_unfinished:
            raise NotImplementedError('--save_unfinished is only supported on vanilla Whisper models.')

        if resume_files:
            raise NotImplementedError('--resume_input is currently only supported on vanilla Whisper models.')

        if is_faster_whisper:
            model_type_name = 'Faster-Whisper'
            from .faster_whisper import load_faster_whisper as load_model_func
            model_name_kwarg = dict(model_size_or_path=model_name)
        elif is_mlx_whisper:
            model_type_name = 'MLX Whisper'
            from .mlx_whisper import load_mlx_whisper as load_model_func
            model_name_kwarg = dict(model_name=model_name)
        else:
            model_type_name = 'Hugging Face Whisper'
            from .hf_whisper import load_hf_whisper as load_model_func
            model_name_kwarg = dict(model_name=model_name)

        if args.get('transcribe_method') == 'transcribe_minimal':
            warnings.warn(f'{model_type_name} models already run on a version of transcribe_minimal. '
                          '--transcribe_method "transcribe_minimal" will be ignored.')
        if args.get('refine'):
            raise NotImplementedError(f'--refine is not supported for {model_type_name} models.')
        if strings_to_locate:
            raise NotImplementedError(f'--locate is not supported for {model_type_name} models.')

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

    if denoiser_outputs and len(denoiser_outputs) != len(inputs):
        raise ValueError(f'--denoiser_outputs and inputs do not match in count. '
                         f'Got {len(denoiser_outputs)} and {len(inputs)}')

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
            elif isinstance(val, dict):
                return str({k: val_to_str(v) for k, v in val.items()})
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
        if unsupported_align_fmts := [
            _ext for p in alignments
            if not p.startswith('text=') and (_ext := splitext(p)[-1].lower()) not in ('.json', '.txt')
        ]:
            raise NotImplementedError(
                f'Unsupported format(s) for alignment: {unsupported_align_fmts}'
            )
        if len(inputs) != len(alignments):
            raise NotImplementedError(
                f'Got {len(inputs)} audio file(s) but specified {len(alignments)} input(s) to align.'
            )
    else:
        alignments = ['']*len(inputs)

    def finalize_outputs(input_file: str, _output: str = None, _alignment: str = None) -> List[str]:
        _curr_output_formats = curr_output_formats.copy()
        basename, ext = splitext(_output or url_to_path(input_file))
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

        for paths in final_outputs:
            for path in paths:
                if not is_allow_overwrite(path):
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

    if args['vad'] and args['vad_onnx']:
        args['vad'] = dict(onnx=args['vad_onnx'])

    if resume_files and len(inputs) != len(resume_files):
        raise ValueError(f'--resume_input and inputs do not match in count. '
                         f'Got {len(resume_files)} and {len(inputs)}')

    if debug:
        print('Input(s)  ->  Outputs(s)')
        for i, (input_audio, output_paths, alignment) in enumerate(zip(inputs, final_outputs, alignments)):
            dm_output = f' {denoiser_outputs[i]} ->' if denoiser_outputs else ''
            if alignment:
                if alignment.startswith('text='):
                    if len(alignment) > 25:
                        alignment = alignment[5:22] + '...'
                    alignment = f' + text="{alignment}"'
                else:
                    alignment = f' + "{alignment}"'
            if resume_files:
                resume_info = f' + "{resume_files[i]}"'
            else:
                resume_info = ''
            print(f'"{input_audio}"{resume_info}{alignment}  ->{dm_output}  {output_paths}')
        print('')

    if show_curr_task:
        model_from_str = '' if model_dir is None else f' from {model_dir}'
        model_loading_str = (
            f'{model_type_name} {model_name} model {model_from_str}'
        )
        print(f'Loading {model_loading_str}\r', end='\n' if debug else '')
    else:
        model_loading_str = ''

    alignments = args['align']
    model = None

    def _load_model():
        nonlocal model
        if model is None and _cache is not None and 'model' in _cache:
            if _cache['model']['type'] == model_type_name and _cache['model']['name'] == model_name:
                model = _cache['model']['instance']
                if model_loading_str:
                    print(f"Reuse {_cache['model']['fullname'] or 'previous model'}   ")
            else:
                del _cache['model']
                import gc
                gc.collect()
        if model is None:
            model_options = dict(
                device=args.get('device'),
                download_root=model_dir,
                dq=dq,
            )
            model_options.update(model_name_kwarg)
            model_options = isolate_useful_options(model_options, load_model_func)
            update_options_with_args('model_option', model_options)
            model = call_method_with_options(load_model_func, model_options)
            if model_loading_str:
                print(f'Loaded {model_loading_str}  ')
            if _cache is not None and _cache.get('persist'):
                _cache['model'] = dict(
                    fullname=model_loading_str, name=model_name, type=model_type_name, instance=model
                )
        return model

    for i, (input_audio, output_paths) in enumerate(zip(inputs, final_outputs)):
        skip_output = False
        if isinstance(input_audio, str) and is_json(input_audio):
            result = WhisperResult(input_audio)
        else:
            model = _load_model()
            args['regroup'] = False
            args['audio'] = input_audio
            if resume_files:
                args['resume'] = resume_files[i]
            if denoiser_outputs:
                args['denoiser_options']['save_path'] = denoiser_outputs[i]
            transcribe_method = args.get('transcribe_method')
            text = None
            if alignments and (text := alignments[i]):
                if text.endswith('.json'):
                    text = WhisperResult(text)
                elif text.endswith('.txt'):
                    with open(text, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif text.startswith('text='):
                    text = text[5:]
                args['text'] = text
                transcribe_method = 'align'
            if strings_to_locate and (text := strings_to_locate[i]):
                args['text'] = text
                transcribe_method = 'locate'
                skip_output = args['verbose'] = True
            transcribe_method = getattr(model, transcribe_method)
            transcribe_options = isolate_useful_options(args, transcribe_method)
            if not text and not is_hf_whisper:
                decoding_options = (
                    isolate_useful_options(args, model.transcribe_original if is_faster_whisper else DecodingOptions)
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

        if result.unfinished_start != -1:
            result.save_as_json(splitext(output_paths[0])[0] + '-UNFINISHED.json')
            break
        elif delete_resume and 'resume' in args and os.path.isfile(args['resume']):
            os.remove(args['resume'])
            print(f'Removed: {os.path.abspath(args["resume"])}')


def cli(cmd: str = None):
    cache = {}
    while True:
        error = None
        try:
            _cli(cmd=cmd, _cache=cache)
        except RuntimeError as e:
            if not str(e).startswith('FFmpeg'):
                raise e
            error = e
        except ValueError as e:
            error = e
        if cache.get('persist'):
            if error is not None:
                print(f'Error: {error}')
        else:
            if error is not None:
                raise error
            break
        cmd = None
