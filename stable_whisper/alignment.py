from typing import TYPE_CHECKING, Union, List, Callable

import torch
import numpy as np
from tqdm import tqdm

from whisper.tokenizer import get_tokenizer
from whisper.utils import format_timestamp, make_safe
from whisper.audio import (
    SAMPLE_RATE, N_FRAMES, N_SAMPLES, N_FFT, pad_or_trim, log_mel_spectrogram
)

from .result import WhisperResult
from .timing import add_word_timestamps_stable, split_word_tokens
from .audio import load_audio
from .utils import warn_compatibility_issues

if TYPE_CHECKING:
    from whisper.model import Whisper

__all__ = ['align']


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
        remove_instant_words: bool = False
) -> WhisperResult:

    warn_compatibility_issues(ignore_compatibility)

    if isinstance(text, WhisperResult):
        if language is None:
            language = text.language
        text = text.all_tokens()
    elif isinstance(text, str) and not text.startswith(' '):
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

    word_intervals = 30
    finished_tokens = 0

    result = []

    with tqdm(total=total_tokens, unit='token', disable=verbose is not False) as tqdm_pbar:

        def update_pbar(finish: bool = False):
            nonlocal finished_tokens
            if finish:
                finished_tokens = tqdm_pbar.total
            tqdm_pbar.update(finished_tokens - tqdm_pbar.n)
            if progress_callback is not None:
                progress_callback(seek=finished_tokens, total=total_tokens)

        while words and seek_sample < total_samples:
            curr_words = words[:word_intervals]
            curr_word_tokens = word_tokens[:word_intervals]
            words = words[word_intervals:]
            word_tokens = word_tokens[word_intervals:]

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
            total_duration = round(total_samples / SAMPLE_RATE, 2)
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
