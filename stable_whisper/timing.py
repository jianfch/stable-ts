import warnings
import string
import torch
import numpy as np
from typing import TYPE_CHECKING, List, Callable, Optional, Union
from itertools import chain
from dataclasses import dataclass

from .whisper_compatibility import (
    TOKENS_PER_SECOND,
    N_SAMPLES_PER_TOKEN,
    median_filter,
    dtw,
    merge_punctuations,
    disable_sdpa
)

if TYPE_CHECKING:
    from .whisper_compatibility import Whisper, Tokenizer


@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


def _new_cache(audio_features=None, extras: int = None) -> dict:
    return dict(
        audio_features=audio_features,
        jump_indices=None,
        text_token_probs=None,
        qks=None,
        extra_caches=[_new_cache() for _ in range(extras)] if extras else None
    )


def _compute_qks(
        model: "Whisper",
        tokenizer: "Tokenizer",
        text_tokens: List[int],
        mel: torch.Tensor,
        tokens: torch.tensor,
        cache: dict
):
    # install hooks on the cross attention layers to retrieve the attention weights
    cache['qks'] = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: cache['qks'].__setitem__(index, outs[-1])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad(), disable_sdpa():
        if (audio_features := cache['audio_features']) is None:
            audio_features = cache['audio_features'] = model.encoder(mel.unsqueeze(0))
        logits = model.decoder(tokens.unsqueeze(0), audio_features)[0]
        sampled_logits = logits[len(tokenizer.sot_sequence):, : tokenizer.eot]
        token_probs = sampled_logits.softmax(dim=-1)
        cache['text_token_probs'] = token_probs[np.arange(len(text_tokens)), text_tokens].tolist()

    for hook in hooks:
        hook.remove()


def _compute_atten_weights(
        model: "Whisper",
        tokenizer: "Tokenizer",
        text_tokens: List[int],
        mel: torch.Tensor,
        num_samples: int,
        tokens: torch.tensor,
        cache: dict,
        medfilt_width: int = 7,
        qk_scale: float = 1.0,
        dynamic_heads_count: Optional[int] = None
) -> torch.Tensor:
    if cache['qks'] is None:
        _compute_qks(model, tokenizer, text_tokens, mel, tokens, cache)
    QKs = cache['qks']
    if getattr(model, 'missing_alignment_heads', False) and not dynamic_heads_count:
        dynamic_heads_count = 6
    if dynamic_heads_count:
        max_qk_len = round(num_samples / N_SAMPLES_PER_TOKEN)
        if not cache.get('is_processed_qks'):
            QKs = torch.cat([qk[0, :, len(tokenizer.sot_sequence): -1, : max_qk_len] for qk in QKs])
            QKs = cache['qks'] = (QKs * qk_scale).softmax(dim=-1)
            cache['is_processed_qks'] = True

        if cache['jump_indices'] is None:
            peaks = QKs.topk(1, dim=-1).indices
        else:
            jump_indices = np.pad(cache['jump_indices'], (0, 1), constant_values=max_qk_len)
            peaks = jump_indices[:-1] + ((jump_indices[1:] - jump_indices[:-1]) * 0.5)
            peaks = torch.from_numpy(peaks).to(QKs.device)[None, :, None]
        distances = (peaks.expand_as(QKs) - torch.arange(QKs.size(-1), device=QKs.device)).abs() / 1500
        scores = (distances * QKs).sum(dim=-1)
        heads = [score.topk(dynamic_heads_count, largest=False).indices for score in scores.T]
        weights = torch.stack([QKs[_h, i] for i, _h in enumerate(heads)], dim=1)
    else:
        weights = torch.cat([QKs[_l][:, _h] for _l, _h in model.alignment_heads.indices().T], dim=0)
        weights = weights[:, len(tokenizer.sot_sequence): -1, : round(num_samples / N_SAMPLES_PER_TOKEN)]
        weights = (weights * qk_scale).softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = median_filter(weights, medfilt_width)

    return weights


def _compute_jump_indices(
        model: "Whisper",
        cache: dict,
        extra_models: List["Whisper"] = None,
        **kwargs
):
    weights = _compute_atten_weights(model, cache=cache, **kwargs)
    if extra_models:
        extra_weights = [weights]
        for mi, other_model in enumerate(extra_models):
            m = _compute_atten_weights(other_model, cache=cache['extra_caches'][mi], **kwargs)
            extra_weights.append(m)
        weights = torch.cat(extra_weights, dim=0)
        extra_text_token_probs = [c['text_token_probs'] for c in cache['extra_caches']] + [cache['text_token_probs']]
        cache['text_token_probs'] = torch.tensor(
            extra_text_token_probs,
            device=extra_weights[0].device
        ).mean(dim=0).tolist()

    matrix = weights.mean(dim=0)
    text_indices, time_indices = dtw(-matrix)

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    cache['jump_indices'] = time_indices[jumps].clip(min=0)


# modified version of whisper.timing.find_alignment
def find_alignment_stable(
        model: "Whisper",
        tokenizer: "Tokenizer",
        text_tokens: List[int],
        mel: torch.Tensor,
        num_samples: int,
        *,
        medfilt_width: int = 7,
        qk_scale: float = 1.0,
        ts_num: int = 0,
        ts_noise: float = None,
        token_split=None,
        audio_features: torch.Tensor = None,
        extra_models: List["Whisper"] = None,
        dynamic_heads: Optional[Union[bool, int, str]] = None
) -> List[WordTiming]:
    if extra_models and (invalid_model_types := set(map(type, extra_models)) - {type(model)}):
        raise NotImplementedError(f'Got unsupported model type(s): {invalid_model_types}')

    if ts_num:
        warnings.warn('``ts_num`` is deprecated and will be removed in future versions.',
                      stacklevel=2)
    if ts_noise:
        warnings.warn('``ts_noise`` is deprecated and will be removed in future versions.',
                      stacklevel=2)
    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    ).to(model.device)

    if token_split is None:
        words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    else:
        words, word_tokens = token_split
        words.append(tokenizer.decode([tokenizer.eot]))
        word_tokens.append([tokenizer.eot])
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    if dynamic_heads:
        if dynamic_heads is True:
            dynamic_heads_count = 6
            dynamic_iterations = None
        elif isinstance(dynamic_heads, int):
            dynamic_heads_count = dynamic_heads
            dynamic_iterations = None
        else:
            assert ',' in dynamic_heads
            dynamic_heads = dynamic_heads.split(',')
            dynamic_heads_count = int(dynamic_heads[0])
            dynamic_iterations = int(dynamic_heads[1])
    else:
        dynamic_heads_count = dynamic_iterations = None
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        text_tokens=text_tokens,
        mel=mel,
        num_samples=num_samples,
        tokens=tokens,
        qk_scale=qk_scale,
        medfilt_width=medfilt_width,
        extra_models=extra_models,
        dynamic_heads_count=dynamic_heads_count
    )
    cache = _new_cache(audio_features=audio_features, extras=0 if extra_models is None else len(extra_models))
    for _ in range(dynamic_iterations or 1):
        _compute_jump_indices(cache=cache, **kwargs)
    jump_times = cache['jump_indices'] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(cache['text_token_probs'][i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]


def _split_tokens(tokens: List[int], tokenizer: "Tokenizer"):
    split_by_space = getattr(tokenizer, 'language_code', tokenizer.language) not in {"zh", "ja", "th", "lo", "my"}
    text = tokenizer.decode_with_timestamps(tokens)
    words = []
    word_tokens = []
    curr_tokens = []
    is_append = False
    for token in tokens:
        curr_tokens.append(token)
        curr_text = tokenizer.decode(curr_tokens)
        is_whole = token >= tokenizer.eot
        if not is_whole:
            is_whole = text[:len(curr_text)] == curr_text
            if is_whole and split_by_space:
                is_append = not (curr_text.startswith(" ") or curr_text.strip() in string.punctuation)

        if is_whole:
            if is_append and len(words) != 0:
                words[-1] += curr_text
                word_tokens[-1].extend(curr_tokens)
            else:
                words.append(curr_text)
                word_tokens.append(curr_tokens)
            text = text[len(curr_text):]
            curr_tokens = []

    if len(curr_tokens) != 0:
        words.append(curr_text if len(text) == 0 else text)
        word_tokens.append(curr_tokens)
    elif len(text) != 0:
        words[-1] += text

    return words, word_tokens


def split_word_tokens(segments: List[dict],
                      tokenizer: "Tokenizer",
                      *,
                      padding: (str, int) = None,
                      split_callback: Callable = None,
                      pad_first_seg: bool = True):
    if padding is not None:
        if isinstance(padding, str):
            padding = tokenizer.encode(padding)
        else:
            padding = [padding]
    tokens = []
    seg_indices = []
    words = []
    word_tokens = []
    for i, s in enumerate(segments):
        temp_word_tokens = [t for t in s['tokens'] if not isinstance(t, int) or t < tokenizer.eot]
        curr_words, curr_word_tokens = (
            _split_tokens(temp_word_tokens, tokenizer)
            if split_callback is None else
            split_callback(temp_word_tokens, tokenizer)
        )
        assert len(curr_words) == len(curr_word_tokens), \
            f'word count and token group count do not match, {len(curr_words)} and {len(curr_word_tokens)}'
        if (
                padding is not None and
                curr_word_tokens[0][0] != padding and
                (len(tokens) == 0 or tokens[-1] != padding) and
                (pad_first_seg or i != 0)
        ):
            tokens.extend(padding)
            words.append(None)
            word_tokens.append(padding)
        seg_indices.extend([i] * len(curr_words))
        tokens.extend(list(chain.from_iterable(curr_word_tokens)))
        words.extend(curr_words)
        word_tokens.extend(curr_word_tokens)

    return tokens, (words, word_tokens), seg_indices


def pop_empty_alignment(alignment: List[WordTiming], seg_indices: Optional[List[int]] = None):
    if seg_indices is not None:
        seg_idx_pos = len(seg_indices)
        empty_wts = {}
        for i in reversed(range(len(alignment))):
            assert seg_idx_pos != -1
            if alignment[i].word is None:
                empty_wts[seg_indices[seg_idx_pos]] = alignment.pop(i)
            else:
                seg_idx_pos -= 1
        return empty_wts

    return list(reversed([alignment.pop(i) for i in reversed(range(len(alignment))) if alignment[i].word is None]))


# modified version of whisper.timing.add_word_timestamps
def add_word_timestamps_stable(
        *,
        segments: List[dict],
        model: "Whisper",
        tokenizer: "Tokenizer",
        mel: torch.Tensor,
        num_samples: int,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        audio_features: torch.Tensor = None,
        ts_num: int = 0,
        ts_noise: float = None,
        min_word_dur: float = 0.1,
        split_callback: Callable = None,
        gap_padding: Optional[str] = ' ...',
        pad_first_seg: bool = True,
        **kwargs,
):
    if len(segments) == 0:
        return

    if min_word_dur is None:
        min_word_dur = 0

    if prepend_punctuations is None:
        prepend_punctuations = "\"'“¿([{-"

    if append_punctuations is None:
        append_punctuations = "\"'.。,，!！?？:：”)]}、"

    def align():
        for seg in segments:
            seg['words'] = []

        text_tokens, token_split, seg_indices = split_word_tokens(
            segments,
            tokenizer,
            padding=gap_padding,
            split_callback=split_callback,
            pad_first_seg=pad_first_seg
        )

        alignment = find_alignment_stable(model, tokenizer, text_tokens, mel, num_samples,
                                          **kwargs,
                                          token_split=token_split,
                                          audio_features=audio_features,
                                          ts_num=ts_num,
                                          ts_noise=ts_noise)
        alt_beginning_alignment = pop_empty_alignment(alignment, seg_indices)

        merge_punctuations(alignment, prepend_punctuations, append_punctuations)

        time_offset = segments[0]["seek"]

        assert len(alignment) == len(seg_indices)
        assert (gap_padding is None or len(segments) == len(alt_beginning_alignment) + (1, 0)[pad_first_seg])
        for i, timing in zip(seg_indices, alignment):
            if len(timing.tokens) != 0:
                start = timing.start
                end = timing.end
                if (
                        len(segments[i]['words']) == 0 and
                        ((end - start) < min_word_dur) and
                        i in alt_beginning_alignment
                ):
                    start = alt_beginning_alignment[i].start
                segments[i]['words'].append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + start, 3),
                        end=round(time_offset + end, 3),
                        probability=timing.probability,
                        tokens=timing.tokens
                    )
                )

    align()

    for segment in segments:
        if len(words := segment["words"]) > 0:
            # adjust the segment-level timestamps based on the word-level timestamps
            segment["start"] = words[0]["start"]
            segment["end"] = words[-1]["end"]
