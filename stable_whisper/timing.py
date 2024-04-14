import warnings
import string
import torch
import numpy as np
from typing import TYPE_CHECKING, List, Callable, Optional, Tuple
from itertools import chain

from whisper.audio import TOKENS_PER_SECOND, N_SAMPLES_PER_TOKEN
from whisper.timing import WordTiming, median_filter, dtw, merge_punctuations

if TYPE_CHECKING:
    from whisper.tokenizer import Tokenizer
    from whisper.model import Whisper


def _compute_qks(
        model: "Whisper",
        tokenizer: "Tokenizer",
        text_tokens: List[int],
        mel: torch.Tensor,
        num_samples: int,
        tokens: torch.tensor,
        medfilt_width: int = 7,
        qk_scale: float = 1.0,
        audio_features: torch.Tensor = None,
) -> Tuple[torch.Tensor, List[float]]:

    # install hooks on the cross attention layers to retrieve the attention weights
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad():
        if audio_features is None:
            audio_features = model.encoder(mel.unsqueeze(0))
        logits = model.decoder(tokens.unsqueeze(0), audio_features)[0]
        sampled_logits = logits[len(tokenizer.sot_sequence):, : tokenizer.eot]
        token_probs = sampled_logits.softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]
        text_token_probs = text_token_probs.tolist()

    for hook in hooks:
        hook.remove()

    # heads * tokens * frames
    weights = torch.cat([QKs[_l][:, _h] for _l, _h in model.alignment_heads.indices().T], dim=0)
    weights = weights[:, len(tokenizer.sot_sequence): -1, : round(num_samples / N_SAMPLES_PER_TOKEN)]
    weights = (weights * qk_scale).softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = median_filter(weights, medfilt_width)

    return weights, text_token_probs


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
        extra_models: List["Whisper"] = None
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

    kwargs = dict(
        tokenizer=tokenizer,
        text_tokens=text_tokens,
        mel=mel,
        num_samples=num_samples,
        tokens=tokens,
        qk_scale=qk_scale,
        medfilt_width=medfilt_width
    )

    weights, text_token_probs = _compute_qks(model, audio_features=audio_features, **kwargs)

    if extra_models:
        extra_weights = [weights]
        extra_text_token_probs = [text_token_probs]
        for other_model in extra_models:
            m, p = _compute_qks(other_model, **kwargs)
            extra_weights.append(m)
            extra_text_token_probs.append(p)
        weights = torch.cat(extra_weights, dim=0)
        text_token_probs = torch.tensor(extra_text_token_probs, device=extra_weights[0].device).mean(dim=0).tolist()

    matrix = weights.mean(dim=0)
    text_indices, time_indices = dtw(-matrix)

    if token_split is None:
        words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    else:
        words, word_tokens = token_split
        words.append(tokenizer.decode([tokenizer.eot]))
        word_tokens.append([tokenizer.eot])
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps].clip(min=0) / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
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
                      split_callback: Callable = None):
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
                (len(tokens) == 0 or tokens[-1] != padding)
        ):
            tokens.extend(padding)
            words.append(None)
            word_tokens.append(padding)
        seg_indices.extend([i] * len(curr_words))
        tokens.extend(list(chain.from_iterable(curr_word_tokens)))
        words.extend(curr_words)
        word_tokens.extend(curr_word_tokens)

    return tokens, (words, word_tokens), seg_indices


def pop_empty_alignment(alignment: List[WordTiming]):
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

        text_tokens, token_split, seg_indices = split_word_tokens(segments, tokenizer,
                                                                  padding=gap_padding, split_callback=split_callback)

        alignment = find_alignment_stable(model, tokenizer, text_tokens, mel, num_samples,
                                          **kwargs,
                                          token_split=token_split,
                                          audio_features=audio_features,
                                          ts_num=ts_num,
                                          ts_noise=ts_noise)
        alt_beginning_alignment = pop_empty_alignment(alignment)

        merge_punctuations(alignment, prepend_punctuations, append_punctuations)

        time_offset = segments[0]["seek"]

        assert len(alignment) == len(seg_indices)
        assert (gap_padding is None or len(segments) == len(alt_beginning_alignment))
        for i, timing in zip(seg_indices, alignment):
            if len(timing.tokens) != 0:
                start = timing.start
                end = timing.end
                if (
                        len(segments[i]['words']) == 0 and
                        ((end - start) < min_word_dur) and
                        len(alt_beginning_alignment)
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
