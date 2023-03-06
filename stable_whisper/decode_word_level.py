import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from whisper.utils import compression_ratio
from whisper.decoding import DecodingTask, BeamSearchDecoder, GreedyDecoder
from whisper.tokenizer import Tokenizer
from typing import List, Tuple, Union
from whisper import DecodingOptions, DecodingResult


def _suppress_ts(ts_logits: torch.Tensor, suppress_ts_mask: torch.Tensor = None):
    if suppress_ts_mask is not None:
        ts_logits[:, suppress_ts_mask] = -np.inf


def _ts_topk(ts_logits: torch.Tensor, k: int, prev_ts: torch.Tensor = None, has_ts_batch=False) -> torch.Tensor:
    if has_ts_batch:
        ts_logits = ts_logits.softmax(dim=-1).mean(dim=-2).unsqueeze(-2)
    temp_ts = torch.stack(torch.topk(ts_logits, k, dim=-1), 0)
    temp_ts.unsqueeze_(-2)
    return temp_ts if prev_ts is None else torch.cat([prev_ts, temp_ts], dim=-2)


# modified version of whisper.decoding.GreedyDecoder
class GreedyDecoderWordLevel(GreedyDecoder):
    def __init__(self, *args, **kwargs):
        self.ts_num: int = kwargs.pop('ts_num', 10)
        self.suppress_ts_mask: torch.Tensor = kwargs.pop('suppress_ts_mask', None)
        self.timestamp_begin = kwargs.pop('timestamp_begin', 50364)
        super(GreedyDecoderWordLevel, self).__init__(*args, **kwargs)
        self.ts = None

    def _suppress_ts(self, logits: torch.Tensor):
        _suppress_ts(logits[:, self.timestamp_begin:],
                     suppress_ts_mask=self.suppress_ts_mask)

    def update_with_ts(self, tokens: torch.Tensor, logits: torch.Tensor, sum_logprobs: torch.Tensor, ts: torch.Tensor) \
            -> Tuple[torch.Tensor, bool]:
        self.ts = ts

        self._suppress_ts(logits)

        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: torch.Tensor, sum_logprobs: torch.Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist(), self.ts.transpose(1, 0)[None]


# modified version of whisper.decoding.BeamSearchDecoder
class BeamSearchDecoderWordLevel(BeamSearchDecoder):

    def __init__(self, *args, **kwargs):
        self.ts_num: int = kwargs.pop('ts_num', 10)
        self.suppress_ts_mask: torch.Tensor = kwargs.pop('suppress_ts_mask', None)
        self.timestamp_begin = kwargs.pop('timestamp_begin', 50364)
        super(BeamSearchDecoderWordLevel, self).__init__(*args, **kwargs)
        self.ts = None
        self.finished_ts_ls = None

    def reset(self):
        self.finished_sequences = None
        self.finished_ts_ls = None

    def _suppress_ts(self, logits: torch.Tensor):
        _suppress_ts(logits[:, self.timestamp_begin:],
                     suppress_ts_mask=self.suppress_ts_mask)

    def update_with_ts(self, tokens: torch.Tensor, logits: torch.Tensor, sum_logprobs: torch.Tensor, ts: torch.Tensor) \
            -> Tuple[torch.Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        self.ts = ts

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]
            self.finished_ts_ls = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences, finished_ts_ls = [], [], [], []

        self._suppress_ts(logprobs)

        for i in range(n_audio):
            scores, sources, finished, finished_ts = {}, {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                    finished_ts[sequence] = self.ts[:, sources[sequence]]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)
            finished_ts_ls.append(finished_ts)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)
        self.ts = self.ts[:, source_indices]

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished, \
            prev_ts_ls, new_ts_ls in \
                zip(self.finished_sequences, finished_sequences,
                    self.finished_ts_ls, finished_ts_ls):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]
                prev_ts_ls[seq] = new_ts_ls[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: torch.Tensor, sum_logprobs: torch.Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        self.ts = self.ts.reshape(self.ts.shape[0], *preceding_tokens.shape[:2], *self.ts.shape[2:])
        sum_logprobs = sum_logprobs.cpu()
        for i, (sequences, ts_) in \
                enumerate(zip(self.finished_sequences, self.finished_ts_ls)):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    seq_tuple = tuple(sequence)
                    sequences[seq_tuple] = sum_logprobs[i][j].item()
                    ts_[seq_tuple] = self.ts[:, i, j]
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[torch.Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        final_ts: List[List[torch.Tensor]] = [
            list(sequences.values()) for sequences in self.finished_ts_ls
        ]
        return tokens, sum_logprobs, final_ts


# modified version of whisper.decoding.DecodingTask
class DecodingTaskWordLevel(DecodingTask):

    def __init__(self, *args, **kwargs):
        self.ts_num: int = kwargs.pop('ts_num', None) or 10
        self.ts_batch_size: int = kwargs.pop('ts_batch_size', None)
        self.ts_batch_noise: float = kwargs.pop('ts_batch_noise')
        if self.ts_batch_noise is None:
            self.ts_batch_noise = 0.25
        self.suppress_ts_mask: torch.Tensor = kwargs.pop('suppress_ts_mask', None)
        self.suppress_word_ts: bool = kwargs.pop('suppress_word_ts', True)
        self.sync_empty: bool = kwargs.pop('sync_empty', False)
        super(DecodingTaskWordLevel, self).__init__(*args, **kwargs)
        if hasattr(self.decoder, 'beam_size'):
            self.decoder = BeamSearchDecoderWordLevel(self.decoder.beam_size,
                                                      self.decoder.eot,
                                                      self.inference,
                                                      self.decoder.patience,
                                                      ts_num=self.ts_num,
                                                      suppress_ts_mask=self.suppress_ts_mask,
                                                      timestamp_begin=self.tokenizer.timestamp_begin)
        else:
            self.decoder = GreedyDecoderWordLevel(self.decoder.temperature,
                                                  self.decoder.eot,
                                                  ts_num=self.ts_num,
                                                  suppress_ts_mask=self.suppress_ts_mask,
                                                  timestamp_begin=self.tokenizer.timestamp_begin)

    # modified version of whisper.DecodingTask._main_loop
    def _main_loop(self, audio_features: torch.Tensor, tokens: torch.Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: torch.Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        if self.ts_batch_size:
            extra_audio_features = audio_features.repeat_interleave(self.ts_batch_size, 0)
            torch.manual_seed(0)
            audio_features = torch.cat([audio_features,
                                        extra_audio_features *
                                        (1 - (torch.rand_like(extra_audio_features) * self.ts_batch_noise))],
                                       dim=0)

        try:
            for i in range(self.sample_len):
                if self.ts_batch_size:
                    logits = self.inference.logits(tokens.repeat_interleave(audio_features.shape[0], 0),
                                                   audio_features)
                else:
                    logits = self.inference.logits(tokens, audio_features)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:1 if self.ts_batch_size else None, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                ts_logits = logits[:, self.tokenizer.timestamp_begin:].clone()
                if self.suppress_word_ts:
                    _suppress_ts(ts_logits, self.suppress_ts_mask)
                ts = _ts_topk(ts_logits, k=self.ts_num, prev_ts=self.decoder.ts, has_ts_batch=bool(self.ts_batch_size))
                if self.sync_empty:
                    del ts_logits
                    torch.cuda.synchronize(audio_features.device)
                    torch.cuda.empty_cache()

                if self.ts_batch_size:
                    logits = logits[:1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update_with_ts(tokens, logits, sum_logprobs, ts)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    # modified version of whisper.DecodingTask.run
    @torch.no_grad()
    def run(self, mel: torch.Tensor) \
            -> Union[List[DecodingResult], Tuple[List[DecodingResult], List[List[int]]]]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: torch.Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: torch.Tensor = torch.tensor([self.initial_tokens]).expand(n_audio, -1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(audio_features=features, language=language, language_probs=probs)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs, ts = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[torch.Tensor]] = [
            [t[self.sample_begin: (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
        ]
        ts: List[List[torch.Tensor]] = [[t[:, :tokens[i][j].shape[-1]] for j, t in enumerate(s)] for i, s in
                                        enumerate(ts)]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        ts: List[List[int]] = [t[i].tolist() for i, t in zip(selected, ts)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
                   DecodingResult(
                       audio_features=features,
                       language=language,
                       tokens=tokens,
                       text=text,
                       avg_logprob=avg_logprob,
                       no_speech_prob=no_speech_prob,
                       temperature=self.options.temperature,
                       compression_ratio=compression_ratio(text),
                   )
                   for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
               ], ts


# modified version of whisper.decoding.decode
@torch.no_grad()
def decode_word_level(model: "Whisper", mel: torch.Tensor, options: DecodingOptions = DecodingOptions(),
                      ts_num: int = None, ts_batch_size: int = None, ts_batch_noise: float = None,
                      suppress_ts_mask: torch.Tensor = None, suppress_word_ts=False, sync_empty=False) -> \
        Union[DecodingResult, List[DecodingResult], tuple]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        The Whisper model modified instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    ts_num: int
        Number of additional top timestamp predictions to save for each word for postprocessing stabilization (default: 10)

    ts_batch_size: int
        Number of extra noisy samples to use for word-level timestamps. (Default: None)
        The logits of these samples are averaged for decoding word-level timestamps.

    ts_batch_noise: float
        Percentage of noise to add to the samples for [ts_batch_size]. (Default: 0.25)

    suppress_ts_mask: (list, torch.Tensor)
        Mask suppress to timestamp token(s) for decoding

    suppress_word_ts: bool
        Use suppress_ts_mask to suppress timestamp tokens of words

    sync_empty: bool
        Whether to synchronize CUDA device and empty cache after each prediction.

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    result, ts = DecodingTaskWordLevel(model, options,
                                       ts_num=ts_num,
                                       ts_batch_size=ts_batch_size,
                                       ts_batch_noise=ts_batch_noise,
                                       suppress_ts_mask=suppress_ts_mask,
                                       suppress_word_ts=suppress_word_ts,
                                       sync_empty=sync_empty).run(mel)

    if single:
        result = result[0]
        ts_tokens = ts[0][1]
        ts_logits = ts[0][0]
    else:
        ts_tokens = [ts_[1] for ts_ in ts]
        ts_logits = [ts_[0] for ts_ in ts]

    return result, ts_tokens, ts_logits
