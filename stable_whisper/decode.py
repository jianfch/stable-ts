from typing import TYPE_CHECKING, List, Union
from dataclasses import replace

import torch
import numpy as np

from .whisper_compatibility import DecodingTask, DecodingOptions, DecodingResult


if TYPE_CHECKING:
    from .whisper_compatibility import Whisper


def _suppress_ts(ts_logits: torch.Tensor, ts_token_mask: torch.Tensor = None):
    if ts_token_mask is not None:
        ts_logits[:, ts_token_mask] = -np.inf


# modified version of whisper.decoding.DecodingTask
class DecodingTaskStable(DecodingTask):

    def __init__(self, *args, **kwargs):
        self.ts_token_mask: torch.Tensor = kwargs.pop('ts_token_mask', None)
        self.audio_features: torch.Tensor = kwargs.pop('audio_features', None)
        super(DecodingTaskStable, self).__init__(*args, **kwargs)

    def _get_audio_features(self, mel: torch.Tensor):
        if self.audio_features is None:
            audio_features = super()._get_audio_features(mel)
            self.audio_features = audio_features.detach().clone()
            return audio_features
        return self.audio_features.clone()

    # modified version of whisper.DecodingTask._main_loop
    def _main_loop(self, audio_features: torch.Tensor, tokens: torch.Tensor):
        n_batch = tokens.shape[0]
        sum_logprobs: torch.Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # suppress timestamp tokens where the audio is silent so that decoder ignores those timestamps
                _suppress_ts(logits[:, self.tokenizer.timestamp_begin:], self.ts_token_mask)

                logits.nan_to_num_(-np.inf)
                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs


# modified version of whisper.decoding.decode
@torch.no_grad()
def decode_stable(model: "Whisper",
                  mel: torch.Tensor,
                  options: DecodingOptions = None,
                  ts_token_mask: torch.Tensor = None,
                  audio_features: torch.Tensor = None,
                  **kwargs, ) -> \
        Union[DecodingResult, List[DecodingResult], tuple]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model : whisper.model.Whisper
        An instance of Whisper ASR model.
    mel : torch.Tensor,
        A tensor containing the Mel spectrogram(s). ``mel.shape`` must be (80, 3000) or (*, 80, 3000).
    options : whisper.decode.DecodingOptions, default whisper.decode.DecodingOptions()
        A dataclass that contains all necessary options for decoding 30-second segments
    ts_token_mask : torch.Tensor, optional
        Mask for suppressing to timestamp token(s) for decoding.
    audio_features : torch.Tensor, optional
        Reused ``audio_feature`` from encoder for fallback.

    Returns
    -------
    whisper.decode.DecodingResult or list whisper.decode.DecodingResult
        The result(s) of decoding contained in ``whisper.decode.DecodingResult`` dataclass instance(s).
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if options is None:
        options = DecodingOptions()

    if kwargs:
        options = replace(options, **kwargs)

    task = DecodingTaskStable(model, options, ts_token_mask=ts_token_mask, audio_features=audio_features)
    result = task.run(mel)

    return result[0] if single else result, task.audio_features
