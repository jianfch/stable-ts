import warnings
from typing import Union, Optional, List

import numpy as np
import torch
import torchaudio

from ..default import is_allow_overwrite


def save_audio_tensor(
        audio: Union[np.ndarray, torch.Tensor],
        path: str,
        sr: int,
        verbose: Optional[bool] = False,
        silent_timings: Optional[Union[np.ndarray, List[dict]]] = None,
        channel: Optional[str] = 'l',
        overwrite: Optional[bool] = None
):
    """
    Save ``audio`` to ``path`` with sections muted according to ``silent_timing`` on ``channel``.
    """
    if not is_allow_overwrite(path, overwrite):
        return
    if channel is not None and channel not in ('l', 'r'):
        raise ValueError(f'``split`` must be "l" or "r" but got "{channel}".')
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    else:
        audio = audio.cpu()
    dims = audio.ndim
    if dims == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if 0 == dims or dims > 2:
        warnings.warn(f'{dims}D audio Tensor not supported.')
        return
    if dims == 1:
        audio = audio[None]

    if silent_timings is not None:
        audio = audio.mean(dim=0) if audio.shape[0] == 2 else audio[0]
        audio_copy = audio.clone()
        for t in silent_timings:
            s, e = (t['start'], t['end']) if isinstance(t, dict) else t
            s = round(s * sr)
            e = round(e * sr)
            audio_copy[s:e] = 0
        if channel is None:
            audio = audio_copy[None]
        else:
            audio = (audio_copy, audio) if channel == 'l' else (audio, audio_copy)
            audio = torch.stack(audio, dim=0)

    try:
        torchaudio.save(path, audio, sr)
    except ValueError as e:
        warnings.warn(f'Failed to save audio to "{path}". Error: {e}', stacklevel=2)
    else:
        if verbose is not None:
            print(f'Saved: "{path}"')
