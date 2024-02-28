import warnings
from itertools import chain
from typing import List, Union, Optional, Tuple

import numpy as np
import torch

from whisper.audio import TOKENS_PER_SECOND


def is_ascending_sequence(
        seq: List[Union[int, float]],
        verbose=True
) -> bool:
    """
    Return boolean for whether a sequence of numbers is in ascending order.
    """
    is_ascending = True
    for idx, (i, j) in enumerate(zip(seq[:-1], seq[1:])):
        if i > j:
            is_ascending = False
            if verbose:
                print(f'[Index{idx}]:{i} > [Index{idx + 1}]:{j}')
            else:
                break

    return is_ascending


def valid_ts(
        ts: List[dict],
        warn=True
) -> bool:
    """
    Return boolean for whether a list of timestamps is in ascending order.
    """
    valid = is_ascending_sequence(list(chain.from_iterable([s['start'], s['end']] for s in ts)), False)
    if warn and not valid:
        warnings.warn(message='Found timestamp(s) out of order.')
    return valid


def mask2timing(
        silence_mask: (np.ndarray, torch.Tensor),
        time_offset: float = 0.0,
        second_per_unit: Optional[float] = None,
        min_start: Optional[float] = None,
        max_end: Optional[float] = None
) -> (Tuple[np.ndarray, np.ndarray], None):
    """
    Return array of start timestamps and array of end timestamps corresponding to where ``silence_mask`` is ``True``.
    """
    if silence_mask is None or not silence_mask.any() or not len(silence_mask):
        return
    assert silence_mask.ndim == 1
    if isinstance(silence_mask, torch.Tensor):
        silence_mask = silence_mask.cpu().numpy().copy()
    elif not isinstance(silence_mask, np.ndarray):
        raise NotImplementedError(f'Expected torch.Tensor or numpy.ndarray, but got {type(silence_mask)}')

    mask = np.concatenate(([False], silence_mask, [False]))
    silent_starts = np.logical_and(~mask[:-2], mask[1:-1]).nonzero()[0]
    silent_ends = (np.logical_and(mask[1:-1], ~mask[2:]).nonzero()[0] + 1)
    if second_per_unit is None:
        silent_starts = silent_starts / TOKENS_PER_SECOND
        silent_ends = silent_ends / TOKENS_PER_SECOND
    else:
        silent_starts = silent_starts * second_per_unit
        silent_ends = silent_ends * second_per_unit
    if time_offset:
        silent_starts += time_offset
        silent_ends += time_offset
    if min_start is not None and silent_starts[0] < min_start:
        assert min_start <= silent_ends[0]
        silent_starts[0] = min_start
    if max_end is not None and silent_ends[-1] > max_end:
        assert max_end >= silent_starts[-1]
        silent_ends[-1] = max_end
    return silent_starts, silent_ends


def timing2mask(
        silent_starts: np.ndarray,
        silent_ends: np.ndarray,
        size: int,
        time_offset: float = None,
        units_per_second: Optional[int] = None
) -> torch.Tensor:
    """
    Return Tensor of booleans that is ``True`` corresponding to ``silent_starts`` and ``silent_ends``.
    """
    if units_per_second is None:
        units_per_second = TOKENS_PER_SECOND
    assert len(silent_starts) == len(silent_ends)
    ts_token_mask = torch.zeros(size, dtype=torch.bool)
    if time_offset:
        silent_starts = (silent_starts - time_offset).clip(min=0)
        silent_ends = (silent_ends - time_offset).clip(min=0)
    mask_i = (silent_starts * units_per_second).round().astype(np.int32)
    mask_e = (silent_ends * units_per_second).round().astype(np.int32)
    for mi, me in zip(mask_i, mask_e):
        ts_token_mask[mi:me+1] = True

    return ts_token_mask


class SetTorchThread:
    def __init__(self, temp_thread_count: int):
        self.original_thread_count = torch.get_num_threads()
        self.temp_thread_count = temp_thread_count

    def __enter__(self):
        torch.set_num_threads(self.temp_thread_count)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_num_threads(self.original_thread_count)
