import numpy as np
import torch
from torch.nn.functional import interpolate, avg_pool1d, pad
from torchaudio.functional import highpass_biquad, lowpass_biquad


def prep_wf_mask(wf: (torch.Tensor, np.ndarray), sr: int, output_size: int = None, kernel_size: int = 5) \
        -> torch.Tensor:
    """
    Preprocesses waveform to be processed into timestamp suppression mask.
    """
    if isinstance(wf, np.ndarray):
        wf = torch.from_numpy(wf).float()
    else:
        wf = wf.float()
    wf = highpass_biquad(lowpass_biquad(wf, sr, 3000), sr, 200).abs()
    if output_size is not None:
        wf = interpolate(wf[None, None],
                         size=output_size,
                         mode='linear',
                         align_corners=False)[0]
    assert kernel_size % 2, f'kernel_size must be odd but got {kernel_size}'
    p = kernel_size // 2
    return avg_pool1d(pad(wf.mul(255).round(), (p, p), 'reflect'), kernel_size=kernel_size, stride=1)[0]


def remove_lower_quantile(prepped_mask: torch.Tensor,
                          upper_quantile: float = None,
                          lower_quantile: float = None,
                          lower_threshold: float = None) -> torch.Tensor:
    """
    Removes lower quantile of amplitude from waveform image
    """
    if upper_quantile is None:
        upper_quantile = 0.85
    if lower_quantile is None:
        lower_quantile = 0.15
    if lower_threshold is None:
        lower_threshold = 0.15
    prepped_mask = prepped_mask.clone()
    mx = torch.quantile(prepped_mask, upper_quantile)
    mn = torch.quantile(prepped_mask, lower_quantile)
    mn_threshold = (mx - mn) * lower_threshold + mn
    prepped_mask[prepped_mask < mn_threshold] = 0
    return prepped_mask


def finalize_mask(prepped_mask: torch.Tensor, suppress_middle=True,
                  max_index: (list, int) = None) -> torch.Tensor:
    """
    Returns a PyTorch Tensor mask of sections with amplitude zero
    """

    prepped_mask = prepped_mask.bool()

    if not suppress_middle:
        nonzero_indices = prepped_mask.nonzero().flatten()
        prepped_mask[nonzero_indices[0]:nonzero_indices[-1] + 1] = True
    if max_index is not None:
        prepped_mask[max_index + 1:] = False

    return ~prepped_mask
