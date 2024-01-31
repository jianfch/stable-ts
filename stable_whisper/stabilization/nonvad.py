from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .utils import mask2timing, timing2mask
from ..audio.utils import audio_to_tensor_resample

from whisper.audio import N_SAMPLES_PER_TOKEN


NONVAD_SAMPLE_RATES = (16000,)


def audio2loudness(
        audio_tensor: torch.Tensor,
        samples_per_unit: int = None
) -> (torch.Tensor, None):
    assert audio_tensor.dim() == 1, f'waveform must be 1D, but got {audio_tensor.dim()}D'
    audio_tensor = audio_tensor.abs()
    k = int(audio_tensor.numel() * 0.001)
    if k:
        top_values, _ = torch.topk(audio_tensor, k)
        threshold = top_values[-1]
    else:
        threshold = audio_tensor.quantile(0.999, dim=-1)
    if samples_per_unit is None:
        samples_per_unit = N_SAMPLES_PER_TOKEN
    if (token_count := round(audio_tensor.shape[-1] / samples_per_unit)+1) > 2:
        if threshold < 1e-5:
            return torch.zeros(token_count, dtype=audio_tensor.dtype, device=audio_tensor.device)
        audio_tensor = audio_tensor / min(1., threshold * 1.75)
        audio_tensor = F.interpolate(
            audio_tensor[None, None],
            size=token_count,
            mode='linear',
            align_corners=False
        )[0, 0]
        return audio_tensor


def wav2mask(
        audio: (torch.Tensor, np.ndarray, str, bytes),
        q_levels: int = 20,
        k_size: int = 5,
        sr: int = None
) -> (Tuple[torch.Tensor, Tuple[np.ndarray, np.ndarray]], None):
    """
    Generate 1D mask from waveform for suppressing timestamp tokens.
    """
    audio = audio_to_tensor_resample(audio, sr, NONVAD_SAMPLE_RATES)
    loudness_tensor = audio2loudness(audio)
    if loudness_tensor is None:
        return
    p = k_size // 2 if k_size else 0
    if p and p < loudness_tensor.shape[-1]:
        assert k_size % 2, f'kernel_size must be odd but got {k_size}'
        mask = torch.avg_pool1d(
            F.pad(
                loudness_tensor[None],
                (p, p),
                'reflect'
            ),
            kernel_size=k_size,
            stride=1
        )[0]
    else:
        mask = loudness_tensor.clone()

    if q_levels:
        mask = mask.mul(q_levels).round()

    mask = mask.bool()

    if not mask.any():  # entirely silent
        return ~mask
    temp_timings = mask2timing(mask)
    s, e = temp_timings
    se_mask = (e - s) > 0.1
    s = s[se_mask]
    e = e[se_mask]
    mask = ~timing2mask(s, e, loudness_tensor.shape[-1])

    if not mask.any():  # no silence
        return

    return mask


def audio2timings(
        audio: (torch.Tensor, np.ndarray, str, bytes),
        q_levels: int = 20,
        k_size: int = 5,
        sr: int = None
) -> (Tuple[np.ndarray, np.ndarray], None):
    return mask2timing(
        wav2mask(audio, q_levels=q_levels, k_size=k_size, sr=sr)
    )


def visualize_mask(
        loudness_tensor: torch.Tensor,
        silence_mask: torch.Tensor = None,
        width: int = 1500,
        height: int = 200,
        output: str = None,
):
    no_silence = silence_mask is None or not silence_mask.any()
    assert no_silence or silence_mask.shape[0] == loudness_tensor.shape[0]
    if loudness_tensor.shape[0] < 2:
        raise NotImplementedError(f'audio size, {loudness_tensor.shape[0]}, is too short to visualize')
    else:
        width = loudness_tensor.shape[0] if width == -1 else width
        im = torch.zeros((height, width, 3), dtype=torch.uint8)
        mid = round(height / 2)
        for i, j in enumerate(loudness_tensor.tolist()):
            j = round(abs(j) * mid)
            if j == 0 or width <= i:
                continue
            im[mid - j:mid + 1, i] = 255
            im[mid + 1:mid + j + 1, i] = 255
        if not no_silence:
            im[:, silence_mask[:width], 1:] = 0
        im = im.cpu().numpy()
        if output and not output.endswith('.png'):
            output += '.png'
        try:
            from PIL import Image
        except ModuleNotFoundError:
            try:
                import cv2
            except ModuleNotFoundError:
                raise ModuleNotFoundError('Failed to import "PIL" or "cv2" to visualize suppression mask. '
                                          'Try "pip install Pillow" or "pip install opencv-python"')
            else:
                im = im[..., [2, 1, 0]]
                if isinstance(output, str):
                    cv2.imwrite(output, im)
                else:
                    cv2.imshow('image', im)
                    cv2.waitKey(0)
        else:
            im = Image.fromarray(im)
            if isinstance(output, str):
                im.save(output)
            else:
                im.show(im)
        if output:
            print(f'Save: {output}')
