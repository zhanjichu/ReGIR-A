"""Paired metrics matching the original main.py evaluation protocol."""

from __future__ import annotations

from pathlib import Path


def image_metrics(reference: str | Path, restored: str | Path, ssim_mode: str = "gray") -> dict[str, float]:
    import cv2
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    original = cv2.imread(str(reference))
    result = cv2.imread(str(restored))
    if original is None or result is None:
        raise ValueError(f"Could not decode paired images: {reference}, {restored}")
    if original.shape != result.shape:
        raise ValueError(f"Image shape mismatch: {original.shape} != {result.shape}")
    psnr = float(peak_signal_noise_ratio(original, result, data_range=255))
    if ssim_mode == "gray":
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        channel_axis = None
    elif ssim_mode == "rgb":
        channel_axis = -1
    else:
        raise ValueError("ssim_mode must be 'gray' or 'rgb'.")
    height, width = original.shape[:2]
    win_size = min(7, height if height % 2 else height - 1, width if width % 2 else width - 1)
    if win_size < 3:
        raise ValueError("SSIM requires image width and height of at least 3 pixels.")
    ssim = float(structural_similarity(
        original, result, data_range=255, channel_axis=channel_axis, win_size=win_size,
    ))
    return {"psnr": psnr, "ssim": ssim}


def ordered_f1(expected: list[str], predicted: list[str]) -> float:
    """2 * LCS / (|G| + |P|), with the sample's annotated tool order."""
    if not expected and not predicted:
        return 1.0
    previous = [0] * (len(predicted) + 1)
    for target in expected:
        current = [0]
        for j, actual in enumerate(predicted, 1):
            current.append(previous[j - 1] + 1 if target == actual else max(previous[j], current[-1]))
        previous = current
    return 2.0 * previous[-1] / (len(expected) + len(predicted))
