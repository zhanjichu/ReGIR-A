from __future__ import annotations

from dataclasses import dataclass
import math

from agent.decisions import Decision


@dataclass(frozen=True)
class Reward:
    format: float
    reflect: float
    quality: float
    total: float


def paper_reward(
    decision: Decision | None, *, executable: bool,
    is_reflection: bool = False,
    before: dict[str, float] | None = None,
    after: dict[str, float] | None = None,
    weights: tuple[float, float, float] = (0.15, 0.15, 0.70),
    psnr_norm: float = 30.0,
) -> Reward:
    """Q2 correctness is a positive normalized PSNR/SSIM gain."""
    if psnr_norm <= 0 or len(weights) != 3 or any(
        not math.isfinite(weight) or weight < 0 for weight in weights
    ):
        raise ValueError("Invalid reward weights or PSNR normalization.")
    valid = decision is not None and executable
    format_reward = float(valid)
    quality_reward = 0.0
    if valid and decision.tool is not None:
        if before is None or after is None:
            raise ValueError("Executed tools require before/after quality measurements.")
        values = [before["psnr"], before["ssim"], after["psnr"], after["ssim"]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite quality measurement; do not silently clip PSNR.")
        quality_reward = 0.5 * (
            (after["psnr"] - before["psnr"]) / psnr_norm
            + after["ssim"] - before["ssim"]
        )
    reflect_reward = float(
        valid and is_reflection and decision.kind in {"error", "continue"} and quality_reward > 0
    )
    total = sum(weight * value for weight, value in zip(
        weights, (format_reward, reflect_reward, quality_reward),
    ))
    return Reward(format_reward, reflect_reward, quality_reward, total)
