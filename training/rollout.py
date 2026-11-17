from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.decisions import Decision, parse_decision
from agent.metrics import image_metrics
from agent.prompts import QUERY_REFLECT, QUERY_SELECT
from training.rewards import Reward, paper_reward


@dataclass
class Step:
    images: tuple[Path, ...]
    query: str
    tokens: Any
    decision: Decision | None
    reward: Reward
    old_logps: Any = None
    reference_logps: Any = None


@dataclass
class Trajectory:
    steps: list[Step]
    output: Path
    status: str

    @property
    def reward(self) -> float:
        return sum(step.reward.total for step in self.steps)


class RolloutRunner:
    def __init__(
        self, agent, executor, *, max_cycles: int = 6,
        weights=(0.15, 0.15, 0.70), psnr_norm: float = 30.0,
        ssim_mode: str = "gray", quality_fn=image_metrics,
    ):
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive.")
        self.agent, self.executor = agent, executor
        self.max_cycles, self.weights, self.psnr_norm = max_cycles, tuple(weights), psnr_norm
        self.ssim_mode, self.quality_fn = ssim_mode, quality_fn

    def rollout(self, sample: dict, workspace: Path) -> Trajectory:
        source, reference = Path(sample["input_image"]), Path(sample["reference_image"])
        previous_input = previous_output = source
        previous_tool = None
        steps = []
        status = "max_cycles_reached"
        for index in range(self.max_cycles):
            initial = index == 0
            images = (source,) if initial else (previous_input, previous_output)
            query = QUERY_SELECT if initial else QUERY_REFLECT.format(previous_tool=previous_tool)
            response, tokens = self.agent.sample(images, query)
            decision = parse_decision(response)
            if initial and (decision is None or decision.kind != "select" or decision.image_ref != "image"):
                decision = None
            if not initial and decision is not None and decision.kind == "select":
                decision = None
            executable = decision is not None
            before = after = None
            restored = None
            current_input = previous_input if decision and decision.kind == "error" else previous_output
            if decision and decision.tool:
                try:
                    restored = self.executor.apply(decision.tool, current_input, workspace, index)
                except (RuntimeError, OSError):
                    executable = False
                if executable:
                    before = self.quality_fn(reference, current_input, ssim_mode=self.ssim_mode)
                    after = self.quality_fn(reference, restored, ssim_mode=self.ssim_mode)
            reward = paper_reward(
                decision, executable=executable, is_reflection=not initial,
                before=before, after=after, weights=self.weights, psnr_norm=self.psnr_norm,
            )
            steps.append(Step(images, query, tokens, decision, reward))
            if decision is None:
                status = "invalid_decision"
                break
            if not executable:
                status = "tool_failed"
                break
            if decision.kind == "success":
                status = "finished"
                break
            previous_input, previous_output = current_input, restored
            previous_tool = decision.tool
        return Trajectory(steps, previous_output, status)
