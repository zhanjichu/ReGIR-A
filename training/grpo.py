"""Trajectory GRPO with environment rewards."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path

from agent.config import load_config, require_mapping
from agent.data import _image_path, read_records
from agent.toolbox import ToolExecutor
from training.policy import RolloutPolicy
from training.rollout import RolloutRunner


def group_advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    if len(rewards) < 2 or not all(math.isfinite(reward) for reward in rewards):
        raise ValueError("A GRPO group requires at least two finite rewards.")
    mean = sum(rewards) / len(rewards)
    deviation = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
    return [(reward - mean) / (deviation + epsilon) for reward in rewards]


def token_loss(new, old, reference, advantage, clip_range, beta):
    ratio = (new - old).exp()
    objective = (ratio * advantage).minimum(ratio.clamp(1 - clip_range, 1 + clip_range) * advantage)
    delta = reference - new
    kl = delta.exp() - delta - 1
    return -objective + beta * kl


def load_samples(config):
    source = config.get("train_file")
    if not source:
        raise ValueError("Set data.train_file in a private configuration.")
    root = Path(source).resolve().parent
    samples = read_records(source)
    for sample in samples:
        for key in ("input_image", "reference_image"):
            sample[key] = _image_path(sample[key], root, config.get("image_path_mappings"))
            if not Path(sample[key]).is_file():
                raise FileNotFoundError("A configured training image is unavailable.")
        if not isinstance(sample.get("metadata", {}), dict):
            raise ValueError("Sample metadata must be a mapping.")
    limit = config.get("max_samples")
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("max_samples must be positive.")
        samples = samples[:int(limit)]
    return samples


def train(config):
    model_config = require_mapping(config, "model")
    data_config = require_mapping(config, "data")
    settings = require_mapping(config, "training")
    environment = require_mapping(config, "environment")
    lora = require_mapping(config, "lora")
    group_size = int(settings.get("num_generations", 8))
    max_cycles = int(environment.get("max_cycles", 6))
    if group_size < 2 or max_cycles < 1:
        raise ValueError("Invalid group size or cycle limit.")
    if not model_config.get("sft_adapter") and not model_config.get("sft_merged", False):
        raise ValueError("Stage II requires an SFT adapter or an explicitly merged SFT model.")
    samples = load_samples(data_config)
    executor = ToolExecutor(environment["tool_config"])
    executor.validate()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed
    from agent.modeling import load_adapter, load_model, load_processor

    seed = int(settings.get("seed", 42))
    set_seed(seed)
    model = load_model(model_config, device_map=model_config.get("device_map"))
    model = load_adapter(model, model_config.get("sft_adapter"), merge=True)
    resume = settings.get("resume_from_checkpoint")
    if resume:
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(lora.get("rank", 16)), lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=0.0, target_modules=lora.get("target_modules", "all-linear"),
            bias="none", task_type="CAUSAL_LM",
        ))
    if not model_config.get("device_map"):
        model.to(settings.get("device", "cuda"))
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if settings.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    processor = load_processor(model_config)
    policy = RolloutPolicy(
        model, processor, temperature=float(settings.get("temperature", 0.8)),
        max_new_tokens=int(settings.get("max_completion_length", 512)),
    )
    runner = RolloutRunner(
        policy, executor, max_cycles=max_cycles,
        weights=environment.get("reward_weights", [0.15, 0.15, 0.70]),
        psnr_norm=float(environment.get("psnr_norm", 30)),
        ssim_mode=environment.get("ssim_mode", "gray"),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings.get("learning_rate", 5e-7)),
    )
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    epoch_start = position_start = global_step = 0
    if resume:
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        epoch_start, position_start, global_step = state["epoch"], state["position"], state["step"]
        torch.set_rng_state(state["rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def save(epoch, position):
        checkpoint = output / f"checkpoint-{global_step}"
        checkpoint.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint)
        processor.save_pretrained(checkpoint)
        torch.save({
            "optimizer": optimizer.state_dict(), "epoch": epoch, "position": position,
            "step": global_step, "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }, checkpoint / "training_state.pt")

    epochs = int(settings.get("num_train_epochs", 3))
    beta = float(settings.get("kl_beta", 0.05))
    clip_range = float(settings.get("clip_range", 0.2))
    if not 0 < clip_range < 1 or beta < 0:
        raise ValueError("Invalid clipping or KL coefficient.")
    epoch = epoch_start
    position = position_start
    for epoch in range(epoch_start, epochs):
        order = list(range(len(samples)))
        random.Random(seed + epoch).shuffle(order)
        for position in range(position_start if epoch == epoch_start else 0, len(order)):
            with tempfile.TemporaryDirectory(prefix="regira-grpo-") as directory:
                trajectories = []
                for member in range(group_size):
                    workspace = Path(directory) / str(member)
                    workspace.mkdir()
                    trajectories.append(runner.rollout(samples[order[position]], workspace))
                advantages = group_advantages([trajectory.reward for trajectory in trajectories])
                model.eval()
                with torch.no_grad():
                    for trajectory in trajectories:
                        for step in trajectory.steps:
                            if len(step.tokens):
                                step.old_logps = policy.token_logps(step).detach().cpu()
                                with model.disable_adapter():
                                    step.reference_logps = policy.token_logps(step).detach().cpu()
                optimizer.zero_grad(set_to_none=True)
                model.train()
                total_loss = 0.0
                trained_tokens = 0
                for trajectory, advantage in zip(trajectories, advantages):
                    length = sum(len(step.tokens) for step in trajectory.steps)
                    if not length:
                        continue
                    for step in trajectory.steps:
                        if not len(step.tokens):
                            continue
                        new = policy.token_logps(step)
                        loss = token_loss(
                            new, step.old_logps.to(new.device), step.reference_logps.to(new.device),
                            advantage, clip_range, beta,
                        ).sum() / (group_size * length)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite GRPO loss.")
                        loss.backward()
                        total_loss += loss.detach().item()
                        trained_tokens += len(step.tokens)
                if trained_tokens:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(settings.get("max_grad_norm", 1.0)),
                    )
                    optimizer.step()
                global_step += 1
                metrics = {
                    "step": global_step, "loss": total_loss,
                    "reward": sum(trajectory.reward for trajectory in trajectories) / group_size,
                    "trained_tokens": trained_tokens,
                }
                print(json.dumps(metrics))
                with (output / "training_metrics.jsonl").open("a", encoding="utf8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
            interval = int(settings.get("save_steps", 50))
            if interval > 0 and global_step % interval == 0:
                save(epoch, position + 1)
            maximum = settings.get("max_steps")
            if maximum is not None and global_step >= int(maximum):
                save(epoch, position + 1)
                return
        position_start = 0
    save(epochs, 0)
    model.save_pretrained(output)
    processor.save_pretrained(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
