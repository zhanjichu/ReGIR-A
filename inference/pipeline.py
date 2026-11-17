"""Iterative image restoration inference and optional reference evaluation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agent.config import load_config, require_mapping
from agent.decisions import Decision, parse_decision
from agent.metrics import image_metrics
from agent.prompts import QUERY_REFLECT, QUERY_SELECT, SYSTEM_PROMPT
from agent.toolbox import ToolExecutor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def predict(
    images: list[Image.Image],
    model,
    processor,
    query: str,
    max_new_tokens: int = 512,
) -> str:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": query})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
    ).to(model.device)
    import torch

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        output_ids[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _apply_decision(
    decision: Decision,
    previous_input: Path,
    previous_output: Path,
    executor: ToolExecutor,
    workspace: Path,
    step: int,
) -> tuple[Path, Path]:
    if decision.tool is None:
        raise ValueError("A restoration decision must include a tool name.")
    if decision.kind == "error" and decision.image_ref == "image1":
        source = previous_input
        next_reference = previous_input
    elif decision.kind == "continue" and decision.image_ref == "image2":
        source = previous_output
        next_reference = previous_output
    else:
        raise ValueError(f"Unsupported image reference in decision: {decision.image_ref}")
    restored = executor.apply(decision.tool, source, workspace, step)
    return next_reference, restored


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def restore_one(
    source: Path,
    destination: Path,
    model,
    processor,
    executor: ToolExecutor,
    workspace: Path,
    max_reflections: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "input": source.name,
        "actions": [],
        "status": "unchanged",
    }
    first_output = predict(
        [_open_rgb(source)],
        model,
        processor,
        QUERY_SELECT,
        max_new_tokens=max_new_tokens,
    )
    first_decision = parse_decision(first_output)
    record["actions"].append({"stage": "select", "response": first_output})
    if first_decision is None or first_decision.kind != "select":
        shutil.copy2(source, destination)
        record["status"] = "invalid_initial_decision"
        record["output"] = destination.name
        record["tool_calls"] = 0
        return record

    previous_input = source
    previous_output = executor.apply(
        first_decision.tool,
        source,
        workspace,
        step=0,
    )
    previous_tool = first_decision.tool
    record["actions"][-1]["tool"] = previous_tool
    record["status"] = "max_reflections_reached"

    for reflection_index in range(max_reflections):
        response = predict(
            [_open_rgb(previous_input), _open_rgb(previous_output)],
            model,
            processor,
            QUERY_REFLECT.format(previous_tool=previous_tool),
            max_new_tokens=max_new_tokens,
        )
        decision = parse_decision(response)
        record["actions"].append(
            {"stage": "reflect", "response": response, "tool": None if decision is None else decision.tool}
        )
        if decision is None:
            record["status"] = "invalid_reflection_decision"
            break
        if decision.kind == "success":
            record["status"] = "restored_successfully"
            break
        if decision.kind not in {"error", "continue"}:
            record["status"] = "invalid_reflection_decision"
            break

        next_reference, next_output = _apply_decision(
            decision,
            previous_input,
            previous_output,
            executor,
            workspace,
            step=reflection_index + 1,
        )
        previous_input = next_reference
        previous_output = next_output
        previous_tool = decision.tool

    shutil.copy2(previous_output, destination)
    record["tool_calls"] = sum(action.get("tool") is not None for action in record["actions"])
    record["selected_step_file"] = str(previous_output.relative_to(workspace))
    record["output"] = destination.name
    return record


def evaluate_directories(
    reference_dir: str | Path, restored_dir: str | Path, ssim_mode: str = "gray",
) -> dict[str, Any]:
    """Compute paired PSNR/SSIM after inference; references never guide decisions."""
    references = Path(reference_dir)
    restored = Path(restored_dir)
    psnr_values = []
    ssim_values = []
    skipped = []
    for source in sorted(path for path in references.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        candidate = restored / source.name
        if not candidate.is_file():
            skipped.append({"image": source.name, "reason": "missing restoration"})
            continue
        try:
            scores = image_metrics(source, candidate, ssim_mode=ssim_mode)
        except ValueError as exc:
            skipped.append({"image": source.name, "reason": str(exc)})
            continue
        psnr_values.append(scores["psnr"])
        ssim_values.append(scores["ssim"])
    metrics: dict[str, Any] = {
        "paired_images": len(psnr_values), "ssim_mode": ssim_mode, "skipped": skipped,
    }
    if psnr_values:
        metrics["mean_psnr"] = float(np.mean(psnr_values))
    if ssim_values:
        metrics["mean_ssim"] = float(np.mean(ssim_values))
    return metrics


def run_inference(
    config: dict[str, Any],
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    tools_config: str | Path | None = None,
    max_reflections: int | None = None,
    reference_dir: str | Path | None = None,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    model_config = require_mapping(config, "model")
    inference_config = require_mapping(config, "inference")
    source_dir = Path(input_dir).resolve()
    target_dir = Path(output_dir).resolve()
    if source_dir == target_dir or (reference_dir and target_dir == Path(reference_dir).resolve()):
        raise ValueError("Output directory must differ from input and reference directories.")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {source_dir}")
    if reference_dir is not None and not Path(reference_dir).is_dir():
        raise NotADirectoryError(f"Reference directory not found: {reference_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found in {source_dir}")
    existing = [p.name for p in images if (target_dir / p.name).exists()]
    if existing:
        raise FileExistsError(f"Use a fresh output directory; files already exist: {existing[:5]}")

    command_config = tools_config or inference_config["tool_config"]
    executor = ToolExecutor(command_config)
    executor.validate()
    if max_cycles is not None and max_reflections is not None:
        raise ValueError("Choose max_cycles or max_reflections, not both.")
    configured_cycles = max_cycles if max_cycles is not None else inference_config.get("max_cycles")
    reflections = int(
        (int(configured_cycles) - 1 if configured_cycles is not None else
         inference_config.get("max_reflections", 5)) if max_reflections is None else max_reflections
    )
    if reflections < 0:
        raise ValueError("max_reflections must be zero or greater.")
    max_new_tokens = int(inference_config.get("max_new_tokens", 512))
    from agent.modeling import load_inference_model

    model, processor = load_inference_model(model_config)
    records: dict[str, Any] = {}
    started_at = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="regir-a-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for index, source in enumerate(images):
            image_workspace = temporary_root / f"image_{index:06d}"
            image_workspace.mkdir()
            records[source.name] = restore_one(
                source,
                target_dir / source.name,
                model,
                processor,
                executor,
                image_workspace,
                reflections,
                max_new_tokens,
            )

    summary: dict[str, Any] = {
        "image_count": len(images),
        "elapsed_seconds": time.perf_counter() - started_at,
        "max_tool_calls": reflections + 1,
        "records": records,
    }
    if reference_dir is not None:
        summary["metrics"] = evaluate_directories(
            reference_dir, target_dir, ssim_mode=inference_config.get("ssim_mode", "gray"),
        )
    with (target_dir / "run_records.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--input-dir", help="Override inference.input_dir in YAML.")
    parser.add_argument("--output-dir", help="Override inference.output_dir in YAML.")
    parser.add_argument("--tools-config")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-reflections", type=int)
    limits.add_argument("--max-cycles", type=int)
    parser.add_argument("--reference-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = require_mapping(config, "inference")
    input_dir = args.input_dir or settings.get("input_dir")
    output_dir = args.output_dir or settings.get("output_dir")
    if not input_dir or not output_dir:
        parser.error("Set inference.input_dir/output_dir in YAML or pass directory arguments.")
    run_inference(
        config,
        input_dir,
        output_dir,
        tools_config=args.tools_config,
        max_reflections=args.max_reflections,
        reference_dir=args.reference_dir or settings.get("reference_dir"),
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
