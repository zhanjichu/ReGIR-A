"""Qwen2.5-VL model and adapter loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def resolve_dtype(name: str) -> torch.dtype:
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return supported[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch_dtype {name!r}; choose one of {sorted(supported)}.") from exc


def load_processor(model_config: dict[str, Any]):
    processor = AutoProcessor.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    max_pixels = model_config.get("image_max_pixels")
    image_processor = getattr(processor, "image_processor", None)
    if max_pixels is not None and image_processor is not None and hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = int(max_pixels)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor


def load_model(model_config: dict[str, Any], device_map: str | dict[str, int] | None = None):
    kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(model_config.get("torch_dtype", "bfloat16")),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config["name_or_path"],
        **kwargs,
    )
    model.config.use_cache = False
    return model


def load_adapter(model, adapter_path: str | None, *, merge: bool = True):
    if adapter_path is None or not str(adapter_path).strip():
        return model
    from peft import PeftModel

    adapter = str(Path(adapter_path).expanduser()) if Path(adapter_path).exists() else adapter_path
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model.merge_and_unload() if merge else model


def load_inference_model(model_config: dict[str, Any]):
    model = load_model(model_config, device_map="auto")
    for adapter_key in ("sft_adapter", "grpo_adapter"):
        model = load_adapter(model, model_config.get(adapter_key), merge=True)
    model.config.use_cache = True
    model.eval()
    return model, load_processor(model_config)
