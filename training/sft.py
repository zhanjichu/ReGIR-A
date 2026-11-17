"""LoRA supervised fine-tuning for Qwen2.5-VL."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent.config import load_config, require_mapping
from agent.data import load_image_dataset, normalize_messages


class MultimodalSFTCollator:
    """Build Qwen-VL batches and compute loss only on assistant completions."""

    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def _render(self, row: dict[str, Any], include_answer: bool) -> tuple[str, list[Any]]:
        messages = normalize_messages(row["messages"])
        images = row["images"]
        if not images:
            raise ValueError("SFT examples must contain at least one image.")
        if not any(message["role"] == "system" for message in messages):
            from agent.prompts import SYSTEM_PROMPT

            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        assistant_indices = [i for i, item in enumerate(messages) if item["role"] == "assistant"]
        if not assistant_indices:
            raise ValueError("SFT example has no assistant target.")
        final_assistant = assistant_indices[-1]
        selected = messages if include_answer else messages[:final_assistant]
        if include_answer and final_assistant != len(messages) - 1:
            selected = messages[: final_assistant + 1]
        selected = deepcopy(selected)
        for message in selected:
            message["content"] = [{"type": "text", "text": message["content"]}]
        first_user = next(message for message in selected if message["role"] == "user")
        first_user["content"] = [{"type": "image"} for _ in images] + first_user["content"]
        text = self.processor.apply_chat_template(
            selected,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )
        return text, images

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts = []
        image_batches = []
        prompt_lengths = []

        for row in features:
            prompt_text, images = self._render(row, include_answer=False)
            full_text, _ = self._render(row, include_answer=True)
            prompt_input = self.processor(
                text=[prompt_text],
                images=[images],
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            )
            prompt_lengths.append(int(prompt_input["attention_mask"][0].sum().item()))
            full_texts.append(full_text)
            image_batches.append(images)

        batch = self.processor(
            text=full_texts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        labels = batch["input_ids"].clone()
        for row_index, prompt_length in enumerate(prompt_lengths):
            length = int(batch["attention_mask"][row_index].sum().item())
            if length > self.max_length:
                raise ValueError("Sample exceeds cutoff_len; resize images or increase cutoff_len.")
            if prompt_length >= length:
                raise ValueError(
                    "The configured cutoff_len leaves no assistant tokens to train on. "
                    "Increase cutoff_len or shorten the prompt."
                )
            token_positions = batch["attention_mask"][row_index].nonzero().flatten()
            labels[row_index, token_positions[:prompt_length]] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def _load_datasets(data_config: dict[str, Any]):
    train = load_image_dataset(
        data_config["train_file"],
        max_samples=data_config.get("max_samples"),
        path_mappings=data_config.get("image_path_mappings"),
    )
    validation_file = data_config.get("validation_file")
    if validation_file:
        validation = load_image_dataset(
            validation_file, path_mappings=data_config.get("image_path_mappings"),
        )
        return train, validation

    validation_split = float(data_config.get("validation_split", 0.0))
    if validation_split <= 0:
        return train, None
    if len(train) < 2:
        raise ValueError("At least two records are required when validation_split is enabled.")
    split = train.train_test_split(test_size=validation_split, seed=42)
    return split["train"], split["test"]


def train(config: dict[str, Any]) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed
    from agent.modeling import load_model, load_processor

    model_config = require_mapping(config, "model")
    data_config = require_mapping(config, "data")
    lora_config = require_mapping(config, "lora")
    train_config = require_mapping(config, "training")

    set_seed(int(train_config.get("seed", 42)))
    train_dataset, eval_dataset = _load_datasets(data_config)
    model = load_model(model_config)
    if train_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_config.get("rank", 16)),
            lora_alpha=int(lora_config.get("alpha", 32)),
            lora_dropout=float(lora_config.get("dropout", 0.05)),
            target_modules=lora_config.get("target_modules", "all-linear"),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    processor = load_processor(model_config)
    arguments = TrainingArguments(
        output_dir=train_config["output_dir"],
        logging_dir=train_config.get("logging_dir"),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_config.get("learning_rate", 1e-4)),
        num_train_epochs=float(train_config.get("num_train_epochs", 1)),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.0)),
        bf16=bool(train_config.get("bf16", False)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(train_config.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(train_config.get("save_steps", 500)),
        save_total_limit=train_config.get("save_total_limit"),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(train_config.get("eval_steps", 500)) if eval_dataset is not None else None,
        report_to=train_config.get("report_to", "none"),
        dataloader_num_workers=int(train_config.get("dataloader_num_workers", 0)),
        ddp_timeout=int(train_config.get("ddp_timeout", 1800)),
        remove_unused_columns=False,
        overwrite_output_dir=False,
        seed=int(train_config.get("seed", 42)),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultimodalSFTCollator(
            processor,
            max_length=int(train_config.get("cutoff_len", 8192)),
        ),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=train_config.get("resume_from_checkpoint"))
    trainer.save_model(train_config["output_dir"])
    processor.save_pretrained(train_config["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    train(config)


if __name__ == "__main__":
    main()
