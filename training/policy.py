from __future__ import annotations

from pathlib import Path

from PIL import Image

from agent.prompts import SYSTEM_PROMPT


class RolloutPolicy:
    def __init__(self, model, processor, *, temperature=0.8, max_new_tokens=512):
        if temperature <= 0 or max_new_tokens < 1:
            raise ValueError("Invalid generation configuration.")
        self.model, self.processor = model, processor
        self.temperature, self.max_new_tokens = temperature, max_new_tokens

    def inputs(self, paths: tuple[Path, ...], query: str):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": query}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self.processor(
            text=[prompt], images=images, return_tensors="pt",
        ).to(self.model.device)

    def sample(self, paths: tuple[Path, ...], query: str):
        import torch

        self.model.eval()
        inputs = self.inputs(paths, query)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=self.temperature, top_p=1.0, top_k=0,
                repetition_penalty=1.0, use_cache=True,
            )
        tokens = output[0, inputs["input_ids"].shape[1]:].detach().clone().cpu()
        text = self.processor.batch_decode(
            [tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        return text, tokens

    def token_logps(self, step):
        import torch
        import torch.nn.functional as F

        inputs = dict(self.inputs(step.images, step.query))
        tokens = step.tokens.to(self.model.device).unsqueeze(0)
        prompt_length = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], tokens], dim=1)
        inputs["attention_mask"] = torch.cat([
            inputs["attention_mask"], torch.ones_like(tokens),
        ], dim=1)
        # Keep pixel_values and image_grid_thw in the forward pass.
        logits = self.model(**inputs, use_cache=False).logits
        logits = logits[:, prompt_length - 1:-1, :].float() / self.temperature
        return F.log_softmax(logits, dim=-1).gather(2, tokens.unsqueeze(-1)).squeeze(0).squeeze(-1)
