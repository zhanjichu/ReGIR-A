# Training data

Training annotations and datasets are not included in this source release. Availability will be announced in the repository README. Replace all example paths with your own paths and provide real annotations.

## Stage I: SFT

Set `data.train_file` in `configs/sft.yaml` to a JSON array or JSONL file. Each record contains `messages` (system/user/assistant roles) and an `images` list. Use the system prompt and queries from `agent/prompts.py`. Image paths may be absolute or relative to the annotation file.

For Q1, use one image and `QUERY_SELECT`; an example target is `<code>de_haze(image)</code>`. For Q2, supply the pre-tool and post-tool images in that order, and use `QUERY_REFLECT` formatted with the actual previous tool. The three possible targets are `Restored Successfully.`, `Error previous tool: <code>de_haze(image1)</code>`, and `Continue to restore: <code>de_haze(image2)</code>`. Tool names and targets must reflect your actual expert annotations.

## Stage II: GRPO

Set `data.train_file` in `configs/grpo.yaml` to a JSON array or JSONL file. Each record contains:

```json
[
  {
    "input_image": "/path/to/your/data/degraded/sample.png",
    "reference_image": "/path/to/your/data/reference/sample.png"
  }
]
```

Reference images must correspond to the degraded inputs and have matching dimensions. They are used for training rewards. Relative paths are resolved from the annotation file directory. Optional `metadata` must be a mapping; the current reward implementation does not read expert branch labels from it.
