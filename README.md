# ReGIR-A

**ReGIR-A: A Reverse-Degradation-Order Guided Agent with Iterative Reflection for Adaptive Optical Remote Sensing Image Restoration**

Jichu Zhan, Zhentao Zou, Yue Zhou, and Xue Jiang

**Accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026.**

DOI: [10.1109/TGRS.2026.3753988](https://doi.org/10.1109/TGRS.2026.3753988)

## Environment

Python 3.11 or later and a Linux/CUDA environment are recommended. Run the commands below from the repository root.

```bash
git clone https://github.com/zhanjichu/ReGIR-A.git
cd ReGIR-A
conda create -n regir-a python=3.11 -y
conda activate regir-a
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Install a PyTorch build compatible with your CUDA drivers.

## Preparation

The backbone is [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). Download the base model from **Hugging Face** and prepare the Stage I SFT and Stage II GRPO adapters. Enter their locations directly in the corresponding YAML configuration files. Paths beginning with `/path/to/your/` are examples to replace with your own paths. See [data/README.md](data/README.md) for more details about dataset.

### Restoration tools

Install the following tools independently and adapt their inference scripts into callable tools for the agent. The action names must match `agent/prompts.py`. 

| Task | Tool / upstream repository | Agent action |
| --- | --- | --- |
| Gaussian denoising, motion and defocus deblurring | [Restormer](https://github.com/swz30/Restormer) | `de_gaussian`, `de_motion_blur`, `de_forced_blur` |
| JPEG artifact removal | [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | `de_jpeg` |
| Destriping | [ASCNet](https://github.com/xdFai/ASCNet) | `de_stripe` |
| Inpainting | [LaMa](https://github.com/advimman/lama) | `inpainting` |
| Dehazing | [DehazeFormer](https://github.com/IDKiro/DehazeFormer) | `de_haze` |
| Cloud removal | [ACA-CRNet](https://github.com/huangwenwenlili/ACA-CRNet)  | `de_cloud` |

Use each tool's publicly released pretrained weights, choosing the checkpoint for the relevant task. Keep these restoration models frozen: ReGIR-A does not retrain or fine-tune them. The two training stages below train the VLM agent adapters. Follow the upstream repositories for weight downloads and tool-specific dependencies. If needing to install more tools, please modify `agent/prompts.py` accordingly.

Adapt each tool's inference script to load its public pretrained weights and accept input/output directories, preserving image dimensions and filenames. Enter the wrapper commands and paths in [configs/tools.example.yaml](configs/tools.example.yaml), replacing the examples and configuring all eight actions. `{input_dir}` and `{output_dir}` are filled automatically.

## Evaluation

Edit [configs/inference.yaml](configs/inference.yaml) directly. Fill in the base model, both adapter checkpoints, the tool YAML, and your input/output image directories. For paired PSNR/SSIM evaluation, set `inference.reference_dir` to a directory containing reference images with matching filenames. Set it to `null` for restoration without paired evaluation. Then run:

```bash
python -m scripts.infer --config configs/inference.yaml
```

Use a fresh output directory. Restored images and `run_records.json` are written there. Directory arguments such as `--input-dir`, `--output-dir`, and `--reference-dir` can override the YAML values.

The agent returns the latest restored image. Reference images are used only for PSNR/SSIM evaluation after restoration. `max_cycles: 6` permits one initial tool call and up to five further calls. 

## Two-stage training

Fill in the model, training-data, and output paths directly in the YAML files below. The example paths are placeholders. See [data/README.md](data/README.md) for the expected annotation fields. Do not commit your private datasets, credentials, or machine-specific configurations.

**Stage I: supervised fine-tuning**

Edit [configs/sft.yaml](configs/sft.yaml) directly: set `model.name_or_path`, `data.train_file`, and `training.output_dir`; optionally set `data.validation_file` and `training.logging_dir`. Each training record needs `messages` and `images`. Effective global batch size also depends on GPU count and gradient accumulation.

```bash
python -m scripts.train_sft --config configs/sft.yaml
```

**Stage II: GRPO**

Edit [configs/grpo.yaml](configs/grpo.yaml): set the base model, `model.sft_adapter` to the Stage I checkpoint, `data.train_file`, `environment.tool_config`, and `training.output_dir`. Each record needs `input_image` and `reference_image`; the restoration toolbox must be configured before training. For an already merged SFT model, set `model.name_or_path` to that model, `model.sft_adapter: null`, and `model.sft_merged: true`. 

```bash
python -m scripts.train_grpo --config configs/grpo.yaml
```

Full training and evaluation require the intended checkpoints, annotations, tool wrappers, and GPU environment.

## Citation

```bibtex
@article{zhan2026regira,
  title   = {{ReGIR-A}: A Reverse-Degradation-Order Guided Agent with Iterative Reflection for Adaptive Optical Remote Sensing Image Restoration},
  author  = {Zhan, Jichu and Zou, Zhentao and Zhou, Yue and Jiang, Xue},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  doi     = {10.1109/TGRS.2026.3753988}
}
```

## License

Code is released under the [MIT License](LICENSE). Third-party models, weights, and imagery retain their respective licenses.
