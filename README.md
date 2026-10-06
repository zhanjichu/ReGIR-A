# ReGIR-A: A Reverse-Degradation-Order Guided Agent with Iterative Reflection for Adaptive Optical Remote Sensing Image Restoration

This repository contains the official implementation of **ReGIR-A**, a reasoning-guided iterative restoration agent for remote sensing image restoration.

> **ReGIR-A: A Reverse-Degradation-Order Guided Agent with Iterative Reflection for Adaptive Optical Remote Sensing Image Restoration**
> *Submitted to IEEE Transactions on Geoscience and Remote Sensing (TGRS)*

The project is currently under review. **The source code, training scripts, pretrained models, and datasets will be released after the paper is officially accepted/published.**

---

## Overview

Remote sensing images often suffer from multiple and coupled degradations, such as haze, blur, noise, low illumination, and adverse atmospheric conditions. Existing restoration methods typically rely on predefined degradation assumptions or fixed restoration pipelines, making them less flexible when facing unknown or mixed degradations.

**ReGIR-A** introduces an agent-based restoration framework that enables the model to:

* identify and reason about 8 kinds of mixed and coupled degradation types of RS image degradations;
* dynamically select appropriate restoration operations;
* perform restoration in an iterative manner;
* reflect on intermediate restoration results;
* determine whether further restoration is necessary.

The overall framework combines **vision-language reasoning, restoration tools, iterative reflection, and adaptive restoration planning** to improve restoration performance under diverse degradation conditions.

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

---

## Environment

### Requirements

We recommend using **Conda** to create an isolated environment.

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/ReGIR-A.git
cd ReGIR-A
```

> Replace `USERNAME` with the GitHub account that hosts this repository.

### 2. Create the Conda environment

```bash
conda create -n regir-a python=3.10 -y
conda activate regir-a
```

### 3. Install the required packages

After the complete source code is released, the required dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The final `requirements.txt` will be provided together with the released code.

---

## Preparation

Before running ReGIR-A, the following components will need to be prepared.

### 1. Pretrained VLM

ReGIR-A uses a vision-language model *Qwen2.5-VL-7B* as the reasoning and decision-making component.

The released version will provide detailed instructions for:

* configuring the model path;
* loading the fine-tuned LoRA weights.

The exact model configuration will be provided upon release.

---

### 2. Restoration Toolbox

ReGIR-A interacts with a set of specialized restoration models/tools.

The toolbox will include restoration models for different degradation types, such as:

* Denoising;
* Deblurring;
* Dehazing;
* Cloud Removal;
* JPEG Removal;
* Inpainting;
* Destriping
* other remote sensing image restoration operations.

The corresponding pretrained weights and setup instructions will be released together with the code.

---

### 3. MD-RS Dataset

The proposed **MD-RS** benchmark will be released for evaluating multi-degradation remote sensing image restoration.

The dataset contains images with diverse degradation conditions and supports evaluation of restoration performance under both individual and coupled degradations.

Detailed instructions for:

* dataset downloading;
* directory organization;
* degradation annotations;
* evaluation protocols

will be provided after the dataset is released.

---

## Usage

Detailed training and inference instructions will be provided after the official code release.

---

## License

The source code and dataset will be released under an appropriate open-source license after publication.

Please check the repository for the final license and usage terms.

---

## Acknowledgements

We thank the authors of the existing remote sensing image restoration and vision-language model projects whose open-source implementations contributed to this work.

The corresponding third-party models and datasets remain subject to their original licenses and terms of use.

---

**Code, pretrained models, training scripts, and the MD-RS dataset are coming soon.**
