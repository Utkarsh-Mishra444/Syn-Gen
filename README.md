[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

# **Generative AI: Learning From Synthetic Data**

A research project that evaluates the effectiveness of synthetic data in improving machine learning classifiers by generating over **90,000 images** for the CIFAR-10 dataset using **StyleGAN-XL** and **Stable Diffusion XL Turbo**. Advanced prompt engineering techniques, including clustering visual attributes and leveraging GEMMA 2B, were used to increase diversity and quality. **ResNet-34** models were trained on varying synthetic-to-real data ratios to uncover insights into the role of synthetic data in real-world machine learning.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Data Usage](#data-usage)
- [References](#references)

---

## Project Overview

This project investigates the use of generative models like **StyleGAN-XL** and **Stable Diffusion** to create synthetic data for machine learning. The generated datasets were used to train **ResNet-34** classifiers on CIFAR-10, with experiments exploring how varying the proportion of synthetic data impacts performance. Findings highlight the potential of synthetic data to improve model generalization and reveal specific patterns in classifier performance.

---

## Key Features

- **Synthetic Data Generation**:
  - **60,000 images from StyleGAN-XL**.
  - **30,000 images from Stable Diffusion**:
    - **Class Label Prompts**: Images generated using CIFAR-10 class names.
    - **Clustered Attributes**: Prompts incorporating diverse attributes from the Visual Genome dataset.
    - **GEMMA Prompts**: Descriptive prompts generated with GEMMA 2B.

- **Classifier Training**:
  - **ResNet-34** models (see `models/resnet34_model.py`) trained on synthetic data alone and in combination with real CIFAR-10 data.
  - Experimented with varying synthetic-to-real data ratios (e.g., 0%, 50%, 100% synthetic).

- **Evaluation and Analysis**:
  - Confusion matrices to analyze class-specific errors.
  - Accuracy vs. synthetic data ratio plots to visualize performance trends.

---

## Methodology

### 1. **Data Generation**
- **StyleGAN-XL**:
  - Generated 60,000 CIFAR-10 images using a pretrained StyleGAN-XL model.
  - Synthetic images labeled according to CIFAR-10 classes.

- **Stable Diffusion XL Turbo**:
  - Generated 30,000 CIFAR-10 images using:
    1. **Class Labels** (e.g., “A photorealistic picture of a cat”).
    2. **Clustered Attributes** (attributes extracted from `data/attribute_processing/` scripts).
    3. **GEMMA Prompts** (detailed prompts from GEMMA 2B).

### 2. **Classifier Training**
- **ResNet-34** models (from `models/resnet34_model.py`) were trained on:
  - Purely synthetic datasets (StyleGAN-XL or Stable Diffusion outputs).
  - Mixed datasets with various synthetic-to-real ratios.

### 3. **Evaluation**
- **Confusion Matrices**: Generated to identify class-wise misclassifications.
- **Accuracy-vs-Ratio Graphs**: Studied how performance changes with different amounts of synthetic data.

---

## Results

### Key Findings
1. **StyleGAN-XL** data aligned more closely with CIFAR-10, yielding better synthetic-only training results.
2. Incorporating synthetic data (e.g., a 50-50 mix of real and synthetic) consistently boosted accuracy.
3. **GEMMA-enhanced** prompts introduced additional diversity but required prompt engineering for best results.

---

## Getting Started

Below is the **correct directory structure** for reference:

```
data/
├── attribute_processing/
│   ├── extract_vg_attributes.py
│   ├── group_attributes_by_class.py
├── json/
│   ├── Object_Attributes_Visual_Genome.json
│   ├── Simple_Wordnet.json
│   ├── Simple_Wordnet_2.json
│   ├── Synset_Matched_Attributes.json

models/
├── resnet18_model.py
├── resnet34_model.py

scripts/
├── generate_cifar10_dataset_with_stylegan_xl.py
├── generate_confusion_matrices.py
├── generate_gemma_prompts.py
├── generate_gemma_prompts_4bit.py
├── generate_images_from_prompts_sdxl-turbo.py
├── train_mixed_datasets.py

README.md
```

### Prerequisites
- Python 3.x
- Common machine learning libraries (e.g., PyTorch, NumPy, etc.)

### Steps

1. **Generate Synthetic Data**  
   - **StyleGAN-XL**:
     ```bash
     python scripts/generate_cifar10_dataset_with_stylegan_xl.py
     ```
   - **Stable Diffusion XL Turbo**:
     ```bash
     python scripts/generate_images_from_prompts_sdxl-turbo.py
     ```
   - (Optional) Use `generate_gemma_prompts.py` / `generate_gemma_prompts_4bit.py` for advanced prompt creation.
   - (Note) Refer scripts for exact usage information

2. **Train Mixed Datasets**  
   - Combine your real CIFAR-10 images with synthetic ones at a desired ratio and run:
     ```bash
     python scripts/train_mixed_datasets.py
     ```
   - Adjust hyperparameters and model selection (e.g., `resnet18_model.py` or `resnet34_model.py`) as needed.

3. **Generate Confusion Matrices**  
   - After training, evaluate your models with:
     ```bash
     python scripts/generate_confusion_matrices.py
     ```
   - This script will produce confusion matrices and basic evaluation metrics
---

## Data Usage

Organize your synthetic data under `data/` as needed, for instance:
```
data/
├── StyleGAN_XL/
│   ├── airplane/
│   ├── automobile/
│   ...
├── SDXL_Turbo/
│   ├── Class_Labels/
│   ├── Clustered_Attributes/
│   ├── GEMMA_Prompts/
```
You can also manage attribute-based prompt generation using:
```
data/attribute_processing/
├── extract_vg_attributes.py
├── group_attributes_by_class.py
```
And store JSON definitions in `data/json/`.

---

## References
1. [Krishna, Ranjay, et al. "Visual genome: Connecting language and vision using crowdsourced dense image annotations." *International Journal of Computer Vision* 123 (2017): 32-73.](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)  
2. [Sauer, Axel, Katja Schwarz, and Andreas Geiger. "Stylegan-xl: Scaling stylegan to large diverse datasets." *ACM SIGGRAPH 2022* conference proceedings. 2022.](https://github.com/autonomousvision/stylegan-xl)  
3. [Sauer, Axel, et al. "Adversarial diffusion distillation." *European Conference on Computer Vision*. Springer, Cham, 2025.](https://stability.ai/research/adversarial-diffusion-distillation)
4. [Team, Gemma, et al. "Gemma: Open models based on gemini research and technology." arXiv preprint arXiv:2403.08295 (2024).](https://arxiv.org/abs/2403.08295)  
