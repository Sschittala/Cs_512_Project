# Hierarchical Self-Prediction as a Regularizer for Deep Neural Networks

Implementation of a hierarchical self-prediction regularization method for improving robustness in deep neural networks using ResNet-18 on CIFAR-10 and CIFAR-10-C.

---

## Overview

Deep neural networks perform well on standard image classification tasks but often fail under distributional shift and corrupted input conditions.

This project investigates whether enforcing internal consistency between intermediate layers can improve robustness to image corruptions.

The approach augments a standard ResNet-18 architecture with auxiliary prediction heads that attempt to predict deeper feature representations from earlier layers. During training, the model is penalized when predicted and actual feature activations disagree.

The goal is to encourage more stable internal feature representations under corruption.

---

## Model Architecture

### Baseline Model

- Standard ResNet-18
- Modified for CIFAR-10 image size (32x32)
- Trained using standard cross-entropy loss

### Experimental Model

The experimental architecture adds auxiliary prediction heads between residual stages.

Each prediction head:
- Takes an intermediate feature tensor
- Predicts the next residual stage activation
- Uses MSE loss between predicted and actual activations

Total loss:

```math
L_{total} = L_{CE} + \lambda_{12}L_{12} + \lambda_{23}L_{23} + \lambda_{34}L_{34}
```

where:
- `L_CE` = cross-entropy classification loss
- `Lij` = self-prediction consistency loss
- `λij` = weighting hyperparameters

The auxiliary heads are only used during training and are disabled during inference.

---

## Datasets

### CIFAR-10
Used for:
- Training
- Validation
- Clean accuracy evaluation

### CIFAR-10-C
Used for:
- Robustness evaluation
- Corruption testing across severity levels 1–5

Corruption categories include:
- Gaussian noise
- Blur
- Fog
- Snow
- JPEG compression
- Pixelation
- Contrast changes
- Elastic transformations

---

## Training Details

- Optimizer: SGD
- Learning Rate: 0.1
- Momentum: 0.9
- Weight Decay: 5e-4
- Epochs: 25
- LR Scheduler: MultiStepLR
- Framework: PyTorch

Automatic mixed precision (AMP) was used when CUDA was available.

---

## Hyperparameter Search

The project explored both:
- Shared λ sweeps
- Layer-wise λ sweeps

Final robustness-oriented configuration:

```python
lambda12 = 100
lambda23 = 200
lambda34 = 50
```

---

## Results

### Baseline ResNet-18

| Metric | Result |
|---|---|
| Clean CIFAR-10 Accuracy | 91.93% |
| CIFAR-10-C Avg Accuracy | 71.22% |

### Hierarchical Self-Prediction Model

| Metric | Result |
|---|---|
| Clean CIFAR-10 Accuracy | 85.82% |
| CIFAR-10-C Avg Accuracy | 72.92% |

The experimental model improved robustness under moderate and severe corruptions, particularly at higher severity levels.

Largest improvement:
- Severity 5 accuracy increased from 54.38% to 60.82%

---

## Repository Structure

```text
.
├── base_model
│   ├── accuracy_x_severity-levels-general.png
│   ├── accuracy_x_severity-levels-per-type.png
│   ├── base_model.ipynb
│   └── cifar10c_results.csv
│
├── exp_model
│   ├── accuracy_x_severity-levels-general.png
│   ├── accuracy_x_severity-levels-per-type.png
│   ├── compare.png
│   ├── final_result_1.png
│   ├── experimental_model.ipynb
│   └── cifar10c_results.csv
│
├── data
│   └── CIFAR-10 dataset files
│
└── README.md
```

---

## File Descriptions

### `base_model/base_model.ipynb`
Notebook containing:
- Baseline ResNet-18 implementation
- CIFAR-10 training
- CIFAR-10-C robustness evaluation
- Robustness visualization generation

### `exp_model/experimental_model.ipynb`
Notebook containing:
- Hierarchical self-prediction architecture
- Auxiliary prediction heads
- Layer-wise regularization losses
- Hyperparameter sweeps
- Robustness evaluation

### Visualization Outputs

Generated plots include:
- Average accuracy across corruption severities
- Per-corruption robustness curves
- Baseline vs experimental comparisons
- Final severity comparison plots

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sschittala/Cs_512_Project.git
cd Cs_512_Project
```
---

## Required Python Packages

```text
torch
torchvision
numpy
pandas
matplotlib
tqdm
```

---

## Running the Project

### Baseline Model

Open and run:

```text
base_model/base_model.ipynb
```

### Experimental Model

Open and run:

```text
exp_model/experimental_model.ipynb
```

---

## Key Findings

- Hierarchical self-prediction improved corruption robustness
- Stronger regularization generally improved robustness
- Increased robustness came at the cost of lower clean accuracy
- Early and intermediate layer consistency contributed most to robustness improvements

The method acts as a robustness-oriented structural regularizer rather than a universally better replacement for the baseline model.

---

## Future Work

Possible extensions include:
- Testing on larger datasets
- Combining with adversarial training
- Adaptive self-prediction weighting
- Scaling to deeper architectures
- Applying to transformer-based vision models

---

## Authors

- Tyler Strach
- Sai Chittala
- Miles Woodcock-Girard
