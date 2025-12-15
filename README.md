# Dropout and the Geometry of the Loss Landscape

**A Curvature-Based Study on CIFAR-10 and MNIST**

**Authors:** Noah Silverberg, Eddie Cavallin, Pearl Mallick

## Project Overview

This project investigates the geometric mechanisms of dropout regularization by analyzing the "flatness" of the loss landscape around convergent minima. We train ResNet-110 models on CIFAR-10 and MLPs on MNIST across a range of dropout rates and apply discrete curvature metrics--specifically Diffusion Curvature and Ollivier-Ricci Curvature--to characterize the local geometry.

Our findings suggest that while dropout improves generalization, it primarily does so by reducing the magnitude of local loss fluctuations ("magnitude flatness") rather than altering the intrinsic diffusion geometry of the basin.

## Repository Structure

  * `experiments.py`: The main entry point for training models, sampling curvature, and saving results.
  * `curvature.py`: Contains implementations for loss landscape sampling (on hyperspheres or using filter normalization) and curfature computations (Diffusion and Ollivier-Ricci).
  * `loss_landscape_master/`: Adapted code from https://github.com/tomgoldstein/loss-landscape for loss landscape visualization and model definitions (ResNet, MLP).

## Usage

The entire experimental pipeline—including training, curvature sampling, and evaluation—is encapsulated in the `experiments.py` script.

### Recreating Results

You can run the experiments by importing and calling the `run_experiment` function.

**1. CIFAR-10 (ResNet-110)**
To replicate the primary results on CIFAR-10 as described in the report (dropout rates 0.0–0.4, up to 700 epochs for 0.4):

```python
from experiments import run_experiment

# Runs ResNet-110 on CIFAR-10 (Default settings)
run_experiment(
    dataset='cifar10',
    network='resnet110',
    epochs=700,
    dropout_rates=[0.0, 0.1, 0.2, 0.3, 0.4]
)
```

**2. MNIST (MLP)**
To replicate the MNIST experiments:

```python
from experiments import run_experiment

run_experiment(
    dataset='mnist',
    network='mlpmnist',
    epochs=20,
    dropout_rates=[0.0, 0.1, 0.2, 0.3, 0.4]
)
```

### Outputs

Results (accuracy, loss, curvature metrics) are saved to `results/experiment_results.json`. Checkpoints are saved in `checkpoints/` every 10 epochs and at the end of training.

## Requirements

  * Python 3.12
  * PyTorch
  * TorchVision
  * NumPy
  * SciPy
  * POT
  * mpi4pytorch (for loss landscape visualization)