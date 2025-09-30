# GNNs for Fluid Simulations

**Description:** A machine learning project for computational fluid dynamics (CFD) using Graph Neural Networks. The system learns to simulate fluid flow around objects by training on OpenFOAM-generated datasets and uses finite volume method-inspired GNN architectures. This project was for my Master's thesis at the University of Cambridge.

**Date Completed:** August 2025

---

## Key Features

- **End-to-End CFD ML Pipeline**: Full workflow from mesh generation to OpenFOAM simulation, through model training, to autoregressive rollout simulations.
- **Complete Training Infrastructure**: Distributed GPU training with checkpoint resuming, gradient monitoring, and comprehensive logging (Weights & Biases integration).
- **Modular Design**: Dataset preprocessing is independent of the model architecture, so new datasets can be plugged in and used with the existing models and training pipeline.

---

## Project Structure

```plaintext
src/
├── models/               # Graph neural network architectures
├── datasets/             # Dataset preprocessing methods
├── utils/                # Utilities for logging, normalization, geometry, loss functions, etc.
├── train.py              # Main training entry point
├── rollout.py            # Model evaluation and rollout script
├── preproc.py            # Data preprocessing script
└── sweep.py              # Hyperparameter sweep script

scripts/                  # Shell and SLURM scripts for training, rollout, etc.
config/                   # JSON configuration files for training, rollout, preprocessing
generate/                 # OpenFOAM case generation, mesh creation, simulation running
analysis/ <not in repo>   # Post-processing, visualization, and performance analysis scripts
_data/    <not in repo>   # Generated datasets, trained models, and analysis outputs
tests/    <not in repo>   # Tests for santiy checks on preprocessing and model design.

report.pdf                # Report submitted for master's thesis (poor image quality is due to compression)
```

---

## Workflow & Key Scripts

### 1. Data Generation Pipeline
```bash
# Generate meshes, run OpenFOAM simulations, convert to HDF5
./scripts/generate.sh <machine> mesh    # Generate computational meshes
./scripts/generate.sh <machine> sim     # Run OpenFOAM fluid simulations
./scripts/generate.sh <machine> conv    # Convert results to HDF5 format
```
Note that the machines are defined in `src/utils/config.py`

### 2. Machine Learning Pipeline
```bash
# Preprocess data (prepare features)
./scripts/preproc.sh <config_name>

# Train neural network models
./scripts/train.sh <config_name> [--debug]

# Evaluate trained models (rollout simulations)
./scripts/rollout.sh <config_name>

# Hyperparameter sweeps
./scripts/sweep.sh <sweep_config>
```

### 3. Configuration System
- **`config/train.json`** - Training parameters (model, optimizer, learning rate, etc.)
- **`config/rollout.json`** - Evaluation settings (which model to load, timesteps to simulate)
- **`config/preproc.json`** - Preprocessing options (datasets, statistics computation)

### Weights & Biases Integration

The project uses [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking, logging, and visualization. To use wandb, you need to export your API key as an environment variable before running any training or evaluation scripts:

```bash
export WANDB_API_KEY=your_wandb_api_key_here
```

Additionally, lines 163 and 175 will need to be set to the desired wandb project/"entity" in `src/utils/logging.py`.

---

## Model Architectures

All models written as PyG modules. The project implements several neural network architectures for fluid dynamics, based on the work of [Pfaff et al.](https://doi.org/10.48550/arXiv.2010.03409) on [MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) and, more recently, by [Li et al.](https://doi.org/10.48550/arXiv.2309.10050) on [Finite-Volume-Graph-Network](https://github.com/Litianyu141/Finite-Volume-Graph-Network).
- **FVGN** (`models/Fvgn.py`) - Finite Volume Graph Network (FVGN)
- **MGN** (`models/Mgn.py`) - Hybrid FVGN-MeshGraphNet approach (FVGN-Direct in report)
- **Flux** (`models/Flux.py`) - Utilises face flux directly (FVGN-Flux in report)
- **Conservative** (`models/Conservative.py`) - FVM-inspired message passing
- **VertPot** (`models/VertPot.py`) - Uses potential at vertex to contruct conservative face fluxes
- **StreamFunc** (`models/StreamFunc.py`) - Constructs velocity from streamfunction

Several variants of each architecture may exist, labelled A through to Z. 
