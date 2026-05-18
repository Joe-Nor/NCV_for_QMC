# Data Format Specification

## Overview

This document describes the data formats used in the RSSE control variates pipeline, including MCMC output, training data, and model checkpoints.

## MCMC Output Format

### Binary Format (.bin)

The Fortran MCMC sampler outputs data in binary format with the following structure:

#### File Header
```
- n_samples (integer*4): Number of configurations
- n_sites (integer*4): Number of lattice sites
- beta (real*8): Inverse temperature
- lattice_type (integer*4): Lattice geometry identifier
```

#### Per-Configuration Data
For each configuration, the following fields are stored sequentially:

```
- nh (integer*4): Number of operators in the string
- operators (integer*4 array): Operator type indices [1..nh]
- bonds (integer*4 array): Bond indices [1..nh]
- weight (real*8): Configuration weight w(C)
- sign (real*8): Sign of the weight (+1 or -1)
- energy (real*8): Local energy estimator
```

### Text Format (.txt)

For debugging and inspection, data can also be saved in text format:

```
# Header
n_samples = 100000
n_sites = 4
beta = 8.0
lattice = 2x2

# Configuration 1
nh = 12
operators: 1 2 1 2 1 2 1 2 1 2 1 2
bonds: 0 1 2 3 0 1 2 3 0 1 2 3
weight: 0.00123456
sign: 1.0
energy: -0.567890

# Configuration 2
...
```

## Training Data Format

### Preprocessed Data (.npz)

After preprocessing, training data is stored in NumPy compressed format:

```python
{
    'operator_strings': np.ndarray,  # Shape: (n_samples, max_nh), dtype: int32
    'nh_values': np.ndarray,         # Shape: (n_samples,), dtype: int32
    'weights': np.ndarray,           # Shape: (n_samples,), dtype: float64
    'signs': np.ndarray,             # Shape: (n_samples,), dtype: float64
    'energies': np.ndarray,          # Shape: (n_samples,), dtype: float64
    'parity_prefixes': np.ndarray,   # Shape: (n_samples, max_nh), dtype: int32
    'metadata': dict                 # Additional information
}
```

#### Field Descriptions

- **operator_strings**: Integer-encoded operator sequences, padded to max_nh
  - Padding value: 0 (reserved for padding token)
  - Operator types: 1, 2, 3, ... (depends on model)

- **nh_values**: Actual length of each operator string (before padding)

- **weights**: Absolute value of configuration weights |w(C)|

- **signs**: Sign of configuration weights (+1 or -1)

- **energies**: Local energy estimators H(C)

- **parity_prefixes**: Cumulative parity information for each position
  - Computed using Fortran library
  - Used for parity-aware embeddings

- **metadata**: Dictionary containing:
  ```python
  {
      'beta': float,
      'n_sites': int,
      'lattice_type': str,
      'n_bonds': int,
      'operator_vocab_size': int,
      'max_nh': int,
      'generation_date': str,
      'sampler_version': str
  }
  ```

## Model Checkpoint Format

### PyTorch Checkpoint (.pt)

Model checkpoints are saved using PyTorch's standard format:

```python
{
    'model_state_dict': OrderedDict,  # Model parameters
    'optimizer_state_dict': dict,     # Optimizer state
    'epoch': int,                     # Training epoch
    'train_loss': float,              # Training loss
    'val_loss': float,                # Validation loss
    'config': dict,                   # Model configuration
    'metadata': dict                  # Additional information
}
```

#### Model Configuration

```python
config = {
    'd_model': 128,           # Model dimension
    'n_heads': 4,             # Number of attention heads
    'n_layers': 6,            # Number of transformer layers
    'd_ff': 512,              # Feed-forward dimension
    'dropout': 0.1,           # Dropout rate
    'max_seq_len': 256,       # Maximum sequence length
    'vocab_size': 10,         # Operator vocabulary size
    'parity_emb_dim': 32,     # Parity embedding dimension
    'use_parity': True,       # Whether to use parity embeddings
}
```

## Operator Encoding

### Operator Types

For the Heisenberg model on a square lattice:

```
0: PAD (padding token)
1: S+S- (spin raising/lowering on bond)
2: S-S+ (spin lowering/raising on bond)
3: SzSz (diagonal interaction)
4: Identity (no operator)
```

### Bond Encoding

Bonds are numbered sequentially:
- 2D square lattice (L×L): bonds 0 to 2L²-1
  - Horizontal bonds: 0 to L²-1
  - Vertical bonds: L² to 2L²-1

Example for 2×2 lattice:
```
Sites:     Bonds:
0--1       0--1
|  |       |  |
2--3       2--3
           (horizontal: 0,1,2,3)
           (vertical: 4,5,6,7)
```

## File Organization

### Recommended Directory Structure

```
data/
├── raw/                    # Raw MCMC output
│   ├── 2x2_beta8.0.bin
│   ├── 2x2_beta10.0.bin
│   └── ...
├── processed/              # Preprocessed training data
│   ├── 2x2_beta8.0_train.npz
│   ├── 2x2_beta8.0_val.npz
│   ├── 2x2_beta8.0_test.npz
│   └── ...
└── metadata/               # Data generation logs
    ├── 2x2_beta8.0_info.json
    └── ...

checkpoints/
├── 2x2_beta8.0/
│   ├── numerator_even.pt
│   ├── denominator_even.pt
│   ├── config.json
│   └── training_log.txt
└── ...
```

## Data Loading Example

### Python

```python
import numpy as np
import torch

# Load preprocessed data
data = np.load('data/processed/2x2_beta8.0_train.npz')
operator_strings = data['operator_strings']
nh_values = data['nh_values']
weights = data['weights']
signs = data['signs']
energies = data['energies']
parity_prefixes = data['parity_prefixes']
metadata = data['metadata'].item()

print(f"Loaded {len(operator_strings)} configurations")
print(f"Beta = {metadata['beta']}, Lattice = {metadata['lattice_type']}")

# Load model checkpoint
checkpoint = torch.load('checkpoints/2x2_beta8.0/numerator_even.pt')
model_state = checkpoint['model_state_dict']
config = checkpoint['config']
print(f"Model trained for {checkpoint['epoch']} epochs")
print(f"Validation loss: {checkpoint['val_loss']:.6f}")
```

## Data Validation

### Sanity Checks

When loading data, perform these validation checks:

1. **Shape consistency**:
   ```python
   assert operator_strings.shape[0] == len(nh_values)
   assert operator_strings.shape[1] >= nh_values.max()
   ```

2. **Value ranges**:
   ```python
   assert np.all(nh_values > 0)
   assert np.all(operator_strings >= 0)
   assert np.all(weights >= 0)
   assert np.all(np.abs(signs) == 1)
   ```

3. **Padding correctness**:
   ```python
   for i, nh in enumerate(nh_values):
       assert np.all(operator_strings[i, nh:] == 0)  # Padding
       assert np.all(operator_strings[i, :nh] > 0)   # Valid operators
   ```

## Conversion Utilities

The repository provides utilities for format conversion:

```bash
# Binary to text
python scripts/convert_data.py --input data.bin --output data.txt --format txt

# Binary to NPZ
python scripts/preprocess_data.py --input data.bin --output data.npz

# Inspect data
python scripts/inspect_data.py --input data.npz --show-stats
```

## Next Steps

- [Method Description](method.md)
- [Reproducing Paper Results](reproducibility.md)
- [API Documentation](api.md)
