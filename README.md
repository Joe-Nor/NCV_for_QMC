# Neural Autoregressive Control Variates for RSSE Quantum Monte Carlo

Code accompanying the paper
[Neural Autoregressive Control Variates for the Quantum Monte Carlo Sign Problem](https://arxiv.org/abs/2605.26814).
The repository implements autoregressive neural-network control variates for
variance reduction in sign-problem quantum Monte Carlo simulations using the
RSSE (Resummation-based Stochastic Series Expansion) method.

## Overview

This repository contains the implementation of neural network control variates for variance reduction in quantum Monte Carlo calculations. The method uses autoregressive Transformer models to learn probability distributions over uncolored graph operator strings, which are then used as control variates to reduce the variance of sign-problem-affected observables.

## Key Features

- **Fortran MCMC sampler**: Efficient RSSE implementation with O(1) incremental loop counting
- **Transformer models**: Autoregressive models with parity prefix embeddings
- **Control variate estimation**: Jackknife-based variance reduction
- **Reproducible examples**: Complete pipeline for 2×2 and 3×1 lattices

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- gfortran (for Fortran sampler)

### Setup
```bash
# Clone repository
git clone https://github.com/Joe-Nor/NCV_for_QMC.git
cd NCV_for_QMC

# Install Python dependencies
pip install -r requirements.txt

# Compile Fortran libraries
cd src
make
cd ..

# Compile Fortran MCMC sampler
cd fortran
gfortran -O3 -o rsse_update_loops_cursor_optimized_v3.x rsse_update_loops_cursor_optimized_v3.f90
cd ..
```

## Quick Start

### 1. Generate MCMC data
```bash
mkdir -p data/raw
cd fortran
RSSE_OUTDIR=../data/raw ./rsse_update_loops_cursor_optimized_v3.x
cd ..
```

The sampler reads `fortran/rsse_input.in` and `fortran/seed.in`.

### 2. Train models
```bash
# Train numerator model (even parity)
python python/train/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py \
    --parity even \
    --data_glob "data/raw/*.bin" \
    --auto_nh_window 1 \
    --output_dir checkpoints/numerator/even

# Train numerator model (odd parity)
python python/train/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py \
    --parity odd \
    --data_glob "data/raw/*.bin" \
    --auto_nh_window 1 \
    --output_dir checkpoints/numerator/odd

# Train denominator models
python python/train/denumerator/train_transformer_parity_sign_v2_pe_nh_window_de_aug.py \
    --parity even \
    --data_glob "data/raw/*.bin" \
    --auto_nh_window 1 \
    --output_dir checkpoints/denominator/even

python python/train/denumerator/train_transformer_parity_sign_v2_pe_nh_window_de_aug.py \
    --parity odd \
    --data_glob "data/raw/*.bin" \
    --auto_nh_window 1 \
    --output_dir checkpoints/denominator/odd
```

### 3. Evaluate control variates
```bash
python python/train/compute_energy_jackknife_Cov.py \
    --data_train data/raw/train.bin \
    --data_test data/raw/test.bin \
    --ckpt_num_even checkpoints/numerator/even/best_model.pt \
    --ckpt_num_odd checkpoints/numerator/odd/best_model.pt \
    --ckpt_denom_even checkpoints/denominator/even/best_model.pt \
    --ckpt_denom_odd checkpoints/denominator/odd/best_model.pt
```

## Public Checkpoints

Curated inference checkpoints used for the reported neural-control-variate
results are available from the GitHub release:

- [v1.0.0 release](https://github.com/Joe-Nor/NCV_for_QMC/releases/tag/v1.0.0)
- [public_checkpoints_ncv.tar.gz](https://github.com/Joe-Nor/NCV_for_QMC/releases/download/v1.0.0/public_checkpoints_ncv.tar.gz)
- [SHA256 checksum](https://github.com/Joe-Nor/NCV_for_QMC/releases/download/v1.0.0/public_checkpoints_ncv.tar.gz.sha256)

The archive contains 84 inference checkpoints for the `2x-2` and `3x1`
systems. See [Public Checkpoint Preparation](docs/checkpoints.md) for the
selection rule and manifest format.

## Documentation

- [Installation Guide](docs/installation.md)
- [Method Description](docs/method.md)
- [Data Format Specification](docs/data_format.md)
- [Public Checkpoint Preparation](docs/checkpoints.md)
- [Associated arXiv paper](https://arxiv.org/abs/2605.26814)

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@misc{qiao2026neuralautoregressivecontrolvariates,
  title         = {Neural Autoregressive Control Variates for the Quantum Monte Carlo Sign Problem},
  author        = {Bei Qiao and Lei Wang},
  year          = {2026},
  eprint        = {2605.26814},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.str-el},
  doi           = {10.48550/arXiv.2605.26814},
  url           = {https://arxiv.org/abs/2605.26814}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

See the accompanying paper for acknowledgments and funding information.
