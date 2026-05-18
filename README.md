# RSSE Control Variates for Quantum Monte Carlo

Transformer-based control variates for mitigating the sign problem in quantum Monte Carlo simulations using the RSSE (Resummation-based Stochastic Series Expansion) method.

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
git clone https://github.com/username/rsse-control-variates.git
cd rsse-control-variates

# Install Python dependencies
pip install -r requirements.txt

# Compile Fortran libraries
cd src
make
cd ..

# Compile Fortran MCMC sampler
cd fortran
gfortran -O3 -o rsse_sampler rsse_update_loops_cursor_optimized_v3.f90
cd ..
```

## Quick Start

### 1. Generate MCMC data
```bash
cd src/fortran
./rsse_sampler --lattice 2x2 --beta 8.0 --samples 100000 --output ../../data/sample/
```

### 2. Train models
```bash
# Train numerator model (even parity)
python python/nh_window/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py \
    --data data/sample/train.npz \
    --output checkpoints/2x2_beta8.0/

# Train denominator model
python python/nh_window/denumerator/train_transformer_parity_sign_v2_pe_nh_window_de_aug.py \
    --data data/sample/train.npz \
    --output checkpoints/2x2_beta8.0/
```

### 3. Evaluate control variates
```bash
python python/nh_window/compute_energy_jackknife.py \
    --test-data data/sample/test.npz \
    --numerator-ckpt checkpoints/2x2_beta8.0/numerator_even.pt \
    --denominator-ckpt checkpoints/2x2_beta8.0/denominator_even.pt \
    --output results/energy.json
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Method Description](docs/method.md)
- [Data Format Specification](docs/data_format.md)
- [Reproducing Paper Results](docs/reproducibility.md)

## Pre-trained Checkpoints

Pre-trained models for various lattice sizes and temperatures are available:
- [Download from Zenodo](https://zenodo.org/record/XXXXXX)

See [checkpoints/README.md](checkpoints/README.md) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2026rsse,
  title={Transformer-based Control Variates for Quantum Monte Carlo Sign Problem},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your.email@institution.edu](mailto:your.email@institution.edu).

## Acknowledgments

This work was supported by [funding sources].
