# Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Python**: 3.8 or higher
- **Fortran Compiler**: gfortran 9.0 or higher (for MCMC sampler)
- **Memory**: At least 8GB RAM recommended
- **Storage**: At least 10GB free space for data and checkpoints

### Python Dependencies
The following Python packages are required:
- PyTorch >= 2.0.0 (with CUDA support recommended for training)
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- tqdm >= 4.65.0

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Joe-Nor/NCV_for_QMC.git
cd NCV_for_QMC
```

### 2. Set Up Python Environment

We recommend using a virtual environment or conda:

#### Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using conda:
```bash
conda create -n rsse python=3.10
conda activate rsse
```

### 3. Compile Fortran Components

#### Compile the parity prefix libraries:
```bash
cd src
make
cd ..
```

The Makefile should compile the Fortran libraries and create Python-callable shared objects.

#### Compile the MCMC sampler:
```bash
cd fortran
gfortran -O3 -o rsse_update_loops_cursor_optimized_v3.x rsse_update_loops_cursor_optimized_v3.f90
cd ..
```

### 4. Verify Installation

Run a quick test to verify the installation:

```bash
python -c "import torch; import numpy; print('Installation successful!')"
```

Test the Fortran sampler:
```bash
cd fortran
mkdir -p ../data/raw
RSSE_OUTDIR=../data/raw ./rsse_update_loops_cursor_optimized_v3.x
cd ..
```

## GPU Support

For GPU-accelerated training, ensure you have:
1. NVIDIA GPU with CUDA Compute Capability 3.5 or higher
2. CUDA Toolkit 11.7 or higher
3. PyTorch with CUDA support

Install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Fortran Compilation Issues

**Problem**: `gfortran: command not found`

**Solution**: Install gfortran:
- Ubuntu/Debian: `sudo apt-get install gfortran`
- macOS: `brew install gcc`
- Windows: Install MinGW-w64

**Problem**: Compilation errors with f2py

**Solution**: Ensure NumPy is installed and f2py is available:
```bash
pip install numpy
python -m numpy.f2py --help
```

### PyTorch Installation Issues

**Problem**: PyTorch installation fails or is very slow

**Solution**: Use conda for PyTorch installation:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Memory Issues

**Problem**: Out of memory during training

**Solution**: 
- Reduce batch size in training scripts
- Use gradient accumulation
- Enable mixed precision training (FP16)

## Next Steps

After successful installation, proceed to:
- [Quick Start Guide](../README.md#quick-start)
- [Method Description](method.md)
- [Associated arXiv Paper](https://arxiv.org/abs/2605.26814)
