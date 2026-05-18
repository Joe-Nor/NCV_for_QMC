# Examples

This directory contains example scripts demonstrating how to use the RSSE control variates package.

## Available Examples

### 1. Training a Numerator Model

**File:** `train_numerator_example.py`

Demonstrates how to train a numerator model (for even parity configurations).

```bash
python examples/train_numerator_example.py \
    --data-dir data/processed \
    --lattice 2x2 \
    --beta 8.0 \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 6 \
    --batch-size 64 \
    --epochs 100 \
    --output-dir checkpoints
```

**Key features:**
- Configurable model architecture
- Automatic data loading
- Checkpoint saving
- Progress monitoring

### 2. Computing Energy with Control Variates

**File:** `compute_energy_example.py`

Shows how to compute energy estimates using trained models and jackknife resampling.

```bash
python examples/compute_energy_example.py \
    --test-data data/processed/2x2_beta8.0_test.npz \
    --numerator-ckpt checkpoints/2x2_beta8.0/numerator_even.pt \
    --denominator-ckpt checkpoints/2x2_beta8.0/denominator_even.pt \
    --n-blocks 20 \
    --output results/2x2_beta8.0_cv.json
```

**Key features:**
- Baseline computation (no CV)
- Control variate construction
- Jackknife uncertainty quantification
- JSON output for results

### 3. Quick Test Pipeline

**File:** `../scripts/quick_test.sh`

End-to-end pipeline for testing on a small system.

```bash
bash scripts/quick_test.sh
```

**What it does:**
1. Generates MCMC data (small sample)
2. Preprocesses data
3. Trains lightweight models
4. Computes energy estimates
5. Generates comparison plots

**Expected runtime:** ~30 minutes on GPU, ~2 hours on CPU

## Example Workflow

### Complete Pipeline for 2×2 Lattice

```bash
# 1. Generate MCMC data
cd fortran
./rsse_update_loops_cursor_optimized_v3.x << EOF
8.0
2
2
1000000
42
../data/raw/2x2_beta8.0.bin
EOF
cd ..

# 2. Preprocess data
python scripts/preprocess_data.py \
    --input data/raw/2x2_beta8.0.bin \
    --output data/processed/2x2_beta8.0 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15

# 3. Train numerator model
python python/nh_window/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py \
    --data data/processed/2x2_beta8.0_train.npz \
    --val-data data/processed/2x2_beta8.0_val.npz \
    --output checkpoints/2x2_beta8.0/ \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 6 \
    --batch-size 64 \
    --epochs 100

# 4. Train denominator model
python python/nh_window/denumerator/train_transformer_parity_sign_v2_pe_nh_window_de_aug.py \
    --data data/processed/2x2_beta8.0_train.npz \
    --val-data data/processed/2x2_beta8.0_val.npz \
    --output checkpoints/2x2_beta8.0/ \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 6 \
    --batch-size 64 \
    --epochs 100

# 5. Compute energy with CV
python python/nh_window/compute_energy_jackknife.py \
    --test-data data/processed/2x2_beta8.0_test.npz \
    --numerator-ckpt checkpoints/2x2_beta8.0/numerator_even.pt \
    --denominator-ckpt checkpoints/2x2_beta8.0/denominator_even.pt \
    --output results/2x2_beta8.0_cv.json \
    --n-blocks 20

# 6. Plot results
python python/analysis/plot_sign_energy_vs_beta.py \
    --results-dir results/ \
    --lattice 2x2 \
    --output figures/energy_vs_beta_2x2.pdf
```

## Customization

### Adjusting Model Size

For smaller systems or faster training:
```bash
--d-model 64 \
--n-heads 4 \
--n-layers 4 \
--d-ff 256 \
--batch-size 32 \
--epochs 50
```

For larger systems or better accuracy:
```bash
--d-model 256 \
--n-heads 8 \
--n-layers 8 \
--d-ff 1024 \
--batch-size 128 \
--epochs 200
```

### Using Different Data Splits

```bash
python scripts/preprocess_data.py \
    --input data/raw/2x2_beta8.0.bin \
    --output data/processed/2x2_beta8.0 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Changing Jackknife Blocks

More blocks = better uncertainty estimates but slower:
```bash
--n-blocks 50  # More accurate
--n-blocks 10  # Faster
```

## Expected Results

### 2×2 Square Lattice, β=8.0

**Without control variates:**
- Energy: -0.718 ± 0.008
- Average sign: ~0.85

**With control variates:**
- Energy: -0.718 ± 0.002
- Variance reduction factor: ~0.3 (70% reduction)

### Performance Metrics

**Training time (per model):**
- GPU (RTX 3090): ~2-3 hours
- CPU (16 cores): ~1-2 days

**Inference time:**
- GPU: ~5 minutes for 100k samples
- CPU: ~30 minutes for 100k samples

## Troubleshooting

### Issue: CUDA out of memory

**Solution:** Reduce batch size
```bash
--batch-size 32  # or even 16
```

### Issue: Training loss not decreasing

**Solution:** Adjust learning rate
```bash
--lr 5e-5  # Lower learning rate
# or
--lr 5e-4  # Higher learning rate
```

### Issue: Poor variance reduction

**Possible causes:**
- Model underfitting (train longer or use larger model)
- Insufficient training data (generate more MCMC samples)
- Data quality issues (check MCMC equilibration)

## Additional Resources

- [Installation Guide](../docs/installation.md)
- [Method Description](../docs/method.md)
- [Data Format Specification](../docs/data_format.md)
- [Reproducing Paper Results](../docs/reproducibility.md)

## Contributing

If you create useful example scripts, please consider contributing them back to the repository!
