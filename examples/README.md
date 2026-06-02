# Examples

This directory contains lightweight command builders for the main training and
evaluation entry points. The production pipeline streams RSSE binary files
directly; no separate `.npz` preprocessing step is required.

## Compile

```bash
cd src
make
cd ../fortran
gfortran -O3 -o rsse_update_loops_cursor_optimized_v3.x rsse_update_loops_cursor_optimized_v3.f90
cd ..
```

## Generate Data

The sampler reads `fortran/rsse_input.in` and `fortran/seed.in`.

```bash
mkdir -p data/raw
cd fortran
RSSE_OUTDIR=../data/raw ./rsse_update_loops_cursor_optimized_v3.x
cd ..
```

For paper-scale results, use independent train/test files and record the seed,
input file, command line, and git commit.

## Train One Numerator Model

```bash
python examples/train_numerator_example.py \
  --parity even \
  --data-glob "data/raw/*.bin" \
  --output-dir checkpoints/numerator/even \
  --epochs 50
```

The script prints the corresponding command for
`python/nh_window/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py`.
Add `--run` to execute it.

Train `parity=odd` separately, and repeat with
`python/nh_window/denumerator/train_transformer_parity_sign_v2_pe_nh_window_de_aug.py`
for denominator models.

## Evaluate Energy

```bash
python examples/compute_energy_example.py \
  --data-train data/raw/train.bin \
  --data-test data/raw/test.bin \
  --ckpt-num-even checkpoints/numerator/even/best_model.pt \
  --ckpt-num-odd checkpoints/numerator/odd/best_model.pt \
  --ckpt-denom-even checkpoints/denominator/even/best_model.pt \
  --ckpt-denom-odd checkpoints/denominator/odd/best_model.pt
```

The script prints the corresponding command for
`python/nh_window/compute_energy_jackknife_Cov.py`. Add `--run` to execute it.

## Additional Resources

- [Installation Guide](../docs/installation.md)
- [Method Description](../docs/method.md)
- [Data Format Specification](../docs/data_format.md)
- [Associated arXiv Paper](https://arxiv.org/abs/2605.26814)
