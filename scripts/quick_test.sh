#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_GLOB="${DATA_GLOB:-data/raw/*.bin}"
DEVICE="${DEVICE:-cpu}"

cd "${PROJECT_ROOT}"

echo "RSSE NCV quick check"
echo "Project root: ${PROJECT_ROOT}"
echo

echo "[1/4] Python imports"
python - <<'PY'
import numpy
import torch
print(f"numpy={numpy.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

echo
echo "[2/4] Fortran helper libraries"
if [ ! -f src/parity_prefix_lib.so ] || [ ! -f src/parity_prefix_candidates_lib.so ]; then
  echo "Shared libraries are missing. Build them with:"
  echo "  cd src && make"
else
  python - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from parity_prefix_wrapper import compute_parity_prefix
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates
print("parity helper imports OK")
PY
fi

echo
echo "[3/4] Sampler"
if [ ! -f fortran/rsse_update_loops_cursor_optimized_v3.x ]; then
  echo "Sampler executable is missing. Build it with:"
  echo "  cd fortran && gfortran -O3 -o rsse_update_loops_cursor_optimized_v3.x rsse_update_loops_cursor_optimized_v3.f90"
else
  echo "Sampler executable found: fortran/rsse_update_loops_cursor_optimized_v3.x"
fi

echo
echo "[4/4] Command templates"
echo "Generate data:"
echo "  mkdir -p data/raw && cd fortran && RSSE_OUTDIR=../data/raw ./rsse_update_loops_cursor_optimized_v3.x"
echo
echo "Train numerator even:"
echo "  python python/train/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py --parity even --data_glob '${DATA_GLOB}' --auto_nh_window 1 --output_dir checkpoints/numerator/even --num_epochs 10"
echo
echo "Train denominator even:"
echo "  python python/train/denumerator/train_transformer_parity_sign_v2_pe_nh_window_de_aug.py --parity even --data_glob '${DATA_GLOB}' --auto_nh_window 1 --output_dir checkpoints/denominator/even --num_epochs 10"
echo
echo "Evaluate after training all four parity models:"
echo "  python python/train/compute_energy_jackknife_Cov.py --data_train data/raw/train.bin --data_test data/raw/test.bin --ckpt_num_even checkpoints/numerator/even/best_model.pt --ckpt_num_odd checkpoints/numerator/odd/best_model.pt --ckpt_denom_even checkpoints/denominator/even/best_model.pt --ckpt_denom_odd checkpoints/denominator/odd/best_model.pt --device ${DEVICE}"
echo
echo "Associated paper: https://arxiv.org/abs/2605.26814"
