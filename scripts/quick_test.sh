#!/bin/bash
# Quick reproduction script for testing the pipeline on a small system

set -e  # Exit on error

echo "=========================================="
echo "RSSE Control Variates - Quick Test"
echo "=========================================="
echo ""

# Configuration
LATTICE="2x2"
BETA="8.0"
N_SAMPLES=10000  # Small for quick test
SEED=42

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Create directories
mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}" "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Configuration:"
echo "  Lattice: ${LATTICE}"
echo "  Beta: ${BETA}"
echo "  Samples: ${N_SAMPLES}"
echo ""

# Step 1: Generate MCMC data
echo "=========================================="
echo "Step 1: Generate MCMC Data"
echo "=========================================="
echo ""

FORTRAN_DIR="${PROJECT_ROOT}/fortran"
MCMC_BINARY="${FORTRAN_DIR}/rsse_update_loops_cursor_optimized_v3.x"

if [ ! -f "${MCMC_BINARY}" ]; then
    echo "Error: MCMC sampler not found at ${MCMC_BINARY}"
    echo "Please compile the Fortran code first:"
    echo "  cd fortran"
    echo "  gfortran -O3 -o rsse_update_loops_cursor_optimized_v3.x rsse_update_loops_cursor_optimized_v3.f90"
    exit 1
fi

OUTPUT_FILE="${RAW_DIR}/${LATTICE}_beta${BETA}.bin"

echo "Running MCMC sampler..."
echo "This may take a few minutes..."
echo ""

# Note: This is a placeholder - actual MCMC input format may differ
# Users should adjust based on their Fortran code's input requirements
cat > "${FORTRAN_DIR}/temp_input.txt" << EOF
${BETA}
2
2
${N_SAMPLES}
${SEED}
${OUTPUT_FILE}
EOF

# Run MCMC (commented out - users should uncomment and adjust)
# cd "${FORTRAN_DIR}"
# ./rsse_update_loops_cursor_optimized_v3.x < temp_input.txt
# rm temp_input.txt
# cd "${PROJECT_ROOT}"

echo "MCMC data generation complete (or skipped if already exists)"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Step 2: Preprocess data
echo "=========================================="
echo "Step 2: Preprocess Data"
echo "=========================================="
echo ""

if [ -f "${OUTPUT_FILE}" ]; then
    echo "Preprocessing MCMC data..."
    # python scripts/preprocess_data.py \
    #     --input "${OUTPUT_FILE}" \
    #     --output "${PROCESSED_DIR}/${LATTICE}_beta${BETA}" \
    #     --train-ratio 0.7 \
    #     --val-ratio 0.15 \
    #     --test-ratio 0.15 \
    #     --seed ${SEED}
    echo "Data preprocessing complete (or skipped)"
else
    echo "Warning: Raw data file not found, skipping preprocessing"
fi
echo ""

# Step 3: Train models
echo "=========================================="
echo "Step 3: Train Models"
echo "=========================================="
echo ""

TRAIN_DATA="${PROCESSED_DIR}/${LATTICE}_beta${BETA}_train.npz"
VAL_DATA="${PROCESSED_DIR}/${LATTICE}_beta${BETA}_val.npz"
OUTPUT_DIR="${CHECKPOINT_DIR}/${LATTICE}_beta${BETA}"

if [ -f "${TRAIN_DATA}" ]; then
    echo "Training numerator model..."
    # python python/nh_window/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py \
    #     --data "${TRAIN_DATA}" \
    #     --val-data "${VAL_DATA}" \
    #     --output "${OUTPUT_DIR}" \
    #     --d-model 64 \
    #     --n-heads 4 \
    #     --n-layers 4 \
    #     --d-ff 256 \
    #     --batch-size 32 \
    #     --epochs 20 \
    #     --lr 1e-4 \
    #     --seed ${SEED}

    echo "Training denominator model..."
    # python python/nh_window/denumerator/train_transformer_parity_sign_v2_pe_nh_window_de_aug.py \
    #     --data "${TRAIN_DATA}" \
    #     --val-data "${VAL_DATA}" \
    #     --output "${OUTPUT_DIR}" \
    #     --d-model 64 \
    #     --n-heads 4 \
    #     --n-layers 4 \
    #     --d-ff 256 \
    #     --batch-size 32 \
    #     --epochs 20 \
    #     --lr 1e-4 \
    #     --seed ${SEED}

    echo "Model training complete (or skipped)"
else
    echo "Warning: Training data not found, skipping model training"
fi
echo ""

# Step 4: Compute energy with CV
echo "=========================================="
echo "Step 4: Compute Energy with CV"
echo "=========================================="
echo ""

TEST_DATA="${PROCESSED_DIR}/${LATTICE}_beta${BETA}_test.npz"
NUMERATOR_CKPT="${OUTPUT_DIR}/numerator_even.pt"
DENOMINATOR_CKPT="${OUTPUT_DIR}/denominator_even.pt"
RESULT_FILE="${RESULTS_DIR}/${LATTICE}_beta${BETA}_cv.json"

if [ -f "${TEST_DATA}" ] && [ -f "${NUMERATOR_CKPT}" ] && [ -f "${DENOMINATOR_CKPT}" ]; then
    echo "Computing energy with control variates..."
    # python python/nh_window/compute_energy_jackknife.py \
    #     --test-data "${TEST_DATA}" \
    #     --numerator-ckpt "${NUMERATOR_CKPT}" \
    #     --denominator-ckpt "${DENOMINATOR_CKPT}" \
    #     --output "${RESULT_FILE}" \
    #     --n-blocks 10 \
    #     --seed ${SEED}

    echo "Energy computation complete (or skipped)"
else
    echo "Warning: Required files not found, skipping energy computation"
fi
echo ""

# Summary
echo "=========================================="
echo "Quick Test Complete"
echo "=========================================="
echo ""
echo "Note: This script contains placeholders for the actual commands."
echo "Uncomment the relevant lines and adjust paths as needed."
echo ""
echo "Expected outputs:"
echo "  - Raw data: ${RAW_DIR}/"
echo "  - Processed data: ${PROCESSED_DIR}/"
echo "  - Checkpoints: ${CHECKPOINT_DIR}/"
echo "  - Results: ${RESULTS_DIR}/"
echo ""
echo "For full reproduction, see: docs/reproducibility.md"
echo "=========================================="
