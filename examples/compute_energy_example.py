#!/usr/bin/env python3
"""
Example: Compute energy with control variates using jackknife resampling

This script demonstrates how to:
1. Load trained models
2. Compute control variate estimates
3. Use jackknife for uncertainty quantification
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

def main():
    parser = argparse.ArgumentParser(
        description='Compute energy with control variates'
    )

    # Data arguments
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data (.npz file)')
    parser.add_argument('--numerator-ckpt', type=str, required=True,
                        help='Path to numerator model checkpoint')
    parser.add_argument('--denominator-ckpt', type=str, required=True,
                        help='Path to denominator model checkpoint')

    # Computation arguments
    parser.add_argument('--n-blocks', type=int, default=20,
                        help='Number of jackknife blocks')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for model inference')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check if files exist
    test_data_path = Path(args.test_data)
    numerator_ckpt_path = Path(args.numerator_ckpt)
    denominator_ckpt_path = Path(args.denominator_ckpt)

    if not test_data_path.exists():
        print(f"Error: Test data not found at {test_data_path}")
        sys.exit(1)

    if not numerator_ckpt_path.exists():
        print(f"Error: Numerator checkpoint not found at {numerator_ckpt_path}")
        sys.exit(1)

    if not denominator_ckpt_path.exists():
        print(f"Error: Denominator checkpoint not found at {denominator_ckpt_path}")
        sys.exit(1)

    print("="*60)
    print("Computing Energy with Control Variates")
    print("="*60)
    print(f"Test data: {test_data_path}")
    print(f"Numerator model: {numerator_ckpt_path}")
    print(f"Denominator model: {denominator_ckpt_path}")
    print(f"Jackknife blocks: {args.n_blocks}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*60)

    # Load test data
    print("\nLoading test data...")
    test_data = np.load(test_data_path)
    n_samples = len(test_data['operator_strings'])
    print(f"Test samples: {n_samples}")

    # Load models
    print("\nLoading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    numerator_ckpt = torch.load(numerator_ckpt_path, map_location=device)
    denominator_ckpt = torch.load(denominator_ckpt_path, map_location=device)

    print(f"Numerator model trained for {numerator_ckpt.get('epoch', 'unknown')} epochs")
    print(f"Denominator model trained for {denominator_ckpt.get('epoch', 'unknown')} epochs")

    # Compute baseline (no CV)
    print("\n" + "-"*60)
    print("Baseline (No Control Variates)")
    print("-"*60)

    signs = test_data['signs']
    energies = test_data['energies']
    weights = test_data['weights']

    # Weighted averages
    mean_sign = np.average(signs, weights=weights)
    mean_energy_sign = np.average(energies * signs, weights=weights)
    energy_baseline = mean_energy_sign / mean_sign

    # Jackknife uncertainty
    block_size = n_samples // args.n_blocks
    energy_jackknife = []

    for i in range(args.n_blocks):
        # Exclude block i
        mask = np.ones(n_samples, dtype=bool)
        mask[i*block_size:(i+1)*block_size] = False

        mean_sign_i = np.average(signs[mask], weights=weights[mask])
        mean_energy_sign_i = np.average(energies[mask] * signs[mask], weights=weights[mask])
        energy_i = mean_energy_sign_i / mean_sign_i
        energy_jackknife.append(energy_i)

    energy_jackknife = np.array(energy_jackknife)
    energy_err_baseline = np.sqrt((args.n_blocks - 1) * np.var(energy_jackknife))

    print(f"Energy (baseline): {energy_baseline:.6f} ± {energy_err_baseline:.6f}")
    print(f"Average sign: {mean_sign:.6f}")

    # TODO: Implement actual CV computation
    # This would involve:
    # 1. Computing log-probabilities from models
    # 2. Constructing control variates
    # 3. Finding optimal coefficients
    # 4. Computing CV-corrected estimates with jackknife

    print("\n" + "-"*60)
    print("With Control Variates")
    print("-"*60)
    print("NOTE: This is an example script.")
    print("For actual CV computation, use:")
    print(f"  python python/nh_window/compute_energy_jackknife.py \\")
    print(f"    --test-data {test_data_path} \\")
    print(f"    --numerator-ckpt {numerator_ckpt_path} \\")
    print(f"    --denominator-ckpt {denominator_ckpt_path} \\")
    print(f"    --n-blocks {args.n_blocks}")
    if args.output:
        print(f"    --output {args.output}")
    print("-"*60)

    # Save results if output specified
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'test_data': str(test_data_path),
            'n_samples': int(n_samples),
            'n_blocks': args.n_blocks,
            'baseline': {
                'energy': float(energy_baseline),
                'energy_err': float(energy_err_baseline),
                'mean_sign': float(mean_sign),
            },
            # CV results would go here
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
