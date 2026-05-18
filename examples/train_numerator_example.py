#!/usr/bin/env python3
"""
Example: Train a numerator model for 2x2 lattice at beta=8.0

This script demonstrates the complete training pipeline for the numerator model
(even parity configurations).
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
        description='Train numerator model for RSSE control variates'
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--lattice', type=str, default='2x2',
                        help='Lattice size (e.g., 2x2, 3x1)')
    parser.add_argument('--beta', type=float, default=8.0,
                        help='Inverse temperature')

    # Model arguments
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=512,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Construct file paths
    data_prefix = f"{args.lattice}_beta{args.beta}"
    train_data = Path(args.data_dir) / f"{data_prefix}_train.npz"
    val_data = Path(args.data_dir) / f"{data_prefix}_val.npz"

    # Check if data exists
    if not train_data.exists():
        print(f"Error: Training data not found at {train_data}")
        print(f"Please run data preprocessing first:")
        print(f"  python scripts/preprocess_data.py --input data/raw/{data_prefix}.bin")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir) / data_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Training Numerator Model")
    print("="*60)
    print(f"Lattice: {args.lattice}")
    print(f"Beta: {args.beta}")
    print(f"Training data: {train_data}")
    print(f"Validation data: {val_data}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    train_dataset = np.load(train_data)
    val_dataset = np.load(val_data) if val_data.exists() else None

    print(f"Training samples: {len(train_dataset['operator_strings'])}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset['operator_strings'])}")

    # Model configuration
    config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'max_seq_len': train_dataset['operator_strings'].shape[1],
        'vocab_size': int(train_dataset['operator_strings'].max()) + 1,
    }

    print("\nModel configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # TODO: Implement actual training loop
    # This is a placeholder - actual implementation would be in the training script
    print("\n" + "="*60)
    print("NOTE: This is an example script.")
    print("For actual training, use:")
    print(f"  python python/nh_window/numerator/train_transformer_parity_sign_v2_pe_nh_window_aug.py \\")
    print(f"    --data {train_data} \\")
    print(f"    --val-data {val_data} \\")
    print(f"    --output {output_dir} \\")
    print(f"    --d-model {args.d_model} \\")
    print(f"    --n-heads {args.n_heads} \\")
    print(f"    --n-layers {args.n_layers} \\")
    print(f"    --batch-size {args.batch_size} \\")
    print(f"    --epochs {args.epochs} \\")
    print(f"    --lr {args.lr}")
    print("="*60)

if __name__ == '__main__':
    main()
