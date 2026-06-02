#!/usr/bin/env python3
"""Build or run a numerator-model training command."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(args):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "python" / "nh_window" / "numerator" / "train_transformer_parity_sign_v2_pe_nh_window_aug.py"
    return [
        sys.executable,
        str(script),
        "--parity",
        args.parity,
        "--data_glob",
        args.data_glob,
        "--auto_nh_window",
        "1" if args.auto_nh_window else "0",
        "--d_model",
        str(args.d_model),
        "--nhead",
        str(args.nhead),
        "--num_layers",
        str(args.num_layers),
        "--dim_feedforward",
        str(args.dim_feedforward),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--output_dir",
        args.output_dir,
    ]


def main():
    parser = argparse.ArgumentParser(description="Numerator training example")
    parser.add_argument("--parity", choices=["even", "odd"], default="even")
    parser.add_argument("--data-glob", dest="data_glob", default="data/raw/*.bin")
    parser.add_argument("--output-dir", dest="output_dir", default="checkpoints/numerator/even")
    parser.add_argument("--d-model", dest="d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", dest="num_layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", dest="dim_feedforward", type=int, default=512)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-auto-nh-window", dest="auto_nh_window", action="store_false")
    parser.set_defaults(auto_nh_window=True)
    parser.add_argument("--run", action="store_true", help="Execute the command instead of only printing it")
    args = parser.parse_args()

    cmd = build_command(args)
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
