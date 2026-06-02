#!/usr/bin/env python3
"""Build or run the joint-CV energy evaluation command."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(args):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "python" / "nh_window" / "compute_energy_jackknife_Cov.py"
    return [
        sys.executable,
        str(script),
        "--data_train",
        args.data_train,
        "--data_test",
        args.data_test,
        "--ckpt_num_even",
        args.ckpt_num_even,
        "--ckpt_num_odd",
        args.ckpt_num_odd,
        "--ckpt_denom_even",
        args.ckpt_denom_even,
        "--ckpt_denom_odd",
        args.ckpt_denom_odd,
        "--n_bins",
        str(args.n_bins),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]


def main():
    parser = argparse.ArgumentParser(description="Joint-CV energy evaluation example")
    parser.add_argument("--data-train", dest="data_train", required=True)
    parser.add_argument("--data-test", dest="data_test", required=True)
    parser.add_argument("--ckpt-num-even", dest="ckpt_num_even", required=True)
    parser.add_argument("--ckpt-num-odd", dest="ckpt_num_odd", required=True)
    parser.add_argument("--ckpt-denom-even", dest="ckpt_denom_even", required=True)
    parser.add_argument("--ckpt-denom-odd", dest="ckpt_denom_odd", required=True)
    parser.add_argument("--n-bins", dest="n_bins", type=int, default=100)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true", help="Execute the command instead of only printing it")
    args = parser.parse_args()

    cmd = build_command(args)
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
