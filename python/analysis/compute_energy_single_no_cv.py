#!/usr/bin/env python3
"""Compute a no-control-variate energy estimate from one RSSE binary file."""

import argparse
import glob
import os
import struct
import sys

import numpy as np


def load_meta(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        fmt_version = {b"RSSE": 2, b"RSS3": 3, b"RSS4": 4}.get(magic)
        if fmt_version is None:
            raise ValueError(f"Unsupported magic {magic!r}")
        version = struct.unpack("<i", f.read(4))[0]
        if version != fmt_version:
            raise ValueError(f"Magic/version mismatch: magic={magic!r}, version={version}")
        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
        beta, _ = struct.unpack("<2d", f.read(16))
    return {"nb": nb, "nn": nn, "beta": beta, "mm": mm, "lx": lx, "ly": ly, "fmt_version": fmt_version}


def read_samples(path, fmt_version):
    samples = []
    with open(path, "rb") as f:
        f.seek(44)
        while True:
            if fmt_version == 4:
                record = f.read(8)
                if len(record) < 8:
                    break
                nh, parity = struct.unpack("<2i", record)
                f.read(4 * nh)
            else:
                record = f.read(8)
                if len(record) < 8:
                    break
                nh, _K = struct.unpack("<2i", record)
                if fmt_version == 2:
                    if nh > 0:
                        pp = np.frombuffer(f.read(nh), dtype="<i1")
                        parity = int(pp[-1])
                    else:
                        parity = 0
                    f.read(4 * nh)
                else:
                    parity_bytes = f.read(4)
                    if len(parity_bytes) < 4:
                        break
                    parity = struct.unpack("<i", parity_bytes)[0]
                    f.read(4 * nh)

            sign = 1 if parity == 0 else -1
            samples.append({"nh": nh, "sign": sign})
    return samples


def binning_analysis(samples, meta, n_bins=100):
    N = len(samples)
    if N == 0:
        raise ValueError("No samples found")
    n_bins = min(n_bins, N)
    bin_size = N // n_bins
    nhs_arr = np.array([s["nh"] * s["sign"] for s in samples])
    s_arr = np.array([s["sign"] for s in samples])
    E_bins = []

    for i in range(n_bins):
        start, end = i * bin_size, (i + 1) * bin_size
        nhs_bin = nhs_arr[start:end].mean()
        s_bin = s_arr[start:end].mean()
        if abs(s_bin) > 1e-10:
            E_bin = -nhs_bin / (meta["beta"] * meta["nn"] * s_bin) + meta["nb"] / (4.0 * meta["nn"])
            E_bins.append(E_bin)

    if not E_bins:
        raise ValueError("All bins have near-zero average sign")
    E_bins = np.array(E_bins)
    return E_bins.mean(), E_bins.std(ddof=1) / np.sqrt(len(E_bins))


def choose_data_file(args):
    if args.data:
        return args.data
    if args.data_glob:
        pattern = args.data_glob
    elif args.beta is not None:
        pattern = os.path.join("data", "raw", f"*beta{args.beta:.3f}*.bin")
    else:
        raise SystemExit("Provide --data, --data_glob, or --beta.")

    files = sorted(glob.glob(pattern))
    if not files and args.beta is not None:
        files = sorted(glob.glob(os.path.join("data", "raw", f"*beta{args.beta:g}*.bin")))
    if not files:
        raise SystemExit(f"No data file matched: {pattern}")
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to one RSSE binary file")
    ap.add_argument("--data_glob", help="Glob for RSSE binary files; first match is used")
    ap.add_argument("--beta", type=float, help="Convenience selector for data/raw/*beta*.bin")
    ap.add_argument("--n_bins", type=int, default=100)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data_file = choose_data_file(args)
    print(f"Processing no-CV estimate from: {data_file}")

    meta = load_meta(data_file)
    samples = read_samples(data_file, meta["fmt_version"])
    E_mean, E_se = binning_analysis(samples, meta, n_bins=args.n_bins)

    print(f"  E/N = {E_mean:.10f} +/- {E_se:.10f}")

    beta_out = args.beta if args.beta is not None else meta["beta"]
    with open(args.output, "w") as f:
        f.write(f"{beta_out} {E_mean} {E_se}\n")


if __name__ == "__main__":
    main()
