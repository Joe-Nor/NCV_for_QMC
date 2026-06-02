#!/usr/bin/env python3
"""
Test n_h window selection for numerator training.
Analyze n_h * count distribution to determine optimal window.
"""

import argparse
import os
import struct
import numpy as np
from typing import Dict, Tuple


def scan_nh_histogram(filepath: str) -> Dict:
    """Scan n_h histogram from V2, V3, or V4 binary file."""
    hist = {}
    nh_values = []

    with open(filepath, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic not in (b"RSSE", b"RSS3", b"RSS4"):
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack("<i", f.read(4))[0]
        if version not in (2, 3, 4):
            raise ValueError(f"Only V2/V3/V4 format supported, got version {version}")
        fmt_version = version

        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
        beta, surface_n = struct.unpack("<2d", f.read(16))

        print(f"File: {filepath}")
        print(f"Format: V{fmt_version}")
        print(f"Lattice: {lx}x{ly}, nn={nn}, nb={nb}, mm={mm}")
        print(f"Beta: {beta:.3f}, surface_n: {surface_n:.6f}\n")

        # Scan all records
        while True:
            nh_bytes = f.read(4)
            if len(nh_bytes) < 4:
                break

            nh = struct.unpack("<i", nh_bytes)[0]

            if fmt_version == 2:
                skip_bytes = 4 + nh + 4 * nh
            elif fmt_version == 3:
                skip_bytes = 4 + 4 + 4 * nh
            else:
                skip_bytes = 4 + 4 * nh
            f.seek(skip_bytes, 1)

            hist[nh] = hist.get(nh, 0) + 1
            nh_values.append(nh)

    nh_array = np.array(nh_values)
    return {
        'hist': hist,
        'total_samples': len(nh_values),
        'nh_min': int(nh_array.min()),
        'nh_max': int(nh_array.max()),
        'mean_nh': float(nh_array.mean()),
        'std_nh': float(nh_array.std()),
    }


def choose_window(hist: Dict[int, int], tail_mass: float) -> Tuple[int, int]:
    """Choose n_h window based on n_h-weighted cumulative distribution."""
    sorted_nh = sorted(hist.keys())

    # n_h-weighted mass for numerator
    mass_dict = {n: float(n * hist[n]) for n in sorted_nh}
    total_mass = sum(mass_dict.values())

    # Choose nmin: cumulative from left
    cumsum = 0
    nmin = sorted_nh[0]
    for n in sorted_nh:
        cumsum += mass_dict[n]
        if cumsum / total_mass >= tail_mass:
            nmin = n
            break

    # Choose nmax: cumulative from right
    cumsum = 0
    nmax = sorted_nh[-1]
    for n in reversed(sorted_nh):
        cumsum += mass_dict[n]
        if cumsum / total_mass >= tail_mass:
            nmax = n
            break

    return nmin, nmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.environ.get("RSSE_SAMPLE_PATH"),
        help="Path to an RSSE binary sample file. Defaults to RSSE_SAMPLE_PATH.",
    )
    args = parser.parse_args()
    if not args.data:
        raise SystemExit("Provide --data or set RSSE_SAMPLE_PATH.")
    filepath = args.data

    print("Scanning n_h distribution...\n")
    stats = scan_nh_histogram(filepath)

    hist = stats['hist']
    total = stats['total_samples']

    print(f"Total samples: {total}")
    print(f"n_h range: [{stats['nh_min']}, {stats['nh_max']}]")
    print(f"Mean n_h: {stats['mean_nh']:.2f}")
    print(f"Std n_h: {stats['std_nh']:.2f}\n")

    # Compute n_h-weighted distribution
    nh_weighted = {n: float(n * hist[n]) for n in hist.keys()}
    total_weight = sum(nh_weighted.values())

    print("n_h-weighted distribution (numerator):")
    print(f"Total n_h weight: {total_weight:.0f}\n")

    # Print distribution
    print(f"{'n_h':>4s} {'count':>8s} {'count%':>7s} {'weight':>12s} {'weight%':>8s}")
    for n in sorted(hist.keys()):
        cnt = hist[n]
        wgt = nh_weighted[n]
        print(f"{n:4d} {cnt:8d} {cnt/total*100:6.2f}% {wgt:12.0f} {wgt/total_weight*100:7.2f}%")

    print()

    # Test different tail mass thresholds
    tail_masses = [0.001, 0.0001, 0.00001]

    print("Window selection for different tail mass thresholds:")
    for tail_mass in tail_masses:
        nmin, nmax = choose_window(hist, tail_mass)

        # Calculate actual excluded mass
        left_mass = sum(nh_weighted[n] for n in hist.keys() if n < nmin)
        right_mass = sum(nh_weighted[n] for n in hist.keys() if n > nmax)
        excluded_mass = left_mass + right_mass
        excluded_frac = excluded_mass / total_weight

        print(f"  tail_mass={tail_mass:.5f}: nmin={nmin:2d}, nmax={nmax:2d}, "
              f"excluded={excluded_frac*100:.4f}%")


if __name__ == '__main__':
    main()
