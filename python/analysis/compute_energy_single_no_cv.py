#!/usr/bin/env python3
"""
Compute energy for a single beta value with CV.
"""
import os, sys, struct, glob
import numpy as np

sys.path.append('/home/user_beiqiao/private/homefile/rsse_tri/python/analysis')


def load_v2_meta(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"RSSE"
        version = struct.unpack("<i", f.read(4))[0]
        assert version == 2
        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
        beta, _ = struct.unpack("<2d", f.read(16))
    return {"nb": nb, "nn": nn, "beta": beta, "mm": mm}


def read_samples(path):
    samples = []
    with open(path, "rb") as f:
        f.seek(44)
        while True:
            nh_k = f.read(8)
            if len(nh_k) < 8:
                break
            nh, K = struct.unpack("<2i", nh_k)
            if nh > 0:
                pp = np.frombuffer(f.read(nh), dtype="<i1")
                f.read(4 * nh)
                parity = int(pp[-1])
            else:
                parity = 0
            sign = 1 if parity == 0 else -1
            samples.append({"nh": nh, "sign": sign})
    return samples


def binning_analysis(samples, meta, n_bins=100):
    N = len(samples)
    nhs_arr = np.array([s["nh"] * s["sign"] for s in samples])
    s_arr = np.array([s["sign"] for s in samples])

    bin_size = N // n_bins
    E_bins = []

    for i in range(n_bins):
        start, end = i * bin_size, (i + 1) * bin_size
        nhs_bin = nhs_arr[start:end].mean()
        s_bin = s_arr[start:end].mean()

        if abs(s_bin) > 1e-10:
            E_bin = -nhs_bin / (meta["beta"] * meta["nn"] * s_bin) + meta["nb"] / (4.0 * meta["nn"])
            E_bins.append(E_bin)

    E_bins = np.array(E_bins)
    return E_bins.mean(), E_bins.std(ddof=1) / np.sqrt(n_bins)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    beta = args.beta
    data_base = f"/home/user_beiqiao/private/datafile/rsse_data/fortran3/3x1/beta{beta}/test"

    test_files = sorted(glob.glob(f"{data_base}/*_seed1*{beta}_*.bin"))

    if not test_files:
        print(f"No test file found for beta={beta}")
        sys.exit(1)

    print(f"Processing beta={beta} (no CV)...")
    print(f"  File: {test_files[0]}")

    meta = load_v2_meta(test_files[0])
    samples = read_samples(test_files[0])
    E_mean, E_se = binning_analysis(samples, meta, n_bins=100)

    print(f"  E/N = {E_mean:.10f} ± {E_se:.10f}")

    with open(args.output, 'w') as f:
        f.write(f"{beta} {E_mean} {E_se}\n")
