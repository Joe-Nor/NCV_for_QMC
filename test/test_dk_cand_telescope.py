"""Telescope-sum self-check for compute_parity_prefix_candidates.

For every opstring x (post-augmentation):
  sum over real operator steps t=0..nh-1 of
      deltaK_candidates[t, b_t]   where b_t = (x[t]/2) - 1   (0-indexed bond)
  must equal  K - nn.

EOS step (row nh) is *not* included: no operator is chosen there.

Also checks that deltaK_candidates[t, b_t] == deltaK_prefix[t] (the chosen bond
at step t reproduces the actual ΔK), which is a sanity check on the Fortran
save/restore.

Usage:
    # Real data:
    python test_dk_cand_telescope.py --data /path/to/file.bin --n 200

    # Synthetic (no data file needed):
    python test_dk_cand_telescope.py --synthetic --lx 3 --ly -3 --n 200 --nh_max 40
"""
import os, sys, struct, argparse, math
import numpy as np

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_base, "src"))
sys.path.insert(0, os.path.join(_base, "python", "nh_window", "denumerator"))

from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates
from train_transformer_parity_sign_v2_pe_nh_window_de_aug import (
    build_bsites, build_spatial_bond_perm, build_pointgroup_bond_perm,
)


def load_meta_and_samples(path, max_N):
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4); _ = struct.unpack("<i", f.read(4))[0]
        fmt = {"RSSE": 2, "RSS3": 3, "RSS4": 4}[magic.decode()]
        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
        beta, _ = struct.unpack("<2d", f.read(16))
    offsets, pos = [], 44
    with open(path, "rb") as f:
        while pos < file_size:
            offsets.append(pos); f.seek(pos)
            nh = struct.unpack("<i", f.read(4))[0]
            if fmt == 2:   pos += 8 + nh + 4 * nh
            elif fmt == 4: pos += 8 + 4 * nh
            else:          pos += 12 + 4 * nh
    N = min(max_N, len(offsets)) if max_N else len(offsets)
    samples = []
    with open(path, "rb") as f:
        for idx in range(N):
            f.seek(offsets[idx])
            if fmt == 4:
                nh, _ = struct.unpack("<2i", f.read(8))
            else:
                nh, K_stored = struct.unpack("<2i", f.read(8))
            if nh > 0:
                if fmt == 2:
                    _pp = np.frombuffer(f.read(nh), dtype="<i1").copy()
                    x = np.frombuffer(f.read(4 * nh), dtype="<i4").copy()
                elif fmt == 4:
                    x = np.frombuffer(f.read(4 * nh), dtype="<i4").copy()
                else:
                    _ = struct.unpack("<i", f.read(4))[0]
                    x = np.frombuffer(f.read(4 * nh), dtype="<i4").copy()
            else:
                x = np.array([], dtype=np.int32)
            samples.append({"x": x, "nh": nh})
    return {"lx": lx, "ly": ly, "nn": nn, "nb": nb, "beta": beta}, samples


def apply_aug(x, perm=None):
    """Re-label opstring bonds via a 0-indexed permutation row."""
    if perm is None or len(x) == 0:
        return x
    b_old = (x // 2).astype(np.int64) - 1
    b_new = perm[b_old]
    return (2 * (b_new + 1)).astype(x.dtype)


def check_sample(x, bsites, nn, nb, tag):
    """Run the two invariants for a single opstring."""
    pp, dkp, K, cand = compute_parity_prefix_candidates(x, bsites, nn, nb)
    nh = len(x)
    errs = []

    if nh == 0:
        # Empty prefix: K must equal nn; cand shape must be (1, nb) all -1.
        if K != nn:
            errs.append(f"{tag} nh=0: K={K} != nn={nn}")
        if cand.shape != (1, nb) or not np.all(cand == -1):
            errs.append(f"{tag} nh=0: cand shape/values wrong ({cand.shape})")
        return errs

    if cand.shape != (nh + 1, nb):
        errs.append(f"{tag}: cand shape {cand.shape} != ({nh+1},{nb})")
        return errs

    # (1) Consistency of the chosen bond at each step with deltaK_prefix.
    b_t = (x // 2).astype(np.int64) - 1            # 0-indexed bonds, length nh
    chosen_dk = cand[np.arange(nh), b_t]           # cand at (t, b_t) for t=0..nh-1
    if not np.all(chosen_dk.astype(np.int32) == dkp.astype(np.int32)):
        mism = int(np.sum(chosen_dk.astype(np.int32) != dkp.astype(np.int32)))
        errs.append(f"{tag}: chosen_dk != deltaK_prefix at {mism}/{nh} steps")

    # (2) Telescope sum: Σ_{t=0..nh-1} dk_cand[t, b_t] = K - nn.
    tele = int(chosen_dk.sum())
    if tele != K - nn:
        errs.append(f"{tag}: tele_sum={tele} != K-nn={K - nn} (K={K}, nn={nn})")

    # (3) Values must be in {-1, 0, +1}.
    if not np.all(np.isin(cand, [-1, 0, 1])):
        errs.append(f"{tag}: cand has out-of-range values")

    return errs


def gen_synthetic_samples(lx, ly, nb, n, nh_max, seed=0):
    """Random opstrings. Bond values are op = 2*(b_1indexed), nh ~ uniform[0, nh_max]."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        nh = int(rng.integers(0, nh_max + 1))
        if nh == 0:
            x = np.array([], dtype=np.int32)
        else:
            b = rng.integers(1, nb + 1, size=nh, dtype=np.int32)
            x = (2 * b).astype(np.int32)
        out.append({"x": x, "nh": nh})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate random opstrings instead of reading from --data")
    ap.add_argument("--lx", type=int, default=3, help="(synthetic only)")
    ap.add_argument("--ly", type=int, default=-3, help="(synthetic only)")
    ap.add_argument("--nh_max", type=int, default=40, help="(synthetic only)")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    if args.synthetic:
        bsites_tmp, nn, nb = build_bsites(args.lx, args.ly)
        lx, ly = args.lx, args.ly
        samples = gen_synthetic_samples(lx, ly, nb, args.n, args.nh_max)
        print(f"synthetic lattice: lx={lx}, ly={ly}, nn={nn}, nb={nb}, N={len(samples)}")
    else:
        if not args.data:
            raise SystemExit("must supply --data or --synthetic")
        meta, samples = load_meta_and_samples(args.data, args.n)
        lx, ly, nn, nb = meta["lx"], meta["ly"], meta["nn"], meta["nb"]
        print(f"lattice: lx={lx}, ly={ly}, nn={nn}, nb={nb}, N={len(samples)}")

    bsites, nn_b, nb_b = build_bsites(lx, ly)
    assert nn_b == nn and nb_b == nb, f"build_bsites mismatch: {nn_b},{nb_b} vs {nn},{nb}"

    # Build augmentation tables (only if applicable).
    try:
        sp_perm = build_spatial_bond_perm(lx, ly, bsites, nb) if lx >= 3 and abs(ly) >= 3 else None
    except Exception as e:
        print(f"spatial_perm disabled: {e}"); sp_perm = None
    try:
        pg_perm = build_pointgroup_bond_perm(lx, ly, bsites, nb) if lx == abs(ly) else None
    except Exception as e:
        print(f"pg_perm disabled: {e}"); pg_perm = None
    print(f"spatial shifts: {0 if sp_perm is None else sp_perm.shape[0]}, "
          f"pg elements: {0 if pg_perm is None else pg_perm.shape[0]}")

    rng = np.random.default_rng(0)
    total_errs = []
    for i, s in enumerate(samples):
        x = s["x"]; nh = s["nh"]

        # (a) Raw opstring.
        total_errs += check_sample(x, bsites, nn, nb, f"s{i} raw")

        # (b) Cyclic shift (if nh>1), random k.
        if nh > 1:
            k = int(rng.integers(1, nh))
            xc = np.roll(x, -k)
            total_errs += check_sample(xc, bsites, nn, nb, f"s{i} cyc(k={k})")

        # (c) Spatial translation (if available).
        if sp_perm is not None and nh > 0:
            k_sp = int(rng.integers(0, sp_perm.shape[0]))
            xs = apply_aug(x, sp_perm[k_sp])
            total_errs += check_sample(xs, bsites, nn, nb, f"s{i} sp(k={k_sp})")

        # (d) Pointgroup (if available).
        if pg_perm is not None and nh > 0:
            k_pg = int(rng.integers(0, pg_perm.shape[0]))
            xp = apply_aug(x, pg_perm[k_pg])
            total_errs += check_sample(xp, bsites, nn, nb, f"s{i} pg(k={k_pg})")

        # (e) All three composed (same order as collate: cyc → sp → pg).
        xa = x.copy()
        if nh > 1:
            k = int(rng.integers(1, nh))
            xa = np.roll(xa, -k)
        if sp_perm is not None and nh > 0:
            k_sp = int(rng.integers(0, sp_perm.shape[0]))
            xa = apply_aug(xa, sp_perm[k_sp])
        if pg_perm is not None and nh > 0:
            k_pg = int(rng.integers(0, pg_perm.shape[0]))
            xa = apply_aug(xa, pg_perm[k_pg])
        total_errs += check_sample(xa, bsites, nn, nb, f"s{i} aug")

    if total_errs:
        print(f"\nFAILED: {len(total_errs)} errors")
        for e in total_errs[:20]:
            print("  " + e)
        sys.exit(1)
    else:
        print(f"\nAll checks passed across {len(samples)} samples × (raw + cyc + sp + pg + aug).")


if __name__ == "__main__":
    main()
