#!/usr/bin/env python3
"""
D₆ Orbit Consistency Diagnostic
================================
For each test sample x, compute log q_θ(g·x) for all 12 g ∈ D₆.
Since log f(g·x) = log f(x) (K and nh are D₆-invariant):

    std_{g∈D₆}[log q_θ(g·x) − log f(x)] = std_{g∈D₆}[log q_θ(g·x)]

This directly measures the model's symmetry violation within each orbit.

Large std → group-average scoring will help.
Small std → model already near-symmetric, averaging has limited benefit.

Usage:
    python orbit_consistency_d6.py \
        --ckpt_even /path/to/numerator/even_NH/best_model.pt \
        --ckpt_odd  /path/to/numerator/odd_NH/best_model.pt \
        --data_test /path/to/test/file.bin \
        --max_N 50000
"""

import os, sys, math, argparse, struct, time
import numpy as np
import torch
import torch.nn.functional as F

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, '..', '..', '..', 'src'))

import train_transformer_parity_sign_v2_pe_nh_window_aug as tps
from parity_prefix_wrapper import compute_parity_prefix
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates


# ============================================================================
# Model loading
# ============================================================================

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    dk_mlp_head = bool(ckpt.get("dk_mlp_head", saved_args.get("dk_mlp_head", 0)))
    dk_head_dk = int(ckpt.get("dk_head_dk", saved_args.get("dk_head_dk", 32)))
    dk_head_hidden = int(ckpt.get("dk_head_hidden", saved_args.get("dk_head_hidden", 256)))
    dk_head_centering = bool(saved_args.get("dk_head_centering", False))
    dk_head_bond_emb = int(saved_args.get("dk_head_bond_emb", 0))
    model = tps.AutoregressiveTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_layers=saved_args["num_layers"],
        dim_feedforward=saved_args["dim_feedforward"],
        dropout=0.0,
        max_len=saved_args["max_len"],
        dk_mlp_head=dk_mlp_head,
        dk_head_dk=dk_head_dk,
        dk_head_hidden=dk_head_hidden,
        dk_head_centering=dk_head_centering,
        dk_head_bond_emb=dk_head_bond_emb,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    nmin = ckpt.get("nmin")
    nmax = ckpt.get("nmax")
    print(f"  Loaded: d={saved_args['d_model']}, L={saved_args['num_layers']}, "
          f"ep={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"  Window: [{nmin}, {nmax}], mlp_head={dk_mlp_head}, bond_emb={dk_head_bond_emb}")
    return model, ckpt, nmin, nmax


# ============================================================================
# Data loading
# ============================================================================

def load_v2_meta(path):
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic in (b"RSSE", b"RSS3", b"RSS4"), f"Unknown magic: {magic}"
        fmt_version = struct.unpack("<i", f.read(4))[0]
        lx, ly, _, nb, mm = struct.unpack("<5i", f.read(20))
        beta = struct.unpack("<d", f.read(8))[0]
        f.read(8)  # surface_n
        offsets, pos = [], 44
        while pos < file_size:
            offsets.append(pos)
            f.seek(pos)
            nh = struct.unpack("<i", f.read(4))[0]
            if fmt_version == 2:
                pos += 8 + nh + 4 * nh
            elif fmt_version == 3:
                pos += 12 + 4 * nh
            else:
                pos += 8 + 4 * nh
    return {"path": path, "nb": nb, "mm": mm, "beta": beta, "offsets": offsets,
            "n": len(offsets), "fmt_version": fmt_version, "lx": lx, "ly": ly}


def read_raw_samples(meta, bsites, nn, nb, max_N=None, need_candidates=False):
    fmt_version = meta["fmt_version"]
    Ntot = meta["n"]
    N = min(max_N, Ntot) if max_N else Ntot
    start = Ntot - N

    samples = []
    with open(meta["path"], "rb") as f:
        for idx in range(start, Ntot):
            f.seek(meta["offsets"][idx])
            if fmt_version == 4:
                nh, _sp = struct.unpack("<2i", f.read(8))
                K_file = None
            else:
                nh, K_file = struct.unpack("<2i", f.read(8))

            if nh > 0:
                if fmt_version == 2:
                    f.read(nh)  # skip stored parity_prefix
                elif fmt_version == 3:
                    f.read(4)  # skip stored parity
                x_dense = np.frombuffer(f.read(4 * nh), dtype="<i4").copy()

                if need_candidates:
                    pp, dkp, K_out, cand = compute_parity_prefix_candidates(
                        x_dense, bsites, nn, nb)
                else:
                    pp, dkp, K_out = compute_parity_prefix(x_dense, bsites, nn, nb)
                    cand = None
                parity = int(pp[-1])
                K = int(K_out)
            else:
                x_dense = np.array([], dtype=np.int32)
                pp = np.array([], dtype=np.int8)
                dkp = np.array([], dtype=np.int32)
                parity = 0
                K = K_file if K_file is not None else 0
                cand = None

            samples.append({
                "x_dense": x_dense, "nh": nh, "K": K, "parity": parity,
                "sign": 1 if parity == 0 else -1,
                "parity_prefix": pp, "deltaK_prefix": dkp,
                "deltaK_candidates": cand,
            })
    return samples


# ============================================================================
# D₆ transformation
# ============================================================================

def apply_d6_to_samples(samples, perm_row, bsites, nn, nb, need_candidates=False):
    """Apply a single D₆ element to all samples, recomputing parity_prefix etc.

    Also verifies K-invariance as a sanity check.
    """
    transformed = []
    for s in samples:
        x = s["x_dense"]
        nh = s["nh"]
        if nh > 0:
            b_old = (x // 2).astype(np.int64) - 1
            b_new = perm_row[b_old]
            gx = (2 * (b_new + 1)).astype(x.dtype)

            if need_candidates:
                pp, dkp, K_out, cand = compute_parity_prefix_candidates(
                    gx, bsites, nn, nb)
            else:
                pp, dkp, K_out = compute_parity_prefix(gx, bsites, nn, nb)
                cand = None

            assert int(K_out) == s["K"], \
                f"K changed under D₆: {s['K']} → {int(K_out)}"
        else:
            gx = x.copy()
            pp = np.array([], dtype=np.int8)
            dkp = np.array([], dtype=np.int32)
            cand = None

        transformed.append({
            "x_dense": gx, "nh": nh, "K": s["K"], "parity": s["parity"],
            "sign": s["sign"], "parity_prefix": pp, "deltaK_prefix": dkp,
            "deltaK_candidates": cand,
        })
    return transformed


# ============================================================================
# Tensor building & scoring (from mean_nhs_cv_pe.py)
# ============================================================================

def precompute_eval_tensors(samples, need_candidates=False, nb_bonds=None):
    N = len(samples)
    if N == 0:
        return None
    max_nh = max(s["nh"] for s in samples)
    max_seq = max_nh + 2
    input_seq_len = max_seq - 1

    tokens = np.full((N, max_seq), 0, dtype=np.int64)
    padding_mask = np.ones((N, max_seq), dtype=np.bool_)
    prefix_parity = np.zeros((N, input_seq_len), dtype=np.int64)
    prefix_len = np.zeros((N, input_seq_len), dtype=np.int64)
    deltaK_prefix = np.ones((N, input_seq_len), dtype=np.int64)

    dk_cand = None
    if need_candidates and nb_bonds is not None:
        V = nb_bonds + tps.OPERATOR_OFFSET
        dk_cand = np.zeros((N, input_seq_len, V), dtype=np.float32)

    for i, s in enumerate(samples):
        nh = s["nh"]
        seq_len = nh + 2
        tokens[i, 0] = 1  # BOS
        if nh > 0:
            tokens[i, 1:1+nh] = s["x_dense"] // 2 + 2  # op_to_token vectorized
        tokens[i, 1+nh] = 2  # EOS
        padding_mask[i, :seq_len] = False

        if nh > 0:
            pp = s["parity_prefix"]
            end = min(1 + nh, input_seq_len)
            prefix_parity[i, 1:end] = pp[:end-1]
            if end < input_seq_len:
                prefix_parity[i, end:] = int(pp[-1])

            prefix_len[i, 1:end] = np.arange(1, end)
            if end < input_seq_len:
                prefix_len[i, end:] = nh

            dkp = s["deltaK_prefix"]
            deltaK_prefix[i, 1:end] = dkp[:end-1] + 1

        if need_candidates and dk_cand is not None:
            cand = s.get("deltaK_candidates")
            if cand is not None:
                n_rows = min(cand.shape[0], input_seq_len)
                dk_cand[i, :n_rows,
                        tps.OPERATOR_OFFSET:tps.OPERATOR_OFFSET + nb_bonds] = cand[:n_rows]

    out = {
        "tokens": torch.from_numpy(tokens),
        "padding_mask": torch.from_numpy(padding_mask),
        "prefix_parity": torch.from_numpy(prefix_parity),
        "prefix_len": torch.from_numpy(prefix_len),
        "deltaK_prefix": torch.from_numpy(deltaK_prefix),
    }
    if dk_cand is not None:
        out["dk_candidates"] = torch.from_numpy(dk_cand)
    return out


@torch.no_grad()
def score_logq_batch(model, tokens, padding_mask, prefix_parity, prefix_len,
                     deltaK_prefix, target_parity, nmin, nmax, device,
                     dk_candidates=None):
    tokens = tokens.to(device)
    padding_mask = padding_mask.to(device)
    prefix_parity = prefix_parity.to(device)
    prefix_len = prefix_len.to(device)
    deltaK_prefix = deltaK_prefix.to(device)
    if dk_candidates is not None:
        dk_candidates = dk_candidates.to(device)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    input_padding_mask = padding_mask[:, :-1]

    logits = model(inputs, padding_mask=input_padding_mask, prefix_parity=prefix_parity,
                   deltaK_prefix=deltaK_prefix, dk_candidates=dk_candidates)
    logits = tps.apply_token_mask(logits, input_padding_mask)
    logits = tps.apply_nh_window_mask(
        logits, prefix_len, prefix_parity, target_parity,
        nmin, nmax, input_padding_mask=input_padding_mask)

    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(padding_mask[:, 1:], 0.0)
    return lp.sum(dim=1)


def compute_logq_all(model, target_parity, samples, device, nmin, nmax,
                     batch_size, need_candidates, nb_bonds):
    precomp = precompute_eval_tensors(samples, need_candidates, nb_bonds)
    if precomp is None:
        return np.array([])
    N = len(samples)
    results = np.zeros(N, dtype=np.float64)
    tk = precomp["tokens"]
    pm = precomp["padding_mask"]
    pp = precomp["prefix_parity"]
    pl = precomp["prefix_len"]
    dk = precomp["deltaK_prefix"]
    dkc = precomp.get("dk_candidates")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dkc_b = dkc[start:end] if dkc is not None else None
        lq = score_logq_batch(model, tk[start:end], pm[start:end], pp[start:end],
                              pl[start:end], dk[start:end], target_parity,
                              nmin, nmax, device, dk_candidates=dkc_b)
        results[start:end] = lq.cpu().numpy()
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="D₆ orbit consistency diagnostic for augmented models")
    ap.add_argument("--ckpt_even", required=True,
                    help="Even-parity model checkpoint (file or dir)")
    ap.add_argument("--ckpt_odd", required=True,
                    help="Odd-parity model checkpoint (file or dir)")
    ap.add_argument("--data_test", required=True,
                    help="Test .bin file")
    ap.add_argument("--max_N", type=int, default=None,
                    help="Max samples to use (last N from file)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--save_npz", type=str, default=None,
                    help="Save per-sample results to .npz file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    def pick_ckpt(path_or_dir):
        if os.path.isfile(path_or_dir):
            return path_or_dir
        best = os.path.join(path_or_dir, "best_model.pt")
        if os.path.isfile(best):
            return best
        raise FileNotFoundError(f"No checkpoint at {path_or_dir}")

    print("[model] Loading even model...")
    model_even, ckpt_even, nmin_e, nmax_e = load_model(pick_ckpt(args.ckpt_even), device)
    print("[model] Loading odd model...")
    model_odd, ckpt_odd, nmin_o, nmax_o = load_model(pick_ckpt(args.ckpt_odd), device)

    assert nmin_e == nmin_o and nmax_e == nmax_o, \
        f"Window mismatch: [{nmin_e},{nmax_e}] vs [{nmin_o},{nmax_o}]"
    nmin, nmax = nmin_e, nmax_e

    need_candidates = bool(getattr(model_even, 'dk_mlp_head', False)) or \
                      bool(getattr(model_odd, 'dk_mlp_head', False))

    # ------------------------------------------------------------------
    # Load data & build lattice
    # ------------------------------------------------------------------
    meta = load_v2_meta(args.data_test)
    lx, ly, nb = meta["lx"], meta["ly"], meta["nb"]
    print(f"\n[data] lx={lx}, ly={ly}, nb={nb}, beta={meta['beta']:.3f}")
    print(f"[data] Total samples in file: {meta['n']}")

    bsites, nn, nb_bonds = tps.build_bsites(lx, ly)
    assert nb == nb_bonds, f"nb mismatch: file={nb}, built={nb_bonds}"

    pg_perm = tps.build_pointgroup_bond_perm(lx, ly, bsites, nb)
    n_group = pg_perm.shape[0]
    print(f"[D₆] {n_group} group elements")

    print(f"\n[data] Reading test samples...")
    t0 = time.time()
    samples = read_raw_samples(meta, bsites, nn, nb, max_N=args.max_N,
                               need_candidates=need_candidates)
    N = len(samples)
    print(f"[data] {N} samples read in {time.time()-t0:.1f}s")

    # Filter to nh window BEFORE any computation
    n_before = len(samples)
    samples = [s for s in samples
               if (nmin is None or s["nh"] >= nmin) and
                  (nmax is None or s["nh"] <= nmax)]
    N = len(samples)
    n_out = n_before - N
    if n_out > 0:
        print(f"[data] Filtered to nh window [{nmin}, {nmax}]: {N} kept, {n_out} removed")

    even_idx = [i for i, s in enumerate(samples) if s["parity"] == 0]
    odd_idx = [i for i, s in enumerate(samples) if s["parity"] == 1]
    print(f"[data] Even: {len(even_idx)}, Odd: {len(odd_idx)}\n")

    # ------------------------------------------------------------------
    # Score all 12 D₆ images for every sample
    # ------------------------------------------------------------------
    logq_orbit = np.full((n_group, N), np.nan, dtype=np.float64)

    for g_idx in range(n_group):
        t1 = time.time()
        label = "e" if g_idx < 6 else "σ"
        k = g_idx if g_idx < 6 else g_idx - 6
        gname = f"C6^{k}" if g_idx < 6 else f"σ·C6^{k}"

        if g_idx == 0:
            g_samples = samples
        else:
            g_samples = apply_d6_to_samples(
                samples, pg_perm[g_idx], bsites, nn, nb,
                need_candidates=need_candidates)

        # Even-parity samples
        if even_idx:
            even_g = [g_samples[i] for i in even_idx]
            lq = compute_logq_all(model_even, 0, even_g, device, nmin, nmax,
                                  args.batch_size, need_candidates, nb_bonds)
            for j, idx in enumerate(even_idx):
                logq_orbit[g_idx, idx] = lq[j]

        # Odd-parity samples
        if odd_idx:
            odd_g = [g_samples[i] for i in odd_idx]
            lq = compute_logq_all(model_odd, 1, odd_g, device, nmin, nmax,
                                  args.batch_size, need_candidates, nb_bonds)
            for j, idx in enumerate(odd_idx):
                logq_orbit[g_idx, idx] = lq[j]

        dt = time.time() - t1
        finite = np.isfinite(logq_orbit[g_idx])
        print(f"  g={g_idx:2d} ({gname:8s})  finite={finite.sum():6d}/{N}  "
              f"logq=[{np.nanmin(logq_orbit[g_idx]):8.2f}, "
              f"{np.nanmax(logq_orbit[g_idx]):8.2f}]  {dt:.1f}s")

    # ------------------------------------------------------------------
    # Compute per-sample orbit statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("D₆ Orbit Consistency Results")
    print(f"{'='*70}")

    all_finite = np.all(np.isfinite(logq_orbit), axis=0)
    n_valid = int(all_finite.sum())
    print(f"\nSamples with all {n_group} logq finite: {n_valid}/{N}")

    if n_valid == 0:
        print("ERROR: no valid samples.")
        return

    logq_valid = logq_orbit[:, all_finite]  # (12, n_valid)
    nh_valid = np.array([samples[i]["nh"] for i in range(N)])[all_finite]
    K_valid = np.array([samples[i]["K"] for i in range(N)])[all_finite]
    parity_valid = np.array([samples[i]["parity"] for i in range(N)])[all_finite]

    orbit_std = np.std(logq_valid, axis=0, ddof=0)
    orbit_range = np.ptp(logq_valid, axis=0)
    orbit_mean = np.mean(logq_valid, axis=0)

    print(f"\n--- std_g[log q(g·x)] over {n_valid} samples ---")
    print(f"  mean   = {orbit_std.mean():.6f}")
    print(f"  median = {np.median(orbit_std):.6f}")
    for p in [5, 25, 75, 95, 99]:
        print(f"  p{p:<2d}    = {np.percentile(orbit_std, p):.6f}")
    print(f"  max    = {orbit_std.max():.6f}")

    print(f"\n--- max_g − min_g [log q(g·x)] ---")
    print(f"  mean   = {orbit_range.mean():.6f}")
    print(f"  median = {np.median(orbit_range):.6f}")
    print(f"  max    = {orbit_range.max():.6f}")

    # Break down by parity
    for par, par_name in [(0, "even"), (1, "odd")]:
        mask = (parity_valid == par)
        n_par = int(mask.sum())
        if n_par == 0:
            continue
        s = orbit_std[mask]
        print(f"\n  [{par_name}] n={n_par}  "
              f"mean_std={s.mean():.6f}  median_std={np.median(s):.6f}  "
              f"p95={np.percentile(s, 95):.6f}  max={s.max():.6f}")

    # Break down by nh
    nh_vals = sorted(set(nh_valid))
    print(f"\n--- Per-nh breakdown ---")
    print(f"  {'nh':>4s}  {'count':>6s}  {'mean_std':>10s}  {'med_std':>10s}  "
          f"{'p95_std':>10s}  {'max_std':>10s}  {'mean_range':>10s}")
    for nh in nh_vals:
        mask = (nh_valid == nh)
        n_nh = int(mask.sum())
        if n_nh == 0:
            continue
        s = orbit_std[mask]
        r = orbit_range[mask]
        print(f"  {nh:4d}  {n_nh:6d}  {s.mean():10.6f}  {np.median(s):10.6f}  "
              f"{np.percentile(s, 95):10.6f}  {s.max():10.6f}  {r.mean():10.6f}")

    # Potential gain from group-averaging
    logq_avg = np.log(np.mean(np.exp(logq_valid), axis=0))
    logq_id = logq_valid[0]  # identity element
    delta_avg = logq_avg - logq_id
    print(f"\n--- Group-average log q vs identity log q ---")
    print(f"  mean  Δ = log q_avg − log q_id = {delta_avg.mean():.6f}")
    print(f"  std   Δ = {delta_avg.std():.6f}")
    print(f"  median Δ = {np.median(delta_avg):.6f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.save_npz:
        np.savez(args.save_npz,
                 logq_orbit=logq_orbit,
                 orbit_std=orbit_std,
                 orbit_range=orbit_range,
                 orbit_mean=orbit_mean,
                 nh=np.array([s["nh"] for s in samples]),
                 K=np.array([s["K"] for s in samples]),
                 parity=np.array([s["parity"] for s in samples]),
                 all_finite=all_finite,
                 nmin=nmin, nmax=nmax, n_group=n_group)
        print(f"\n[saved] {args.save_npz}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    mean_std = orbit_std.mean()
    med_std = np.median(orbit_std)
    print(f"\n{'='*70}")
    if mean_std > 0.1:
        print(f"VERDICT: mean orbit std = {mean_std:.4f} (median = {med_std:.4f}) — LARGE")
        print("  → D₆ group-average scoring should give significant improvement.")
    elif mean_std > 0.01:
        print(f"VERDICT: mean orbit std = {mean_std:.4f} (median = {med_std:.4f}) — MODERATE")
        print("  → D₆ group-average scoring may give modest improvement.")
    else:
        print(f"VERDICT: mean orbit std = {mean_std:.4f} (median = {med_std:.4f}) — SMALL")
        print("  → Model is already near-symmetric; group-average has limited benefit.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
