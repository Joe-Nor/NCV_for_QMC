#!/usr/bin/env python3
"""
Compute energy with jackknife error estimation using n_h window models.
Nonlinear c* optimization: directly minimize jackknife SE of the ratio estimator,
instead of the linear (delta method) approximation used in compute_energy_jackknife_Cov.py.

Comparison:
  - Linear (Cov):    min_{c1,c2} Var(A - R0*B - c1*h_A + c2*R0*h_B)
  - Nonlinear (this): min_{c1,c2} SE_JK[ (A_cv)/(B_cv) ]  where A_cv=A-c1*h_A, B_cv=B-c2*h_B
"""
import os, sys, argparse, struct, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix as _compute_pp
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates as _compute_pp_cand

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numerator'))
import train_transformer_parity_sign_v2_pe_nh_window_aug as tps_num

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'denumerator'))
import train_transformer_parity_sign_v2_pe_nh_window_de_aug as tps_denom


# ============================================================================
# Reuse infrastructure from compute_energy_jackknife_Cov.py
# ============================================================================

def ed_exact_energy(lx, ly, beta):
    """Compute exact E_std/N via full ED for small lattices (nn <= 12)."""
    if lx == 3 and ly == 1:
        bonds = [(0, 1), (1, 2), (2, 0)]
        nn = 3
    else:
        Lyabs = abs(ly)
        nn = lx * Lyabs
        raw = []
        for y1 in range(Lyabs):
            for x1 in range(lx):
                s = x1 + y1 * lx
                raw.append((s, (x1+1)%lx + y1*lx))
                raw.append((s, x1 + ((y1+1)%Lyabs)*lx))
                raw.append((s, (x1+1)%lx + ((y1+1)%Lyabs)*lx))
        seen = set()
        bonds = []
        for (i, j) in raw:
            edge = (min(i, j), max(i, j))
            if edge not in seen:
                seen.add(edge)
                bonds.append((i, j))
    if nn > 12:
        return None
    dim = 1 << nn
    H = np.zeros((dim, dim), dtype=np.float64)
    for (i, j) in bonds:
        for x in range(dim):
            si = 0.5 if ((x >> i) & 1) else -0.5
            sj = 0.5 if ((x >> j) & 1) else -0.5
            H[x, x] += si * sj
            if si != sj:
                y = x ^ ((1 << i) | (1 << j))
                H[x, y] += 0.5
    evals = np.linalg.eigvalsh(H)
    w = np.exp(-beta * evals)
    Z = w.sum()
    E = (w * evals).sum() / Z
    return E / nn


def pick_latest_checkpoint(path_or_dir):
    if os.path.isfile(path_or_dir):
        return path_or_dir
    best = os.path.join(path_or_dir, "best_model.pt")
    if os.path.isfile(best):
        return best
    cands = glob.glob(os.path.join(path_or_dir, "*.pt")) + glob.glob(os.path.join(path_or_dir, "*.pth"))
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1] if cands else None


def load_model(ckpt_path, tps_module, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    dk_mlp_head = bool(ckpt.get("dk_mlp_head", args.get("dk_mlp_head", 0)))
    dk_head_dk = int(ckpt.get("dk_head_dk", args.get("dk_head_dk", 32)))
    dk_head_hidden = int(ckpt.get("dk_head_hidden", args.get("dk_head_hidden", 256)))
    model = tps_module.AutoregressiveTransformer(
        vocab_size=ckpt["vocab_size"], d_model=args["d_model"], nhead=args["nhead"],
        num_layers=args["num_layers"], dim_feedforward=args["dim_feedforward"],
        dropout=0.0, max_len=args["max_len"],
        dk_mlp_head=dk_mlp_head, dk_head_dk=dk_head_dk, dk_head_hidden=dk_head_hidden,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt.get("nmin"), ckpt.get("nmax")


def load_v2_meta(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (b"RSSE", b"RSS3", b"RSS4"):
            raise ValueError(f"Unknown magic {magic!r}")
        version = struct.unpack("<i", f.read(4))[0]
        if version not in (2, 3, 4):
            raise ValueError(f"Unsupported version {version}")
        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
        beta, _ = struct.unpack("<2d", f.read(16))
    return {"nb": nb, "nn": nn, "beta": beta, "mm": mm, "lx": lx, "ly": ly, "fmt_version": version}


def build_bsites(lx, ly):
    """Build bond-to-site mapping, same logic as Fortran makelattice()."""
    if lx == 3 and ly == 1:
        nn, nb = 3, 3
        bsites = np.zeros((2, nb), dtype=np.int32, order='F')
        bsites[0, 0] = 1; bsites[1, 0] = 2
        bsites[0, 1] = 2; bsites[1, 1] = 3
        bsites[0, 2] = 3; bsites[1, 2] = 1
        return bsites, nn, nb
    is_tri_pbc = (ly < 0)
    Lyabs = abs(ly)
    if is_tri_pbc:
        nn = lx * Lyabs
        # Generate all directed bonds, then deduplicate
        raw = []
        for y1 in range(Lyabs):
            for x1 in range(lx):
                s = 1 + x1 + y1 * lx
                x2 = (x1 + 1) % lx; y2 = y1
                raw.append((s, 1 + x2 + y2*lx))
                x2 = x1; y2 = (y1 + 1) % Lyabs
                raw.append((s, 1 + x2 + y2*lx))
                x2 = (x1 + 1) % lx; y2 = (y1 + 1) % Lyabs
                raw.append((s, 1 + x2 + y2*lx))
        # Remove duplicate undirected edges (keeps first occurrence)
        seen = set()
        unique = []
        for (s1, s2) in raw:
            edge = (min(s1, s2), max(s1, s2))
            if edge not in seen:
                seen.add(edge)
                unique.append((s1, s2))
        nb = len(unique)
        bsites = np.zeros((2, nb), dtype=np.int32, order='F')
        for i, (s1, s2) in enumerate(unique):
            bsites[0, i] = s1
            bsites[1, i] = s2
        return bsites, nn, nb
    nn = lx * ly
    nb = 2 * nn
    bsites = np.zeros((2, nb), dtype=np.int32, order='F')
    for y1 in range(ly):
        for x1 in range(lx):
            s = 1 + x1 + y1 * lx
            x2 = (x1 + 1) % lx; y2 = y1
            bsites[0, s-1] = s; bsites[1, s-1] = 1 + x2 + y2*lx
            x2 = x1; y2 = (y1 + 1) % ly
            bsites[0, s-1+nn] = s; bsites[1, s-1+nn] = 1 + x2 + y2*lx
    return bsites, nn, nb


# Module-level lattice info, set by read_samples()
_mod_bsites = None
_mod_nn = None
_mod_nb = None


def precompute_logfact(M):
    return np.array([math.lgamma(n + 1.0) for n in range(M + 1)], dtype=np.float64)


def compute_logf(nh, K, beta, logfact):
    return K * math.log(2.0) + nh * math.log(beta / 2.0) - logfact[nh]


def read_samples(path, beta, logfact, max_N=None, need_candidates=False):
    global _mod_bsites, _mod_nn, _mod_nb
    file_size = os.path.getsize(path)
    # Detect format version from header
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"RSSE":
            fmt_version = 2
        elif magic == b"RSS3":
            fmt_version = 3
        elif magic == b"RSS4":
            fmt_version = 4
        else:
            raise ValueError(f"Unknown magic {magic!r}")
        version = struct.unpack("<i", f.read(4))[0]
        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))

    # Build bsites for parity_prefix + deltaK_prefix computation
    _bsites, _nn, _nb = build_bsites(lx, ly)
    _mod_bsites, _mod_nn, _mod_nb = _bsites, _nn, _nb

    def _compute(x):
        if need_candidates:
            return _compute_pp_cand(x, _bsites, _nn, _nb)
        pp, dkp, K_out = _compute_pp(x, _bsites, _nn, _nb)
        return pp, dkp, K_out, None

    offsets, pos = [], 44
    with open(path, "rb") as f:
        while pos < file_size:
            offsets.append(pos)
            f.seek(pos)
            nh = struct.unpack("<i", f.read(4))[0]
            if fmt_version == 2:
                pos += 8 + nh + 4 * nh
            elif fmt_version == 4:
                pos += 8 + 4 * nh  # nh(4) + parity(4) + opstring(4*nh)
            else:
                pos += 12 + 4 * nh  # nh(4) + K(4) + parity(4) + opstring(4*nh)
    Ntot = len(offsets)
    N = min(max_N, Ntot) if max_N else Ntot
    start = Ntot - N
    samples = []
    with open(path, "rb") as f:
        for idx in range(start, Ntot):
            f.seek(offsets[idx])
            cand = None
            if fmt_version == 4:
                nh, stored_parity = struct.unpack("<2i", f.read(8))
                if nh > 0:
                    x = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                    pp, dkp, K, cand = _compute(x)
                    parity = int(pp[-1])
                else:
                    pp, x, parity, K = np.array([], np.int8), np.array([], np.int32), 0, _nn
                    dkp = np.array([], dtype=np.int32)
                    if need_candidates:
                        cand = np.full((1, _nb), -1.0, dtype=np.float32)
            else:
                nh, K = struct.unpack("<2i", f.read(8))
                if nh > 0:
                    if fmt_version == 2:
                        pp = np.frombuffer(f.read(nh), dtype="<i1").copy()
                        x = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                        pp, dkp, K, cand = _compute(x)
                    else:
                        stored_parity = struct.unpack("<i", f.read(4))[0]
                        x = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                        pp, dkp, K, cand = _compute(x)
                    parity = int(pp[-1])
                else:
                    pp, x, parity = np.array([], np.int8), np.array([], np.int32), 0
                    dkp = np.array([], dtype=np.int32)
                    if need_candidates:
                        cand = np.full((1, _nb), -1.0, dtype=np.float32)
            samples.append({"x_dense": x, "nh": nh, "K": K, "parity": parity,
                          "sign": 1 if parity == 0 else -1,
                          "_logf": compute_logf(nh, K, beta, logfact),
                          "parity_prefix": pp, "deltaK_prefix": dkp,
                          "deltaK_candidates": cand, "idx": len(samples)})
    return samples


@torch.no_grad()
def compute_logq_batch(model, tokens, padding_mask, prefix_parity, prefix_len, deltaK_prefix,
                       target_parity, nmin, nmax, device, tps_module, dk_candidates=None):
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
    logits = tps_module.apply_token_mask(logits, input_padding_mask)
    logits = tps_module.apply_nh_window_mask(
        logits, prefix_len, prefix_parity, target_parity,
        nmin, nmax, input_padding_mask=input_padding_mask
    )
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(padding_mask[:, 1:], 0.0)
    return lp.sum(dim=1)


def compute_logq_for_samples(model, target_parity, samples, nmin, nmax,
                             batch_size, device, tps_module, need_candidates=False):
    results = np.zeros(len(samples), dtype=np.float64)
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        batch = samples[start:end]
        # bsites=None: skip recompute; samples already have the needed fields.
        out = tps_module.collate_fn_parity_v2_aug(
            batch, bsites=None, nn_sites=_mod_nn, nb_bonds=_mod_nb,
            augment=False, compute_candidates=need_candidates,
        )
        tokens, padding_mask, _, prefix_parity, prefix_len, deltaK_prefix, dk_candidates, raw = out
        lq = compute_logq_batch(model, tokens, padding_mask, prefix_parity, prefix_len,
                                deltaK_prefix, target_parity, nmin, nmax, device, tps_module,
                                dk_candidates=dk_candidates).cpu().numpy()
        for i, s in enumerate(raw):
            results[s["idx"]] = lq[i]
    return results


# ============================================================================
# Jackknife SE computation (used as objective for optimization)
# ============================================================================

def jackknife_se_from_bins(A_bins, B_bins, beta, nn, nb, eps=1e-14):
    """Compute jackknife SE for ratio E = -A/(beta*nn*B) + nb/(4*nn)."""
    n_bins = len(A_bins)
    A_full = A_bins.mean()
    B_full = B_bins.mean()

    if abs(B_full) < eps:
        return np.inf, np.nan, np.nan, np.nan

    E_full = -A_full / (beta * nn * B_full) + nb / (4.0 * nn)

    E_loo = []
    for i in range(n_bins):
        A_loo = (n_bins * A_full - A_bins[i]) / (n_bins - 1)
        B_loo = (n_bins * B_full - B_bins[i]) / (n_bins - 1)
        if abs(B_loo) > eps:
            E_loo.append(-A_loo / (beta * nn * B_loo) + nb / (4.0 * nn))

    # If too many bins were skipped, the result is unreliable
    if len(E_loo) < n_bins * 0.9:
        return np.inf, E_full, np.nan, np.nan

    E_loo = np.array(E_loo)
    # Check for non-finite values
    if not np.all(np.isfinite(E_loo)):
        return np.inf, E_full, np.nan, np.nan

    n_loo = len(E_loo)
    E_jk_mean = E_loo.mean()
    E_se = np.sqrt((n_loo - 1) / n_loo * ((E_loo - E_jk_mean)**2).sum())
    E_bc = n_loo * E_full - (n_loo - 1) * E_jk_mean

    return E_se, E_full, E_bc, E_jk_mean


# ============================================================================
# Core: nonlinear c* optimization
# ============================================================================

def optimize_c_nonlinear(nhs_arr, s_arr, h_num, h_denom, n_bins, beta, nn, nb,
                         c_linear_num, c_linear_denom):
    """
    Find (c1, c2) that minimize jackknife SE of the ratio estimator directly.

    Uses the linear-optimal (c1, c2) from the Cov method as starting point,
    then runs Nelder-Mead to search for better values.
    """
    N = len(nhs_arr)
    bin_size = N // n_bins

    # Precompute per-bin raw means (these don't change)
    A_raw_bins = np.zeros(n_bins)
    B_raw_bins = np.zeros(n_bins)
    hA_bins = np.zeros(n_bins)
    hB_bins = np.zeros(n_bins)
    for i in range(n_bins):
        sl = slice(i * bin_size, (i + 1) * bin_size)
        A_raw_bins[i] = nhs_arr[sl].mean()
        B_raw_bins[i] = s_arr[sl].mean()
        hA_bins[i] = h_num[sl].mean()
        hB_bins[i] = h_denom[sl].mean()

    # Compute scale for bounding: use std of h to set reasonable c range
    h_num_scale = max(np.std(h_num[np.isfinite(h_num)]), 1e-10)
    h_den_scale = max(np.std(h_denom[np.isfinite(h_denom)]), 1e-10)
    B_mean = B_raw_bins.mean()

    def objective(c):
        c1, c2 = c
        A_bins = A_raw_bins - c1 * hA_bins
        B_bins = B_raw_bins - c2 * hB_bins
        # Safety: reject if B_cv mean is too close to zero (< 10% of raw B mean)
        if abs(B_bins.mean()) < 0.1 * abs(B_mean):
            return np.inf
        se, _, _, _ = jackknife_se_from_bins(A_bins, B_bins, beta, nn, nb)
        return se

    # Starting point: linear-optimal from Cov method
    x0 = np.array([c_linear_num, c_linear_denom])
    se_linear = objective(x0)

    # If linear c* gives inf (e.g. B_cv too small), fall back to no-CV SE
    if not np.isfinite(se_linear):
        se_no_cv, _, _, _ = jackknife_se_from_bins(A_raw_bins, B_raw_bins, beta, nn, nb)
        se_linear = se_no_cv
        x0 = np.array([0.0, 0.0])

    print(f"\n  Nonlinear optimization...")
    print(f"  Starting from linear c*: c_num={x0[0]:.6f}, c_denom={x0[1]:.6f}, SE={se_linear:.10f}")

    # Multi-start: try several perturbations around x0
    best_se = se_linear
    best_c = x0.copy()

    # Bounded optimization using L-BFGS-B
    # Bound c within reasonable range: |c| < 10 * max(|c_linear|, 1)
    c_bound_num = max(abs(x0[0]) * 10, 10.0)
    c_bound_den = max(abs(x0[1]) * 10, 10.0)
    bounds = [(-c_bound_num, c_bound_num), (-c_bound_den, c_bound_den)]

    # Main optimization from linear starting point
    for method in ['Nelder-Mead', 'Powell']:
        result = minimize(objective, x0, method=method,
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-12})
        if np.isfinite(result.fun) and result.fun < best_se:
            # Verify result is within reasonable bounds
            if abs(result.x[0]) < c_bound_num and abs(result.x[1]) < c_bound_den:
                best_se = result.fun
                best_c = result.x.copy()

    # Also try grid search around the linear optimum
    scales = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    for s1 in scales:
        for s2 in scales:
            c_try = np.array([x0[0] * s1, x0[1] * s2])
            se_try = objective(c_try)
            if np.isfinite(se_try) and se_try < best_se:
                best_se = se_try
                best_c = c_try.copy()
            # Also run optimizer from this point
            if np.isfinite(se_try):
                result = minimize(objective, c_try, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12})
                if np.isfinite(result.fun) and result.fun < best_se:
                    if abs(result.x[0]) < c_bound_num and abs(result.x[1]) < c_bound_den:
                        best_se = result.fun
                        best_c = result.x.copy()

    # Try c_denom = 0 (only correct numerator)
    result_num_only = minimize(lambda c: objective([c[0], 0.0]), [x0[0]],
                               method='Nelder-Mead', options={'maxiter': 2000})
    if np.isfinite(result_num_only.fun) and result_num_only.fun < best_se:
        best_se = result_num_only.fun
        best_c = np.array([result_num_only.x[0], 0.0])

    # Try c_num = 0 (only correct denominator)
    result_denom_only = minimize(lambda c: objective([0.0, c[0]]), [x0[1]],
                                  method='Nelder-Mead', options={'maxiter': 2000})
    if np.isfinite(result_denom_only.fun) and result_denom_only.fun < best_se:
        best_se = result_denom_only.fun
        best_c = np.array([0.0, result_denom_only.x[0]])

    improvement = se_linear / best_se if best_se > 0 else float('inf')
    print(f"  Nonlinear optimal:      c_num={best_c[0]:.6f}, c_denom={best_c[1]:.6f}, SE={best_se:.10f}")
    print(f"  Improvement over linear: {improvement:.4f}x")

    return best_c[0], best_c[1], se_linear, best_se


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_train", required=True)
    ap.add_argument("--data_test", required=True)
    ap.add_argument("--ckpt_num_even", required=True)
    ap.add_argument("--ckpt_num_odd", required=True)
    ap.add_argument("--ckpt_denom_even", required=True)
    ap.add_argument("--ckpt_denom_odd", required=True)
    ap.add_argument("--n_bins", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    meta_test = load_v2_meta(args.data_test)
    meta_train = load_v2_meta(args.data_train)
    for key in ["beta", "nn", "nb"]:
        if meta_test[key] != meta_train[key]:
            raise ValueError(f"Metadata mismatch: {key}")
    meta = meta_test
    logfact = precompute_logfact(meta["mm"] + 10)

    # Load models
    print("Loading models...")
    model_num_even, nmin_num_e, nmax_num_e = load_model(pick_latest_checkpoint(args.ckpt_num_even), tps_num, device)
    model_num_odd, nmin_num_o, nmax_num_o = load_model(pick_latest_checkpoint(args.ckpt_num_odd), tps_num, device)
    model_denom_even, nmin_den_e, nmax_den_e = load_model(pick_latest_checkpoint(args.ckpt_denom_even), tps_denom, device)
    model_denom_odd, nmin_den_o, nmax_den_o = load_model(pick_latest_checkpoint(args.ckpt_denom_odd), tps_denom, device)

    assert nmin_num_e == nmin_num_o and nmax_num_e == nmax_num_o, "Numerator window mismatch"
    assert nmin_den_e == nmin_den_o and nmax_den_e == nmax_den_o, "Denominator window mismatch"
    nmin_num, nmax_num = nmin_num_e, nmax_num_e
    nmin_den, nmax_den = nmin_den_e, nmax_den_e
    print(f"Numerator window: [{nmin_num}, {nmax_num}]")
    print(f"Denominator window: [{nmin_den}, {nmax_den}]")

    need_num = bool(getattr(model_num_even, 'dk_mlp_head', False)) or \
               bool(getattr(model_num_odd, 'dk_mlp_head', False))
    need_den = bool(getattr(model_denom_even, 'dk_mlp_head', False)) or \
               bool(getattr(model_denom_odd, 'dk_mlp_head', False))
    need_candidates = need_num or need_den
    if need_candidates:
        print(f"ΔK-candidates enabled (at least one model uses MLP head)")

    # ===== TRAIN: compute linear c* (as baseline) =====
    print(f"\n[TRAIN] Loading {args.data_train}...")
    train_samples = read_samples(args.data_train, meta["beta"], logfact,
                                  need_candidates=need_candidates)
    N_train = len(train_samples)
    print(f"  N_train = {N_train}")

    print("  Computing logq (numerator)...")
    lq_ne_tr = compute_logq_for_samples(model_num_even, 0, train_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)
    lq_no_tr = compute_logq_for_samples(model_num_odd, 1, train_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)
    print("  Computing logq (denominator)...")
    lq_de_tr = compute_logq_for_samples(model_denom_even, 0, train_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)
    lq_do_tr = compute_logq_for_samples(model_denom_odd, 1, train_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)

    # Build train arrays
    A_tr = np.array([s["nh"] * s["sign"] for s in train_samples])
    B_tr = np.array([s["sign"] for s in train_samples])
    logf_tr = np.array([s["_logf"] for s in train_samples])
    logq_num_tr = np.array([lq_ne_tr[i] if train_samples[i]["parity"]==0 else lq_no_tr[i] for i in range(N_train)])
    logq_den_tr = np.array([lq_de_tr[i] if train_samples[i]["parity"]==0 else lq_do_tr[i] for i in range(N_train)])

    logw_A_tr = logq_num_tr - logf_tr
    logw_B_tr = logq_den_tr - logf_tr
    C_num = -np.median(logw_A_tr[np.isfinite(logw_A_tr)])
    C_denom = -np.median(logw_B_tr[np.isfinite(logw_B_tr)])

    h_num_tr = np.exp(lq_ne_tr - logf_tr + C_num) - np.exp(lq_no_tr - logf_tr + C_num)
    h_den_tr = np.exp(lq_de_tr - logf_tr + C_denom) - np.exp(lq_do_tr - logf_tr + C_denom)

    # Linear c* (Cov method baseline)
    R0 = A_tr.mean() / B_tr.mean()
    X_tr = A_tr - R0 * B_tr
    u1 = h_num_tr
    u2 = -R0 * h_den_tr
    u1c, u2c, Xc = u1 - u1.mean(), u2 - u2.mean(), X_tr - X_tr.mean()
    Sigma = np.array([[(u1c*u1c).mean(), (u1c*u2c).mean()],
                      [(u2c*u1c).mean(), (u2c*u2c).mean()]])
    bvec = np.array([(u1c*Xc).mean(), (u2c*Xc).mean()])
    c_lin = np.linalg.solve(Sigma, bvec)
    c_lin_num, c_lin_denom = c_lin[0], c_lin[1]

    print(f"\n  Linear c* (Cov method): c_num={c_lin_num:.6f}, c_denom={c_lin_denom:.6f}")
    print(f"  R0 = {R0:.6f}, C_num = {C_num:.6f}, C_denom = {C_denom:.6f}")

    # Nonlinear optimization on TRAIN data
    n_bins_opt = 100  # bins for optimization objective
    c_nl_num, c_nl_denom, se_lin_train, se_nl_train = optimize_c_nonlinear(
        A_tr, B_tr, h_num_tr, h_den_tr, n_bins_opt,
        meta["beta"], meta["nn"], meta["nb"], c_lin_num, c_lin_denom
    )

    del train_samples, lq_ne_tr, lq_no_tr, lq_de_tr, lq_do_tr
    del A_tr, B_tr, logf_tr, h_num_tr, h_den_tr

    # ===== TEST: apply both linear and nonlinear c* =====
    print(f"\n[TEST] Loading {args.data_test}...")
    test_samples = read_samples(args.data_test, meta["beta"], logfact,
                                 need_candidates=need_candidates)
    N = len(test_samples)
    print(f"  N_test = {N}")

    print("  Computing logq (numerator)...")
    lq_ne = compute_logq_for_samples(model_num_even, 0, test_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)
    lq_no = compute_logq_for_samples(model_num_odd, 1, test_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)
    print("  Computing logq (denominator)...")
    lq_de = compute_logq_for_samples(model_denom_even, 0, test_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)
    lq_do = compute_logq_for_samples(model_denom_odd, 1, test_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)

    nhs_arr = np.array([s["nh"] * s["sign"] for s in test_samples])
    s_arr = np.array([s["sign"] for s in test_samples])
    logf_arr = np.array([s["_logf"] for s in test_samples])

    h_num = np.exp(lq_ne - logf_arr + C_num) - np.exp(lq_no - logf_arr + C_num)
    h_denom = np.exp(lq_de - logf_arr + C_denom) - np.exp(lq_do - logf_arr + C_denom)

    # Build bins for all three methods
    bin_size = N // args.n_bins
    A_raw_bins = np.zeros(args.n_bins)
    B_raw_bins = np.zeros(args.n_bins)
    hA_bins = np.zeros(args.n_bins)
    hB_bins = np.zeros(args.n_bins)

    for i in range(args.n_bins):
        sl = slice(i * bin_size, (i + 1) * bin_size)
        A_raw_bins[i] = nhs_arr[sl].mean()
        B_raw_bins[i] = s_arr[sl].mean()
        hA_bins[i] = h_num[sl].mean()
        hB_bins[i] = h_denom[sl].mean()

    # Method 1: No CV
    se_raw, E_raw, E_bc_raw, _ = jackknife_se_from_bins(
        A_raw_bins, B_raw_bins, meta["beta"], meta["nn"], meta["nb"])

    # Method 2: Linear c* (Cov)
    A_lin = A_raw_bins - c_lin_num * hA_bins
    B_lin = B_raw_bins - c_lin_denom * hB_bins
    se_lin, E_lin, E_bc_lin, _ = jackknife_se_from_bins(
        A_lin, B_lin, meta["beta"], meta["nn"], meta["nb"])

    # Method 3: Nonlinear c*
    A_nl = A_raw_bins - c_nl_num * hA_bins
    B_nl = B_raw_bins - c_nl_denom * hB_bins
    se_nl, E_nl, E_bc_nl, _ = jackknife_se_from_bins(
        A_nl, B_nl, meta["beta"], meta["nn"], meta["nb"])

    exact = ed_exact_energy(meta["lx"], meta["ly"], meta["beta"])
    print(f"\n{'='*70}")
    print(f"Jackknife Analysis (n_bins={args.n_bins}, bin_size={bin_size}, N={N})")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'E/N (bc)':>14} {'SE':>14}", end="")
    if exact is not None:
        print(f" {'Error':>12} {'sigma':>8}", end="")
    print(f" {'SE impr':>8}")
    print(f"{'-'*70}")

    for label, E_bc, se in [
        ("No CV", E_bc_raw, se_raw),
        ("Linear c* (Cov)", E_bc_lin, se_lin),
        ("Nonlinear c* (this work)", E_bc_nl, se_nl),
    ]:
        impr = se_raw / se if se > 0 else float('inf')
        print(f"{label:<30} {E_bc:>14.10f} {se:>14.10f}", end="")
        if exact is not None:
            err = abs(E_bc - exact)
            sig = err / se if se > 0 else float('inf')
            print(f" {err:>12.10f} {sig:>8.2f}", end="")
        print(f" {impr:>8.2f}x")

    print(f"{'-'*70}")
    if exact is not None:
        print(f"Exact (ED): {exact:.10f}")
    else:
        print(f"Exact: not available (nn > 12)")
    print(f"\nLinear  c*: c_num={c_lin_num:.6f}, c_denom={c_lin_denom:.6f}")
    print(f"Nonlinear c*: c_num={c_nl_num:.6f}, c_denom={c_nl_denom:.6f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
