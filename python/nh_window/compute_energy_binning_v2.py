#!/usr/bin/env python3
"""
Compute energy with binning analysis using n_h window models.
Numerator and denominator can have different window sizes.
"""
import os, sys, argparse, struct, glob, math
import numpy as np
import torch
import torch.nn.functional as F

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix as _compute_pp
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates as _compute_pp_cand

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numerator'))
import train_transformer_parity_sign_v2_pe_nh_window_aug as tps_num

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'denumerator'))
import train_transformer_parity_sign_v2_pe_nh_window_de_aug as tps_denom


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
    nmin = ckpt.get("nmin", None)
    nmax = ckpt.get("nmax", None)
    return model, nmin, nmax


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
    import math
    return np.array([math.lgamma(n + 1.0) for n in range(M + 1)], dtype=np.float64)


def compute_logf(nh, K, beta, logfact):
    import math
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
                          "sign": 1 if parity == 0 else -1, "_logf": compute_logf(nh, K, beta, logfact),
                          "parity_prefix": pp, "deltaK_prefix": dkp,
                          "deltaK_candidates": cand, "idx": len(samples)})
    return samples


@torch.no_grad()
def compute_logq_batch(model, tokens, padding_mask, prefix_parity, prefix_len, deltaK_prefix, target_parity,
                       nmin, nmax, device, tps_module, dk_candidates=None):
    """Compute per-sample log q(x) for a batch with n_h window masking."""
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


def compute_c_star(samples, logq_even, logq_odd, is_numerator=True):
    """Compute c_star from train data, handling window-out samples."""
    N = len(samples)
    if is_numerator:
        obs = np.array([s["nh"] * s["sign"] for s in samples])
    else:
        obs = np.array([s["sign"] for s in samples])
    s_arr = np.array([s["sign"] for s in samples])
    logf_arr = np.array([s["_logf"] for s in samples])
    logq = np.array([logq_even[i] if samples[i]["parity"]==0 else logq_odd[i] for i in range(N)])

    logw = logq - logf_arr
    finite = np.isfinite(logw)
    if finite.sum() == 0:
        raise ValueError("No finite samples for c* estimation")

    C = -np.median(logw[finite])
    h = np.exp(logq_even - logf_arr + C) - np.exp(logq_odd - logf_arr + C)

    h0 = obs - obs.mean()
    hc = h - h.mean()
    c_star = (h0 * hc).mean() / max((hc * hc).mean(), 1e-30)
    return c_star, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_train_num", required=True, help="Train data for numerator c_star")
    ap.add_argument("--data_train_denom", required=True, help="Train data for denominator c_star")
    ap.add_argument("--data_test", required=True, help="Test data for binning")
    ap.add_argument("--ckpt_num_even", required=True)
    ap.add_argument("--ckpt_num_odd", required=True)
    ap.add_argument("--ckpt_denom_even", required=True)
    ap.add_argument("--ckpt_denom_odd", required=True)
    ap.add_argument("--n_bins", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    meta = load_v2_meta(args.data_test)
    logfact = precompute_logfact(meta["mm"] + 10)

    print("Loading models...")
    model_num_even, nmin_num_e, nmax_num_e = load_model(pick_latest_checkpoint(args.ckpt_num_even), tps_num, device)
    model_num_odd, nmin_num_o, nmax_num_o = load_model(pick_latest_checkpoint(args.ckpt_num_odd), tps_num, device)
    model_denom_even, nmin_den_e, nmax_den_e = load_model(pick_latest_checkpoint(args.ckpt_denom_even), tps_denom, device)
    model_denom_odd, nmin_den_o, nmax_den_o = load_model(pick_latest_checkpoint(args.ckpt_denom_odd), tps_denom, device)

    if nmin_num_e != nmin_num_o or nmax_num_e != nmax_num_o:
        raise ValueError(f"Numerator window mismatch: even [{nmin_num_e},{nmax_num_e}] vs odd [{nmin_num_o},{nmax_num_o}]")
    if nmin_den_e != nmin_den_o or nmax_den_e != nmax_den_o:
        raise ValueError(f"Denominator window mismatch: even [{nmin_den_e},{nmax_den_e}] vs odd [{nmin_den_o},{nmax_den_o}]")

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

    # Step 1: Compute c_star from train data
    print(f"\n[TRAIN NUM] Computing c_star_num from {args.data_train_num}...")
    train_samples_num = read_samples(args.data_train_num, meta["beta"], logfact,
                                      need_candidates=need_num)
    print(f"  N_train_num = {len(train_samples_num)}")

    logq_num_even_train = compute_logq_for_samples(model_num_even, 0, train_samples_num, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)
    logq_num_odd_train = compute_logq_for_samples(model_num_odd, 1, train_samples_num, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)
    c_star_num, C_num = compute_c_star(train_samples_num, logq_num_even_train, logq_num_odd_train, is_numerator=True)
    print(f"  c_star_num = {c_star_num:.6f}, C_num = {C_num:.6f}")
    del train_samples_num, logq_num_even_train, logq_num_odd_train

    print(f"\n[TRAIN DENOM] Computing c_star_denom from {args.data_train_denom}...")
    train_samples_denom = read_samples(args.data_train_denom, meta["beta"], logfact,
                                        need_candidates=need_den)
    print(f"  N_train_denom = {len(train_samples_denom)}")

    logq_denom_even_train = compute_logq_for_samples(model_denom_even, 0, train_samples_denom, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)
    logq_denom_odd_train = compute_logq_for_samples(model_denom_odd, 1, train_samples_denom, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)
    c_star_denom, C_denom = compute_c_star(train_samples_denom, logq_denom_even_train, logq_denom_odd_train, is_numerator=False)
    print(f"  c_star_denom = {c_star_denom:.6f}, C_denom = {C_denom:.6f}")
    del train_samples_denom, logq_denom_even_train, logq_denom_odd_train

    # Step 2: Load test data and compute logq
    print(f"\n[TEST] Loading {args.data_test}...")
    test_samples = read_samples(args.data_test, meta["beta"], logfact,
                                 need_candidates=need_candidates)
    N = len(test_samples)
    print(f"  N_test = {N}")

    print("  Computing logq (numerator)...")
    logq_num_even = compute_logq_for_samples(model_num_even, 0, test_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)
    logq_num_odd = compute_logq_for_samples(model_num_odd, 1, test_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, need_candidates=need_num)

    print("  Computing logq (denominator)...")
    logq_denom_even = compute_logq_for_samples(model_denom_even, 0, test_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)
    logq_denom_odd = compute_logq_for_samples(model_denom_odd, 1, test_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, need_candidates=need_den)

    # Build arrays
    nhs_arr = np.array([s["nh"] * s["sign"] for s in test_samples])
    s_arr = np.array([s["sign"] for s in test_samples])
    logf_arr = np.array([s["_logf"] for s in test_samples])
    logq_num = np.array([logq_num_even[i] if test_samples[i]["parity"]==0 else logq_num_odd[i] for i in range(N)])
    logq_denom = np.array([logq_denom_even[i] if test_samples[i]["parity"]==0 else logq_denom_odd[i] for i in range(N)])

    # Apply CV using c_star from train: h = (q_even - q_odd) / f
    logw_num = logq_num - logf_arr
    logw_denom = logq_denom - logf_arr

    h_num = np.exp(logq_num_even - logf_arr + C_num) - np.exp(logq_num_odd - logf_arr + C_num)
    nhs_cv = nhs_arr - c_star_num * h_num

    h_denom = np.exp(logq_denom_even - logf_arr + C_denom) - np.exp(logq_denom_odd - logf_arr + C_denom)
    s_cv = s_arr - c_star_denom * h_denom

    # Diagnostic: per-component variance reduction
    finite_num = np.isfinite(logw_num)
    finite_denom = np.isfinite(logw_denom)
    print(f"\n{'='*60}")
    print(f"Component Diagnostics (N={N})")
    print(f"{'='*60}")
    print(f"Numerator <n_h*s>:")
    print(f"  Window-out samples: {(~finite_num).sum()}/{N}")
    print(f"  Raw:  mean={nhs_arr.mean():.6f}, var={nhs_arr.var():.4f}, SE={nhs_arr.std()/np.sqrt(N):.6f}")
    print(f"  CV:   mean={nhs_cv.mean():.6f}, var={nhs_cv.var():.4f}, SE={nhs_cv.std()/np.sqrt(N):.6f}")
    if nhs_cv.var() > 0:
        print(f"  Var reduction: {nhs_arr.var()/nhs_cv.var():.2f}x")
    print(f"Denominator <s>:")
    print(f"  Window-out samples: {(~finite_denom).sum()}/{N}")
    print(f"  Raw:  mean={s_arr.mean():.6f}, var={s_arr.var():.4f}, SE={s_arr.std()/np.sqrt(N):.6f}")
    print(f"  CV:   mean={s_cv.mean():.6f}, var={s_cv.var():.4f}, SE={s_cv.std()/np.sqrt(N):.6f}")
    if s_cv.var() > 0:
        print(f"  Var reduction: {s_arr.var()/s_cv.var():.2f}x")
    print(f"{'='*60}")

    # Binning analysis
    bin_size = N // args.n_bins
    E_bins_cv = []
    E_bins_raw = []

    for i in range(args.n_bins):
        start = i * bin_size
        end = start + bin_size

        # With CV
        nhs_bin_cv = nhs_cv[start:end].mean()
        s_bin_cv = s_cv[start:end].mean()
        if abs(s_bin_cv) > 1e-10:
            E_bin_cv = -nhs_bin_cv / (meta["beta"] * meta["nn"] * s_bin_cv) + meta["nb"] / (4.0 * meta["nn"])
            E_bins_cv.append(E_bin_cv)

        # No CV
        nhs_bin_raw = nhs_arr[start:end].mean()
        s_bin_raw = s_arr[start:end].mean()
        if abs(s_bin_raw) > 1e-10:
            E_bin_raw = -nhs_bin_raw / (meta["beta"] * meta["nn"] * s_bin_raw) + meta["nb"] / (4.0 * meta["nn"])
            E_bins_raw.append(E_bin_raw)

    E_bins_cv = np.array(E_bins_cv)
    E_bins_raw = np.array(E_bins_raw)

    E_mean_cv = E_bins_cv.mean()
    E_std_cv = E_bins_cv.std(ddof=1)
    E_se_cv = E_std_cv / np.sqrt(args.n_bins)

    E_mean_raw = E_bins_raw.mean()
    E_std_raw = E_bins_raw.std(ddof=1)
    E_se_raw = E_std_raw / np.sqrt(args.n_bins)

    print(f"\n{'='*60}")
    print(f"Binning Analysis (n_bins = {args.n_bins}, bin_size = {bin_size})")
    print(f"{'='*60}")
    print(f"No CV:")
    print(f"  E/N = {E_mean_raw:.10f} ± {E_se_raw:.10f}")
    print(f"")
    print(f"With CV:")
    print(f"  E/N = {E_mean_cv:.10f} ± {E_se_cv:.10f}")
    print(f"")
    print(f"Exact: -0.249999847")
    print(f"Error (no CV):  {abs(E_mean_raw - (-0.249999847)):.10f} ({abs(E_mean_raw - (-0.249999847))/E_se_raw:.2f} sigma)")
    print(f"Error (with CV): {abs(E_mean_cv - (-0.249999847)):.10f} ({abs(E_mean_cv - (-0.249999847))/E_se_cv:.2f} sigma)")
    print(f"")
    print(f"SE improvement: {E_se_raw / E_se_cv:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
