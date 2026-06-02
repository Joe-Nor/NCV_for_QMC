#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerator ⟨n_h * s⟩ CV estimation using n_h-weighted models with n_h window.

Computes E[n_h*s], Var[n_h*s], and variance reduction using CV g = s * q/f
where q is trained with n_h-weighted loss and n_h window masking.
"""

import os, glob, math, argparse, struct, sys as _sys
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
import train_transformer_parity_sign_v2_pe_nh_window_aug as tps

_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix as _compute_pp
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates as _compute_pp_cand


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


def pick_latest_checkpoint(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        return path_or_dir
    best = os.path.join(path_or_dir, "best_model.pt")
    if os.path.isfile(best):
        return best
    cands = glob.glob(os.path.join(path_or_dir, "*.pt")) + glob.glob(os.path.join(path_or_dir, "*.pth"))
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1] if cands else None


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model from checkpoint, reading architecture from saved args.

    Detects MLP ΔK-candidate head from ckpt metadata and reconstructs it if set.
    """
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
    sd = {k: v for k, v in ckpt["model_state_dict"].items()
          if not k.startswith("delta_k_head.")}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    nmin = ckpt.get("nmin", None)
    nmax = ckpt.get("nmax", None)
    print(f"  Loaded: d_model={saved_args['d_model']}, num_layers={saved_args['num_layers']}, "
          f"epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"  n_h window: [{nmin}, {nmax}]")
    if dk_mlp_head:
        print(f"  dk_mlp_head: d_k={dk_head_dk}, hidden={dk_head_hidden}, "
              f"centering={dk_head_centering}, bond_emb={dk_head_bond_emb}")
    return model, ckpt


def load_v2_meta(path: str) -> Dict:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic in (b"RSSE", b"RSS3", b"RSS4"), f"Unknown magic: {magic}"
        version = struct.unpack("<i", f.read(4))[0]
        assert version in (2, 3, 4), f"Only V2/V3/V4 supported, got {version}"
        fmt_version = version
        lx, ly, _, nb, mm = struct.unpack("<5i", f.read(20))
        beta = struct.unpack("<d", f.read(8))[0]
        f.read(8)
        offsets, pos = [], 44
        while pos < file_size:
            offsets.append(pos)
            f.seek(pos)
            nh = struct.unpack("<i", f.read(4))[0]
            if fmt_version == 2:
                pos += 8 + nh + 4 * nh
            elif fmt_version == 3:
                pos += 12 + 4 * nh  # nh(4) + K(4) + parity(4) + opstring(4*nh)
            else:  # V4
                pos += 8 + 4 * nh   # nh(4) + parity(4) + opstring(4*nh)
    return {"path": path, "nb": nb, "mm": mm, "beta": beta, "offsets": offsets,
            "n": len(offsets), "fmt_version": fmt_version, "lx": lx, "ly": ly}


def precompute_logfact(M: int) -> np.ndarray:
    return np.array([math.lgamma(n + 1.0) for n in range(M + 1)], dtype=np.float64)


def compute_logf(nh: int, K: int, beta: float, logfact: np.ndarray) -> float:
    return K * math.log(2.0) + nh * math.log(beta / 2.0) - logfact[nh]


@torch.no_grad()
def score_logq_batch(model, tokens, padding_mask, prefix_parity, prefix_len, deltaK_prefix, target_parity,
                     nmin, nmax, device, dk_candidates=None):
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
    logits = tps.apply_token_mask(logits, input_padding_mask)
    logits = tps.apply_nh_window_mask(
        logits, prefix_len, prefix_parity, target_parity,
        nmin, nmax, input_padding_mask=input_padding_mask
    )

    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(padding_mask[:, 1:], 0.0)

    return lp.sum(dim=1)


def read_samples(meta: Dict, beta: float, logfact: np.ndarray, max_N: int = None,
                 need_candidates: bool = False) -> List[Dict]:
    """Read samples from a V2 / V3 / V4 binary file.

    If need_candidates=True, also compute per-(position, bond) ΔK candidates
    for each sample (needed when any model has the MLP ΔK head).
    """
    global _mod_bsites, _mod_nn, _mod_nb
    fmt_version = meta["fmt_version"]
    Ntot = meta["n"]
    N = min(max_N, Ntot) if max_N else Ntot
    start = Ntot - N
    # Build bsites for parity_prefix + deltaK_prefix computation
    bsites, nn, nb = build_bsites(meta["lx"], meta["ly"])
    _mod_bsites, _mod_nn, _mod_nb = bsites, nn, nb

    def _compute(x):
        if need_candidates:
            pp, dkp, K_out, cand = _compute_pp_cand(x, bsites, nn, nb)
            return pp, dkp, K_out, cand
        pp, dkp, K_out = _compute_pp(x, bsites, nn, nb)
        return pp, dkp, K_out, None

    samples = []
    with open(meta["path"], "rb") as f:
        for idx in range(start, Ntot):
            f.seek(meta["offsets"][idx])
            if fmt_version == 4:
                nh, stored_parity = struct.unpack("<2i", f.read(8))
                K = None
            else:
                nh, K = struct.unpack("<2i", f.read(8))
            cand = None
            if nh > 0:
                if fmt_version == 2:
                    parity_prefix = np.frombuffer(f.read(nh), dtype="<i1").copy()
                    x_dense = np.frombuffer(f.read(4 * nh), dtype="<i4").copy()
                    parity_prefix, dkp, _k, cand = _compute(x_dense)
                elif fmt_version == 4:
                    x_dense = np.frombuffer(f.read(4 * nh), dtype="<i4").copy()
                    parity_prefix, dkp, _k, cand = _compute(x_dense)
                    K = int(_k)
                else:
                    stored_parity = struct.unpack("<i", f.read(4))[0]
                    x_dense = np.frombuffer(f.read(4 * nh), dtype="<i4").copy()
                    parity_prefix, dkp, _k, cand = _compute(x_dense)
                parity = int(parity_prefix[-1])
            else:
                parity_prefix, x_dense, parity = np.array([], np.int8), np.array([], np.int32), 0
                dkp = np.array([], dtype=np.int32)
                if K is None:
                    K = 0
                if need_candidates:
                    cand = np.full((1, nb), -1.0, dtype=np.float32)
            samples.append({
                "x_dense": x_dense, "nh": nh, "K": K, "parity": parity,
                "sign": 1 if parity == 0 else -1,
                "_logf": compute_logf(nh, K, beta, logfact),
                "parity_prefix": parity_prefix, "deltaK_prefix": dkp,
                "deltaK_candidates": cand, "idx": len(samples),
            })
    return samples


def precompute_eval_tensors(samples, need_candidates: bool = False, nb_bonds: int = None):
    """Pre-compute all padded tensors once for evaluation.

    Avoids repeated Python-level padding and token conversion per batch,
    and skips redundant Fortran parity_prefix recomputation.

    If need_candidates=True, also builds a (N, input_seq_len, V) float32 tensor
    `dk_candidates` populated on the bond slice [OPERATOR_OFFSET:OPERATOR_OFFSET+nb_bonds]
    for MLP-head inference.
    """
    N = len(samples)
    max_nh = max(s["nh"] for s in samples)
    max_seq = max_nh + 2  # BOS + nh tokens + EOS
    input_seq_len = max_seq - 1

    tokens = np.full((N, max_seq), 0, dtype=np.int64)       # PAD=0
    padding_mask = np.ones((N, max_seq), dtype=np.bool_)     # True=pad
    prefix_parity = np.zeros((N, input_seq_len), dtype=np.int64)
    prefix_len = np.zeros((N, input_seq_len), dtype=np.int64)
    deltaK_prefix = np.ones((N, input_seq_len), dtype=np.int64)  # default index 1 (delta=0)

    dk_cand_tensor = None
    if need_candidates:
        assert nb_bonds is not None, "nb_bonds required when need_candidates=True"
        V = nb_bonds + tps.OPERATOR_OFFSET
        dk_cand_tensor = np.zeros((N, input_seq_len, V), dtype=np.float32)

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

        if nh > 0:
            end = min(1 + nh, input_seq_len)
            prefix_len[i, 1:end] = np.arange(1, end)
            if end < input_seq_len:
                prefix_len[i, end:] = nh

        if nh > 0:
            dkp = s["deltaK_prefix"]
            end = min(1 + nh, input_seq_len)
            deltaK_prefix[i, 1:end] = dkp[:end-1] + 1  # delta+1 → indices 0,1,2

        if need_candidates:
            cand = s.get("deltaK_candidates")
            if cand is not None:
                n_rows = min(cand.shape[0], input_seq_len)
                dk_cand_tensor[i, :n_rows,
                               tps.OPERATOR_OFFSET:tps.OPERATOR_OFFSET + nb_bonds] = cand[:n_rows]

    out = {
        "tokens": torch.from_numpy(tokens),
        "padding_mask": torch.from_numpy(padding_mask),
        "prefix_parity": torch.from_numpy(prefix_parity),
        "prefix_len": torch.from_numpy(prefix_len),
        "deltaK_prefix": torch.from_numpy(deltaK_prefix),
    }
    if need_candidates:
        out["dk_candidates"] = torch.from_numpy(dk_cand_tensor)
    return out


def compute_logq_for_all(model, target_parity, samples, device, nmin, nmax, batch_size=256, precomputed=None):
    """Compute logq for all samples using the specified model."""
    if precomputed is None:
        precomputed = precompute_eval_tensors(samples)
    N = len(samples)
    results = np.zeros(N, dtype=np.float64)
    tk = precomputed["tokens"]
    pm = precomputed["padding_mask"]
    pp = precomputed["prefix_parity"]
    pl = precomputed["prefix_len"]
    dk = precomputed["deltaK_prefix"]
    dkc = precomputed.get("dk_candidates", None)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dkc_batch = dkc[start:end] if dkc is not None else None
        lq = score_logq_batch(model, tk[start:end], pm[start:end], pp[start:end], pl[start:end],
                              dk[start:end], target_parity, nmin, nmax, device,
                              dk_candidates=dkc_batch).cpu().numpy()
        results[start:end] = lq
    return results


def build_arrays(samples, logq_even_arr, logq_odd_arr):
    """Build numpy arrays from samples and logq results."""
    N = len(samples)
    nhs_arr = np.empty(N, np.float64)  # n_h * s
    s_arr = np.empty(N, np.float64)
    logf_arr = np.empty(N, np.float64)
    logq_arr = np.empty(N, np.float64)
    nh_arr = np.empty(N, np.int32)
    K_arr = np.empty(N, np.int32)

    for i, s in enumerate(samples):
        s_arr[i] = s["sign"]
        nhs_arr[i] = s["nh"] * s["sign"]
        logf_arr[i] = s["_logf"]
        nh_arr[i] = s["nh"]
        K_arr[i] = s["K"]
        logq_arr[i] = logq_even_arr[i] if s["parity"] == 0 else logq_odd_arr[i]

    logw_arr = logq_arr - logf_arr
    return nhs_arr, s_arr, logf_arr, logq_arr, logw_arr, logq_even_arr, logq_odd_arr, nh_arr, K_arr


def print_diagnostics(label, nhs_arr, s_arr, logq_arr, logf_arr, logw_arr,
                      logq_even_arr, logq_odd_arr, nh_arr, K_arr):
    """Print diagnostic statistics for a dataset, robust to -inf from nh-window masking."""
    N = len(s_arr)
    print(f"\n{'='*60}")
    print(f"[{label}] N={N}")
    print(f"  nh: min={nh_arr.min()}, max={nh_arr.max()}, mean={nh_arr.mean():.2f}, std={nh_arr.std():.2f}")
    print(f"  K:  min={K_arr.min()}, max={K_arr.max()}, mean={K_arr.mean():.2f}, std={K_arr.std():.2f}")
    print(f"  n_h*s: mean={nhs_arr.mean():.4f}, std={nhs_arr.std():.4f}")

    even_mask = (s_arr > 0)
    odd_mask = (s_arr < 0)
    n_even, n_odd = int(even_mask.sum()), int(odd_mask.sum())
    print(f"\n  n_even={n_even}, n_odd={n_odd}")

    # matched-q diagnostics: only samples whose matched logq is finite
    finite_matched = np.isfinite(logq_arr) & np.isfinite(logf_arr) & np.isfinite(logw_arr)
    print(f"  finite matched samples = {finite_matched.sum()}/{N}")

    if finite_matched.any():
        print(f"\n  logq: mean={logq_arr[finite_matched].mean():.4f}, std={logq_arr[finite_matched].std():.4f}")
        print(f"  logf: mean={logf_arr[finite_matched].mean():.4f}, std={logf_arr[finite_matched].std():.4f}")
        print(f"  logw: mean={logw_arr[finite_matched].mean():.4f}, std={logw_arr[finite_matched].std():.4f}")
    else:
        print("\n  logq/logf/logw: no finite matched samples")

    even_finite = even_mask & finite_matched
    odd_finite = odd_mask & finite_matched

    if even_finite.any():
        print(f"  logw|even: mean={logw_arr[even_finite].mean():.4f}, std={logw_arr[even_finite].std():.4f}")
    else:
        print("  logw|even: no finite samples")

    if odd_finite.any():
        print(f"  logw|odd:  mean={logw_arr[odd_finite].mean():.4f}, std={logw_arr[odd_finite].std():.4f}")
    else:
        print("  logw|odd:  no finite samples")

    # exp(-inf)=0 is mathematically fine here, so these can be computed on all samples
    w_raw = np.exp(logw_arr)
    Ew_even = float(w_raw[even_mask].mean()) if n_even > 0 else float("nan")
    Ew_odd = float(w_raw[odd_mask].mean()) if n_odd > 0 else float("nan")
    print(f"\n  E[w|even] = {Ew_even:.6e}")
    print(f"  E[w|odd]  = {Ew_odd:.6e}")
    if np.isfinite(Ew_even) and np.isfinite(Ew_odd) and Ew_odd != 0.0:
        print(f"  E[w|even]/E[w|odd] = {Ew_even / Ew_odd:.6f} (should be ~1.0)")
    else:
        print("  E[w|even]/E[w|odd] = nan/inf (ill-defined)")

    # h(x) = (q+ - q-) / f for numerator CV
    # With nh-window, some samples can have both logq_even = logq_odd = -inf.
    # Those must be excluded when choosing C_diff.
    max_logq = np.maximum(logq_even_arr, logq_odd_arr)
    valid_any = np.isfinite(max_logq) & np.isfinite(logf_arr)

    print(f"\n  h = (q+ - q-) / f:")
    print(f"    valid_any samples   = {valid_any.sum()}/{N}")
    print(f"    both q's = -inf     = {(~valid_any).sum()}/{N}")

    if valid_any.sum() >= 2:
        C_diff = -float(np.median(max_logq[valid_any]))
        h_arr = np.exp(logq_even_arr - logf_arr + C_diff) - np.exp(logq_odd_arr - logf_arr + C_diff)

        # all-sample rho: window-out samples contribute h=0
        if np.std(h_arr) > 0 and np.std(nhs_arr) > 0:
            rho_all = np.corrcoef(nhs_arr, h_arr)[0, 1]
            print(f"    Corr(n_h*s, h) [all]       = {rho_all:.6f}")
        else:
            print(f"    Corr(n_h*s, h) [all]       = nan")

        # valid_any rho: excludes samples with both q's = -inf
        h_valid = h_arr[valid_any]
        nhs_valid = nhs_arr[valid_any]
        if np.std(h_valid) > 0 and np.std(nhs_valid) > 0:
            rho_valid = np.corrcoef(nhs_valid, h_valid)[0, 1]
            print(f"    Corr(n_h*s, h) [valid_any] = {rho_valid:.6f}  (rho, key CV metric)")
        else:
            print(f"    Corr(n_h*s, h) [valid_any] = nan")
    else:
        print(f"    Corr(n_h*s, h) = nan (not enough valid samples)")


def main():
    ap = argparse.ArgumentParser(description="Mean sign CV with separate train/test files")
    ap.add_argument("--data_train", required=True, help="V2 .bin file for c_star estimation")
    ap.add_argument("--data_test", required=True, help="V2 .bin file for evaluation")
    ap.add_argument("--ckpt_even", required=True, help="Even model checkpoint (file or dir)")
    ap.add_argument("--ckpt_odd", required=True, help="Odd model checkpoint (file or dir)")
    ap.add_argument("--max_N_train", type=int, default=None, help="Max samples from train file")
    ap.add_argument("--max_N_test", type=int, default=None, help="Max samples from test file")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--h_clip_factor", type=float, default=0.0,
                    help="Zero out h for samples with |h/nh| > factor * median(|h/nh|_train)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load metadata for both files
    meta_train = load_v2_meta(args.data_train)
    meta_test = load_v2_meta(args.data_test)

    # Verify same lattice (nb) and beta
    assert meta_train["nb"] == meta_test["nb"], \
        f"nb mismatch: train={meta_train['nb']}, test={meta_test['nb']}"
    assert abs(meta_train["beta"] - meta_test["beta"]) < 1e-10, \
        f"beta mismatch: train={meta_train['beta']}, test={meta_test['beta']}"

    beta = meta_train["beta"]
    mm_max = max(meta_train["mm"], meta_test["mm"])


    # Load models
    print(f"\n[model] Loading even model...")
    model_even, ckpt_even = load_model_from_checkpoint(pick_latest_checkpoint(args.ckpt_even), device)
    print(f"[model] Loading odd model...")
    model_odd, ckpt_odd = load_model_from_checkpoint(pick_latest_checkpoint(args.ckpt_odd), device)

    # Extract n_h window parameters with strict consistency check
    nmin_even = ckpt_even.get("nmin", None)
    nmax_even = ckpt_even.get("nmax", None)

    nmin_odd = ckpt_odd.get("nmin", None)
    nmax_odd = ckpt_odd.get("nmax", None)

    if nmin_even != nmin_odd or nmax_even != nmax_odd:
        raise ValueError(
            f"Window mismatch: even [{nmin_even}, {nmax_even}] vs odd [{nmin_odd}, {nmax_odd}]"
        )

    nmin = nmin_even
    nmax = nmax_even
    print(f"[config] n_h window: [{nmin}, {nmax}]")

    # logfact large enough for both files
    logfact = precompute_logfact(mm_max)

    # ==================== Train file: compute c_star ====================
    need_candidates = bool(getattr(model_even, 'dk_mlp_head', False)) or \
                      bool(getattr(model_odd, 'dk_mlp_head', False))
    nb_bonds_eval = meta_train['nb']
    if need_candidates:
        print(f"[config] ΔK-candidates enabled (at least one model uses MLP head)")

    print(f"\n[train] Reading samples from {os.path.basename(args.data_train)}...")
    train_samples = read_samples(meta_train, beta, logfact, args.max_N_train,
                                  need_candidates=need_candidates)
    n_train = len(train_samples)
    print(f"[train] Read {n_train} samples, computing logq...")

    train_precomputed = precompute_eval_tensors(train_samples,
                                                 need_candidates=need_candidates,
                                                 nb_bonds=nb_bonds_eval)
    train_logq_even = compute_logq_for_all(model_even, 0, train_samples, device, nmin, nmax, args.batch_size, train_precomputed)
    print(f"[train] logq_even done")
    train_logq_odd = compute_logq_for_all(model_odd, 1, train_samples, device, nmin, nmax, args.batch_size, train_precomputed)
    print(f"[train] logq_odd done")

    nhs_train, s_train, logf_train, logq_train, logw_train, _, _, nh_train, K_train = \
        build_arrays(train_samples, train_logq_even, train_logq_odd)

    print_diagnostics("Train data", nhs_train, s_train, logq_train, logf_train, logw_train,
                      train_logq_even, train_logq_odd, nh_train, K_train)

    # Control variate: h = (q_even - q_odd) / f = exp(logq_even - logf + C) - exp(logq_odd - logf + C)
    # At nmax, both models give finite logq → h uses both.
    # For nh < nmax, opposite logq = -inf → exp(-inf) = 0 → h = s * q_matched / f.
    # This preserves E[h] = Z_even - Z_odd = 0 (unbiased).

    finite_train = np.isfinite(logw_train)
    n_train_total = len(logw_train)
    n_train_finite = int(finite_train.sum())
    print(f"\n[filter train] total={n_train_total}, finite={n_train_finite}, filtered={n_train_total - n_train_finite}")

    if n_train_finite == 0:
        raise ValueError("No finite samples in training set for c* estimation")

    C = -float(np.median(logw_train[finite_train]))

    h_train = np.exp(train_logq_even - logf_train + C) - np.exp(train_logq_odd - logf_train + C)

    # Zero non-finite h
    bad = ~np.isfinite(h_train)
    if bad.any():
        h_train[bad] = 0.0
        print(f"[h filter train] {int(bad.sum())}/{n_train} non-finite zeroed")

    # h clip: |h| scales with nh, so threshold on |h/nh|
    med_h_per_nh = None
    if args.h_clip_factor > 0:
        ok = (nh_train > 0) & (h_train != 0)
        if ok.any():
            ratio = np.abs(h_train[ok]) / nh_train[ok]
            med_h_per_nh = float(np.median(ratio))
            per_sample_thr = args.h_clip_factor * med_h_per_nh * nh_train
            above = np.abs(h_train) > np.maximum(per_sample_thr, 1e-30)
            n_zeroed = int(above.sum())
            if n_zeroed > 0:
                h_train[above] = 0.0
            print(f"[h clip train] factor={args.h_clip_factor}, "
                  f"median(|h/nh|)={med_h_per_nh:.6g}, zeroed={n_zeroed}/{n_train}")

    # c_star = Cov(n_h*s, h) / Var(h)
    obs0 = nhs_train - nhs_train.mean()
    h0 = h_train - h_train.mean()
    var_h = (h0 * h0).mean()
    c_star = float((obs0 * h0).mean() / max(var_h, 1e-30))

    if np.std(h_train) > 0 and np.std(nhs_train) > 0:
        corr_nhs_h_train = np.corrcoef(nhs_train, h_train)[0, 1]
    else:
        corr_nhs_h_train = float("nan")

    print(f"\n[c_star estimation]")
    print(f"  C = {C:.6f}")
    print(f"  c* = {c_star:.6f}")
    print(f"  Corr(n_h*s, h) on all train samples = {corr_nhs_h_train:.6f}")

    del train_samples, train_logq_even, train_logq_odd

    # ==================== Test file: evaluate ====================
    print(f"\n[test] Reading samples from {os.path.basename(args.data_test)}...")
    test_samples = read_samples(meta_test, beta, logfact, args.max_N_test,
                                 need_candidates=need_candidates)
    n_test = len(test_samples)
    print(f"[test] Read {n_test} samples, computing logq...")

    test_precomputed = precompute_eval_tensors(test_samples,
                                                need_candidates=need_candidates,
                                                nb_bonds=nb_bonds_eval)
    test_logq_even = compute_logq_for_all(model_even, 0, test_samples, device, nmin, nmax, args.batch_size, test_precomputed)
    print(f"[test] logq_even done")
    test_logq_odd = compute_logq_for_all(model_odd, 1, test_samples, device, nmin, nmax, args.batch_size, test_precomputed)
    print(f"[test] logq_odd done")

    nhs_test, s_test, logf_test, logq_test, logw_test, _, _, nh_test, K_test = \
        build_arrays(test_samples, test_logq_even, test_logq_odd)

    print_diagnostics("Test data", nhs_test, s_test, logq_test, logf_test, logw_test,
                      test_logq_even, test_logq_odd, nh_test, K_test)

    # h on test data using C from train
    h_test = np.exp(test_logq_even - logf_test + C) - np.exp(test_logq_odd - logf_test + C)

    # Zero non-finite h
    bad = ~np.isfinite(h_test)
    if bad.any():
        h_test[bad] = 0.0
        print(f"[h filter test] {int(bad.sum())}/{n_test} non-finite zeroed")

    # Apply nh-scaled h clip threshold from train
    if med_h_per_nh is not None:
        per_sample_thr = args.h_clip_factor * med_h_per_nh * nh_test
        above = np.abs(h_test) > np.maximum(per_sample_thr, 1e-30)
        n_zeroed = int(above.sum())
        if n_zeroed > 0:
            h_test[above] = 0.0
        print(f"[h clip test] factor={args.h_clip_factor}, "
              f"median(|h/nh|)={med_h_per_nh:.6g}, zeroed={n_zeroed}/{n_test}")

    nhs_cv_test = nhs_test - c_star * h_test

    # For diagnostics: count finite matched samples
    finite_test = np.isfinite(logw_test)
    n_test_total = len(logw_test)
    n_test_finite = int(finite_test.sum())
    print(f"\n[filter test] total={n_test_total}, finite={n_test_finite}, filtered={n_test_total - n_test_finite}")

    if np.std(h_test) > 0 and np.std(nhs_test) > 0:
        corr_nhs_h_test_all = np.corrcoef(nhs_test, h_test)[0, 1]
    else:
        corr_nhs_h_test_all = float("nan")

    mu_raw = float(nhs_test.mean())
    mu_cv = float(nhs_cv_test.mean())
    se_raw = float(nhs_test.std(ddof=1) / math.sqrt(n_test))
    se_cv = float(nhs_cv_test.std(ddof=1) / math.sqrt(n_test))
    var_raw = float(nhs_test.var(ddof=1))
    var_cv = float(nhs_cv_test.var(ddof=1))
    var_reduction = var_raw / var_cv if var_cv > 0 else float('inf')
    se_reduction = se_raw / se_cv if se_cv > 0 else float('inf')

    Eh_test = float(h_test.mean())
    SEh_test = float(h_test.std(ddof=1) / math.sqrt(n_test)) if n_test > 1 else float("nan")

    print(f"\n{'='*60}")
    print(f"[RESULTS]")
    print(f"  Train: {os.path.basename(args.data_train)} (n={n_train}, mm={meta_train['mm']})")
    print(f"  Test:  {os.path.basename(args.data_test)} (n={n_test}, mm={meta_test['mm']})")
    print(f"{'='*60}")
    print(f"  c_star = {c_star:.10g}")
    print(f"  C      = {C:.10g}")
    print(f"  Corr(n_h*s, h)_train [all]    = {corr_nhs_h_train:.6f}")
    print(f"  Corr(n_h*s, h)_test  [all]    = {corr_nhs_h_test_all:.6f}")
    print(f"  finite matched samples    = {n_test_finite}/{n_test}")

    print(f"\n  E[h]_test  = {Eh_test:.10g}")
    print(f"  SE[h]_test = {SEh_test:.6g}")

    print(f"\n  No CV:   E[n_h*s]      = {mu_raw:.10g}")
    print(f"           Var[n_h*s]    = {var_raw:.10g}")
    print(f"           SE[n_h*s]     = {se_raw:.6g}")

    print(f"\n  With CV: E[n_h*s - c*h]   = {mu_cv:.10g}")
    print(f"           Var[n_h*s - c*h] = {var_cv:.10g}")
    print(f"           SE[n_h*s - c*h]  = {se_cv:.6g}")

    print(f"\n  Variance reduction Var[n_h*s]/Var[n_h*s - c*h] = {var_reduction:.4f}")
    print(f"  SE reduction       SE[n_h*s]/SE[n_h*s - c*h]   = {se_reduction:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
