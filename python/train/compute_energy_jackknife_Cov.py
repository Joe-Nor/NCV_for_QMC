#!/usr/bin/env python3
"""
Compute energy with jackknife error estimation using n_h window models.
Joint CV optimization for ratio estimation.
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


def ed_exact_energy(lx, ly, beta):
    """Compute exact E_std/N via full ED for small lattices (nn <= 12)."""
    # Build bonds
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
    dk_head_centering = bool(args.get("dk_head_centering", False))
    dk_head_bond_emb = int(args.get("dk_head_bond_emb", 0))
    model = tps_module.AutoregressiveTransformer(
        vocab_size=ckpt["vocab_size"], d_model=args["d_model"], nhead=args["nhead"],
        num_layers=args["num_layers"], dim_feedforward=args["dim_feedforward"],
        dropout=0.0, max_len=args["max_len"],
        dk_mlp_head=dk_mlp_head, dk_head_dk=dk_head_dk, dk_head_hidden=dk_head_hidden,
        dk_head_centering=dk_head_centering, dk_head_bond_emb=dk_head_bond_emb,
    )
    sd = {k: v for k, v in ckpt["model_state_dict"].items()
          if not k.startswith("delta_k_head.")}
    model.load_state_dict(sd, strict=True)
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


def precompute_eval_tensors(samples, need_candidates=False, nb_bonds=None, operator_offset=None):
    """Pre-compute all padded tensors once for evaluation.

    Avoids repeated Python-level padding and token conversion per batch,
    and skips redundant Fortran parity_prefix recomputation.

    If need_candidates=True, also builds a (N, input_seq_len, V) float32 tensor
    `dk_candidates` populated on the bond slice.
    """
    N = len(samples)
    max_nh = max(s["nh"] for s in samples)
    max_seq = max_nh + 2  # BOS + nh tokens + EOS
    input_seq_len = max_seq - 1

    # Pre-allocate
    tokens = np.full((N, max_seq), 0, dtype=np.int64)       # PAD=0
    padding_mask = np.ones((N, max_seq), dtype=np.bool_)     # True=pad
    prefix_parity = np.zeros((N, input_seq_len), dtype=np.int64)
    prefix_len = np.zeros((N, input_seq_len), dtype=np.int64)
    deltaK_prefix = np.ones((N, input_seq_len), dtype=np.int64)  # default index 1 (delta=0)

    dk_cand_tensor = None
    if need_candidates:
        assert nb_bonds is not None, "nb_bonds required when need_candidates=True"
        assert operator_offset is not None, "operator_offset required when need_candidates=True"
        V = nb_bonds + operator_offset
        dk_cand_tensor = np.zeros((N, input_seq_len, V), dtype=np.float32)

    for i, s in enumerate(samples):
        nh = s["nh"]
        seq_len = nh + 2  # BOS + nh + EOS

        # Tokens: BOS=1, ops (vectorized), EOS=2
        tokens[i, 0] = 1  # BOS
        if nh > 0:
            tokens[i, 1:1+nh] = s["x_dense"] // 2 + 2  # op_to_token vectorized
        tokens[i, 1+nh] = 2  # EOS
        padding_mask[i, :seq_len] = False

        # Prefix parity
        if nh > 0:
            pp = s["parity_prefix"]
            end = min(1 + nh, input_seq_len)
            prefix_parity[i, 1:end] = pp[:end-1]
            if end < input_seq_len:
                prefix_parity[i, end:] = int(pp[-1])

        # Prefix len
        if nh > 0:
            end = min(1 + nh, input_seq_len)
            prefix_len[i, 1:end] = np.arange(1, end)
            if end < input_seq_len:
                prefix_len[i, end:] = nh

        # DeltaK prefix
        if nh > 0:
            dkp = s["deltaK_prefix"]
            end = min(1 + nh, input_seq_len)
            deltaK_prefix[i, 1:end] = dkp[:end-1] + 1  # delta+1 → indices 0,1,2

        if need_candidates:
            cand = s.get("deltaK_candidates")
            if cand is not None:
                n_rows = min(cand.shape[0], input_seq_len)
                dk_cand_tensor[i, :n_rows,
                               operator_offset:operator_offset + nb_bonds] = cand[:n_rows]

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


def compute_logq_for_samples(model, target_parity, samples, nmin, nmax,
                             batch_size, device, tps_module, precomputed=None):
    """Compute log q(x) for all samples. Uses precomputed tensors if provided."""
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
        lq = compute_logq_batch(
            model, tk[start:end], pm[start:end], pp[start:end], pl[start:end],
            dk[start:end], target_parity, nmin, nmax, device, tps_module,
            dk_candidates=dkc_batch
        ).cpu().numpy()
        results[start:end] = lq

    return results


def compute_joint_c_star(samples, logq_num_even, logq_num_odd, logq_denom_even, logq_denom_odd,
                         h_clip_factor=0.0, R0_override=None):
    """
    Jointly optimize c_num and c_denom to minimize Var(X_cv) where X = A - R0*B.

    Returns: c_num, c_denom, C_num, C_denom, R0, med_num_per_nh, thr_denom
      - med_num_per_nh: median(|h_A/nh|) on train (None if no clip)
      - thr_denom: absolute threshold for h_B (None if no clip)
    """
    N = len(samples)
    nh_arr = np.array([s["nh"] for s in samples], dtype=np.float64)
    A = np.array([s["nh"] * s["sign"] for s in samples])
    B = np.array([s["sign"] for s in samples])
    s_arr = B.copy()
    logf_arr = np.array([s["_logf"] for s in samples])

    logq_num = np.array([logq_num_even[i] if samples[i]["parity"]==0 else logq_num_odd[i] for i in range(N)])
    logq_denom = np.array([logq_denom_even[i] if samples[i]["parity"]==0 else logq_denom_odd[i] for i in range(N)])

    logw_A = logq_num - logf_arr
    logw_B = logq_denom - logf_arr

    finite_A = np.isfinite(logw_A)
    finite_B = np.isfinite(logw_B)

    if finite_A.sum() == 0 or finite_B.sum() == 0:
        raise ValueError("No finite samples for joint c* estimation")

    C_num = -np.median(logw_A[finite_A])
    C_denom = -np.median(logw_B[finite_B])

    # h = (q_even - q_odd) / f
    h_A = np.exp(logq_num_even - logf_arr + C_num) - np.exp(logq_num_odd - logf_arr + C_num)
    h_B = np.exp(logq_denom_even - logf_arr + C_denom) - np.exp(logq_denom_odd - logf_arr + C_denom)

    # Zero non-finite h values
    for label, h in [("h_num", h_A), ("h_denom", h_B)]:
        bad = ~np.isfinite(h)
        if bad.any():
            h[bad] = 0.0
            print(f"  [train] {label}: {int(bad.sum())}/{N} non-finite zeroed")

    # h clip: zero out extreme samples before fitting c*
    med_num_per_nh, thr_denom = None, None
    if h_clip_factor > 0:
        # Numerator: |h_A| scales with nh, so normalize by nh before thresholding
        ok = (nh_arr > 0) & (h_A != 0)
        if ok.any():
            ratio = np.abs(h_A[ok]) / nh_arr[ok]
            med_num_per_nh = float(np.median(ratio))
            per_sample_thr = h_clip_factor * med_num_per_nh * nh_arr
            above = np.abs(h_A) > np.maximum(per_sample_thr, 1e-30)
            n_zeroed = int(above.sum())
            if n_zeroed > 0:
                h_A[above] = 0.0
            print(f"  [h clip train] h_num: factor={h_clip_factor}, "
                  f"median(|h/nh|)={med_num_per_nh:.6g}, zeroed={n_zeroed}/{N}")

        # Denominator: |h_B| does not scale with nh, use global threshold
        ok_d = h_B != 0
        if ok_d.any():
            med_d = float(np.median(np.abs(h_B[ok_d])))
            thr_denom = h_clip_factor * med_d
            above = np.abs(h_B) > thr_denom
            n_zeroed = int(above.sum())
            if n_zeroed > 0:
                h_B[above] = 0.0
            print(f"  [h clip train] h_denom: factor={h_clip_factor}, "
                  f"median(|h|)={med_d:.6g}, thr={thr_denom:.6g}, zeroed={n_zeroed}/{N}")

    if R0_override is not None:
        R0 = R0_override
        print(f"  Using R0 from test data: {R0:.6f}")
    else:
        R0 = A.mean() / B.mean()
    X = A - R0 * B

    u1 = h_A
    u2 = -R0 * h_B

    u1_c = u1 - u1.mean()
    u2_c = u2 - u2.mean()
    X_c = X - X.mean()

    Sigma = np.array([
        [(u1_c * u1_c).mean(), (u1_c * u2_c).mean()],
        [(u2_c * u1_c).mean(), (u2_c * u2_c).mean()]
    ])

    b = np.array([(u1_c * X_c).mean(), (u2_c * X_c).mean()])

    cond = np.linalg.cond(Sigma)
    if cond > 1e12:
        raise ValueError(f"Covariance matrix near-singular (cond={cond:.2e})")

    c = np.linalg.solve(Sigma, b)
    c_num, c_denom = c[0], c[1]

    # Diagnostics
    var_A = A.var()
    var_B = B.var()
    cov_AB = ((A - A.mean()) * (B - B.mean())).mean()
    var_X = X.var()

    X_cv = X - c_num * h_A + R0 * c_denom * h_B
    var_X_cv = X_cv.var()

    print(f"\n  Train diagnostics:")
    print(f"    Var(A) = {var_A:.6f}")
    print(f"    Var(B) = {var_B:.6f}")
    print(f"    Cov(A,B) = {cov_AB:.6f}")
    print(f"    Var(X) = Var(A - R0*B) = {var_X:.6f}")
    print(f"    Var(X_cv) = {var_X_cv:.6f}")
    print(f"    Var reduction in X direction: {var_X / var_X_cv:.2f}x")

    return c_num, c_denom, C_num, C_denom, R0, med_num_per_nh, thr_denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_train", required=True, help="Train data for joint c* estimation")
    ap.add_argument("--data_test", required=True, help="Test data for jackknife")
    ap.add_argument("--ckpt_num_even", required=True)
    ap.add_argument("--ckpt_num_odd", required=True)
    ap.add_argument("--ckpt_denom_even", required=True)
    ap.add_argument("--ckpt_denom_odd", required=True)
    ap.add_argument("--n_bins", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--h_clip_factor", type=float, default=0.0,
                    help="Zero out h for samples with |h| > factor * median(|h_train|)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load and verify metadata consistency
    meta_test = load_v2_meta(args.data_test)
    meta_train = load_v2_meta(args.data_train)

    for key in ["beta", "nn", "nb"]:
        if meta_test[key] != meta_train[key]:
            raise ValueError(f"Metadata mismatch: train {key}={meta_train[key]}, test {key}={meta_test[key]}")

    meta = meta_test
    logfact = precompute_logfact(meta["mm"] + 10)

    print("Loading models...")
    model_num_even, nmin_num_e, nmax_num_e = load_model(pick_latest_checkpoint(args.ckpt_num_even), tps_num, device)
    model_num_odd, nmin_num_o, nmax_num_o = load_model(pick_latest_checkpoint(args.ckpt_num_odd), tps_num, device)
    model_denom_even, nmin_den_e, nmax_den_e = load_model(pick_latest_checkpoint(args.ckpt_denom_even), tps_denom, device)
    model_denom_odd, nmin_den_o, nmax_den_o = load_model(pick_latest_checkpoint(args.ckpt_denom_odd), tps_denom, device)

    if nmin_num_e != nmin_num_o or nmax_num_e != nmax_num_o:
        raise ValueError(f"Numerator window mismatch")
    if nmin_den_e != nmin_den_o or nmax_den_e != nmax_den_o:
        raise ValueError(f"Denominator window mismatch")

    nmin_num, nmax_num = nmin_num_e, nmax_num_e
    nmin_den, nmax_den = nmin_den_e, nmax_den_e
    print(f"Numerator window: [{nmin_num}, {nmax_num}]")
    print(f"Denominator window: [{nmin_den}, {nmax_den}]")

    # Any model using the MLP ΔK head requires per-sample ΔK candidates.
    need_candidates = any(getattr(m, 'dk_mlp_head', False) for m in
                          [model_num_even, model_num_odd, model_denom_even, model_denom_odd])
    if need_candidates:
        print(f"ΔK-candidates enabled (at least one model uses MLP head)")
    nb_bonds_eval = meta['nb']
    # Either tps module works for OPERATOR_OFFSET (same constant in both).
    operator_offset = tps_num.OPERATOR_OFFSET

    import time as _time

    # Step 0: Read test data to compute R0 (more stable with larger sample)
    print(f"\n[TEST] Loading {args.data_test} for R0 estimation...")
    t0 = _time.time()
    test_samples = read_samples(args.data_test, meta["beta"], logfact,
                                 need_candidates=need_candidates)
    N = len(test_samples)
    print(f"  N_test = {N}  ({_time.time()-t0:.1f}s)")

    nhs_test_arr = np.array([s["nh"] * s["sign"] for s in test_samples])
    s_test_arr = np.array([s["sign"] for s in test_samples])
    R0_test = float(nhs_test_arr.mean() / s_test_arr.mean())
    print(f"  R0 from test: {R0_test:.6f}  (<nh*s>={nhs_test_arr.mean():.6f}, <s>={s_test_arr.mean():.6f})")

    # Step 1: Joint optimization on train data (using R0 from test)
    print(f"\n[TRAIN] Loading {args.data_train}...")
    t0 = _time.time()
    train_samples = read_samples(args.data_train, meta["beta"], logfact,
                                  need_candidates=need_candidates)
    print(f"  N_train = {len(train_samples)}  ({_time.time()-t0:.1f}s)")

    t0 = _time.time()
    train_precomputed = precompute_eval_tensors(train_samples,
                                                 need_candidates=need_candidates,
                                                 nb_bonds=nb_bonds_eval,
                                                 operator_offset=operator_offset)
    print(f"  Precompute tensors: {_time.time()-t0:.1f}s")

    print("  Computing logq (numerator)...")
    t0 = _time.time()
    logq_num_even_train = compute_logq_for_samples(model_num_even, 0, train_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, train_precomputed)
    logq_num_odd_train = compute_logq_for_samples(model_num_odd, 1, train_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, train_precomputed)
    print(f"    ({_time.time()-t0:.1f}s)")

    print("  Computing logq (denominator)...")
    t0 = _time.time()
    logq_denom_even_train = compute_logq_for_samples(model_denom_even, 0, train_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, train_precomputed)
    logq_denom_odd_train = compute_logq_for_samples(model_denom_odd, 1, train_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, train_precomputed)
    print(f"    ({_time.time()-t0:.1f}s)")

    print("  Joint optimization (R0 from test)...")
    c_star_num, c_star_denom, C_num, C_denom, R0_used, med_num_per_nh, thr_denom = compute_joint_c_star(
        train_samples, logq_num_even_train, logq_num_odd_train,
        logq_denom_even_train, logq_denom_odd_train,
        h_clip_factor=args.h_clip_factor,
        R0_override=R0_test
    )

    print(f"\n{'='*60}")
    print(f"Joint CV Optimization Results")
    print(f"{'='*60}")
    print(f"c_num = {c_star_num:.6f}, C_num = {C_num:.6f}")
    print(f"c_denom = {c_star_denom:.6f}, C_denom = {C_denom:.6f}")
    print(f"R0 (from test) = {R0_used:.6f}")
    print(f"{'='*60}")

    del train_samples, train_precomputed
    del logq_num_even_train, logq_num_odd_train, logq_denom_even_train, logq_denom_odd_train

    # Step 2: Apply fixed coefficients to test data (already loaded in step 0)
    print(f"\n[TEST] Computing logq on {N} test samples...")

    t0 = _time.time()
    test_precomputed = precompute_eval_tensors(test_samples,
                                                need_candidates=need_candidates,
                                                nb_bonds=nb_bonds_eval,
                                                operator_offset=operator_offset)
    print(f"  Precompute tensors: {_time.time()-t0:.1f}s")

    print("  Computing logq (numerator)...")
    t0 = _time.time()
    logq_num_even = compute_logq_for_samples(model_num_even, 0, test_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, test_precomputed)
    logq_num_odd = compute_logq_for_samples(model_num_odd, 1, test_samples, nmin_num, nmax_num, args.batch_size, device, tps_num, test_precomputed)
    print(f"    ({_time.time()-t0:.1f}s)")

    print("  Computing logq (denominator)...")
    t0 = _time.time()
    logq_denom_even = compute_logq_for_samples(model_denom_even, 0, test_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, test_precomputed)
    logq_denom_odd = compute_logq_for_samples(model_denom_odd, 1, test_samples, nmin_den, nmax_den, args.batch_size, device, tps_denom, test_precomputed)
    print(f"    ({_time.time()-t0:.1f}s)")

    del test_precomputed

    # Build arrays (reuse nhs_test_arr and s_test_arr from step 0)
    nhs_arr = nhs_test_arr
    s_arr = s_test_arr
    logf_arr = np.array([s["_logf"] for s in test_samples])
    logq_num = np.array([logq_num_even[i] if test_samples[i]["parity"]==0 else logq_num_odd[i] for i in range(N)])
    logq_denom = np.array([logq_denom_even[i] if test_samples[i]["parity"]==0 else logq_denom_odd[i] for i in range(N)])

    # Apply CV using h = (q_even - q_odd) / f
    h_num = np.exp(logq_num_even - logf_arr + C_num) - np.exp(logq_num_odd - logf_arr + C_num)
    h_denom = np.exp(logq_denom_even - logf_arr + C_denom) - np.exp(logq_denom_odd - logf_arr + C_denom)

    # Zero non-finite h values
    for label, h in [("h_num", h_num), ("h_denom", h_denom)]:
        bad = ~np.isfinite(h)
        if bad.any():
            h[bad] = 0.0
            print(f"  [test] {label}: {int(bad.sum())}/{N} non-finite zeroed")

    # Apply h clip thresholds from train
    if med_num_per_nh is not None:
        nh_test = np.array([s["nh"] for s in test_samples], dtype=np.float64)
        per_sample_thr = args.h_clip_factor * med_num_per_nh * nh_test
        above = np.abs(h_num) > np.maximum(per_sample_thr, 1e-30)
        n_zeroed = int(above.sum())
        if n_zeroed > 0:
            h_num[above] = 0.0
        print(f"  [h clip test] h_num (nh-scaled): factor={args.h_clip_factor}, "
              f"median(|h/nh|)={med_num_per_nh:.6g}, zeroed={n_zeroed}/{N}")
    if thr_denom is not None:
        above = np.abs(h_denom) > thr_denom
        n_zeroed = int(above.sum())
        if n_zeroed > 0:
            h_denom[above] = 0.0
        print(f"  [h clip test] h_denom: thr={thr_denom:.6g}, zeroed={n_zeroed}/{N}")

    nhs_cv = nhs_arr - c_star_num * h_num
    s_cv = s_arr - c_star_denom * h_denom

    # Component diagnostics
    logw_num = logq_num - logf_arr
    logw_denom = logq_denom - logf_arr
    finite_num = np.isfinite(logw_num)
    finite_denom = np.isfinite(logw_denom)
    print(f"\n{'='*60}")
    print(f"Component Diagnostics (N={N})")
    print(f"{'='*60}")
    print(f"Numerator <n_h*s>:")
    print(f"  Window-out: {(~finite_num).sum()}/{N}")
    print(f"  Raw:  mean={nhs_arr.mean():.6f}, var={nhs_arr.var():.4f}")
    print(f"  CV:   mean={nhs_cv.mean():.6f}, var={nhs_cv.var():.4f}")
    if nhs_cv.var() > 0:
        print(f"  Var reduction: {nhs_arr.var()/nhs_cv.var():.2f}x")
    print(f"Denominator <s>:")
    print(f"  Window-out: {(~finite_denom).sum()}/{N}")
    print(f"  Raw:  mean={s_arr.mean():.6f}, var={s_arr.var():.4f}")
    print(f"  CV:   mean={s_cv.mean():.6f}, var={s_cv.var():.4f}")
    if s_cv.var() > 0:
        print(f"  Var reduction: {s_arr.var()/s_cv.var():.2f}x")
    print(f"{'='*60}")

    # Jackknife analysis
    def jackknife_ratio_from_bins(A_bins, B_bins, beta, nn, nb, eps=1e-14):
        """Jackknife SE for ratio E = -A/(beta*nn*B) + nb/(4*nn). Leave-one-bin-out."""
        n_bins = len(A_bins)
        A_full = A_bins.mean()
        B_full = B_bins.mean()

        if abs(B_full) < eps:
            return np.nan, np.nan, np.nan, np.nan
        E_full = -A_full / (beta * nn * B_full) + nb / (4.0 * nn)

        E_loo = []
        for i in range(n_bins):
            A_loo = (n_bins * A_full - A_bins[i]) / (n_bins - 1)
            B_loo = (n_bins * B_full - B_bins[i]) / (n_bins - 1)
            if abs(B_loo) > eps:
                E_loo.append(-A_loo / (beta * nn * B_loo) + nb / (4.0 * nn))

        if len(E_loo) == 0:
            return E_full, np.nan, np.nan, np.nan

        E_loo = np.array(E_loo)
        n_loo = len(E_loo)
        E_jk_mean = E_loo.mean()
        E_se = np.sqrt((n_loo - 1) / n_loo * ((E_loo - E_jk_mean)**2).sum())

        # Jackknife bias-corrected central value:
        # removes O(1/N) ratio estimator bias
        E_bc = n_loo * E_full - (n_loo - 1) * E_jk_mean

        return E_full, E_jk_mean, E_bc, E_se

    bin_size = N // args.n_bins
    A_bins_raw = np.zeros(args.n_bins)
    B_bins_raw = np.zeros(args.n_bins)
    A_bins_cv = np.zeros(args.n_bins)
    B_bins_cv = np.zeros(args.n_bins)

    for i in range(args.n_bins):
        start = i * bin_size
        end = start + bin_size
        A_bins_raw[i] = nhs_arr[start:end].mean()
        B_bins_raw[i] = s_arr[start:end].mean()
        A_bins_cv[i] = nhs_cv[start:end].mean()
        B_bins_cv[i] = s_cv[start:end].mean()

    E_full_raw, E_jk_raw, E_bc_raw, E_se_raw = jackknife_ratio_from_bins(
        A_bins_raw, B_bins_raw, meta["beta"], meta["nn"], meta["nb"]
    )
    E_full_cv, E_jk_cv, E_bc_cv, E_se_cv = jackknife_ratio_from_bins(
        A_bins_cv, B_bins_cv, meta["beta"], meta["nn"], meta["nb"]
    )

    exact = ed_exact_energy(meta["lx"], meta["ly"], meta["beta"])
    print(f"\n{'='*60}")
    print(f"Jackknife Analysis (n_bins={args.n_bins}, bin_size={bin_size})")
    print(f"{'='*60}")
    print(f"No CV:")
    print(f"  E/N (full)          = {E_full_raw:.10f} ± {E_se_raw:.10f}")
    print(f"  E/N (bias-corrected)= {E_bc_raw:.10f} ± {E_se_raw:.10f}")
    print(f"")
    print(f"With CV:")
    print(f"  E/N (full)          = {E_full_cv:.10f} ± {E_se_cv:.10f}")
    print(f"  E/N (bias-corrected)= {E_bc_cv:.10f} ± {E_se_cv:.10f}")
    print(f"")
    if exact is not None:
        print(f"Exact (ED): {exact:.10f}")
        print(f"Error (no CV):   {abs(E_bc_raw - exact):.10f} ({abs(E_bc_raw - exact)/E_se_raw:.2f} sigma)")
        print(f"Error (with CV): {abs(E_bc_cv - exact):.10f} ({abs(E_bc_cv - exact)/E_se_cv:.2f} sigma)")
    else:
        print(f"Exact: not available (nn > 12)")
    print(f"")
    print(f"SE improvement: {E_se_raw / E_se_cv:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

