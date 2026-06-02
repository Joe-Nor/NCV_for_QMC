#!/usr/bin/env python3
"""
Diagnose h(x) = (q+ - q-)/f outliers.
Find which samples have extreme h values and what their nh, K, logw look like.
"""
import sys, os, struct, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "src"))
sys.path.insert(0, os.path.join(_base, "..", "python", "train", "denumerator"))
from parity_prefix_wrapper import compute_parity_prefix
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates
import train_transformer_parity_sign_v2_pe_nh_window_de_aug as tps


def build_bsites(lx, ly):
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
    nn = lx * ly; nb = 2 * nn
    bsites = np.zeros((2, nb), dtype=np.int32, order='F')
    for y1 in range(ly):
        for x1 in range(lx):
            s = 1 + x1 + y1 * lx
            x2 = (x1 + 1) % lx; y2 = y1
            bsites[0, s-1] = s; bsites[1, s-1] = 1 + x2 + y2*lx
            x2 = x1; y2 = (y1 + 1) % ly
            bsites[0, s-1+nn] = s; bsites[1, s-1+nn] = 1 + x2 + y2*lx
    return bsites, nn, nb


def load_v2_meta(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<i", f.read(4))[0]
        if magic == b"RSSE":
            fmt_version = 2
        elif magic == b"RSS3":
            fmt_version = 3
        elif magic == b"RSS4":
            fmt_version = 4
        else:
            raise ValueError(f"Unknown magic {magic!r}")
        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
        beta, _ = struct.unpack("<2d", f.read(16))
    # Scan offsets
    file_size = os.path.getsize(path)
    offsets, pos = [], 44
    with open(path, "rb") as f:
        while pos < file_size:
            offsets.append(pos)
            f.seek(pos)
            nh = struct.unpack("<i", f.read(4))[0]
            if fmt_version == 2:
                pos += 8 + nh + 4 * nh
            elif fmt_version == 4:
                pos += 8 + 4 * nh   # (nh, parity) + opstring
            else:
                pos += 12 + 4 * nh   # (nh, K, parity) + opstring
    return {"path": path, "nb": nb, "nn": nn, "beta": beta, "mm": mm,
            "lx": lx, "ly": ly, "offsets": offsets, "n": len(offsets), "fmt_version": fmt_version}


def read_samples(meta, beta, logfact, max_N=None, need_candidates=False):
    fmt_version = meta["fmt_version"]
    bsites, nn, nb = build_bsites(meta["lx"], meta["ly"])
    Ntot = meta["n"]
    N = min(max_N, Ntot) if max_N else Ntot
    start = Ntot - N

    def _compute(x):
        if need_candidates:
            return compute_parity_prefix_candidates(x, bsites, nn, nb)
        pp, dkp, K_out = compute_parity_prefix(x, bsites, nn, nb)
        return pp, dkp, K_out, None

    samples = []
    with open(meta["path"], "rb") as f:
        for idx in range(start, Ntot):
            f.seek(meta["offsets"][idx])
            if fmt_version == 4:
                nh, stored_parity = struct.unpack("<2i", f.read(8))
            else:
                nh, K_stored = struct.unpack("<2i", f.read(8))
            cand = None
            if nh > 0:
                if fmt_version == 2:
                    pp = np.frombuffer(f.read(nh), dtype="<i1").copy()
                    x = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                elif fmt_version == 4:
                    x = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                else:
                    stored_parity = struct.unpack("<i", f.read(4))[0]
                    x = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                pp, dkp, K, cand = _compute(x)
                parity = int(pp[-1])
            else:
                pp, x, parity, K = np.array([], np.int8), np.array([], np.int32), 0, nn
                dkp = np.array([], dtype=np.int32)
                if need_candidates:
                    cand = np.full((1, nb), -1.0, dtype=np.float32)
            logf = K * math.log(2.0) + nh * math.log(beta / 2.0) - logfact[nh] if nh > 0 else 0.0
            samples.append({
                "x_dense": x, "nh": nh, "K": K, "parity": parity,
                "sign": 1 if parity == 0 else -1, "_logf": logf,
                "parity_prefix": pp, "deltaK_prefix": dkp,
                "deltaK_candidates": cand, "idx": len(samples),
            })
    return samples, bsites, nn, nb


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    dk_mlp_head = bool(ckpt.get("dk_mlp_head", args.get("dk_mlp_head", 0)))
    dk_head_dk = int(ckpt.get("dk_head_dk", args.get("dk_head_dk", 32)))
    dk_head_hidden = int(ckpt.get("dk_head_hidden", args.get("dk_head_hidden", 256)))
    dk_head_centering = bool(args.get("dk_head_centering", False))
    dk_head_bond_emb = int(args.get("dk_head_bond_emb", 0))
    model = tps.AutoregressiveTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=args["d_model"], nhead=args["nhead"],
        num_layers=args["num_layers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=0.0, max_len=args["max_len"],
        dk_mlp_head=dk_mlp_head, dk_head_dk=dk_head_dk, dk_head_hidden=dk_head_hidden,
        dk_head_centering=dk_head_centering, dk_head_bond_emb=dk_head_bond_emb,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, ckpt.get("nmin"), ckpt.get("nmax")


@torch.no_grad()
def compute_logq_batch(model, batch, target_parity, nmin, nmax, device, bsites, nn_sites, nb_bonds):
    # For MLP-head checkpoints, samples already carry deltaK_candidates (computed in
    # read_samples). We pass compute_candidates=True so the collate builds the tensor.
    need_cand = bool(getattr(model, 'dk_mlp_head', False))
    collate_out = tps.collate_fn_parity_v2_aug(
        batch, bsites=None, nn_sites=nn_sites, nb_bonds=nb_bonds,
        augment=False, compute_candidates=need_cand,
    )
    tokens, padding_mask, _, prefix_parity, prefix_len, deltaK_prefix, dk_candidates, raw = collate_out

    tokens = tokens.to(device)
    padding_mask = padding_mask.to(device)
    prefix_parity = prefix_parity.to(device)
    prefix_len = prefix_len.to(device)
    deltaK_prefix = deltaK_prefix.to(device)
    if dk_candidates is not None:
        dk_candidates = dk_candidates.to(device)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    input_pad = padding_mask[:, :-1]

    logits = model(inputs, padding_mask=input_pad, prefix_parity=prefix_parity,
                   deltaK_prefix=deltaK_prefix, dk_candidates=dk_candidates)
    logits = tps.apply_token_mask(logits, input_pad)
    logits = tps.apply_nh_window_mask(logits, prefix_len, prefix_parity, target_parity,
                                       nmin, nmax, input_padding_mask=input_pad)
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(padding_mask[:, 1:].to(device), 0.0)
    return lp.sum(dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt_even", required=True)
    ap.add_argument("--ckpt_odd", required=True)
    ap.add_argument("--max_N", type=int, default=200000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load models
    print("Loading models...")
    model_even, nmin, nmax = load_model(args.ckpt_even, device)
    model_odd, _, _ = load_model(args.ckpt_odd, device)
    need_candidates = bool(getattr(model_even, 'dk_mlp_head', False)) or \
                      bool(getattr(model_odd, 'dk_mlp_head', False))
    if need_candidates:
        print("ΔK-candidates enabled (at least one model uses MLP head)")
    print(f"  nh window: [{nmin}, {nmax}]")

    # Load data
    meta = load_v2_meta(args.data)
    beta = meta["beta"]
    logfact = np.array([math.lgamma(n + 1.0) for n in range(300)], dtype=np.float64)
    print(f"Reading {args.max_N} samples...")
    samples, bsites, nn, nb = read_samples(meta, beta, logfact, args.max_N,
                                             need_candidates=need_candidates)
    N = len(samples)
    print(f"  N={N}, beta={beta}")

    # Compute logq for even and odd models
    logq_even = np.full(N, -np.inf)
    logq_odd = np.full(N, -np.inf)

    for label, model, target_p, arr in [("even", model_even, 0, logq_even), ("odd", model_odd, 1, logq_odd)]:
        print(f"Computing logq ({label})...")
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch = samples[start:end]
            lq = compute_logq_batch(model, batch, target_p, nmin, nmax, device, bsites, nn, nb)
            for i, s in enumerate(batch):
                arr[s["idx"]] = lq[i]

    # Build arrays
    nh_arr = np.array([s["nh"] for s in samples])
    K_arr = np.array([s["K"] for s in samples])
    sign_arr = np.array([s["sign"] for s in samples])
    logf_arr = np.array([s["_logf"] for s in samples])
    parity_arr = np.array([s["parity"] for s in samples])

    # logw for matched model
    logq_matched = np.where(parity_arr == 0, logq_even, logq_odd)
    logw = logq_matched - logf_arr

    # h = (q+ - q-) / f, with numerical stabilization
    max_logq = np.maximum(logq_even, logq_odd)
    C = -np.median(max_logq[np.isfinite(max_logq)])
    h_arr = np.exp(logq_even - logf_arr + C) - np.exp(logq_odd - logf_arr + C)

    print(f"\n=== Global statistics ===")
    print(f"logw: mean={logw[np.isfinite(logw)].mean():.4f}, std={logw[np.isfinite(logw)].std():.4f}")
    print(f"Corr(s, h) = {np.corrcoef(sign_arr, h_arr)[0,1]:.6f}")

    # Analyze by nh
    print(f"\n=== Per-nh statistics ===")
    print(f"{'nh':>4} {'count':>7} {'K_mean':>7} {'logw_std':>8} {'h_std':>12} {'Corr(s,h)':>10} {'|h|>10x_med':>12}")
    unique_nh = np.unique(nh_arr)
    nh_stats = []
    for nh_val in unique_nh:
        if nmin is not None and nh_val < nmin:
            continue
        if nmax is not None and nh_val > nmax:
            continue
        mask = nh_arr == nh_val
        n_count = mask.sum()
        if n_count < 10:
            continue
        h_sub = h_arr[mask]
        s_sub = sign_arr[mask]
        logw_sub = logw[mask]
        K_sub = K_arr[mask]

        logw_fin = logw_sub[np.isfinite(logw_sub)]
        if len(logw_fin) < 2:
            continue

        h_median = np.median(np.abs(h_sub))
        outlier_threshold = 10 * h_median if h_median > 0 else 1.0
        n_outliers = (np.abs(h_sub) > outlier_threshold).sum()

        corr_sh = np.corrcoef(s_sub, h_sub)[0, 1] if np.std(h_sub) > 0 and np.std(s_sub) > 0 else 0.0

        print(f"{nh_val:4d} {n_count:7d} {K_sub.mean():7.2f} {logw_fin.std():8.4f} "
              f"{np.std(h_sub):12.4e} {corr_sh:10.6f} {n_outliers:12d}")
        nh_stats.append((nh_val, n_count, logw_fin.std(), np.std(h_sub), corr_sh, n_outliers))

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. logw vs nh
    ax = axes[0, 0]
    finite = np.isfinite(logw)
    ax.scatter(nh_arr[finite], logw[finite], s=1, alpha=0.1, c=K_arr[finite], cmap="viridis")
    ax.set_xlabel("nh"); ax.set_ylabel("log(q/f)")
    ax.set_title("log(q/f) vs nh (colored by K)")

    # 2. logw vs K
    ax = axes[0, 1]
    ax.scatter(K_arr[finite], logw[finite], s=1, alpha=0.1, c=nh_arr[finite], cmap="plasma")
    ax.set_xlabel("K"); ax.set_ylabel("log(q/f)")
    ax.set_title("log(q/f) vs K (colored by nh)")

    # 3. std(logw) per nh
    ax = axes[0, 2]
    if nh_stats:
        nhs = [x[0] for x in nh_stats]
        stds = [x[2] for x in nh_stats]
        ax.plot(nhs, stds, "bo-", markersize=4)
    ax.set_xlabel("nh"); ax.set_ylabel("std(logw)")
    ax.set_title("std(log q/f) per nh")

    # 4. |h| distribution (log scale)
    ax = axes[1, 0]
    abs_h = np.abs(h_arr)
    abs_h_pos = abs_h[abs_h > 0]
    if len(abs_h_pos) > 0:
        ax.hist(np.log10(abs_h_pos), bins=80, alpha=0.7)
    ax.set_xlabel("log10(|h|)"); ax.set_ylabel("Count")
    ax.set_title("|h| distribution (log scale)")

    # 5. Corr(s,h) per nh
    ax = axes[1, 1]
    if nh_stats:
        nhs = [x[0] for x in nh_stats]
        corrs = [x[4] for x in nh_stats]
        ax.plot(nhs, corrs, "ro-", markersize=4)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("nh"); ax.set_ylabel("Corr(s, h)")
    ax.set_title("Corr(s, h) per nh")

    # 6. Cumulative variance contribution
    ax = axes[1, 2]
    sorted_idx = np.argsort(np.abs(h_arr))[::-1]
    cumvar = np.cumsum(h_arr[sorted_idx]**2) / np.sum(h_arr**2)
    frac = np.arange(1, len(cumvar)+1) / len(cumvar)
    ax.plot(frac * 100, cumvar * 100)
    ax.set_xlabel("% samples (sorted by |h|)")
    ax.set_ylabel("% of Var(h)")
    ax.set_title("Cumulative variance of h")
    ax.axvline(1, color="r", linestyle="--", alpha=0.5, label="top 1%")
    ax.axvline(10, color="orange", linestyle="--", alpha=0.5, label="top 10%")
    ax.legend()

    fig.suptitle(f"h(x) outlier diagnostics — {os.path.basename(args.data)}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
