#!/usr/bin/env python3
"""
Test: Fix nh, vary bond choices to get different K values.
Compare model log q(x) vs analytical log f(x) for each K.
"""
import sys, os, math, struct, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add paths
_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "src"))
sys.path.insert(0, os.path.join(_base, "..", "python", "train", "denumerator"))
from parity_prefix_wrapper import compute_parity_prefix

import train_transformer_parity_sign_v2_pe_nh_window_de_aug as tps


def build_bsites(lx, ly):
    """Same as in training scripts."""
    is_tri_pbc = (ly < 0)
    Lyabs = abs(ly)
    if is_tri_pbc:
        nn = lx * Lyabs
        nb = 3 * nn
        bsites = np.zeros((2, nb), dtype=np.int32, order='F')
        for y1 in range(Lyabs):
            for x1 in range(lx):
                s = 1 + x1 + y1 * lx
                x2 = (x1 + 1) % lx; y2 = y1
                bsites[0, s-1] = s; bsites[1, s-1] = 1 + x2 + y2*lx
                x2 = x1; y2 = (y1 + 1) % Lyabs
                bsites[0, s-1+nn] = s; bsites[1, s-1+nn] = 1 + x2 + y2*lx
                x2 = (x1 + 1) % lx; y2 = (y1 + 1) % Lyabs
                bsites[0, s-1+2*nn] = s; bsites[1, s-1+2*nn] = 1 + x2 + y2*lx
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


def compute_logf(nh, K, beta):
    return K * math.log(2.0) + nh * math.log(beta / 2.0) - math.lgamma(nh + 1)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = tps.AutoregressiveTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=args["d_model"],
        nhead=args["nhead"],
        num_layers=args["num_layers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=0.0,
        max_len=args["max_len"],
        K_max=args.get("K_max", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    nmin = ckpt.get("nmin")
    nmax = ckpt.get("nmax")
    beta = ckpt.get("beta", None)
    return model, ckpt, nmin, nmax, beta


@torch.no_grad()
def compute_logq_single(model, ops, parity_prefix, K_prefix, target_parity, nmin, nmax, nn_sites, device):
    """Compute log q(x) for a single opstring."""
    nh = len(ops)
    op_tokens = [op // 2 + tps.OPERATOR_OFFSET for op in ops]
    seq = [tps.BOS_ID] + op_tokens + [tps.EOS_ID]
    tokens = torch.tensor([seq], dtype=torch.long, device=device)
    padding_mask = torch.zeros(1, len(seq), dtype=torch.bool, device=device)

    # Build prefix_parity (input_seq_len = len(seq) - 1)
    input_seq_len = len(seq) - 1
    prefix_parity = torch.zeros(1, input_seq_len, dtype=torch.long, device=device)
    if nh > 0:
        pp_tensor = torch.from_numpy(parity_prefix.astype(np.int64))
        end = min(1 + nh, input_seq_len)
        prefix_parity[0, 1:end] = pp_tensor[:end - 1]
        if end < input_seq_len:
            prefix_parity[0, end:] = int(parity_prefix[-1])

    # Build prefix_len
    prefix_len = torch.zeros(1, input_seq_len, dtype=torch.long, device=device)
    if nh > 0:
        end = min(1 + nh, input_seq_len)
        prefix_len[0, 1:end] = torch.arange(1, end)
        if end < input_seq_len:
            prefix_len[0, end:] = nh

    # Build K_prefix
    K_prefix_tensor = torch.full((1, input_seq_len), nn_sites, dtype=torch.long, device=device)
    if nh > 0:
        kp_tensor = torch.from_numpy(K_prefix.astype(np.int64))
        end = min(1 + nh, input_seq_len)
        K_prefix_tensor[0, 1:end] = kp_tensor[:end - 1]
        if end < input_seq_len:
            K_prefix_tensor[0, end:] = int(K_prefix[-1])

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    input_pad = padding_mask[:, :-1]

    logits = model(inputs, input_pad, prefix_parity=prefix_parity, K_prefix=K_prefix_tensor)
    logits = tps.apply_token_mask(logits, input_pad)
    logits = tps.apply_nh_window_mask(logits, prefix_len, prefix_parity, target_parity,
                                       nmin, nmax, input_padding_mask=input_pad)

    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(padding_mask[:, 1:].to(device), 0.0)
    return lp.sum().item()


def generate_samples_fixed_nh(nh, nb, bsites, nn, n_samples, rng):
    """Generate random opstrings with fixed nh, compute their K and parity."""
    samples = []
    for _ in range(n_samples):
        # Random bond choices (uncolored: values = 2*bond_index)
        bonds = rng.integers(0, nb, size=nh)
        ops = (bonds * 2).astype(np.int32)
        pp, kp, K = compute_parity_prefix(ops, bsites, nn, nb)
        parity = int(pp[-1]) if nh > 0 else 0
        samples.append({
            "ops": ops,
            "nh": nh,
            "K": K,
            "parity": parity,
            "parity_prefix": pp,
            "K_prefix": kp,
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_even", required=True)
    ap.add_argument("--ckpt_odd", required=True)
    ap.add_argument("--nh", type=int, default=None, help="Fixed nh to test (auto-detect if not set)")
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load models
    print("Loading even model...")
    model_even, ckpt_even, nmin, nmax, beta = load_model(args.ckpt_even, device)
    print("Loading odd model...")
    model_odd, ckpt_odd, _, _, _ = load_model(args.ckpt_odd, device)

    lx = ckpt_even["lx"]
    ly = ckpt_even["ly"]
    nn_sites = ckpt_even["nn"]
    nb = ckpt_even["nb"]

    if beta is None:
        beta = float(input("Beta not in checkpoint. Enter beta: "))

    bsites, nn_b, nb_b = build_bsites(lx, ly)

    # Auto-detect nh if not specified: use midpoint of [nmin, nmax]
    nh = args.nh if args.nh is not None else (nmin + nmax) // 2
    print(f"Testing with fixed nh={nh}, nmin={nmin}, nmax={nmax}, beta={beta}")
    print(f"Lattice: {lx}x{ly}, nn={nn_sites}, nb={nb}")

    # Generate random samples
    rng = np.random.default_rng(42)
    print(f"Generating {args.n_samples} random opstrings with nh={nh}...")
    samples = generate_samples_fixed_nh(nh, nb, bsites, nn_sites, args.n_samples, rng)

    # Split by parity
    even_samples = [s for s in samples if s["parity"] == 0]
    odd_samples = [s for s in samples if s["parity"] == 1]
    print(f"  Even: {len(even_samples)}, Odd: {len(odd_samples)}")

    # Compute log q and log f for each sample
    results = {"even": [], "odd": []}

    for label, model, target_p, slist in [
        ("even", model_even, 0, even_samples),
        ("odd", model_odd, 1, odd_samples),
    ]:
        print(f"Computing log q for {label} ({len(slist)} samples)...")
        for i, s in enumerate(slist):
            logq = compute_logq_single(
                model, s["ops"], s["parity_prefix"], s["K_prefix"],
                target_p, nmin, nmax, nn_sites, device
            )
            logf = compute_logf(s["nh"], s["K"], beta)
            results[label].append({
                "K": s["K"],
                "logq": logq,
                "logf": logf,
                "logq_minus_logf": logq - logf,
            })
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(slist)}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, label in enumerate(["even", "odd"]):
        data = results[label]
        if len(data) == 0:
            continue
        Ks = np.array([d["K"] for d in data])
        logqs = np.array([d["logq"] for d in data])
        logfs = np.array([d["logf"] for d in data])
        residuals = np.array([d["logq_minus_logf"] for d in data])

        # Panel 1: log q vs log f, colored by K
        ax = axes[row, 0]
        sc = ax.scatter(logfs, logqs, c=Ks, s=5, alpha=0.5, cmap="viridis")
        mn, mx = min(logfs.min(), logqs.min()), max(logfs.max(), logqs.max())
        ax.plot([mn, mx], [mn, mx], "r--", alpha=0.5, label="y=x")
        ax.set_xlabel("log f(x)"); ax.set_ylabel("log q(x)")
        corr = np.corrcoef(logfs, logqs)[0, 1] if len(logfs) > 1 else 0
        ax.set_title(f"Parity={label}, nh={nh} (Corr={corr:.4f})")
        plt.colorbar(sc, ax=ax, label="K")
        ax.legend()

        # Panel 2: residual (log q - log f) vs K
        ax = axes[row, 1]
        ax.scatter(Ks, residuals, s=5, alpha=0.3, color="steelblue" if label == "even" else "orange")
        # Bin by K and show mean ± std
        unique_K = np.unique(Ks)
        K_means, K_stds = [], []
        for k in unique_K:
            mask = Ks == k
            K_means.append(residuals[mask].mean())
            K_stds.append(residuals[mask].std() if mask.sum() > 1 else 0)
        K_means, K_stds = np.array(K_means), np.array(K_stds)
        ax.errorbar(unique_K, K_means, yerr=K_stds, fmt="ro-", markersize=4, capsize=3, label="mean±std")
        ax.axhline(residuals.mean(), color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("K"); ax.set_ylabel("log q - log f")
        ax.set_title(f"Residual vs K (parity={label})")
        ax.legend()

        # Panel 3: histogram of residuals per K bin
        ax = axes[row, 2]
        # Show a few representative K values
        K_counts = {k: (Ks == k).sum() for k in unique_K}
        top_Ks = sorted(K_counts, key=K_counts.get, reverse=True)[:5]
        for k in sorted(top_Ks):
            mask = Ks == k
            ax.hist(residuals[mask], bins=30, alpha=0.5, label=f"K={k} (n={mask.sum()})")
        ax.set_xlabel("log q - log f"); ax.set_ylabel("Count")
        ax.set_title(f"Residual distribution by K (parity={label})")
        ax.legend(fontsize=8)

    fig.suptitle(f"Fixed nh={nh}, {lx}x{ly}, beta={beta}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
