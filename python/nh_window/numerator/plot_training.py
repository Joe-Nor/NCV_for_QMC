#!/usr/bin/env python3
"""
Plot 2x3 figure: top row = loss curves, bottom row = log p(x) distributions.
Loads per-epoch checkpoints for loss, and best model + dataset for log p(x).
"""

import os, sys, glob, struct, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix as _compute_pp
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates as _compute_pp_cand
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import training module (selected by --pe flag in main)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
tps = None  # set in main()


def _read_nn_from_header(path):
    """Read nn (number of sites) from binary data file header."""
    with open(path, "rb") as f:
        magic = f.read(4)
        _ = struct.unpack("<i", f.read(4))[0]
        lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
    return nn


def load_loss_history(ckpt_dir, parity_name):
    """Load train/val loss from all per-epoch checkpoints."""
    pattern = os.path.join(ckpt_dir, f"model_{parity_name}_epoch*.pt")
    files = glob.glob(pattern)
    epochs, train_losses, val_losses = [], [], []
    for f in files:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        epochs.append(ckpt["epoch"])
        train_losses.append(ckpt["train_loss"])
        val_losses.append(ckpt["val_loss"])
    order = np.argsort(epochs)
    return (np.array(epochs)[order], np.array(train_losses)[order],
            np.array(val_losses)[order])


def load_model(ckpt_path, device, use_pe=False):
    """Load model from checkpoint."""
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
    sd = {k: v for k, v in ckpt["model_state_dict"].items()
          if not k.startswith("delta_k_head.")}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    nmin = ckpt.get("nmin", None)
    nmax = ckpt.get("nmax", None)
    return model, ckpt, nmin, nmax


@torch.no_grad()
def compute_logp_batch(model, tokens, padding_mask, prefix_parity, prefix_len, deltaK_prefix, target_parity,
                       nmin, nmax, device, use_pe=False, dk_candidates=None):
    """Compute per-sample log p(x) = sum of log p(x_t | x_{<t})."""
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
    if use_pe:
        logits = model(inputs, padding_mask=input_pad, prefix_parity=prefix_parity,
                       deltaK_prefix=deltaK_prefix, dk_candidates=dk_candidates)
    else:
        logits = model(inputs, padding_mask=input_pad,
                       deltaK_prefix=deltaK_prefix, dk_candidates=dk_candidates)
    logits = tps.apply_token_mask(logits, input_pad)
    logits = tps.apply_nh_window_mask(logits, prefix_len, prefix_parity, target_parity,
                                      nmin, nmax, input_padding_mask=input_pad)

    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(padding_mask[:, 1:].to(device), 0.0)

    return lp.sum(dim=1).cpu().numpy()


def precompute_eval_tensors(samples, need_candidates=False, nb_bonds=None):
    """Pre-compute all padded tensors once for evaluation."""
    N = len(samples)
    max_nh = max(s["nh"] for s in samples)
    max_seq = max_nh + 2
    input_seq_len = max_seq - 1

    tokens = np.full((N, max_seq), 0, dtype=np.int64)
    padding_mask = np.ones((N, max_seq), dtype=np.bool_)
    prefix_parity = np.zeros((N, input_seq_len), dtype=np.int64)
    prefix_len = np.zeros((N, input_seq_len), dtype=np.int64)
    deltaK_prefix = np.ones((N, input_seq_len), dtype=np.int64)

    dk_cand_tensor = None
    if need_candidates:
        assert nb_bonds is not None
        V = nb_bonds + tps.OPERATOR_OFFSET
        dk_cand_tensor = np.zeros((N, input_seq_len, V), dtype=np.float32)

    for i, s in enumerate(samples):
        nh = s["nh"]
        seq_len = nh + 2
        tokens[i, 0] = 1
        if nh > 0:
            tokens[i, 1:1+nh] = s["x_dense"] // 2 + 2
        tokens[i, 1+nh] = 2
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
            deltaK_prefix[i, 1:end] = dkp[:end-1] + 1
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


def compute_logp_for_dataset(model, target_parity, dataset, batch_size, device, nmin, nmax,
                              use_pe=False, nn_sites=None, bsites=None, nb_bonds=None):
    """Compute log p(x) for all samples in a streaming dataset."""
    # Collect all samples from the streaming dataset
    samples = list(dataset)
    if not samples:
        return np.array([], dtype=np.float64)
    need_candidates = bool(getattr(model, 'dk_mlp_head', False))
    # Compute parity_prefix/deltaK_prefix/deltaK_candidates for samples that don't have them (V3/V4)
    if bsites is not None:
        for s in samples:
            need_recompute = (s["nh"] > 0 and (len(s.get("parity_prefix", [])) == 0
                                               or "deltaK_prefix" not in s))
            need_cand_recompute = need_candidates and s.get("deltaK_candidates", None) is None
            if need_recompute or need_cand_recompute:
                if need_candidates:
                    pp, dkp, K, cand = _compute_pp_cand(s["x_dense"], bsites, nn_sites, nb_bonds)
                    s["deltaK_candidates"] = cand
                else:
                    pp, dkp, K = _compute_pp(s["x_dense"], bsites, nn_sites, nb_bonds)
                s["parity_prefix"] = pp
                s["deltaK_prefix"] = dkp
                s["K"] = K
                if need_recompute:
                    s["parity"] = int(pp[-1])
                    s["sign"] = +1 if s["parity"] == 0 else -1
            elif s["nh"] == 0:
                s.setdefault("deltaK_prefix", np.array([], dtype=np.int32))
                if need_candidates and s.get("deltaK_candidates", None) is None:
                    s["deltaK_candidates"] = np.full((1, nb_bonds), -1.0, dtype=np.float32)
    precomputed = precompute_eval_tensors(samples,
                                           need_candidates=need_candidates,
                                           nb_bonds=nb_bonds)
    N = len(samples)
    tk = precomputed["tokens"]
    pm = precomputed["padding_mask"]
    pp = precomputed["prefix_parity"]
    pl = precomputed["prefix_len"]
    dk = precomputed["deltaK_prefix"]
    dkc = precomputed.get("dk_candidates", None)
    all_logp = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dkc_batch = dkc[start:end] if dkc is not None else None
        lp = compute_logp_batch(model, tk[start:end], pm[start:end], pp[start:end], pl[start:end],
                                dk[start:end], target_parity, nmin, nmax, device,
                                use_pe=use_pe, dk_candidates=dkc_batch)
        all_logp.append(lp)
    return np.concatenate(all_logp)


def main():
    global tps
    ap = argparse.ArgumentParser()
    ap.add_argument("--even_dir", required=True, help="Even checkpoint directory")
    ap.add_argument("--odd_dir", required=True, help="Odd checkpoint directory")
    ap.add_argument("--data", required=True, help="Data file for log p(x) computation")
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--title", default=None, help="Figure suptitle")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_first_n", type=int, default=1, help="Skip first N epochs in loss plot")
    ap.add_argument("--pe", action="store_true", help="Use PE model variant")
    ap.add_argument("--paper", action="store_true", help="Publication style: larger fonts, PDF output")
    args = ap.parse_args()

    import train_transformer_parity_sign_v2_pe_nh_window_aug as _tps
    tps = _tps

    device = torch.device(args.device)

    # --- Load loss history ---
    ep_even, tl_even, vl_even = load_loss_history(args.even_dir, "even")
    ep_odd, tl_odd, vl_odd = load_loss_history(args.odd_dir, "odd")

    skip = args.skip_first_n
    mask_e = ep_even > skip
    mask_o = ep_odd > skip
    ep_even, tl_even, vl_even = ep_even[mask_e], tl_even[mask_e], vl_even[mask_e]
    ep_odd, tl_odd, vl_odd = ep_odd[mask_o], tl_odd[mask_o], vl_odd[mask_o]

    best_ep_even = ep_even[np.argmin(vl_even)]
    best_vl_even = vl_even.min()
    best_ep_odd = ep_odd[np.argmin(vl_odd)]
    best_vl_odd = vl_odd.min()

    # --- Load best models and compute log p(x) ---
    print("Loading even model...")
    model_even, ckpt_even, nmin, nmax = load_model(os.path.join(args.even_dir, "best_model.pt"), device, use_pe=True)
    print("Loading odd model...")
    model_odd, ckpt_odd, _, _ = load_model(os.path.join(args.odd_dir, "best_model.pt"), device, use_pe=True)

    data_files = [args.data]
    train_frac = ckpt_even["args"].get("train_fraction", 0.8)
    nn_sites = _read_nn_from_header(args.data)

    # Build bsites for V3 parity_prefix computation
    bsites_arr, nn_b, nb_b = tps.build_bsites(ckpt_even["lx"], ckpt_even["ly"])

    print("Computing log p(x) for even train...")
    ds_even_train = tps.RSSEStreamingDatasetV2(data_files, target_parity=0, split="train",
                                                train_fraction=train_frac, shuffle_buffer=0)
    logp_even_train = compute_logp_for_dataset(model_even, 0, ds_even_train, args.batch_size, device, nmin, nmax, use_pe=True, nn_sites=nn_sites, bsites=bsites_arr, nb_bonds=nb_b)

    print("Computing log p(x) for even val...")
    ds_even_val = tps.RSSEStreamingDatasetV2(data_files, target_parity=0, split="val",
                                              train_fraction=train_frac, shuffle_buffer=0)
    logp_even_val = compute_logp_for_dataset(model_even, 0, ds_even_val, args.batch_size, device, nmin, nmax, use_pe=True, nn_sites=nn_sites, bsites=bsites_arr, nb_bonds=nb_b)

    print("Computing log p(x) for odd train...")
    ds_odd_train = tps.RSSEStreamingDatasetV2(data_files, target_parity=1, split="train",
                                               train_fraction=train_frac, shuffle_buffer=0)
    logp_odd_train = compute_logp_for_dataset(model_odd, 1, ds_odd_train, args.batch_size, device, nmin, nmax, use_pe=True, nn_sites=nn_sites, bsites=bsites_arr, nb_bonds=nb_b)

    print("Computing log p(x) for odd val...")
    ds_odd_val = tps.RSSEStreamingDatasetV2(data_files, target_parity=1, split="val",
                                             train_fraction=train_frac, shuffle_buffer=0)
    logp_odd_val = compute_logp_for_dataset(model_odd, 1, ds_odd_val, args.batch_size, device, nmin, nmax, use_pe=True, nn_sites=nn_sites, bsites=bsites_arr, nb_bonds=nb_b)

    del model_even, model_odd
    torch.cuda.empty_cache()

    # --- Plot ---
    paper = args.paper
    fs_label = 16 if paper else 12
    fs_tick = 13 if paper else 10
    fs_legend = 13 if paper else 9
    fs_title = 15 if paper else 12
    fs_annot = 12 if paper else 8

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # ---- Top row: loss curves ----
    ax = axes[0, 0]
    ax.plot(ep_even, tl_even, "g-o", markersize=3, label="Train")
    ax.plot(ep_even, vl_even, "r-o", markersize=3, label=f"Val (best ep {best_ep_even})")
    ax.axvline(best_ep_even, color="r", linestyle="--", alpha=0.5)
    ax.plot(best_ep_even, best_vl_even, "ro", markersize=8)
    ax.set_xlabel("Epoch", fontsize=fs_label); ax.set_ylabel("Loss (per token)", fontsize=fs_label)
    ax.set_title("Even Model", fontsize=fs_title); ax.legend(fontsize=fs_legend)
    ax.tick_params(labelsize=fs_tick)

    ax = axes[0, 1]
    ax.plot(ep_odd, tl_odd, "g-o", markersize=3, label="Train")
    ax.plot(ep_odd, vl_odd, "r-o", markersize=3, label=f"Val (best ep {best_ep_odd})")
    ax.axvline(best_ep_odd, color="r", linestyle="--", alpha=0.5)
    ax.plot(best_ep_odd, best_vl_odd, "ro", markersize=8)
    ax.set_xlabel("Epoch", fontsize=fs_label); ax.set_ylabel("Loss (per token)", fontsize=fs_label)
    ax.set_title("Odd Model", fontsize=fs_title); ax.legend(fontsize=fs_legend)
    ax.tick_params(labelsize=fs_tick)

    ax = axes[0, 2]
    ax.plot(ep_even, vl_even, "b-o", markersize=3, label=f"Even (best ep {best_ep_even})")
    ax.plot(ep_odd, vl_odd, color="orange", marker="o", markersize=3, label=f"Odd (best ep {best_ep_odd})")
    ax.plot(best_ep_even, best_vl_even, "bo", markersize=8)
    ax.plot(best_ep_odd, best_vl_odd, "o", color="orange", markersize=8)
    ax.annotate(f"{best_vl_even:.6f}", (best_ep_even, best_vl_even),
                textcoords="offset points", xytext=(-10, -15), fontsize=fs_annot, color="blue")
    ax.annotate(f"{best_vl_odd:.6f}", (best_ep_odd, best_vl_odd),
                textcoords="offset points", xytext=(5, 10), fontsize=fs_annot, color="orange")
    ax.set_xlabel("Epoch", fontsize=fs_label); ax.set_ylabel("Loss (per token)", fontsize=fs_label)
    ax.set_title("Val Loss Comparison", fontsize=fs_title); ax.legend(fontsize=fs_legend)
    ax.tick_params(labelsize=fs_tick)

    # ---- Bottom row: log p(x) distributions ----
    logp_even_train_finite = logp_even_train[np.isfinite(logp_even_train)]
    logp_even_val_finite = logp_even_val[np.isfinite(logp_even_val)]
    logp_odd_train_finite = logp_odd_train[np.isfinite(logp_odd_train)]
    logp_odd_val_finite = logp_odd_val[np.isfinite(logp_odd_val)]

    print(f"[filter] even_train: {len(logp_even_train_finite)}/{len(logp_even_train)} finite")
    print(f"[filter] even_val: {len(logp_even_val_finite)}/{len(logp_even_val)} finite")
    print(f"[filter] odd_train: {len(logp_odd_train_finite)}/{len(logp_odd_train)} finite")
    print(f"[filter] odd_val: {len(logp_odd_val_finite)}/{len(logp_odd_val)} finite")

    n_bins = 60

    ax = axes[1, 0]
    ax.hist(logp_even_train_finite, bins=n_bins, alpha=0.6, color="steelblue",
            label=f"Train (n={len(logp_even_train_finite)})")
    ax.hist(logp_even_val_finite, bins=n_bins, alpha=0.6, color="indianred",
            label=f"Val (n={len(logp_even_val_finite)})")
    ax.set_xlabel(r"log $p(x)$", fontsize=fs_label); ax.set_ylabel("Count", fontsize=fs_label)
    ax.set_title(r"Even log $p(x)$ Distribution", fontsize=fs_title); ax.legend(fontsize=fs_legend)
    ax.tick_params(labelsize=fs_tick); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(logp_odd_train_finite, bins=n_bins, alpha=0.6, color="orange",
            label=f"Train (n={len(logp_odd_train_finite)})")
    ax.hist(logp_odd_val_finite, bins=n_bins, alpha=0.6, color="salmon",
            label=f"Val (n={len(logp_odd_val_finite)})")
    ax.set_xlabel(r"log $p(x)$", fontsize=fs_label); ax.set_ylabel("Count", fontsize=fs_label)
    ax.set_title(r"Odd log $p(x)$ Distribution", fontsize=fs_title); ax.legend(fontsize=fs_legend)
    ax.tick_params(labelsize=fs_tick); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    logp_even_all = np.concatenate([logp_even_train_finite, logp_even_val_finite])
    logp_odd_all = np.concatenate([logp_odd_train_finite, logp_odd_val_finite])
    ax.hist(logp_even_all, bins=n_bins, alpha=0.6, color="steelblue",
            label=f"Even (n={len(logp_even_all)})")
    ax.hist(logp_odd_all, bins=n_bins, alpha=0.6, color="orange",
            label=f"Odd (n={len(logp_odd_all)})")
    ax.set_xlabel(r"log $p(x)$", fontsize=fs_label); ax.set_ylabel("Count", fontsize=fs_label)
    ax.set_title(r"Overall log $p(x)$ Distribution", fontsize=fs_title); ax.legend(fontsize=fs_legend)
    ax.tick_params(labelsize=fs_tick); ax.grid(True, alpha=0.3)

    if args.title:
        fig.suptitle(args.title, fontsize=16 if paper else 14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    if paper:
        pdf_path = args.output.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
