#!/usr/bin/env python3
"""Plot log q (model) vs log f (reference measure) for even and odd checkpoints."""

import os, sys, math, struct
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix as _compute_pp
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates as _compute_pp_cand
tps = None  # set in main()

# Module-level lattice info, set by read_samples()
_mod_bsites = None
_mod_nn = None
_mod_nb = None


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


def load_v2_meta(path):
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


def compute_logf(nh, K, beta, logfact):
    return K * math.log(2.0) + nh * math.log(beta / 2.0) - logfact[nh]


def precompute_logfact(M):
    return np.array([math.lgamma(n + 1.0) for n in range(M + 1)], dtype=np.float64)


def read_samples(meta, beta, logfact, max_N=None, need_candidates=False):
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
            return _compute_pp_cand(x, bsites, nn, nb)
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
                parity_prefix = np.array([], np.int8)
                x_dense = np.array([], np.int32)
                parity = 0
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
    sd = {k: v for k, v in ckpt["model_state_dict"].items()
          if not k.startswith("delta_k_head.")}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    nmin = ckpt.get("nmin", None)
    nmax = ckpt.get("nmax", None)
    return model, ckpt, nmin, nmax


@torch.no_grad()
def score_logq_batch(model, tokens, padding_mask, prefix_parity, prefix_len, deltaK_prefix, target_parity,
                     nmin, nmax, device, use_pe=False, dk_candidates=None):
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
    if use_pe:
        logits = model(inputs, padding_mask=input_padding_mask, prefix_parity=prefix_parity,
                       deltaK_prefix=deltaK_prefix, dk_candidates=dk_candidates)
    else:
        logits = model(inputs, padding_mask=input_padding_mask,
                       deltaK_prefix=deltaK_prefix, dk_candidates=dk_candidates)
    logits = tps.apply_token_mask(logits, input_padding_mask)
    logits = tps.apply_nh_window_mask(logits, prefix_len, prefix_parity, target_parity,
                                      nmin, nmax, input_padding_mask=input_padding_mask)

    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(padding_mask[:, 1:].to(device), 0.0)

    return lp.sum(dim=1)


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


def compute_logq_all(model, target_parity, samples, device, nmin, nmax, batch_size=256, use_pe=False, precomputed=None):
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
                              use_pe=use_pe, dk_candidates=dkc_batch).cpu().numpy()
        results[start:end] = lq
    return results


def plot_one_model(ax_scatter, ax_hist, logq, logf, nh, parity_label, ckpt, color, paper=False):
    """Plot scatter and histogram for one model."""
    # Filter finite values
    finite = np.isfinite(logq) & np.isfinite(logf)
    n_total = len(logq)
    n_finite = int(finite.sum())
    n_bad = n_total - n_finite
    frac_bad = n_bad / n_total if n_total > 0 else 0.0

    print(f"[filter {parity_label}] total={n_total}")
    print(f"[filter {parity_label}] finite={n_finite}")
    print(f"[filter {parity_label}] filtered_nonfinite={n_bad} ({frac_bad:.4%})")
    print(f"[filter {parity_label}] Non-finite samples are typically out-of-window test samples assigned zero probability by the model.")

    if n_finite == 0:
        raise ValueError(f"No finite samples remain after filtering logp/logf values for {parity_label}. Check n_h window coverage on the test set.")

    logq_plot = logq[finite]
    logf_plot = logf[finite]
    nh_plot = nh[finite]

    fs_label = 14 if paper else 13
    fs_tick = 12 if paper else 10
    fs_legend = 12 if paper else 11
    fs_annot = 11 if paper else 12
    fs_title = 14 if paper else 13

    # Scatter: logq vs logf
    sc = ax_scatter.scatter(logf_plot, logq_plot, c=nh_plot, cmap="viridis", s=3, alpha=0.3, rasterized=True)
    cb = plt.colorbar(sc, ax=ax_scatter)
    cb.set_label(r"$n_h$", fontsize=fs_label)
    cb.ax.tick_params(labelsize=fs_tick)
    lmin = min(logf_plot.min(), logq_plot.min())
    lmax = max(logf_plot.max(), logq_plot.max())
    ax_scatter.plot([lmin, lmax], [lmin, lmax], "r--", lw=1, label="y = x")
    ax_scatter.set_xlabel(r"$\log |W(X)|$", fontsize=fs_label)
    ax_scatter.set_ylabel(r"$\log q(X)$", fontsize=fs_label)
    ax_scatter.tick_params(labelsize=fs_tick)
    if paper:
        ax_scatter.set_title(f"Parity = {parity_label}", fontsize=fs_title)
    else:
        ax_scatter.set_title(f"Parity = {parity_label} (N={n_finite}/{n_total}, epoch={ckpt['epoch']})", fontsize=fs_title)
    ax_scatter.legend(fontsize=fs_legend)
    rho = np.corrcoef(logf_plot, logq_plot)[0, 1]
    ax_scatter.text(0.05, 0.92, f"Corr = {rho:.4f}", transform=ax_scatter.transAxes, fontsize=fs_annot,
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    # Histogram: log(q/f)
    logw = logq_plot - logf_plot
    ax_hist.hist(logw, bins=80, alpha=0.7, density=True, color=color, label=parity_label)
    ax_hist.set_xlabel(r"$\log q(X) - \log |W(X)|$", fontsize=fs_label)
    ax_hist.set_ylabel("Density", fontsize=fs_label)
    ax_hist.tick_params(labelsize=fs_tick)
    if paper:
        ax_hist.set_title(rf"$\log(q/|W|)$ — {parity_label}", fontsize=fs_title)
    else:
        ax_hist.set_title(rf"$\log(q/|W|)$ — {parity_label}", fontsize=fs_title)
    ax_hist.axvline(0, color="red", ls="--", lw=1)
    ax_hist.legend(fontsize=fs_legend)
    ax_hist.text(0.05, 0.92, f"mean={logw.mean():.3f}\nstd={logw.std():.3f}",
                 transform=ax_hist.transAxes, fontsize=fs_annot, va="top",
                 bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))


def main():
    global tps
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--even_ckpt", required=True, help="Path to even best_model.pt")
    ap.add_argument("--odd_ckpt", required=True, help="Path to odd best_model.pt")
    ap.add_argument("--data", required=True, help="Data file (.bin)")
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--title_prefix", default=None, help="Title prefix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pe", action="store_true", help="Use PE model variant")
    ap.add_argument("--paper", action="store_true", help="Publication style: larger fonts, simplified titles, PDF output")
    ap.add_argument("--combined", action="store_true", help="Combined 1x2 layout: even+odd overlaid")
    cli = ap.parse_args()

    import train_transformer_parity_sign_v2_pe_nh_window_de_aug as _tps
    tps = _tps

    ckpt_even_path = cli.even_ckpt
    ckpt_odd_path  = cli.odd_ckpt
    data_path = cli.data
    output_path = cli.output

    device = torch.device(cli.device)
    print(f"Device: {device}")

    # Load both models
    print("Loading even model...")
    model_even, ckpt_even, nmin, nmax = load_model(ckpt_even_path, device)
    print(f"  epoch={ckpt_even['epoch']}, val_loss={ckpt_even['val_loss']:.6f}")
    print(f"  n_h window: [{nmin}, {nmax}]")

    print("Loading odd model...")
    model_odd, ckpt_odd, _, _ = load_model(ckpt_odd_path, device)
    print(f"  epoch={ckpt_odd['epoch']}, val_loss={ckpt_odd['val_loss']:.6f}")

    # Load data
    print("Loading data...")
    meta = load_v2_meta(data_path)
    beta = meta["beta"]
    logfact = precompute_logfact(meta["mm"])
    need_candidates = bool(getattr(model_even, 'dk_mlp_head', False)) or \
                      bool(getattr(model_odd, 'dk_mlp_head', False))
    if need_candidates:
        print("  ΔK-candidates enabled (at least one model uses MLP head)")
    samples = read_samples(meta, beta, logfact, need_candidates=need_candidates)
    print(f"  N={len(samples)}, beta={beta}")

    # Split by parity
    even_samples = [s for s in samples if s["parity"] == 0]
    odd_samples  = [s for s in samples if s["parity"] == 1]
    print(f"  even: {len(even_samples)}, odd: {len(odd_samples)}")

    # Re-index each group
    for i, s in enumerate(even_samples):
        s["idx"] = i
    for i, s in enumerate(odd_samples):
        s["idx"] = i

    # Compute logq
    print("Precomputing tensors...")
    nb_bonds_eval = meta['nb']
    even_precomputed = precompute_eval_tensors(even_samples,
                                                need_candidates=need_candidates,
                                                nb_bonds=nb_bonds_eval)
    odd_precomputed = precompute_eval_tensors(odd_samples,
                                               need_candidates=need_candidates,
                                               nb_bonds=nb_bonds_eval)
    print("Computing logq (even)...")
    logq_even = compute_logq_all(model_even, 0, even_samples, device, nmin, nmax, use_pe=True, precomputed=even_precomputed)
    print("Computing logq (odd)...")
    logq_odd = compute_logq_all(model_odd, 1, odd_samples, device, nmin, nmax, use_pe=True, precomputed=odd_precomputed)

    logf_even = np.array([s["_logf"] for s in even_samples])
    logf_odd  = np.array([s["_logf"] for s in odd_samples])
    nh_even   = np.array([s["nh"] for s in even_samples])
    nh_odd    = np.array([s["nh"] for s in odd_samples])

    prefix = cli.title_prefix or os.path.basename(os.path.dirname(ckpt_even_path))

    if cli.combined:
        # --- Combined 1x2 layout: even+odd overlaid ---
        fs_label = 14 if cli.paper else 13
        fs_tick = 12 if cli.paper else 10
        fs_legend = 12 if cli.paper else 11
        fs_annot = 11 if cli.paper else 12
        fs_title = 14 if cli.paper else 13

        fig, (ax_sc, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))

        # Filter
        mask_e = np.isfinite(logq_even) & np.isfinite(logf_even)
        mask_o = np.isfinite(logq_odd) & np.isfinite(logf_odd)
        lq_e, lf_e, nh_e = logq_even[mask_e], logf_even[mask_e], nh_even[mask_e]
        lq_o, lf_o, nh_o = logq_odd[mask_o], logf_odd[mask_o], nh_odd[mask_o]

        # Scatter: even (blues) and odd (reds)
        sc_e = ax_sc.scatter(lf_e, lq_e, c=nh_e, cmap="winter", s=3, alpha=0.3, rasterized=True, label="even")
        sc_o = ax_sc.scatter(lf_o, lq_o, c=nh_o, cmap="autumn", s=3, alpha=0.3, rasterized=True, label="odd")
        lmin = min(lf_e.min(), lf_o.min(), lq_e.min(), lq_o.min())
        lmax = max(lf_e.max(), lf_o.max(), lq_e.max(), lq_o.max())
        ax_sc.plot([lmin, lmax], [lmin, lmax], "k--", lw=1.5, label="y = x")
        ax_sc.set_xlabel(r"$\log |W(X)|$", fontsize=fs_label)
        ax_sc.set_ylabel(r"$\log q(X)$", fontsize=fs_label)
        ax_sc.tick_params(labelsize=fs_tick)

        rho_e = np.corrcoef(lf_e, lq_e)[0, 1]
        rho_o = np.corrcoef(lf_o, lq_o)[0, 1]
        ax_sc.text(0.05, 0.92, f"Corr(even) = {rho_e:.4f}\nCorr(odd)  = {rho_o:.4f}",
                   transform=ax_sc.transAxes, fontsize=fs_annot,
                   bbox=dict(boxstyle="round", fc="wheat", alpha=0.8), va="top")
        ax_sc.legend(fontsize=fs_legend, loc="lower right")
        if cli.paper:
            ax_sc.set_title(r"$\log q$ vs $\log |W|$", fontsize=fs_title)
        else:
            ax_sc.set_title(f"logq vs log|W|  (even: {mask_e.sum()}, odd: {mask_o.sum()})", fontsize=fs_title)

        # Histogram: both parities
        logw_e = lq_e - lf_e
        logw_o = lq_o - lf_o
        ax_hist.hist(logw_e, bins=80, alpha=0.6, density=True, color="steelblue", label="even")
        ax_hist.hist(logw_o, bins=80, alpha=0.6, density=True, color="coral", label="odd")
        ax_hist.set_xlabel(r"$\log q(X) - \log |W(X)|$", fontsize=fs_label)
        ax_hist.set_ylabel("Density", fontsize=fs_label)
        ax_hist.tick_params(labelsize=fs_tick)
        ax_hist.text(0.05, 0.92,
                     f"even: mean={logw_e.mean():.3f}, std={logw_e.std():.3f}\n"
                     f"odd:  mean={logw_o.mean():.3f}, std={logw_o.std():.3f}",
                     transform=ax_hist.transAxes, fontsize=fs_annot, va="top",
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
        ax_hist.legend(fontsize=fs_legend)
        if cli.paper:
            ax_hist.set_title(r"$\log(q/|W|)$ distribution", fontsize=fs_title)
        else:
            ax_hist.set_title(r"$\log(q/|W|)$ distribution", fontsize=fs_title)

        if cli.paper:
            fig.suptitle(r"Denominator model ($2\times2$, $\beta=8.0$)", fontsize=fs_title+2)
        else:
            fig.suptitle(prefix, fontsize=14)
    else:
        # --- Original 2x2 layout ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        plot_one_model(axes[0, 0], axes[0, 1], logq_even, logf_even, nh_even, "even", ckpt_even, "C0", paper=cli.paper)
        plot_one_model(axes[1, 0], axes[1, 1], logq_odd,  logf_odd,  nh_odd,  "odd",  ckpt_odd,  "C1", paper=cli.paper)
        if cli.paper:
            fig.suptitle(prefix, fontsize=16)
        else:
            fig.suptitle(
                f"{prefix}  |  Data: {os.path.basename(data_path)}\n"
                f"Even: epoch={ckpt_even['epoch']}, val_loss={ckpt_even['val_loss']:.4f}  |  "
                f"Odd: epoch={ckpt_odd['epoch']}, val_loss={ckpt_odd['val_loss']:.4f}",
                fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
