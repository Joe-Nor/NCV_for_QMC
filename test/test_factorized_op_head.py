#!/usr/bin/env python3
"""Tests for site-bond embedding and logit component diagnostics."""
import sys, os, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'nh_window', 'numerator'))
import train_transformer_parity_sign_v2_pe_nh_window_aug as tps

OPERATOR_OFFSET = tps.OPERATOR_OFFSET
LX, LY = 3, -3
NN = 9
NB = 27
VOCAB = NB + OPERATOR_OFFSET

bsites_arr, nn_sites, nb_bonds = tps.build_bsites(LX, LY)
bond_meta = tps.compute_bond_metadata(bsites_arr, nb_bonds, LX, LY)
assert bond_meta is not None
bond_s1, bond_s2, bond_dir_arr, num_sites, n_dir = bond_meta

B, T = 4, 16


def _rand_inputs(device="cpu"):
    tokens = torch.randint(OPERATOR_OFFSET, VOCAB, (B, T), device=device)
    pad = torch.zeros(B, T, dtype=torch.bool, device=device)
    pp = torch.zeros(B, T, dtype=torch.long, device=device)
    dkp = torch.ones(B, T, dtype=torch.long, device=device)
    dk_cand = torch.zeros(B, T, VOCAB, dtype=torch.float32, device=device)
    dk_cand[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET + NB] = torch.randint(-1, 2, (B, T, NB)).float()
    return tokens, pad, pp, dkp, dk_cand


def test_baseline():
    """No dk_head: vanilla output_proj only."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=False,
    )
    model.eval()
    tokens, pad, pp, dkp, _ = _rand_inputs()
    with torch.no_grad():
        logits = model(tokens, padding_mask=pad, prefix_parity=pp, deltaK_prefix=dkp)
    assert logits.shape == (B, T, VOCAB)
    print("  PASS: baseline forward")


def test_site_bond_residual():
    """dk_mlp_head + site/dir embedding."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
        dk_head_bond_emb=0, dk_head_site_emb=32, dk_head_dir_emb=8,
        bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
        num_sites=num_sites, num_bond_dirs=n_dir,
    )
    model.eval()
    tokens, pad, pp, dkp, dk_cand = _rand_inputs()
    with torch.no_grad():
        logits = model(tokens, padding_mask=pad, prefix_parity=pp,
                       deltaK_prefix=dkp, dk_candidates=dk_cand)
    assert logits.shape == (B, T, VOCAB)
    print("  PASS: site-bond residual forward")


def test_zero_init():
    """dk_mlp_head correction is zero at init."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
        dk_head_site_emb=16, dk_head_dir_emb=4,
        bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
        num_sites=num_sites, num_bond_dirs=n_dir,
    )
    model.eval()
    h = torch.randn(B, T, 64)
    dk_cand = torch.zeros(B, T, VOCAB, dtype=torch.float32)
    dk_cand[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET + NB] = torch.randint(-1, 2, (B, T, NB)).float()
    with torch.no_grad():
        corr = model._compute_bond_correction(h, dk_cand)
    assert corr.abs().max().item() < 1e-6, f"Correction not zero at init: {corr.abs().max().item()}"
    print("  PASS: zero-init verification")


def test_diagnostic_full():
    """diagnose_logit_components with dk + padding + targets."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
        dk_head_site_emb=16, dk_head_dir_emb=4,
        bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
        num_sites=num_sites, num_bond_dirs=n_dir,
    )
    model.eval()
    h = torch.randn(B, T, 64)
    dk_cand = torch.zeros(B, T, VOCAB, dtype=torch.float32)
    dk_cand[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET + NB] = torch.randint(-1, 2, (B, T, NB)).float()
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[:, -2:] = True
    targets = torch.randint(OPERATOR_OFFSET, VOCAB, (B, T))

    diag = model.diagnose_logit_components(
        h, dk_candidates=dk_cand, input_padding_mask=pad_mask, targets=targets)
    for key in ["base_global_std", "base_candidate_spread", "base_common_mode_std",
                "corr_global_std", "corr_candidate_spread", "corr_common_mode_std",
                "corr_candidate_over_base", "corr_common_over_candidate",
                "final_op_entropy_mean", "final_op_entropy_norm_mean",
                "final_op_max_prob_mean", "final_op_candidate_spread",
                "target_nll_mean", "target_rank_mean", "target_rank_p50", "target_rank_p95"]:
        assert key in diag, f"Missing key: {key}"
    assert diag["base_global_std"] > 0
    assert diag["final_op_entropy_norm_mean"] > 0
    print(f"  PASS: full diagnostic (base cand={diag['base_candidate_spread']:.4f}, "
          f"corr cand={diag['corr_candidate_spread']:.4f}, "
          f"corr common/cand={diag['corr_common_over_candidate']:.1f}, "
          f"entropy={diag['final_op_entropy_norm_mean']:.3f}*log(nb), "
          f"rank_mean={diag['target_rank_mean']:.1f})")


def test_diagnostic_no_padding():
    """diagnose_logit_components without padding/targets."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
        dk_head_site_emb=16, dk_head_dir_emb=4,
        bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
        num_sites=num_sites, num_bond_dirs=n_dir,
    )
    model.eval()
    h = torch.randn(B, T, 64)
    dk_cand = torch.zeros(B, T, VOCAB, dtype=torch.float32)
    dk_cand[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET + NB] = torch.randint(-1, 2, (B, T, NB)).float()

    diag = model.diagnose_logit_components(h, dk_candidates=dk_cand)
    assert "base_candidate_spread" in diag
    assert "target_nll_mean" not in diag
    print("  PASS: diagnostic without padding/targets")


def test_diagnostic_baseline():
    """diagnose_logit_components on baseline model (no dk)."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=False,
    )
    model.eval()
    h = torch.randn(B, T, 64)
    diag = model.diagnose_logit_components(h)
    assert "base_global_std" in diag
    assert "corr_global_std" not in diag
    assert "final_op_entropy_mean" in diag
    print("  PASS: baseline diagnostic (no dk)")


def test_gradient_flow():
    """Gradients flow through site-bond correction."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
        dk_head_site_emb=16, dk_head_dir_emb=4,
        bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
        num_sites=num_sites, num_bond_dirs=n_dir,
    )
    model.train()
    tokens, pad, pp, dkp, dk_cand = _rand_inputs()
    logits = model(tokens, padding_mask=pad, prefix_parity=pp,
                   deltaK_prefix=dkp, dk_candidates=dk_cand)
    loss = logits[:, :, OPERATOR_OFFSET:].sum()
    loss.backward()
    assert model.bond_head[0].weight.grad is not None
    assert model.bond_head[0].weight.grad.abs().max() > 0
    print("  PASS: gradient flow through site-bond correction")


if __name__ == "__main__":
    print("Running site-bond embedding & diagnostic tests...")
    test_baseline()
    test_site_bond_residual()
    test_zero_init()
    test_diagnostic_full()
    test_diagnostic_no_padding()
    test_diagnostic_baseline()
    print("\nAll tests passed!")
