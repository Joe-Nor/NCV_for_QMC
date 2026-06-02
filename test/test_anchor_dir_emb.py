#!/usr/bin/env python3
"""Tests for anchor_dir vs sitepair geometry modes in dk_mlp_head."""
import sys, os, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'train', 'numerator'))
import train_transformer_parity_sign_v2_pe_nh_window_aug as tps

OPERATOR_OFFSET = tps.OPERATOR_OFFSET
LX, LY = 3, -3
NN = 9
NB = 27
VOCAB = NB + OPERATOR_OFFSET

bsites_arr, nn_sites, nb_bonds = tps.build_bsites(LX, LY)
bond_meta = tps.compute_bond_metadata(bsites_arr, nb_bonds, LX, LY)
assert bond_meta is not None
bond_s1, bond_s2, bond_dir_arr, bond_anchor_arr, num_sites, n_dir = bond_meta

B, T = 4, 16


def _rand_inputs():
    tokens = torch.randint(OPERATOR_OFFSET, VOCAB, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    pp = torch.zeros(B, T, dtype=torch.long)
    dkp = torch.ones(B, T, dtype=torch.long)
    dk_cand = torch.zeros(B, T, VOCAB, dtype=torch.float32)
    dk_cand[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET + NB] = torch.randint(-1, 2, (B, T, NB)).float()
    return tokens, pad, pp, dkp, dk_cand


def test_bond_metadata_returns_anchor():
    """compute_bond_metadata returns 6-tuple with bond_anchor."""
    assert len(bond_meta) == 6
    assert bond_anchor_arr.shape == (NB,)
    assert bond_anchor_arr.dtype == np.int64
    assert bond_anchor_arr.min() >= 0
    assert bond_anchor_arr.max() < NN
    print(f"  PASS: bond_metadata returns anchor (shape={bond_anchor_arr.shape})")


def test_anchor_plus_dir_reconstructs_endpoint():
    """For each bond, anchor + direction = the other endpoint (mod PBC)."""
    Lyabs = abs(LY)
    directions = [(1, 0), (0, 1), (1, 1)]
    for b in range(NB):
        anchor = int(bond_anchor_arr[b])
        d = int(bond_dir_arr[b])
        ax = anchor % LX
        ay = anchor // LX
        ddx, ddy = directions[d]
        other = (ay + ddy) % Lyabs * LX + (ax + ddx) % LX
        s1, s2 = int(bond_s1[b]), int(bond_s2[b])
        endpoints = {s1, s2}
        assert anchor in endpoints, f"bond {b}: anchor {anchor} not in endpoints {endpoints}"
        assert other in endpoints, f"bond {b}: anchor({ax},{ay})+dir{d}({ddx},{ddy})={other} not in {endpoints}"
        assert other != anchor or s1 == s2, f"bond {b}: anchor==other but s1!=s2"
    print(f"  PASS: anchor+dir reconstructs all {NB} bonds")


def test_anchor_consistent_regardless_of_bsites_order():
    """Flipping s1/s2 in bsites should give the same canonical anchor."""
    flipped = bsites_arr.copy()
    flipped[0, :], flipped[1, :] = bsites_arr[1, :].copy(), bsites_arr[0, :].copy()
    meta2 = tps.compute_bond_metadata(flipped, nb_bonds, LX, LY)
    assert meta2 is not None
    _, _, dir2, anchor2, _, _ = meta2
    for b in range(NB):
        assert anchor2[b] == bond_anchor_arr[b], f"bond {b}: anchor changed when endpoints flipped"
        assert dir2[b] == bond_dir_arr[b], f"bond {b}: dir changed when endpoints flipped"
    print(f"  PASS: anchor/dir invariant under endpoint flip")


def test_sitepair_forward():
    """geom_mode='sitepair' forward pass."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
        dk_head_site_emb=16, dk_head_dir_emb=4,
        dk_head_geom_mode="sitepair",
        bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
        bond_anchor=bond_anchor_arr,
        num_sites=num_sites, num_bond_dirs=n_dir,
    )
    model.eval()
    tokens, pad, pp, dkp, dk_cand = _rand_inputs()
    with torch.no_grad():
        logits = model(tokens, padding_mask=pad, prefix_parity=pp,
                       deltaK_prefix=dkp, dk_candidates=dk_cand)
    assert logits.shape == (B, T, VOCAB)
    print("  PASS: sitepair forward")


def test_anchor_dir_forward():
    """geom_mode='anchor_dir' forward pass."""
    model = tps.AutoregressiveTransformer(
        vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
        dim_feedforward=128, dropout=0.0, max_len=32,
        dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
        dk_head_site_emb=16, dk_head_dir_emb=4,
        dk_head_geom_mode="anchor_dir",
        bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
        bond_anchor=bond_anchor_arr,
        num_sites=num_sites, num_bond_dirs=n_dir,
    )
    model.eval()
    tokens, pad, pp, dkp, dk_cand = _rand_inputs()
    with torch.no_grad():
        logits = model(tokens, padding_mask=pad, prefix_parity=pp,
                       deltaK_prefix=dkp, dk_candidates=dk_cand)
    assert logits.shape == (B, T, VOCAB)
    print("  PASS: anchor_dir forward")


def test_zero_init_both_modes():
    """dk correction is zero at init for both modes."""
    for mode in ["sitepair", "anchor_dir"]:
        model = tps.AutoregressiveTransformer(
            vocab_size=VOCAB, d_model=64, nhead=2, num_layers=1,
            dim_feedforward=128, dropout=0.0, max_len=32,
            dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
            dk_head_site_emb=16, dk_head_dir_emb=4,
            dk_head_geom_mode=mode,
            bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
            bond_anchor=bond_anchor_arr,
            num_sites=num_sites, num_bond_dirs=n_dir,
        )
        model.eval()
        h = torch.randn(B, T, 64)
        dk_cand = torch.zeros(B, T, VOCAB, dtype=torch.float32)
        dk_cand[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET + NB] = torch.randint(-1, 2, (B, T, NB)).float()
        with torch.no_grad():
            corr = model._compute_bond_correction(h, dk_cand)
        assert corr.abs().max().item() < 1e-6, f"{mode}: correction not zero at init"
        # EOS/PAD/BOS = 0
        assert corr[:, :, :OPERATOR_OFFSET].abs().max().item() == 0.0
    print("  PASS: zero-init for both modes, EOS/PAD/BOS=0")


def test_print_anchor_metadata():
    """Print first few bonds for visual verification."""
    print("  Bond metadata (first 6):")
    for b in range(min(6, NB)):
        print(f"    b={b}: s1={bond_s1[b]} s2={bond_s2[b]} "
              f"anchor={bond_anchor_arr[b]} dir={bond_dir_arr[b]}")
    print("  PASS: metadata printed")


if __name__ == "__main__":
    print("Running anchor_dir geometry tests...")
    test_bond_metadata_returns_anchor()
    test_anchor_plus_dir_reconstructs_endpoint()
    test_anchor_consistent_regardless_of_bsites_order()
    test_sitepair_forward()
    test_anchor_dir_forward()
    test_zero_init_both_modes()
    test_print_anchor_metadata()
    print("\nAll tests passed!")
