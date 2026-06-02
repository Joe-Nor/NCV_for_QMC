#!/usr/bin/env python3
"""Minimal forward-pass test for the site-bond candidate embedding ablation.

Run: python3 cyclic_aug_bias/test/test_site_bond_emb.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'python', 'train', 'numerator'))
import torch
import numpy as np
from train_transformer_parity_sign_v2_pe_nh_window_aug import (
    AutoregressiveTransformer, build_bsites, compute_bond_metadata, OPERATOR_OFFSET
)

lx, ly = 3, -3
bsites, nn, nb = build_bsites(lx, ly)
bond_meta = compute_bond_metadata(bsites, nb, lx, ly)
bond_s1, bond_s2, bond_dir_arr, _bond_anchor_arr, num_sites, n_dir = bond_meta
vocab_size = nb + OPERATOR_OFFSET
B, T = 2, 10

x = torch.randint(OPERATOR_OFFSET, vocab_size, (B, T))
dk_cand = torch.zeros(B, T, vocab_size)
dk_cand[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET+nb] = torch.randint(-1, 2, (B, T, nb)).float()

def check_zero_init(model, label):
    model.eval()
    with torch.no_grad():
        h = model.embedding(x) * (model.d_model ** 0.5)
        h = model.pos_encoder(h)
        h = model.transformer(h, mask=model.causal_mask[:T, :T])
        base = model.output_proj(h)
        full = model(x, dk_candidates=dk_cand)
    corr = (full - base)
    maxerr = corr.abs().max().item()
    assert maxerr < 1e-5, f"{label}: zero-init FAILED, max|corr|={maxerr}"
    assert corr[:, :, :OPERATOR_OFFSET].abs().max().item() < 1e-7, \
        f"{label}: special tokens affected"
    print(f"  {label}: OK  max|corr|={maxerr:.2e}")
    model.train()

configs = [
    ("no dk_mlp_head",       dict(dk_mlp_head=False)),
    ("old bond_emb=32",      dict(dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
                                  dk_head_bond_emb=32)),
    ("site=32 dir=8",        dict(dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
                                  dk_head_bond_emb=0,
                                  bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
                                  num_sites=num_sites, num_bond_dirs=n_dir,
                                  dk_head_site_emb=32, dk_head_dir_emb=8)),
    ("all combined",         dict(dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
                                  dk_head_bond_emb=8,
                                  bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
                                  num_sites=num_sites, num_bond_dirs=n_dir,
                                  dk_head_site_emb=32, dk_head_dir_emb=8,
                                  dk_head_bond_res_emb=4)),
    ("centering + site/dir", dict(dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
                                  dk_head_centering=True,
                                  bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
                                  num_sites=num_sites, num_bond_dirs=n_dir,
                                  dk_head_site_emb=32, dk_head_dir_emb=8)),
]

print(f"Lattice: lx={lx}, ly={ly}, nn={nn}, nb={nb}, sites={num_sites}, dirs={n_dir}")
print(f"vocab_size={vocab_size}, B={B}, T={T}\n")

for label, kwargs in configs:
    m = AutoregressiveTransformer(vocab_size=vocab_size, d_model=64, nhead=2,
                                   num_layers=1, **kwargs)
    logits = m(x, dk_candidates=dk_cand) if kwargs.get('dk_mlp_head') else m(x)
    assert logits.shape == (B, T, vocab_size), f"{label}: shape {logits.shape}"
    if kwargs.get('dk_mlp_head'):
        check_zero_init(m, label)
    else:
        print(f"  {label}: OK  shape={logits.shape}")

# Gradient flow test
print("\nGradient flow test (site=32 dir=8)...")
m = AutoregressiveTransformer(vocab_size=vocab_size, d_model=64, nhead=2, num_layers=1,
                               dk_mlp_head=True, dk_head_dk=16, dk_head_hidden=64,
                               bond_s1=bond_s1, bond_s2=bond_s2, bond_dir=bond_dir_arr,
                               num_sites=num_sites, num_bond_dirs=n_dir,
                               dk_head_site_emb=32, dk_head_dir_emb=8)
logits = m(x, dk_candidates=dk_cand)
loss = logits.sum()
loss.backward()
for name, p in m.named_parameters():
    if 'dk_site' in name or 'dk_dir' in name:
        assert p.grad is not None, f"No gradient for {name}"
        print(f"  {name}: grad norm={p.grad.norm().item():.4f}")

print("\nALL TESTS PASSED")
