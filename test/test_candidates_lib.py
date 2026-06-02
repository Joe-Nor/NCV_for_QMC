#!/usr/bin/env python3
"""
Verify parity_prefix_candidates_lib against brute-force computation.

Brute-force: at each position t, for each candidate bond b, call
compute_parity_prefix on prefix[0:t] + [2*b] and read the last deltaK.

Usage:
    cd src
    make clean && make
    cd ../test
    python3 test_candidates_lib.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates


def build_bsites(lx, ly):
    """Build bond-to-site mapping (same as training scripts)."""
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
        raw = []
        for y1 in range(Lyabs):
            for x1 in range(lx):
                s = 1 + x1 + y1 * lx
                x2 = (x1 + 1) % lx; y2 = y1
                raw.append((s, 1 + x2 + y2 * lx))
                x2 = x1; y2 = (y1 + 1) % Lyabs
                raw.append((s, 1 + x2 + y2 * lx))
                x2 = (x1 + 1) % lx; y2 = (y1 + 1) % Lyabs
                raw.append((s, 1 + x2 + y2 * lx))
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

    raise ValueError(f"Unsupported lattice: lx={lx}, ly={ly}")


def brute_force_candidates(x_dense, bsites, nn, nb):
    """Compute candidate ΔK by brute force: O(nh^2 * nb * L).

    For each position t (0..nh), for each bond b (1..nb):
      - Build prefix x_dense[0:t]
      - Append 2*b
      - Call compute_parity_prefix, read last deltaK
    """
    nh = len(x_dense)
    cand = np.zeros((nh + 1, nb), dtype=np.float32)

    for t in range(nh + 1):
        prefix = x_dense[:t].copy() if t > 0 else np.array([], dtype=np.int32)
        for b_idx in range(nb):
            op_value = np.int32(2 * (b_idx + 1))
            extended = np.append(prefix, op_value).astype(np.int32)
            _, dk_out, _ = compute_parity_prefix(extended, bsites, nn, nb)
            cand[t, b_idx] = float(dk_out[-1])

    return cand


def test_lattice(lx, ly, n_samples=20, max_nh=30, seed=42):
    """Test on random opstrings for a given lattice."""
    bsites, nn, nb = build_bsites(lx, ly)
    rng = np.random.RandomState(seed)

    print(f"\n=== Lattice lx={lx}, ly={ly}: nn={nn}, nb={nb} ===")

    n_pass = 0
    n_fail = 0

    for i in range(n_samples):
        nh = rng.randint(1, max_nh + 1)
        # Random opstring: each entry is 2 * bond (bond in 1..nb)
        bonds = rng.randint(1, nb + 1, size=nh)
        x_dense = (2 * bonds).astype(np.int32)

        # Method 1: brute force
        cand_bf = brute_force_candidates(x_dense, bsites, nn, nb)

        # Method 2: new library
        pp, dkp, K, cand_new = compute_parity_prefix_candidates(
            x_dense, bsites, nn, nb
        )

        # Also verify parity_prefix and deltaK_prefix match original
        pp_ref, dkp_ref, K_ref = compute_parity_prefix(x_dense, bsites, nn, nb)

        pp_ok = np.array_equal(pp, pp_ref)
        dkp_ok = np.array_equal(dkp, dkp_ref)
        K_ok = (K == K_ref)
        cand_ok = np.array_equal(cand_bf, cand_new)

        if pp_ok and dkp_ok and K_ok and cand_ok:
            n_pass += 1
        else:
            n_fail += 1
            print(f"  FAIL sample {i}: nh={nh}")
            if not pp_ok:
                print(f"    parity_prefix mismatch")
                diff = np.where(pp != pp_ref)[0]
                print(f"    diff at positions: {diff[:10]}")
            if not dkp_ok:
                print(f"    deltaK_prefix mismatch")
                diff = np.where(dkp != dkp_ref)[0]
                print(f"    diff at positions: {diff[:10]}")
            if not K_ok:
                print(f"    K mismatch: new={K}, ref={K_ref}")
            if not cand_ok:
                diff_pos = np.where(cand_bf != cand_new)
                n_diff = len(diff_pos[0])
                print(f"    candidates mismatch: {n_diff} entries differ")
                for j in range(min(5, n_diff)):
                    t, b = diff_pos[0][j], diff_pos[1][j]
                    print(f"      pos={t}, bond={b+1}: bf={cand_bf[t,b]:.0f}, "
                          f"new={cand_new[t,b]:.0f}")

            if n_fail >= 5:
                print("  Too many failures, stopping early")
                break

    print(f"  Results: {n_pass} pass, {n_fail} fail out of {n_pass + n_fail}")
    return n_fail == 0


def test_edge_cases(bsites, nn, nb):
    """Test edge cases: nh=1, repeated bonds, etc."""
    print(f"\n=== Edge cases: nn={nn}, nb={nb} ===")

    cases = [
        ("nh=1, bond 1", np.array([2], dtype=np.int32)),
        ("nh=1, bond nb", np.array([2 * nb], dtype=np.int32)),
        ("nh=2, same bond", np.array([2, 2], dtype=np.int32)),
        ("nh=2, different bonds", np.array([2, 4], dtype=np.int32)),
        ("nh=3, all same", np.array([2, 2, 2], dtype=np.int32)),
    ]

    all_ok = True
    for name, x_dense in cases:
        cand_bf = brute_force_candidates(x_dense, bsites, nn, nb)
        _, _, _, cand_new = compute_parity_prefix_candidates(
            x_dense, bsites, nn, nb
        )
        ok = np.array_equal(cand_bf, cand_new)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name} (nh={len(x_dense)})")
        if not ok:
            all_ok = False
            diff_pos = np.where(cand_bf != cand_new)
            for j in range(min(3, len(diff_pos[0]))):
                t, b = diff_pos[0][j], diff_pos[1][j]
                print(f"    pos={t}, bond={b+1}: bf={cand_bf[t,b]:.0f}, "
                      f"new={cand_new[t,b]:.0f}")

    return all_ok


def main():
    all_ok = True

    # 3-site triangle (lx=3, ly=1): nn=3, nb=3
    bsites_3x1, nn_3x1, nb_3x1 = build_bsites(3, 1)
    all_ok &= test_edge_cases(bsites_3x1, nn_3x1, nb_3x1)
    all_ok &= test_lattice(3, 1, n_samples=50, max_nh=20)

    # 2×2 tri PBC (lx=2, ly=-2): nn=4, nb=6
    all_ok &= test_lattice(2, -2, n_samples=50, max_nh=30)

    # 3×3 tri PBC (lx=3, ly=-3): nn=9, nb=27
    all_ok &= test_lattice(3, -3, n_samples=20, max_nh=40)

    print("\n" + "=" * 50)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 50)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
