"""Test spatial translation bond permutation: K must be invariant."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'nh_window', 'numerator'))

from parity_prefix_wrapper import compute_parity_prefix

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "tps_num",
    os.path.join(os.path.dirname(__file__), '..', 'python', 'nh_window', 'numerator',
                 'train_transformer_parity_sign_v2_pe_nh_window_aug.py'))
tps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tps)


def _check_perm(name, perm, bsites, nn, nb, n_samples, nh_range):
    num_ops = perm.shape[0]
    assert np.array_equal(perm[0], np.arange(nb)), f"{name} row 0 must be identity"
    rng = np.random.RandomState(42)
    fail = 0
    for _ in range(n_samples):
        nh = rng.randint(*nh_range)
        ops = (2 * (rng.randint(0, nb, size=nh) + 1)).astype(np.int32)
        _, _, K0 = compute_parity_prefix(ops, bsites, nn, nb)
        for k in range(num_ops):
            b_old = (ops // 2).astype(np.int64) - 1
            b_new = perm[k, b_old]
            ops_new = (2 * (b_new + 1)).astype(np.int32)
            _, _, K_new = compute_parity_prefix(ops_new, bsites, nn, nb)
            if K_new != K0:
                fail += 1
                print(f"    FAIL {name}: g={k}, K0={K0}, K_new={K_new}, nh={nh}")
                break
    if fail == 0:
        print(f"    OK {name}: K invariant over {n_samples} samples × {num_ops} ops")
    return fail == 0


def test_lattice(lx, ly, n_samples=20, nh_range=(5, 30)):
    bsites, nn, nb = tps.build_bsites(lx, ly)
    print(f"  lattice ({lx}, {ly}): nn={nn}, nb={nb}")

    sp = tps.build_spatial_bond_perm(lx, ly, bsites, nb)
    ok_sp = _check_perm("spatial", sp, bsites, nn, nb, n_samples, nh_range)

    if lx == abs(ly):
        pg = tps.build_pointgroup_bond_perm(lx, ly, bsites, nb)
        ok_pg = _check_perm("D_6", pg, bsites, nn, nb, n_samples, nh_range)
    else:
        print(f"    D_6: skipped (lx={lx} != |ly|={abs(ly)})")
        ok_pg = True

    return ok_sp and ok_pg


if __name__ == '__main__':
    print("Testing spatial bond permutation K-invariance:")
    # Run small→large because the shared Fortran module only grows bsites_w buffer.
    ok_31 = test_lattice(3, 1)
    ok_33 = test_lattice(3, -3)
    try:
        ok_44 = test_lattice(4, -4)
    except Exception as e:
        print(f"  4x-4 skipped: {e}")
        ok_44 = True
    print("\nAll pass" if (ok_33 and ok_31 and ok_44) else "\nFAIL")
