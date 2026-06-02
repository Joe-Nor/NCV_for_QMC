"""
Validation of parity_prefix_lib against stored V2 binary data.

Tests:
1. Bit-exact match: recomputed parity_prefix == stored parity_prefix (every position)
2. Cyclic invariance of K: K unchanged after cyclic shift
3. Cyclic invariance of final parity: parity_prefix[-1] unchanged after shift
"""

import sys
import os
import struct
import numpy as np
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix


def build_bsites_triangle_3x1():
    """Build bsites for 3-site triangle (lx=3, ly=1): nn=3, nb=3."""
    bsites = np.zeros((2, 3), dtype=np.int32, order='F')
    bsites[0, 0] = 1; bsites[1, 0] = 2  # bond 1: site 1-2
    bsites[0, 1] = 2; bsites[1, 1] = 3  # bond 2: site 2-3
    bsites[0, 2] = 3; bsites[1, 2] = 1  # bond 3: site 3-1
    return bsites, 3, 3  # bsites, nn, nb


def build_bsites_from_header(lx, ly, nn, nb):
    """Build bsites from lattice parameters, same logic as Fortran makelattice()."""
    # 3-site triangle
    if lx == 3 and ly == 1:
        return build_bsites_triangle_3x1()

    is_tri_pbc = (ly < 0)
    Lyabs = abs(ly)

    if is_tri_pbc:
        nn_check = lx * Lyabs
        nb_check = 3 * nn_check
        bsites = np.zeros((2, nb_check), dtype=np.int32, order='F')

        for y1 in range(Lyabs):
            for x1 in range(lx):
                s = 1 + x1 + y1 * lx

                # dir 1: +x
                x2 = (x1 + 1) % lx
                y2 = y1
                bsites[0, s - 1] = s
                bsites[1, s - 1] = 1 + x2 + y2 * lx

                # dir 2: +y
                x2 = x1
                y2 = (y1 + 1) % Lyabs
                bsites[0, s - 1 + nn_check] = s
                bsites[1, s - 1 + nn_check] = 1 + x2 + y2 * lx

                # dir 3: +x+y (diagonal)
                x2 = (x1 + 1) % lx
                y2 = (y1 + 1) % Lyabs
                bsites[0, s - 1 + 2 * nn_check] = s
                bsites[1, s - 1 + 2 * nn_check] = 1 + x2 + y2 * lx

        return bsites, nn_check, nb_check

    # Square lattice
    nn_check = lx * ly
    nb_check = 2 * nn_check
    bsites = np.zeros((2, nb_check), dtype=np.int32, order='F')

    for y1 in range(ly):
        for x1 in range(lx):
            s = 1 + x1 + y1 * lx

            # +x bonds
            x2 = (x1 + 1) % lx
            y2 = y1
            bsites[0, s - 1] = s
            bsites[1, s - 1] = 1 + x2 + y2 * lx

            # +y bonds
            x2 = x1
            y2 = (y1 + 1) % ly
            bsites[0, s - 1 + nn_check] = s
            bsites[1, s - 1 + nn_check] = 1 + x2 + y2 * lx

    return bsites, nn_check, nb_check


def read_v2_samples(filepath, max_samples=None):
    """Read samples from V2 binary file. Returns header info and list of samples."""
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        assert magic == b'RSSE', f"Invalid magic: {magic}"

        version = struct.unpack('<i', f.read(4))[0]
        assert version == 2, f"Expected V2, got version {version}"

        lx, ly, nn, nb, mm = struct.unpack('<5i', f.read(20))
        beta, surface_n = struct.unpack('<2d', f.read(16))

        header = {
            'lx': lx, 'ly': ly, 'nn': nn, 'nb': nb, 'mm': mm,
            'beta': beta, 'surface_n': surface_n
        }

        samples = []
        count = 0
        while True:
            if max_samples is not None and count >= max_samples:
                break

            nh_k_bytes = f.read(8)
            if len(nh_k_bytes) < 8:
                break

            nh, K = struct.unpack('<2i', nh_k_bytes)

            if nh > 0:
                parity_prefix = np.frombuffer(f.read(nh), dtype=np.int8).copy()
                opstring = np.frombuffer(f.read(4 * nh), dtype=np.int32).copy()
            else:
                parity_prefix = np.array([], dtype=np.int8)
                opstring = np.array([], dtype=np.int32)

            samples.append({
                'nh': nh,
                'K': K,
                'parity_prefix': parity_prefix,
                'opstring': opstring,
            })
            count += 1

    return header, samples


def _check_bit_exact(samples, bsites, nn, nb):
    """Test 1: recomputed parity_prefix matches stored values at every position."""
    n_tested = 0
    n_pass = 0
    n_fail = 0

    for i, s in enumerate(samples):
        if s['nh'] == 0:
            continue

        pp_computed, kp_computed, K_computed = compute_parity_prefix(s['opstring'], bsites, nn, nb)
        pp_stored = s['parity_prefix']

        n_tested += 1
        if np.array_equal(pp_computed, pp_stored):
            n_pass += 1
        else:
            n_fail += 1
            if n_fail <= 5:  # Print first few failures
                mismatch = np.where(pp_computed != pp_stored)[0]
                print(f"  FAIL sample {i}: nh={s['nh']}, K_stored={s['K']}, "
                      f"K_computed={K_computed}, mismatch at positions {mismatch[:10]}")
                print(f"    stored:   {pp_stored[:20]}")
                print(f"    computed: {pp_computed[:20]}")

        # Also check K
        if K_computed != s['K']:
            print(f"  K MISMATCH sample {i}: stored={s['K']}, computed={K_computed}")

    print(f"Test 1 (bit-exact): {n_pass}/{n_tested} passed, {n_fail} failed")
    return n_fail == 0


def _check_cyclic_K_invariance(samples, bsites, nn, nb, max_shifts=5):
    """Test 2: K is invariant under cyclic shifts."""
    n_tested = 0
    n_pass = 0
    n_fail = 0

    for i, s in enumerate(samples):
        if s['nh'] <= 1:
            continue

        _, _, K_orig = compute_parity_prefix(s['opstring'], bsites, nn, nb)

        n_shifts = min(max_shifts, s['nh'])
        shifts = np.random.choice(s['nh'], size=n_shifts, replace=False)
        all_ok = True

        for k in shifts:
            shifted = np.roll(s['opstring'], -k)
            _, _, K_shifted = compute_parity_prefix(shifted, bsites, nn, nb)

            if K_shifted != K_orig:
                all_ok = False
                if n_fail < 5:
                    print(f"  FAIL sample {i}, shift={k}: K_orig={K_orig}, K_shifted={K_shifted}")

        n_tested += 1
        if all_ok:
            n_pass += 1
        else:
            n_fail += 1

    print(f"Test 2 (K invariance): {n_pass}/{n_tested} passed, {n_fail} failed")
    return n_fail == 0


def _check_cyclic_parity_invariance(samples, bsites, nn, nb, max_shifts=5):
    """Test 3: final parity is invariant under cyclic shifts."""
    n_tested = 0
    n_pass = 0
    n_fail = 0

    for i, s in enumerate(samples):
        if s['nh'] <= 1:
            continue

        pp_orig, _, _ = compute_parity_prefix(s['opstring'], bsites, nn, nb)
        parity_orig = pp_orig[-1]

        n_shifts = min(max_shifts, s['nh'])
        shifts = np.random.choice(s['nh'], size=n_shifts, replace=False)
        all_ok = True

        for k in shifts:
            shifted = np.roll(s['opstring'], -k)
            pp_shifted, _, _ = compute_parity_prefix(shifted, bsites, nn, nb)
            parity_shifted = pp_shifted[-1]

            if parity_shifted != parity_orig:
                all_ok = False
                if n_fail < 5:
                    print(f"  FAIL sample {i}, shift={k}: "
                          f"parity_orig={parity_orig}, parity_shifted={parity_shifted}")

        n_tested += 1
        if all_ok:
            n_pass += 1
        else:
            n_fail += 1

    print(f"Test 3 (parity invariance): {n_pass}/{n_tested} passed, {n_fail} failed")
    return n_fail == 0


def test_parity_lib_sample_file():
    """Pytest entry point for an optional RSSE V2 sample file."""
    data_path = os.environ.get("RSSE_SAMPLE_PATH")
    if not data_path:
        pytest.skip("set RSSE_SAMPLE_PATH to run parity-prefix data validation")
    if not os.path.exists(data_path):
        pytest.skip(f"RSSE_SAMPLE_PATH does not exist: {data_path}")

    header, samples = read_v2_samples(data_path)
    bsites, nn, nb = build_bsites_from_header(
        header['lx'], header['ly'], header['nn'], header['nb']
    )
    np.random.seed(42)
    assert _check_bit_exact(samples, bsites, nn, nb)
    assert _check_cyclic_K_invariance(samples, bsites, nn, nb)
    assert _check_cyclic_parity_invariance(samples, bsites, nn, nb)


def main():
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = os.environ.get("RSSE_SAMPLE_PATH")
    if not data_path:
        print("Data file not provided. Pass a path or set RSSE_SAMPLE_PATH.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    header, samples = read_v2_samples(data_path)
    print(f"  Lattice: {header['lx']}x{header['ly']}, nn={header['nn']}, nb={header['nb']}, mm={header['mm']}")
    print(f"  beta={header['beta']}, N={header['surface_n']}")
    print(f"  Samples loaded: {len(samples)}")

    # Build bsites
    bsites, nn, nb = build_bsites_from_header(
        header['lx'], header['ly'], header['nn'], header['nb']
    )
    print(f"  bsites shape: {bsites.shape}, nn={nn}, nb={nb}")
    print()

    # Filter out nh=0 for stats
    nhs = [s['nh'] for s in samples if s['nh'] > 0]
    if nhs:
        print(f"  n_h range: [{min(nhs)}, {max(nhs)}], mean={np.mean(nhs):.1f}")
    print()

    np.random.seed(42)

    ok1 = _check_bit_exact(samples, bsites, nn, nb)
    print()
    ok2 = _check_cyclic_K_invariance(samples, bsites, nn, nb)
    print()
    ok3 = _check_cyclic_parity_invariance(samples, bsites, nn, nb)
    print()

    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
