#!/usr/bin/env python3
"""
Exact Diagonalization for triangular lattice with PBC.

Lattice convention (same as Fortran makelattice with ly < 0):
  - lx × Ly sites, PBC in both directions
  - 3 bond directions per site: +x, +y, +x+y (diagonal)
  - nn = lx * Ly, nb = 3 * nn
  - Site indexing: s = x + y * lx  (0-indexed)

  The triangular lattice is a square lattice with one diagonal
  added per plaquette (+x+y direction).

For nn <= 12: full dense diagonalization (2^nn x 2^nn).
For nn > 12:  S_z-sector decomposition — each sector diagonalized separately.

Usage:
    python ED_tri.py                       # default 3x3
    python ED_tri.py --lx 4 --ly 4         # 4x4 triangle PBC (S_z sectors)
    python ED_tri.py --lx 3 --ly 1 --open  # 3-site open triangle (legacy)
"""

import numpy as np
import argparse
from math import comb
from itertools import combinations


# ============================================================================
# Lattice construction
# ============================================================================

def build_triangular_bonds_pbc(lx, ly):
    """Build bond list for lx × ly triangular lattice with PBC.

    Bond ordering matches Fortran makelattice():
      bonds  0..nn-1:      +x direction
      bonds  nn..2*nn-1:   +y direction
      bonds  2*nn..3*nn-1: +x+y diagonal

    On small lattices (e.g. 2x2) PBC wrapping causes duplicate edges.
    These are removed: only the first occurrence of each undirected edge
    is kept. For lx >= 3 and ly >= 3 no duplicates exist.

    Returns: (bonds, nn) where bonds is list of (i,j) 0-indexed.
    """
    nn = lx * ly
    raw_bonds = []

    # +x direction
    for y1 in range(ly):
        for x1 in range(lx):
            s = x1 + y1 * lx
            x2 = (x1 + 1) % lx
            raw_bonds.append((s, x2 + y1 * lx))

    # +y direction
    for y1 in range(ly):
        for x1 in range(lx):
            s = x1 + y1 * lx
            y2 = (y1 + 1) % ly
            raw_bonds.append((s, x1 + y2 * lx))

    # +x+y diagonal
    for y1 in range(ly):
        for x1 in range(lx):
            s = x1 + y1 * lx
            x2 = (x1 + 1) % lx
            y2 = (y1 + 1) % ly
            raw_bonds.append((s, x2 + y2 * lx))

    # Deduplicate: keep first occurrence of each undirected edge
    seen = set()
    bonds = []
    for (i, j) in raw_bonds:
        edge = (min(i, j), max(i, j))
        if edge not in seen:
            seen.add(edge)
            bonds.append((i, j))

    if len(bonds) < len(raw_bonds):
        n_removed = len(raw_bonds) - len(bonds)
        print(f"  [dedup] Removed {n_removed} duplicate bonds "
              f"({len(raw_bonds)} -> {len(bonds)}) due to PBC wrapping on {lx}x{ly}")

    return bonds, nn


def build_triangle_3site():
    """3-site open triangle (legacy, matches lx=3, ly=1)."""
    return [(0, 1), (1, 2), (2, 0)], 3


def print_bond_structure(bonds, nn, lx, ly):
    """Print bond list."""
    nb = len(bonds)
    print(f"  Bonds ({nb}): {bonds}")
    print(f"  nn = {nn}, nb = {nb}, nb/nn = {nb/nn:.1f}")


# ============================================================================
# Full dense diagonalization (small systems, nn <= ~12)
# ============================================================================

def build_H_dense(nn, bonds, mode="std"):
    """Build full 2^nn x 2^nn Hamiltonian matrix.

    mode:
      "std"  — standard Heisenberg H = sum Si·Sj
      "phys" — SSE physical: H_phys = -sum (H1 - H2) = H_std - nb/4
      "abs"  — SSE absolute:  H_abs = -sum (H1 + H2)
    """
    dim = 1 << nn
    H = np.zeros((dim, dim), dtype=np.float64)

    if mode == "std":
        for (i, j) in bonds:
            for x in range(dim):
                si = 0.5 if ((x >> i) & 1) else -0.5
                sj = 0.5 if ((x >> j) & 1) else -0.5
                H[x, x] += si * sj
                if si != sj:
                    y = x ^ ((1 << i) | (1 << j))
                    H[x, y] += 0.5
    else:
        s_sign = +1.0 if mode == "abs" else -1.0
        for (i, j) in bonds:
            for x in range(dim):
                si = 0.5 if ((x >> i) & 1) else -0.5
                sj = 0.5 if ((x >> j) & 1) else -0.5
                H[x, x] += -(0.25 - si * sj)
                if si != sj:
                    y = x ^ ((1 << i) | (1 << j))
                    H[x, y] += -(s_sign * 0.5)
    return H


def thermal_from_evals(evals, beta):
    """(Z, E, C) from eigenvalue array."""
    w = np.exp(-beta * evals)
    Z = w.sum()
    E = (w * evals).sum() / Z
    E2 = (w * evals**2).sum() / Z
    C = beta**2 * (E2 - E**2)
    return Z, E, C


def ed_report_dense(beta, nn, bonds):
    nb = len(bonds)
    evals_std = np.linalg.eigvalsh(build_H_dense(nn, bonds, "std"))
    evals_phys = np.linalg.eigvalsh(build_H_dense(nn, bonds, "phys"))
    evals_abs = np.linalg.eigvalsh(build_H_dense(nn, bonds, "abs"))

    Zs, Es, Cs = thermal_from_evals(evals_std, beta)
    Zp, Ep, Cp = thermal_from_evals(evals_phys, beta)
    Za, Ea, Ca = thermal_from_evals(evals_abs, beta)

    return {
        "beta": beta, "nn": nn, "nb": nb,
        "E_std_per_site": Es / nn,
        "C_std_per_site": Cs / nn,
        "E_sse_phys_per_site": Ep / nn,
        "avg_sign": Zp / Za,
        "shift_check": Es - Ep,
        "expected_shift": nb / 4.0,
    }


# ============================================================================
# S_z-sector diagonalization (large systems, nn up to ~20)
# ============================================================================

def build_sector_basis(nn, n_up):
    """Build basis states for the S_z sector with n_up up-spins.
    Returns array of ints (bit representations), sorted.
    """
    if n_up == 0:
        return np.array([0], dtype=np.int64)
    if n_up == nn:
        return np.array([(1 << nn) - 1], dtype=np.int64)
    # Generate all states with exactly n_up bits set
    states = []
    for bits in combinations(range(nn), n_up):
        x = 0
        for b in bits:
            x |= (1 << b)
        states.append(x)
    states.sort()
    return np.array(states, dtype=np.int64)


def build_H_sector(nn, bonds, basis, mode="std"):
    """Build Hamiltonian in a fixed S_z sector.

    basis: sorted array of basis state ints.
    Returns: dense matrix (len(basis) x len(basis)).
    """
    dim = len(basis)
    # Map state -> index for fast lookup
    state_to_idx = {}
    for idx, x in enumerate(basis):
        state_to_idx[int(x)] = idx

    H = np.zeros((dim, dim), dtype=np.float64)

    if mode == "std":
        for (i, j) in bonds:
            for idx in range(dim):
                x = int(basis[idx])
                si = 0.5 if ((x >> i) & 1) else -0.5
                sj = 0.5 if ((x >> j) & 1) else -0.5
                H[idx, idx] += si * sj
                if si != sj:
                    y = x ^ ((1 << i) | (1 << j))
                    jdx = state_to_idx[y]
                    H[idx, jdx] += 0.5
    else:
        s_sign = +1.0 if mode == "abs" else -1.0
        for (i, j) in bonds:
            for idx in range(dim):
                x = int(basis[idx])
                si = 0.5 if ((x >> i) & 1) else -0.5
                sj = 0.5 if ((x >> j) & 1) else -0.5
                H[idx, idx] += -(0.25 - si * sj)
                if si != sj:
                    y = x ^ ((1 << i) | (1 << j))
                    jdx = state_to_idx[y]
                    H[idx, jdx] += -(s_sign * 0.5)
    return H


def collect_all_evals_sectors(nn, bonds, mode="std"):
    """Diagonalize each S_z sector and collect all eigenvalues."""
    all_evals = []
    for n_up in range(nn + 1):
        dim_sector = comb(nn, n_up)
        sz = n_up - nn / 2.0
        print(f"    S_z = {sz:+5.1f}  dim = {dim_sector}", end="", flush=True)
        basis = build_sector_basis(nn, n_up)
        H_sec = build_H_sector(nn, bonds, basis, mode)
        evals = np.linalg.eigvalsh(H_sec)
        all_evals.append(evals)
        print(f"  E_min = {evals[0]:.6f}")
    return np.concatenate(all_evals)


def ed_report_sectors(beta, nn, bonds, evals_cache):
    """Compute thermodynamics from pre-computed eigenvalues."""
    nb = len(bonds)
    Zs, Es, Cs = thermal_from_evals(evals_cache["std"], beta)
    Zp, Ep, Cp = thermal_from_evals(evals_cache["phys"], beta)
    Za, Ea, Ca = thermal_from_evals(evals_cache["abs"], beta)

    return {
        "beta": beta, "nn": nn, "nb": nb,
        "E_std_per_site": Es / nn,
        "C_std_per_site": Cs / nn,
        "E_sse_phys_per_site": Ep / nn,
        "avg_sign": Zp / Za,
        "shift_check": Es - Ep,
        "expected_shift": nb / 4.0,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lx", type=int, default=3)
    ap.add_argument("--ly", type=int, default=3)
    ap.add_argument("--open", action="store_true",
                    help="Use 3-site open triangle (lx=3, ly=1 legacy)")
    ap.add_argument("--beta_min", type=float, default=0.5)
    ap.add_argument("--beta_max", type=float, default=5.0)
    ap.add_argument("--beta_steps", type=int, default=19)
    ap.add_argument("--sector_threshold", type=int, default=12,
                    help="Use S_z sectors when nn > this value (default 12)")
    args = ap.parse_args()

    if args.open:
        bonds, nn = build_triangle_3site()
        label = "3-site triangle (open)"
        lx, ly = 3, 1
    else:
        bonds, nn = build_triangular_bonds_pbc(args.lx, args.ly)
        label = f"{args.lx}x{args.ly} triangular PBC"
        lx, ly = args.lx, args.ly

    nb = len(bonds)
    print(f"Lattice: {label}")
    print(f"  nn = {nn}, nb = {nb}")
    print_bond_structure(bonds, nn, lx, ly)

    use_sectors = (nn > args.sector_threshold)
    if use_sectors:
        max_sector = comb(nn, nn // 2)
        print(f"\n  Using S_z-sector decomposition (nn={nn} > {args.sector_threshold})")
        print(f"  Largest sector: S_z=0, dim = {max_sector}")
        print(f"  Total sectors: {nn + 1}\n")

        evals_cache = {}
        for mode in ["std", "phys", "abs"]:
            print(f"  Diagonalizing H_{mode}:")
            evals_cache[mode] = collect_all_evals_sectors(nn, bonds, mode)
            print()
    else:
        print(f"\n  Using full dense diagonalization (dim = {1 << nn})\n")

    betas = np.linspace(args.beta_min, args.beta_max, args.beta_steps)

    print(f"{'beta':>8s}  {'E_std/N':>14s}  {'E_phys/N':>14s}  {'<sign>':>14s}  {'C/N':>10s}")
    print("-" * 68)
    for beta in betas:
        if use_sectors:
            r = ed_report_sectors(beta, nn, bonds, evals_cache)
        else:
            r = ed_report_dense(beta, nn, bonds)
        print(f"{beta:8.3f}  {r['E_std_per_site']:14.10f}  {r['E_sse_phys_per_site']:14.10f}  "
              f"{r['avg_sign']:14.10f}  {r['C_std_per_site']:10.6f}")

    # Verify shift
    if use_sectors:
        r0 = ed_report_sectors(betas[0], nn, bonds, evals_cache)
    else:
        r0 = ed_report_dense(betas[0], nn, bonds)
    print(f"\nShift check: E_std - E_phys = {r0['shift_check']:.10f}  (expect {r0['expected_shift']:.10f})")


if __name__ == "__main__":
    main()
