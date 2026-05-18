"""
Python wrapper for Fortran parity_prefix_candidates_lib.so via ctypes.

Extends parity_prefix_wrapper: in addition to parity_prefix and deltaK_prefix,
computes candidate-wise ΔK for ALL bonds at EACH autoregressive position.

Usage:
    from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates

    # ops_compact: np.array(nh,), int32, uncolored opstring (values = 2*b)
    # bsites: np.array((2, nb), order='F'), int32, bond-to-site mapping (1-indexed)
    parity_prefix, deltaK_prefix, K, deltaK_candidates = \
        compute_parity_prefix_candidates(ops_compact, bsites, nn, nb)

    # deltaK_candidates: np.array((nh+1, nb), float32)
    #   deltaK_candidates[t, b] = ΔK from inserting bond (b+1) after prefix of length t
"""

import ctypes
import numpy as np
import os

# Load shared library
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'parity_prefix_candidates_lib.so')
_lib = ctypes.CDLL(_lib_path)


def compute_parity_prefix_candidates(ops_compact, bsites, nn, nb):
    """
    Compute prefix parities, delta-K prefix, AND candidate-wise ΔK.

    Args:
        ops_compact: np.array(nh,), dtype=int32
            Compact uncolored opstring (values = 2*bond_index, 1-indexed)
        bsites: np.array((2, nb), order='F'), dtype=int32
            Bond-to-site mapping, 1-indexed. Column-major (Fortran order).
        nn: int — number of sites
        nb: int — number of bonds

    Returns:
        parity_prefix: np.array(nh,), dtype=int8
            Prefix parity at each position (0 or 1)
        deltaK_prefix: np.array(nh,), dtype=int32
            Delta K at each step (-1, 0, or +1)
        K: int
            Total loop count after all operators
        deltaK_candidates: np.array((nh+1, nb), dtype=float32)
            Candidate-wise ΔK. Row t = autoregressive position (0..nh).
            Column b = bond index (0..nb-1, mapping to Fortran 1..nb).
            Values are in {-1, 0, +1}.
    """
    nh = len(ops_compact)

    if nh == 0:
        # All sites free → any bond gives ΔK = -1
        cand = np.full((1, nb), -1.0, dtype=np.float32)
        return (np.array([], dtype=np.int8),
                np.array([], dtype=np.int32),
                nn,
                cand)

    # Ensure correct dtypes and contiguity
    ops = np.ascontiguousarray(ops_compact, dtype=np.int32)
    bs = np.asfortranarray(bsites, dtype=np.int32)

    parity_out = np.zeros(nh, dtype=np.int8)
    deltaK_out = np.zeros(nh, dtype=np.int32)
    K_out = ctypes.c_int32(0)
    nh_c = ctypes.c_int32(nh)
    nn_c = ctypes.c_int32(nn)
    nb_c = ctypes.c_int32(nb)

    # Fortran output: deltaK_candidates(nb, nh+1), column-major
    # In Fortran, column j = position j-1, row i = bond i.
    # Allocate as column-major (Fortran order)
    cand_out = np.zeros((nb, nh + 1), dtype=np.int32, order='F')

    _lib.compute_parity_prefix_candidates_(
        ops.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(nh_c),
        bs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(nn_c),
        ctypes.byref(nb_c),
        parity_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        deltaK_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(K_out),
        cand_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    # Transpose to (nh+1, nb) in C order, convert to float32
    deltaK_candidates = cand_out.T.astype(np.float32).copy()

    return parity_out, deltaK_out, K_out.value, deltaK_candidates
