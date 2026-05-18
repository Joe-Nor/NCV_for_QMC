"""
Python wrapper for Fortran parity_prefix_lib.so via ctypes.

Usage:
    from parity_prefix_wrapper import compute_parity_prefix

    # ops_compact: np.array(nh,), int32, uncolored opstring (values = 2*b)
    # bsites: np.array((2, nb), order='F'), int32, bond-to-site mapping (1-indexed)
    parity_prefix, deltaK_prefix, K = compute_parity_prefix(ops_compact, bsites, nn, nb)
"""

import ctypes
import numpy as np
import os

# Load shared library
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parity_prefix_lib.so')
_lib = ctypes.CDLL(_lib_path)


def compute_parity_prefix(ops_compact, bsites, nn, nb):
    """
    Compute prefix parities and delta-K prefix for a compact opstring.

    Args:
        ops_compact: np.array(nh,), dtype=int32
            Compact uncolored opstring (values = 2*bond_index)
        bsites: np.array((2, nb), order='F'), dtype=int32
            Bond-to-site mapping, 1-indexed. Column-major (Fortran order).
        nn: int
            Number of sites
        nb: int
            Number of bonds

    Returns:
        parity_prefix: np.array(nh,), dtype=int8
            Prefix parity at each position (0 or 1)
        deltaK_prefix: np.array(nh,), dtype=int32
            Delta K at each step (-1, 0, or +1)
        K: int
            Total loop count after all operators
    """
    nh = len(ops_compact)
    if nh == 0:
        return np.array([], dtype=np.int8), np.array([], dtype=np.int32), nn

    # Ensure correct dtypes and contiguity
    ops = np.ascontiguousarray(ops_compact, dtype=np.int32)
    # bsites must be Fortran-contiguous (column-major) for Fortran 2D array
    bs = np.asfortranarray(bsites, dtype=np.int32)

    parity_out = np.zeros(nh, dtype=np.int8)
    deltaK_out = np.zeros(nh, dtype=np.int32)
    K_out = ctypes.c_int32(0)
    nh_c = ctypes.c_int32(nh)
    nn_c = ctypes.c_int32(nn)
    nb_c = ctypes.c_int32(nb)

    _lib.compute_parity_prefix_(
        ops.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(nh_c),
        bs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(nn_c),
        ctypes.byref(nb_c),
        parity_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        deltaK_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(K_out),
    )
    return parity_out, deltaK_out, K_out.value
