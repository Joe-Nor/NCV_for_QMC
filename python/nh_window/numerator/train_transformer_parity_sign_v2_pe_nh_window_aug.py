#!/usr/bin/env python3
"""
Numerator ⟨n_h * s⟩ CV Training with n_h-weighted loss, n_h Window,
and Cyclic Data Augmentation.

Train q ∝ n_h * f by minimizing E[-n_h * log q(x)] with automatic n_h window
selection. At nmax, EOS is forced regardless of parity to preserve normalization.

Cyclic augmentation: each sample is randomly cyclic-shifted each epoch,
with parity_prefix recomputed via Fortran library.

Usage:
    python train_transformer_parity_sign_v2_pe_nh_window_aug.py --parity even --data_glob "/path/*.bin" --auto_nh_window 1
"""

import os
import sys
import glob
import math
import random
import argparse
import struct
import functools
import numpy as np
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

# Add cyclic_aug/src to path for parity_prefix_wrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix
from parity_prefix_candidates_wrapper import compute_parity_prefix_candidates

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Special Tokens and Operator Encoding
# ============================================================================

# Special token IDs
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# Operator tokens start at 3 (dense mapping: bond index b -> token b+2)
OPERATOR_OFFSET = 3


def op_to_token(op_value: int) -> int:
    """Convert raw operator value (2*b) to token ID. Dense: b -> b+2."""
    return op_value // 2 + OPERATOR_OFFSET - 1


def token_to_op(token_id: int) -> int:
    """Convert token ID back to raw operator value (2*b)."""
    return (token_id - OPERATOR_OFFSET + 1) * 2


# ============================================================================
# n_h Distribution Scanning and Triangle Patch
# ============================================================================

def scan_nh_histogram_v2(file_paths: List[str], max_scan_samples: int = 200000) -> Dict:
    """
    Fast scan of n_h distribution from V2 binary files.

    Only reads headers and n_h values, does not parse full records.

    Args:
        file_paths: List of .bin file paths
        max_scan_samples: Maximum total samples to scan

    Returns:
        dict with keys: hist, total_samples, nh_min, nh_max, mean_nh, std_nh
    """
    hist = {}
    total = 0
    nh_values = []

    for fpath in file_paths:
        if total >= max_scan_samples:
            break

        with open(fpath, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic == b"RSSE":
                fmt_version = 2
            elif magic == b"RSS3":
                fmt_version = 3
            elif magic == b"RSS4":
                fmt_version = 4
            else:
                continue
            version = struct.unpack("<i", f.read(4))[0]
            if version != fmt_version:
                continue
            f.read(20)  # lx, ly, nn, nb, mm
            f.read(16)  # beta, surface_n

            # Scan records
            while total < max_scan_samples:
                nh_bytes = f.read(4)
                if len(nh_bytes) < 4:
                    break
                nh = struct.unpack("<i", nh_bytes)[0]

                # Skip rest of record
                if fmt_version == 2:
                    skip_bytes = 4 + nh + 4 * nh  # K + parity_prefix + opstring
                elif fmt_version == 3:
                    skip_bytes = 4 + 4 + 4 * nh   # K + parity + opstring
                else:  # V4
                    skip_bytes = 4 + 4 * nh        # parity + opstring
                f.seek(skip_bytes, 1)

                hist[nh] = hist.get(nh, 0) + 1
                nh_values.append(nh)
                total += 1

    if total == 0:
        raise ValueError("No samples found in scan")

    nh_array = np.array(nh_values)
    return {
        'hist': hist,
        'total_samples': total,
        'nh_min': int(nh_array.min()),
        'nh_max': int(nh_array.max()),
        'mean_nh': float(nh_array.mean()),
        'std_nh': float(nh_array.std()),
    }


def choose_nh_window_from_hist(hist: Dict[int, int], left_tail_mass: float, right_tail_mass: float, weighted: bool = True) -> Tuple[int, int]:
    """
    Choose n_h window [nmin, nmax] from histogram based on cumulative tail mass.

    Args:
        hist: {n_h: count}
        left_tail_mass: exclude left tail with cumulative mass <= this threshold
        right_tail_mass: exclude right tail with cumulative mass <= this threshold
        weighted: if True, use n_h * count (numerator); if False, use count (denominator)

    Returns:
        (nmin, nmax)
    """
    if not hist:
        raise ValueError("Empty histogram")

    sorted_nh = sorted(hist.keys())

    if weighted:
        mass_dict = {n: float(n * hist[n]) for n in sorted_nh}
    else:
        mass_dict = {n: float(hist[n]) for n in sorted_nh}

    total_mass = sum(mass_dict.values())
    if total_mass <= 0:
        raise ValueError(f"Total mass <= 0: {total_mass}")

    # Choose nmin: cumulative from left
    cumsum = 0
    nmin = sorted_nh[0]
    for n in sorted_nh:
        cumsum += mass_dict[n]
        if cumsum / total_mass >= left_tail_mass:
            nmin = n
            break

    # Choose nmax: cumulative from right
    cumsum = 0
    nmax = sorted_nh[-1]
    for n in reversed(sorted_nh):
        cumsum += mass_dict[n]
        if cumsum / total_mass >= right_tail_mass:
            nmax = n
            break

    return nmin, nmax




# ============================================================================
# Streaming Dataset (V2 Format Only)
# ============================================================================

class RSSEStreamingDatasetV2(IterableDataset):
    """
    Streaming dataset for V2 format with n_h window filtering.
    """

    def __init__(
        self,
        file_paths: List[str],
        target_parity: int,
        max_samples_per_file: Optional[int] = None,
        shuffle_buffer: int = 10000,
        allow_mixed_mm: bool = False,
        stride: int = 1,
        split: str = "all",
        train_fraction: float = 0.8,
        seed: int = 42,
        nh_min: Optional[int] = None,
        nh_max: Optional[int] = None,
    ):
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be one of ['all','train','val'], got {split}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if not (0.0 <= train_fraction <= 1.0):
            raise ValueError(f"train_fraction must be in [0,1], got {train_fraction}")

        self.file_paths = file_paths
        self.target_parity = target_parity
        self.max_samples_per_file = max_samples_per_file
        self.shuffle_buffer = shuffle_buffer
        self.allow_mixed_mm = allow_mixed_mm
        self.stride = stride
        self.split = split
        self.train_fraction = train_fraction
        self.seed = seed
        self.nh_min = nh_min
        self.nh_max = nh_max
        self._epoch = 0  # Epoch counter for shuffle seed variation

        self.file_metadata: List[Dict] = []
        self.total_samples = 0

        for fpath in file_paths:
            meta = self._load_file_metadata(fpath)
            self.file_metadata.append(meta)
            n_samples = meta["n_samples"]
            if max_samples_per_file:
                n_samples = min(n_samples, max_samples_per_file)
            self.total_samples += n_samples

        if not allow_mixed_mm and len(self.file_metadata) > 1:
            nb_values = [meta["nb"] for meta in self.file_metadata]
            if len(set(nb_values)) > 1:
                print(f"WARNING: Files have different nb values: {set(nb_values)}")
                print(f"         Using max(nb) = {max(nb_values)} for vocab_size calculation")

        self.nb = max(meta["nb"] for meta in self.file_metadata)
        self.lx = self.file_metadata[0]["lx"]
        self.ly = self.file_metadata[0]["ly"]
        self.nn = self.file_metadata[0]["nn"]
        self.beta = self.file_metadata[0]["beta"]

        print("Streaming dataset initialized (V2 format):")
        print(f"  Files: {len(file_paths)}")
        print(f"  Lattice: lx={self.lx}, ly={self.ly}, nn={self.nn}, nb={self.nb}")
        print(f"  Total samples (before filters): {self.total_samples}")
        print(f"  Target parity: {target_parity} ({'even' if target_parity == 0 else 'odd'})")
        print(f"  Stride: {stride}")
        print(f"  Split: {split} (train_fraction={train_fraction})")
        if nh_min is not None or nh_max is not None:
            print(f"  n_h window: [{nh_min}, {nh_max}]")
        print(f"  Shuffle buffer: {shuffle_buffer}")

    def _load_file_metadata(self, filepath: str) -> Dict:
        """Load header and scan samples (V2 or V3 format)."""
        file_size = os.path.getsize(filepath)
        header_size = 44

        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic == b"RSSE":
                fmt_version = 2
            elif magic == b"RSS3":
                fmt_version = 3
            elif magic == b"RSS4":
                fmt_version = 4
            else:
                raise ValueError(f"Invalid magic in {filepath}: {magic}")

            version = struct.unpack("<i", f.read(4))[0]
            if version != fmt_version:
                raise ValueError(f"Magic/version mismatch: magic={magic}, version={version}")

            lx, ly, nn, nb, mm = struct.unpack("<5i", f.read(20))
            _beta, _surface_n = struct.unpack("<2d", f.read(16))

            n_samples, sample_offsets = self._scan_samples(f, file_size, header_size, fmt_version)

        return {
            "filepath": filepath,
            "lx": lx,
            "ly": ly,
            "nn": nn,
            "nb": nb,
            "beta": _beta,
            "n_samples": n_samples,
            "header_size": header_size,
            "sample_offsets": sample_offsets,
            "fmt_version": fmt_version,
        }

    def _scan_samples(self, f, file_size: int, header_size: int, fmt_version: int) -> Tuple[int, List[int]]:
        """Scan V2/V3/V4 file to count samples and record offsets."""
        sample_offsets = []
        pos = header_size
        f.seek(pos)

        while pos < file_size:
            sample_offsets.append(pos)
            if fmt_version == 4:
                # V4: (nh:i4, parity:i4, opstring:4*nh) — no K field
                header_bytes = f.read(8)
                if len(header_bytes) < 8:
                    break
                nh, _parity = struct.unpack("<2i", header_bytes)
                skip_bytes = 4 * nh  # opstring only
            else:
                # V2/V3: (nh:i4, K:i4, ...)
                nh_k_bytes = f.read(8)
                if len(nh_k_bytes) < 8:
                    break
                nh, _K = struct.unpack("<2i", nh_k_bytes)
                if fmt_version == 2:
                    skip_bytes = nh + 4 * nh  # parity_prefix(nh bytes) + opstring(4*nh bytes)
                else:
                    skip_bytes = 4 + 4 * nh   # parity(4 bytes) + opstring(4*nh bytes)
            f.seek(skip_bytes, 1)
            pos = f.tell()

        return len(sample_offsets), sample_offsets

    def _parse_sample_from_file(self, f, fmt_version: int) -> Optional[Dict]:
        """Parse a single V2, V3, or V4 sample from open file."""
        if fmt_version == 4:
            # V4: (nh:i4, parity:i4, opstring:4*nh) — no K field
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return None
            nh, parity = struct.unpack("<2i", header_bytes)
            K = 0

            if nh > 0:
                opstring_bytes = f.read(4 * nh)
                if len(opstring_bytes) < 4 * nh:
                    return None
                x_dense = np.frombuffer(opstring_bytes, dtype="<i4").copy()
            else:
                x_dense = np.array([], dtype=np.int32)

            parity_prefix = np.array([], dtype=np.int8)
        else:
            # V2/V3: (nh:i4, K:i4, ...)
            nh_k_bytes = f.read(8)
            if len(nh_k_bytes) < 8:
                return None

            nh, K = struct.unpack("<2i", nh_k_bytes)

            if nh > 0:
                if fmt_version == 2:
                    parity_prefix_bytes = f.read(nh)
                    if len(parity_prefix_bytes) < nh:
                        return None
                    parity_prefix = np.frombuffer(parity_prefix_bytes, dtype="<i1").copy()
                    parity = int(parity_prefix[-1])
                else:
                    # V3: read stored parity (int32), parity_prefix computed later in collate_fn
                    parity_bytes = f.read(4)
                    if len(parity_bytes) < 4:
                        return None
                    parity = struct.unpack("<i", parity_bytes)[0]
                    parity_prefix = np.array([], dtype=np.int8)

                opstring_bytes = f.read(4 * nh)
                if len(opstring_bytes) < 4 * nh:
                    return None
                x_dense = np.frombuffer(opstring_bytes, dtype="<i4").copy()
            else:
                parity_prefix = np.array([], dtype=np.int8)
                x_dense = np.array([], dtype=np.int32)
                parity = 0

        sign = +1 if parity == 0 else -1

        return {
            "x_dense": x_dense,
            "nh": nh,
            "K": K,
            "sign": sign,
            "parity": parity,
            "parity_prefix": parity_prefix,
            "needs_parity_compute": False,
        }

    def __iter__(self):
        """Stream through data with stride + parity + nh filters."""
        worker = get_worker_info()
        if worker is None:
            wid, nworkers = 0, 1
        else:
            wid, nworkers = worker.id, worker.num_workers

        rng = random.Random(self.seed + wid + self._epoch * 1000003)

        def splitmix64(x: int) -> int:
            x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            z = x
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
            z = z ^ (z >> 31)
            return z & 0xFFFFFFFFFFFFFFFF

        def is_train_by_hash(filtered_idx: int) -> bool:
            key = (filtered_idx ^ (self.seed * 0xD1B54A32D192ED03)) & 0xFFFFFFFFFFFFFFFF
            h = splitmix64(key)
            u = h / 2**64
            return u < self.train_fraction

        buffer = []
        global_raw_idx = 0
        filtered_idx = 0

        for meta in self.file_metadata:
            n_samples = meta["n_samples"]
            fmt_version = meta["fmt_version"]
            if self.max_samples_per_file:
                n_samples = min(n_samples, self.max_samples_per_file)

            with open(meta["filepath"], "rb") as f:
                f.seek(meta["header_size"])

                for _ in range(n_samples):
                    sample = self._parse_sample_from_file(f, fmt_version)
                    if sample is None:
                        break

                    raw_idx = global_raw_idx
                    global_raw_idx += 1

                    # stride filter
                    if raw_idx % self.stride != 0:
                        continue

                    # parity filter (skip for V3 — parity not yet computed)
                    if not sample.get("needs_parity_compute", False):
                        if sample["parity"] != self.target_parity:
                            continue

                    # nh filter
                    if self.nh_min is not None and sample["nh"] < self.nh_min:
                        continue
                    if self.nh_max is not None and sample["nh"] > self.nh_max:
                        continue

                    this_fidx = filtered_idx
                    filtered_idx += 1

                    # online split
                    train_flag = is_train_by_hash(this_fidx)
                    if self.split == "train" and not train_flag:
                        continue
                    if self.split == "val" and train_flag:
                        continue

                    # multi-worker sharding
                    if (this_fidx % nworkers) != wid:
                        continue

                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer:
                        rng.shuffle(buffer)
                        for s in buffer:
                            yield s
                        buffer.clear()

        if buffer:
            rng.shuffle(buffer)
            for s in buffer:
                yield s


def build_bsites(lx, ly):
    """Build bond-to-site mapping, same logic as Fortran makelattice().
    Returns bsites (2, nb) Fortran-order int32 array, nn, nb."""
    # 3-site triangle
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

    # Square lattice
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


def build_spatial_bond_perm(lx, ly, bsites, nb):
    """Build spatial-translation bond permutation table.

    For each shift (sx, sy) relabel site s (1-indexed) at (x, y) as site at
    ((x+sx) % lx, (y+sy) % Lyabs), then look up the new bond index.

    Returns:
        perm: np.int64 array, shape (num_shifts, nb), 0-indexed bond mapping.
              perm[k, b_old] = b_new, with k = sx + sy*lx.
              Row 0 is the identity (shift (0,0)).
    """
    Lyabs = abs(ly)
    edge_to_bond = {}
    for b in range(nb):
        s1 = int(bsites[0, b]); s2 = int(bsites[1, b])
        edge_to_bond[(min(s1, s2), max(s1, s2))] = b

    def shift_site(s, sx, sy):
        s0 = s - 1
        x = s0 % lx
        y = s0 // lx
        return 1 + ((x + sx) % lx) + ((y + sy) % Lyabs) * lx

    num_shifts = lx * Lyabs
    perm = np.zeros((num_shifts, nb), dtype=np.int64)
    for sy in range(Lyabs):
        for sx in range(lx):
            k = sx + sy * lx
            for b in range(nb):
                s1 = int(bsites[0, b]); s2 = int(bsites[1, b])
                s1n = shift_site(s1, sx, sy); s2n = shift_site(s2, sx, sy)
                key = (min(s1n, s2n), max(s1n, s2n))
                if key not in edge_to_bond:
                    raise ValueError(f"shift ({sx},{sy}) on bond {b} ({s1},{s2})→({s1n},{s2n}) not in lattice")
                perm[k, b] = edge_to_bond[key]
    return perm


def build_pointgroup_bond_perm(lx, ly, bsites, nb):
    """Build D_6 point-group bond permutation table for triangular Lx=|Ly| torus.

    Basis: a_1 = (1, 0), a_2 = (-1/2, √3/2). In this basis bonds
    (x+1, y), (x, y+1), (x+1, y+1) are all nearest-neighbour.

    Generators:
      C_6 (60° rotation): (x, y) -> (x - y, x)
      sigma (mirror):     (x, y) -> (x - y, -y)

    Returns:
        perm: np.int64 array, shape (12, nb). Row 0 is identity.
              Rows enumerate {C_6^k, sigma·C_6^k} for k in 0..5.
    """
    if lx != abs(ly):
        raise ValueError(f"D_6 requires lx=|ly|, got lx={lx}, ly={ly}")
    L = lx

    edge_to_bond = {}
    for b in range(nb):
        s1 = int(bsites[0, b]); s2 = int(bsites[1, b])
        edge_to_bond[(min(s1, s2), max(s1, s2))] = b

    def matmul(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
        ]

    E = [[1, 0], [0, 1]]
    C6 = [[1, -1], [1, 0]]
    sigma = [[1, -1], [0, -1]]

    elements = []
    M = E
    for _ in range(6):
        elements.append(M)
        elements.append(matmul(sigma, M))
        M = matmul(C6, M)

    perm = np.zeros((12, nb), dtype=np.int64)
    for g_idx, M in enumerate(elements):
        for b in range(nb):
            s1 = int(bsites[0, b]); s2 = int(bsites[1, b])
            x1 = (s1 - 1) % lx; y1 = (s1 - 1) // lx
            x2 = (s2 - 1) % lx; y2 = (s2 - 1) // lx
            x1n = (M[0][0]*x1 + M[0][1]*y1) % L
            y1n = (M[1][0]*x1 + M[1][1]*y1) % L
            x2n = (M[0][0]*x2 + M[0][1]*y2) % L
            y2n = (M[1][0]*x2 + M[1][1]*y2) % L
            s1n = 1 + x1n + y1n * L
            s2n = 1 + x2n + y2n * L
            key = (min(s1n, s2n), max(s1n, s2n))
            if key not in edge_to_bond:
                raise ValueError(f"D_6 g={g_idx} on bond {b} ({s1},{s2})→({s1n},{s2n}) not in lattice")
            perm[g_idx, b] = edge_to_bond[key]
    return perm


def collate_fn_parity_v2_aug(batch: List[Dict], bsites=None, nn_sites=None, nb_bonds=None, augment=True, cyclic_aug=True, spatial_perm=None, pointgroup_perm=None, compute_candidates=False):
    """Collate function with optional cyclic augmentation + prefix_len for n_h window masking.

    Args:
        bsites, nn_sites, nb_bonds: lattice info for parity_prefix computation.
            Required for V3 data and cyclic augmentation.
        augment: if True, apply random cyclic shift (training). If False, only compute
            parity_prefix for V3 samples without shifting (validation).
        spatial_perm: optional np.int64 (num_shifts, nb) from build_spatial_bond_perm.
            When provided and augment=True, apply a random spatial translation after
            cyclic shift by remapping bond indices; Fortran-side bsites stays fixed.
        pointgroup_perm: optional np.int64 (12, nb) from build_pointgroup_bond_perm.
            When provided and augment=True, apply a random D_6 point-group operation
            after translation by remapping bond indices. Independent of spatial_perm.
        compute_candidates: if True, also compute per-(position, bond) ΔK candidates
            and return a (B, T_input, V) float32 tensor for the MLP bond-logit head.
            Candidates are computed AFTER all augmentations.

    Returns:
        tokens: (batch, max_seq_len)
        padding_mask: (batch, max_seq_len)
        target_parity: (batch,)
        prefix_parity: (batch, input_seq_len)
        prefix_len: (batch, input_seq_len) - number of operators generated up to each position
        deltaK_prefix_tensor: (batch, input_seq_len)
        dk_cand_tensor: (batch, input_seq_len, vocab_size) float32 or None
        raw_samples: list of dicts
    """
    if bsites is not None:
        for sample in batch:
            nh = sample['nh']

            # For nh==0 we still need candidate row 0 if the bias is enabled.
            if nh <= 0:
                if compute_candidates:
                    _, _, _, cand_out = compute_parity_prefix_candidates(
                        sample['x_dense'], bsites, nn_sites, nb_bonds
                    )
                    sample['deltaK_candidates'] = cand_out  # (1, nb)
                continue

            # Cyclic augmentation (training only)
            if augment and cyclic_aug and nh > 1:
                k = np.random.randint(0, nh)
                if k > 0:
                    sample['x_dense'] = np.roll(sample['x_dense'], -k)

            # Spatial translation augmentation (training only, optional)
            if augment and spatial_perm is not None:
                k_sp = np.random.randint(0, spatial_perm.shape[0])
                if k_sp > 0:
                    x = sample['x_dense']
                    b_old = (x // 2).astype(np.int64) - 1
                    b_new = spatial_perm[k_sp, b_old]
                    sample['x_dense'] = (2 * (b_new + 1)).astype(x.dtype)

            # Point-group (D_6) augmentation (training only, optional, independent of spatial)
            if augment and pointgroup_perm is not None:
                k_pg = np.random.randint(0, pointgroup_perm.shape[0])
                if k_pg > 0:
                    x = sample['x_dense']
                    b_old = (x // 2).astype(np.int64) - 1
                    b_new = pointgroup_perm[k_pg, b_old]
                    sample['x_dense'] = (2 * (b_new + 1)).astype(x.dtype)

            # Compute parity_prefix, deltaK_prefix (and optionally candidates) via Fortran.
            if compute_candidates:
                pp_out, dkp_out, K_out, cand_out = compute_parity_prefix_candidates(
                    sample['x_dense'], bsites, nn_sites, nb_bonds
                )
                sample['deltaK_candidates'] = cand_out  # (nh+1, nb) float32
            else:
                pp_out, dkp_out, K_out = compute_parity_prefix(
                    sample['x_dense'], bsites, nn_sites, nb_bonds
                )
            sample['parity_prefix'] = pp_out
            sample['deltaK_prefix'] = dkp_out
            sample['parity'] = int(pp_out[-1])
            sample['K'] = K_out
            sample['sign'] = +1 if sample['parity'] == 0 else -1
            sample['needs_parity_compute'] = False

    sequences = []
    parities = []
    parity_prefixes = []
    deltaK_prefixes = []

    for sample in batch:
        x_dense = sample['x_dense']
        parity = sample['parity']
        parity_prefix = sample['parity_prefix']
        deltaK_prefix = sample.get('deltaK_prefix', np.array([], dtype=np.int32))

        op_tokens = [op_to_token(int(op)) for op in x_dense]
        seq = [BOS_ID] + op_tokens + [EOS_ID]
        sequences.append(seq)
        parities.append(parity)
        parity_prefixes.append(parity_prefix)
        deltaK_prefixes.append(deltaK_prefix)

    max_len = max(len(seq) for seq in sequences)
    padded = []
    masks = []

    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = seq + [PAD_ID] * pad_len
        mask = [False] * len(seq) + [True] * pad_len
        padded.append(padded_seq)
        masks.append(mask)

    tokens = torch.tensor(padded, dtype=torch.long)
    padding_mask = torch.tensor(masks, dtype=torch.bool)
    target_parity = torch.tensor(parities, dtype=torch.long)

    # Build prefix_parity: (batch, input_seq_len)
    input_seq_len = max_len - 1
    prefix_parity = torch.zeros(len(batch), input_seq_len, dtype=torch.long)
    for b, pp in enumerate(parity_prefixes):
        nh = len(pp)
        if nh > 0:
            pp_tensor = torch.from_numpy(pp.astype(np.int64))
            end = min(1 + nh, input_seq_len)
            prefix_parity[b, 1:end] = pp_tensor[:end - 1]
            if end < input_seq_len:
                prefix_parity[b, end:] = int(pp[-1])

    # Build prefix_len: (batch, input_seq_len)
    prefix_len = torch.zeros(len(batch), input_seq_len, dtype=torch.long)
    for b in range(len(batch)):
        nh = len(batch[b]['x_dense'])
        if nh > 0:
            end = min(1 + nh, input_seq_len)
            prefix_len[b, 1:end] = torch.arange(1, end)
            if end < input_seq_len:
                prefix_len[b, end:] = nh

    # Build deltaK_prefix: (batch, input_seq_len)
    # deltaK values are -1, 0, +1 → mapped to indices 0, 1, 2 (delta + 1)
    # Position 0 (BOS): delta=0 → index 1
    # Position 1..nh: deltaK_prefix from Fortran (mapped: delta+1)
    # Position nh+1.. (EOS/PAD): delta=0 → index 1
    deltaK_prefix_tensor = torch.ones((len(batch), input_seq_len), dtype=torch.long)  # default index 1 (delta=0)
    for b, dkp in enumerate(deltaK_prefixes):
        nh = len(dkp)
        if nh > 0:
            dkp_tensor = torch.from_numpy(dkp.astype(np.int64)) + 1  # delta+1 → indices 0,1,2
            end = min(1 + nh, input_seq_len)
            deltaK_prefix_tensor[b, 1:end] = dkp_tensor[:end - 1]
            # post-EOS positions stay at index 1 (delta=0), already default

    # Build dk_candidates tensor for the MLP bond-logit head.
    # Shape (B, input_seq_len, V). Only the [OPERATOR_OFFSET : OPERATOR_OFFSET+nb]
    # slice is populated; special tokens (PAD/BOS/EOS) keep 0. Prediction step t
    # (0 <= t <= nh, including the EOS-prediction step) gets row t of cand.
    dk_cand_tensor = None
    if compute_candidates:
        V = nb_bonds + OPERATOR_OFFSET
        dk_cand_tensor = torch.zeros((len(batch), input_seq_len, V), dtype=torch.float32)
        for b, sample in enumerate(batch):
            cand = sample.get('deltaK_candidates', None)
            if cand is None:
                continue
            n_rows = min(cand.shape[0], input_seq_len)
            cand_t = torch.from_numpy(cand[:n_rows].astype(np.float32, copy=False))
            dk_cand_tensor[b, :n_rows, OPERATOR_OFFSET:OPERATOR_OFFSET + nb_bonds] = cand_t

    return tokens, padding_mask, target_parity, prefix_parity, prefix_len, deltaK_prefix_tensor, dk_cand_tensor, batch


# ============================================================================
# Transformer Model
# ============================================================================

class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        return self.dropout(x)


class AutoregressiveTransformer(nn.Module):
    """Autoregressive Transformer with prefix parity + deltaK embedding.

    Two modes for bond-logit computation:
      - default (dk_mlp_head=False): logits = output_proj(h)
      - dk_mlp_head=True (residual): logits = output_proj(h), and for bond tokens
        only, an additive correction MLP([h ; e(ΔK_b)]) is summed in. The MLP's
        final Linear is zero-initialized so at init the correction is exactly 0
        and the model behaves like the baseline. PAD/BOS/EOS unchanged.
    """

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=128, dropout=0.1, max_len=256,
                 dk_mlp_head=False, dk_head_dk=16, dk_head_hidden=128,
                 dk_head_centering=False, dk_head_bond_emb=0,
                 **kwargs):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.nb_bonds = vocab_size - OPERATOR_OFFSET
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.parity_embedding = nn.Embedding(2, d_model)
        self.deltaK_embedding = nn.Embedding(3, d_model)
        self.pos_encoder = LearnedPositionalEmbedding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.dk_mlp_head = bool(dk_mlp_head)
        self.dk_head_centering = bool(dk_head_centering)
        self.dk_head_bond_emb = int(dk_head_bond_emb)

        if self.dk_mlp_head:
            self.dk_head_embed = nn.Embedding(3, dk_head_dk)
            mlp_in_dim = d_model + dk_head_dk
            if self.dk_head_bond_emb > 0:
                self.bond_id_embed = nn.Embedding(self.nb_bonds, self.dk_head_bond_emb)
                mlp_in_dim += self.dk_head_bond_emb
            self.bond_head = nn.Sequential(
                nn.Linear(mlp_in_dim, dk_head_hidden),
                nn.GELU(),
                nn.Linear(dk_head_hidden, 1),
            )

        causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        self.register_buffer('causal_mask', causal_mask)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.parity_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.deltaK_embedding.weight, mean=0, std=0.02)
        if self.dk_mlp_head:
            nn.init.normal_(self.dk_head_embed.weight, mean=0, std=0.02)
            if self.dk_head_bond_emb > 0:
                nn.init.normal_(self.bond_id_embed.weight, mean=0, std=0.02)
            nn.init.normal_(self.bond_head[0].weight, mean=0, std=0.02)
            nn.init.zeros_(self.bond_head[0].bias)
            # Residual-style zero-init on the output linear so the correction
            # starts at exactly 0 → model behaves as baseline at init.
            nn.init.zeros_(self.bond_head[-1].weight)
            nn.init.zeros_(self.bond_head[-1].bias)

    def _compute_bond_correction(self, h, dk_candidates):
        """Residual MLP bond correction: MLP([h; e(ΔK_b)]) per candidate bond.

        Returns: (B, T, V) with zeros on PAD/BOS/EOS slots and the MLP correction
                 on the bond slots [OPERATOR_OFFSET : OPERATOR_OFFSET + nb_bonds].
        """
        B, T, _ = h.shape
        nb = self.nb_bonds
        bond_slice = dk_candidates[:, :, OPERATOR_OFFSET:OPERATOR_OFFSET + nb]
        dk_idx = (bond_slice + 1).long().clamp(0, 2)
        dk_emb = self.dk_head_embed(dk_idx)
        h_exp = h.unsqueeze(2).expand(B, T, nb, self.d_model)
        cat_parts = [h_exp, dk_emb]
        if self.dk_head_bond_emb > 0:
            bond_ids = torch.arange(nb, device=h.device).view(1, 1, nb).expand(B, T, nb)
            b_emb = self.bond_id_embed(bond_ids)
            cat_parts.append(b_emb)
        cat = torch.cat(cat_parts, dim=-1)
        bond_corr = self.bond_head(cat).squeeze(-1)
        if self.dk_head_centering:
            bond_corr = bond_corr - bond_corr.mean(dim=-1, keepdim=True)
        zero_special = bond_corr.new_zeros(B, T, OPERATOR_OFFSET)
        return torch.cat([zero_special, bond_corr], dim=-1)

    def forward(self, x, padding_mask=None, prefix_parity=None, deltaK_prefix=None,
                dk_candidates=None):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        if prefix_parity is not None:
            x = x + self.parity_embedding(prefix_parity)
        if deltaK_prefix is not None:
            x = x + self.deltaK_embedding(deltaK_prefix)
        x = self.pos_encoder(x)
        h = self.transformer(x, mask=self.causal_mask[:seq_len, :seq_len],
                             src_key_padding_mask=padding_mask)
        logits = self.output_proj(h)
        if self.dk_mlp_head and dk_candidates is not None:
            logits = logits + self._compute_bond_correction(h, dk_candidates)
        return logits


# ============================================================================
# Masking Functions
# ============================================================================

def apply_token_mask(logits, input_padding_mask=None):
    """Mask out illegal tokens (PAD, BOS)."""
    out = logits.clone()
    out[:, :, BOS_ID] = float("-inf")
    if input_padding_mask is None:
        out[:, :, PAD_ID] = float("-inf")
    else:
        nonpad = ~input_padding_mask
        out[:, :, PAD_ID] = out[:, :, PAD_ID].masked_fill(nonpad, float("-inf"))
    return out


def apply_nh_window_mask(logits, prefix_len, prefix_parity, target_parity,
                         nmin, nmax, input_padding_mask=None):
    """
    Apply n_h window mask (vectorized, memory-efficient).

    IMPORTANT: Must be called AFTER apply_token_mask(). Uses additive masking that
    preserves prior constraints (PAD/BOS already masked to -inf).

    When nmax is set: at nmax, EOS is forced regardless of parity. Both even and
    odd q overlap at nh=nmax; each still normalizes to 1 on [nmin, nmax] with
    E[h] = Z_even - Z_odd = 0 exactly (but h has cross-parity contamination at
    the boundary).

    When nmax is None: no upper bound is enforced. Each q's support is restricted
    by parity (q_even only terminates on even parity, q_odd only on odd), so the
    two supports are disjoint. Each q normalizes to 1 on its parity subspace;
    E[h] = 0 exactly with no cross-parity contamination at any position.

    Rules determined by prefix_len:
        - prefix_len < nmin: forbid EOS
        - nmin <= prefix_len (and < nmax if nmax set): EOS only if parity correct
        - prefix_len == nmax (if nmax set): only EOS (regardless of parity)

    Args:
        logits: (B, T, V) - already processed by apply_token_mask
        prefix_len: (B, T) - number of operators generated
        prefix_parity: (B, T) - parity at each position
        target_parity: int - target parity (0 or 1)
        nmin, nmax: int or None - n_h window bounds
        input_padding_mask: (B, T) - True for padded positions

    Returns:
        logits: (B, T, V) with additional masks applied
    """
    B, T, V = logits.shape
    device = logits.device

    # Valid positions (exclude padding)
    if input_padding_mask is not None:
        valid = ~input_padding_mask
    else:
        valid = torch.ones(B, T, dtype=torch.bool, device=device)

    # Stage 1: prefix_len < nmin -> forbid EOS (only if nmin set)
    if nmin is not None:
        mask1 = (prefix_len < nmin) & valid
        logits[:, :, EOS_ID] = logits[:, :, EOS_ID].masked_fill(mask1, float("-inf"))

    # Stage 2: in-window (nmin <= prefix_len < nmax, unbounded sides default True)
    #          -> EOS only if parity correct
    if nmin is not None:
        in_range = (prefix_len >= nmin)
    else:
        in_range = torch.ones_like(valid)
    if nmax is not None:
        in_range = in_range & (prefix_len < nmax)
    mask2 = in_range & valid
    mask2_wrong = mask2 & (prefix_parity != target_parity)
    logits[:, :, EOS_ID] = logits[:, :, EOS_ID].masked_fill(mask2_wrong, float("-inf"))

    # Stage 3: prefix_len == nmax -> only EOS (regardless of parity)
    if nmax is not None:
        mask3 = (prefix_len == nmax) & valid
        if mask3.any():
            logits_backup_eos = logits[:, :, EOS_ID].clone()
            logits[mask3] = float("-inf")
            logits[:, :, EOS_ID] = torch.where(mask3, logits_backup_eos, logits[:, :, EOS_ID])

    # No Stage 4 needed: if nh > nmax, Stage 3 already forces EOS at position nmax,
    # giving lp = -inf for the operator target there. Total logq = -inf + finite = -inf.
    # Setting all logits to -inf here would cause log_softmax → NaN, which is worse.

    return logits


# ============================================================================
# Density-ratio flatness helpers
# ============================================================================

def precompute_logfact_table(M):
    """Precompute log(n!) for n = 0..M as a float32 tensor."""
    return torch.tensor([math.lgamma(n + 1.0) for n in range(M + 1)], dtype=torch.float32)


def compute_logf_batch(raw_samples, beta, logfact_table, device):
    """Compute log f(D) = K*log(2) + nh*log(beta/2) - log(nh!) for each sample."""
    nh = torch.tensor([s['nh'] for s in raw_samples], dtype=torch.long, device=device)
    K = torch.tensor([s['K'] for s in raw_samples], dtype=torch.float32, device=device)
    logfact = logfact_table.to(device)
    log_beta_half = math.log(beta / 2.0)
    return K * math.log(2.0) + nh.float() * log_beta_half - logfact[nh]


def compute_ratio_flatness_loss(logq, log_target, min_batch=8):
    """Compute log(E[(q/target)^2] / E[q/target]^2), penalizing q/target variance.

    Numerically stable via logsumexp. Returns (loss_tensor, n_finite).
    """
    z = logq - log_target
    finite = torch.isfinite(z)
    z_clean = z[finite]
    n = z_clean.numel()
    if n < min_batch:
        return z.new_zeros(()), n
    log_m1 = torch.logsumexp(z_clean, dim=0) - math.log(n)
    log_m2 = torch.logsumexp(2.0 * z_clean, dim=0) - math.log(n)
    return log_m2 - 2.0 * log_m1, n


# ============================================================================
# Training and Evaluation
# ============================================================================

def compute_loss(model, batch, target_parity, pad_id, device, nmin, nmax,
                 beta=0.0, logfact_table=None, ce_reduction='token_mean',
                 lambda_flat=0.0, flat_min_batch=8):
    """Compute n_h-weighted cross-entropy loss with n_h window masking.

    Minimizes E[-n_h * log q(x)] to train q ∝ n_h * f.
    """
    tokens, padding_mask, batch_parity, prefix_parity, prefix_len, deltaK_prefix, dk_candidates, raw_samples = batch

    tokens = tokens.to(device, non_blocking=True)
    padding_mask = padding_mask.to(device, non_blocking=True)
    prefix_parity = prefix_parity.to(device, non_blocking=True)
    prefix_len = prefix_len.to(device, non_blocking=True)
    deltaK_prefix = deltaK_prefix.to(device, non_blocking=True)
    if dk_candidates is not None:
        dk_candidates = dk_candidates.to(device, non_blocking=True)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    input_padding_mask = padding_mask[:, :-1]

    logits = model(inputs, input_padding_mask, prefix_parity=prefix_parity,
                   deltaK_prefix=deltaK_prefix, dk_candidates=dk_candidates)

    logits = apply_token_mask(logits, input_padding_mask)
    logits = apply_nh_window_mask(
        logits, prefix_len, prefix_parity, target_parity,
        nmin, nmax, input_padding_mask=input_padding_mask
    )

    B, T = targets.shape
    valid_mask = ~padding_mask[:, 1:]

    ce_flat = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction='none',
    )
    ce = ce_flat.reshape(B, T)

    # Per-sample sequence NLL and nh weights
    seq_nll = (ce * valid_mask.float()).sum(dim=1)   # (B,)
    nh_batch = torch.tensor([s['nh'] for s in raw_samples], dtype=torch.float32, device=device)
    nh_sum = nh_batch.sum()
    n_valid = valid_mask.float().sum()

    # n_h-weighted CE loss with selected reduction
    if ce_reduction == 'seq_mean':
        loss_ce = (nh_batch * seq_nll).mean()
    else:  # token_mean (default, backward compat)
        loss_ce = (nh_batch * seq_nll).sum() / nh_sum

    # Per-component CE diagnostics (operator vs EOS)
    op_mask = (targets >= OPERATOR_OFFSET) & valid_mask
    eos_mask = (targets == EOS_ID) & valid_mask
    op_ce = ce[op_mask].sum().item() if op_mask.any() else 0.0
    eos_ce = ce[eos_mask].sum().item() if eos_mask.any() else 0.0
    n_op = int(op_mask.sum().item())
    n_eos = int(eos_mask.sum().item())

    # Density-ratio flatness loss: z = logq - log(nh*f) for numerator (q ∝ nh*f)
    loss_flat = torch.zeros((), device=device)
    loss_flat_val = 0.0
    n_finite_z = 0
    z_stats = {}
    if logfact_table is not None:
        logq = -seq_nll                                          # (B,)
        logf = compute_logf_batch(raw_samples, beta, logfact_table, device)  # (B,)
        log_nh = torch.log(nh_batch.clamp_min(1))                # log(nh), safe for nh=0
        log_target = logf + log_nh                               # log(nh * f)
        with torch.no_grad():
            z_det = logq.detach() - log_target
            finite = torch.isfinite(z_det)
            z_clean = z_det[finite]
            n_finite_z = z_clean.numel()
            if n_finite_z > 0:
                z_stats = {
                    'z_mean': z_clean.mean().item(),
                    'z_std': z_clean.std().item() if n_finite_z > 1 else 0.0,
                    'z_max': z_clean.max().item(),
                }
                with torch.no_grad():
                    log_m1 = torch.logsumexp(z_clean, dim=0) - math.log(n_finite_z)
                    log_m2 = torch.logsumexp(2.0 * z_clean, dim=0) - math.log(n_finite_z)
                    z_stats['flat_diag'] = (log_m2 - 2.0 * log_m1).item()
        if lambda_flat > 0.0:
            loss_flat, n_finite_z = compute_ratio_flatness_loss(logq, log_target, flat_min_batch)
            loss_flat_val = loss_flat.item()

    total_loss = loss_ce + lambda_flat * loss_flat

    return {
        "loss": total_loss,
        "loss_ce": loss_ce.item(),
        "loss_flat": loss_flat_val,
        "token_loss": loss_ce.item(),
        "op_ce": op_ce,
        "eos_ce": eos_ce,
        "n_op": n_op,
        "n_eos": n_eos,
        "weight": nh_sum.item(),
        "mean_seq_nll": seq_nll.mean().item(),
        "mean_token_nll": (ce.sum() / n_valid.clamp_min(1)).item(),
        "z_mean": z_stats.get('z_mean', 0.0),
        "z_std": z_stats.get('z_std', 0.0),
        "z_max": z_stats.get('z_max', 0.0),
        "flat_diag": z_stats.get('flat_diag', 0.0),
        "n_finite_z": n_finite_z,
        "batch_size": B,
    }


def train_epoch(model, dataloader, optimizer, target_parity, pad_id, device, nmin, nmax,
                beta=0.0, logfact_table=None, ce_reduction='token_mean',
                lambda_flat=0.0, flat_min_batch=8, epoch=0, flat_warmup_epochs=50,
                global_step=0, grad_accum_steps=1):
    """Train for one epoch with n_h weighting + optional ΔK auxiliary loss + flatness."""
    model.train()
    acc = {k: 0.0 for k in ["loss", "weight",
                             "op_ce", "eos_ce", "n_op", "n_eos",
                             "loss_ce_sum", "loss_flat_sum", "flat_diag_sum",
                             "n_batches", "seq_nll_sum", "token_nll_sum",
                             "z_mean_sum", "z_std_sum", "z_max_max",
                             "n_samples"]}
    acc["z_max_max"] = float("-inf")

    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(dataloader):
        lambda_eff = 0.0 if epoch < flat_warmup_epochs else lambda_flat

        r = compute_loss(model, batch, target_parity, pad_id, device, nmin, nmax,
                         beta=beta, logfact_table=logfact_table, ce_reduction=ce_reduction,
                         lambda_flat=lambda_eff, flat_min_batch=flat_min_batch)
        (r["loss"] / grad_accum_steps).backward()

        if (i + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        w = r["weight"]
        bs = r["batch_size"]
        acc["loss"] += r["token_loss"] * w
        acc["weight"] += w
        acc["op_ce"] += r["op_ce"]
        acc["eos_ce"] += r["eos_ce"]
        acc["n_op"] += r["n_op"]
        acc["n_eos"] += r["n_eos"]
        acc["loss_ce_sum"] += r["loss_ce"] * bs
        acc["loss_flat_sum"] += r["loss_flat"] * bs
        acc["flat_diag_sum"] += r["flat_diag"] * bs
        acc["n_batches"] += 1
        acc["seq_nll_sum"] += r["mean_seq_nll"] * bs
        acc["token_nll_sum"] += r["mean_token_nll"] * bs
        acc["z_mean_sum"] += r["z_mean"] * bs
        acc["z_std_sum"] += r["z_std"] * bs
        if r["z_max"] != 0.0:
            acc["z_max_max"] = max(acc["z_max_max"], r["z_max"])
        acc["n_samples"] += bs

    # Flush remaining accumulated gradients
    if (i + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    if acc["weight"] == 0:
        raise ValueError("Train dataloader produced zero n_h weight.")
    ns = max(acc["n_samples"], 1)
    return {
        "token_loss": acc["loss"] / acc["weight"],
        "op_ce": acc["op_ce"] / acc["n_op"] if acc["n_op"] > 0 else 0.0,
        "eos_ce": acc["eos_ce"] / acc["n_eos"] if acc["n_eos"] > 0 else 0.0,
        "loss_ce": acc["loss_ce_sum"] / ns,
        "loss_flat": acc["loss_flat_sum"] / ns,
        "flat_diag": acc["flat_diag_sum"] / ns,
        "mean_seq_nll": acc["seq_nll_sum"] / ns,
        "mean_token_nll": acc["token_nll_sum"] / ns,
        "z_mean": acc["z_mean_sum"] / ns,
        "z_std": acc["z_std_sum"] / ns,
        "z_max": acc["z_max_max"] if acc["z_max_max"] != float("-inf") else 0.0,
    }, global_step


def evaluate(model, dataloader, target_parity, pad_id, device, nmin, nmax,
             beta=0.0, logfact_table=None, ce_reduction='token_mean',
             lambda_flat=0.0, flat_min_batch=8):
    model.eval()
    acc = {k: 0.0 for k in ["loss", "weight",
                             "op_ce", "eos_ce", "n_op", "n_eos",
                             "loss_ce_sum", "loss_flat_sum", "flat_diag_sum",
                             "n_batches", "seq_nll_sum", "token_nll_sum",
                             "z_mean_sum", "z_std_sum", "z_max_max",
                             "n_samples"]}
    acc["z_max_max"] = float("-inf")

    with torch.inference_mode():
        for batch in dataloader:
            r = compute_loss(model, batch, target_parity, pad_id, device, nmin, nmax,
                             beta=beta, logfact_table=logfact_table, ce_reduction=ce_reduction,
                             lambda_flat=lambda_flat, flat_min_batch=flat_min_batch)
            w = r["weight"]
            bs = r["batch_size"]
            acc["loss"] += r["token_loss"] * w
            acc["weight"] += w
            acc["op_ce"] += r["op_ce"]
            acc["eos_ce"] += r["eos_ce"]
            acc["n_op"] += r["n_op"]
            acc["n_eos"] += r["n_eos"]
            acc["loss_ce_sum"] += r["loss_ce"] * bs
            acc["loss_flat_sum"] += r["loss_flat"] * bs
            acc["flat_diag_sum"] += r["flat_diag"] * bs
            acc["n_batches"] += 1
            acc["seq_nll_sum"] += r["mean_seq_nll"] * bs
            acc["token_nll_sum"] += r["mean_token_nll"] * bs
            acc["z_mean_sum"] += r["z_mean"] * bs
            acc["z_std_sum"] += r["z_std"] * bs
            if r["z_max"] != 0.0:
                acc["z_max_max"] = max(acc["z_max_max"], r["z_max"])
            acc["n_samples"] += bs

    if acc["weight"] == 0:
        raise ValueError("Val dataloader produced zero n_h weight.")
    ns = max(acc["n_samples"], 1)
    return {
        "token_loss": acc["loss"] / acc["weight"],
        "op_ce": acc["op_ce"] / acc["n_op"] if acc["n_op"] > 0 else 0.0,
        "eos_ce": acc["eos_ce"] / acc["n_eos"] if acc["n_eos"] > 0 else 0.0,
        "loss_ce": acc["loss_ce_sum"] / ns,
        "loss_flat": acc["loss_flat_sum"] / ns,
        "flat_diag": acc["flat_diag_sum"] / ns,
        "mean_seq_nll": acc["seq_nll_sum"] / ns,
        "mean_token_nll": acc["token_nll_sum"] / ns,
        "z_mean": acc["z_mean_sum"] / ns,
        "z_std": acc["z_std_sum"] / ns,
        "z_max": acc["z_max_max"] if acc["z_max_max"] != float("-inf") else 0.0,
    }


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Transformer with auto n_h window')

    parser.add_argument('--parity', type=str, required=True, choices=['even', 'odd'])
    parser.add_argument('--data_glob', type=str,
                        default='/home/user_beiqiao/private/datafile/rsse_data/fortran2/*.bin')
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_samples_per_file', type=int, default=None)
    parser.add_argument('--allow_mixed_mm', type=int, default=0)

    # n_h window parameters
    parser.add_argument('--auto_nh_window', type=int, default=0)
    parser.add_argument('--left_tail_mass', type=float, default=1e-4)
    parser.add_argument('--right_tail_mass', type=float, default=1e-4)
    parser.add_argument('--max_scan_samples', type=int, default=200000)
    parser.add_argument('--manual_nmin', type=int, default=None)
    parser.add_argument('--manual_nmax', type=int, default=None)
    parser.add_argument('--show_nh_bins', type=int, default=1)

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=256)

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--shuffle_buffer', type=int, default=10000)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--train_fraction', type=float, default=0.8)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4)
    parser.add_argument('--save_all_checkpoints', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--spatial_aug', action='store_true',
                        help='Enable spatial translation augmentation (requires lx>=3 and |ly|>=3)')
    parser.add_argument('--pointgroup_aug', action='store_true',
                        help='Enable D_6 point-group augmentation (requires lx=|ly| triangular torus)')
    parser.add_argument('--no_cyclic_aug', action='store_true',
                        help='Disable cyclic shift augmentation (enabled by default)')
    parser.add_argument('--dk_mlp_head', type=int, default=0,
                        help='Replace bond-logit computation with MLP([h ; e(ΔK_b)]) per candidate. '
                             'PAD/BOS/EOS scored by a separate small head.')
    parser.add_argument('--dk_head_dk', type=int, default=32,
                        help='ΔK embedding dim for the MLP head.')
    parser.add_argument('--dk_head_hidden', type=int, default=256,
                        help='Hidden dim of the MLP head.')
    parser.add_argument('--dk_head_centering', action='store_true',
                        help='Force bond_corr to be zero-mean across 27 bonds at '
                             'each position; anchors q+/q- to a shared bond-vs-EOS '
                             'baseline.')
    parser.add_argument('--dk_head_bond_emb', type=int, default=0,
                        help='Per-bond identity embedding dim concatenated to MLP head '
                             'input. 0 disables (default).')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_v2_nh_window')

    # Flatness loss parameters
    parser.add_argument('--ce_reduction', type=str, default='token_mean',
                        choices=['token_mean', 'seq_mean'],
                        help='CE reduction: token_mean (default) or seq_mean (per-sequence average).')
    parser.add_argument('--lambda_flat', type=float, default=0.0,
                        help='Weight for density-ratio flatness loss (0=disabled).')
    parser.add_argument('--flat_warmup_epochs', type=int, default=50,
                        help='Number of epochs before enabling flatness loss.')
    parser.add_argument('--flat_min_batch', type=int, default=8,
                        help='Minimum finite z samples per batch to compute flatness loss.')
    parser.add_argument('--early_stop_metric', type=str, default='ce',
                        choices=['ce', 'total'],
                        help='Metric for early stopping: ce (CE only) or total (CE + flat).')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (>1 stabilizes flat loss).')
    parser.add_argument('--train_nmax', type=int, default=None,
                        help='Dataset-level upper nh cutoff: samples with nh > train_nmax '
                             'are skipped during training. Does not affect model masking '
                             '(use with manual_nmax=None for soft cutoff).')

    args = parser.parse_args()

    target_parity = 0 if args.parity == 'even' else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    file_paths = sorted(glob.glob(args.data_glob))
    if not file_paths:
        raise ValueError(f"No files found: {args.data_glob}")
    if args.max_files:
        file_paths = file_paths[:args.max_files]

    print(f"Found {len(file_paths)} files")
    print(f"Parity: {args.parity} (target_parity={target_parity})\n")

    # Determine n_h window
    nmin, nmax = None, None
    if args.manual_nmin is not None:
        nmin = args.manual_nmin
    if args.manual_nmax is not None:
        nmax = args.manual_nmax

    # Auto-scan when needed by auto_nh_window OR train_nmax=-1
    need_auto_scan = (args.auto_nh_window and (nmin is None or nmax is None)) \
                     or (args.train_nmax is not None and args.train_nmax < 0)
    auto_nmax_for_dataset = None

    if need_auto_scan:
        print("Scanning n_h distribution...")
        stats = scan_nh_histogram_v2(file_paths, args.max_scan_samples)
        hist = stats['hist']
        total = stats['total_samples']

        if total <= 0:
            raise ValueError(f"No samples scanned: total_samples={total}")
        if not hist:
            raise ValueError("Empty histogram from scan")

        print(f"  Scanned {total} samples, n_h range: [{min(hist.keys())}, {max(hist.keys())}]")
        print(f"  Mean n_h: {stats['mean_nh']:.2f}")

        # Auto compute nmin/nmax using n_h-weighted cumulative distribution (numerator)
        auto_nmin, auto_nmax = choose_nh_window_from_hist(
            hist, args.left_tail_mass, args.right_tail_mass, weighted=True
        )
        print(f"  Auto n_h window (numerator-weighted): nmin={auto_nmin}, nmax={auto_nmax}")

        if args.auto_nh_window:
            if nmin is None:
                nmin = auto_nmin
            if nmax is None:
                nmax = auto_nmax

        # train_nmax=-1: use auto nmax for dataset filter, auto nmin for model mask
        if args.train_nmax is not None and args.train_nmax < 0:
            auto_nmax_for_dataset = auto_nmax
            if nmin is None:
                nmin = auto_nmin
            print(f"  auto train_nmax → {auto_nmax}, nmin → {nmin}")

    if nmin is not None or nmax is not None:
        print(f"n_h window (model mask): [{nmin}, {nmax}]")
    else:
        print("No n_h window mask applied")

    # Dataset nh_max: train_nmax > 0 → explicit; train_nmax < 0 → auto; None → fall back to model nmax
    if args.train_nmax is not None and args.train_nmax > 0:
        dataset_nmax = args.train_nmax
    elif auto_nmax_for_dataset is not None:
        dataset_nmax = auto_nmax_for_dataset
    else:
        dataset_nmax = nmax

    if dataset_nmax != nmax:
        print(f"train_nmax (dataset filter): {dataset_nmax}  "
              f"(model nmax={nmax}, dataset skips nh > {dataset_nmax})")
    print()

    # Create datasets
    train_dataset = RSSEStreamingDatasetV2(
        file_paths=file_paths,
        target_parity=target_parity,
        max_samples_per_file=args.max_samples_per_file,
        shuffle_buffer=args.shuffle_buffer,
        allow_mixed_mm=bool(args.allow_mixed_mm),
        stride=args.stride,
        split='train',
        train_fraction=args.train_fraction,
        nh_min=nmin,
        nh_max=dataset_nmax,
    )

    val_dataset = RSSEStreamingDatasetV2(
        file_paths=file_paths,
        target_parity=target_parity,
        max_samples_per_file=args.max_samples_per_file,
        shuffle_buffer=0,
        allow_mixed_mm=bool(args.allow_mixed_mm),
        stride=args.stride,
        split='val',
        train_fraction=args.train_fraction,
        nh_min=nmin,
        nh_max=dataset_nmax,
    )

    # Get lattice info
    nb = train_dataset.nb
    lx = train_dataset.lx
    ly = train_dataset.ly
    beta = train_dataset.beta
    nn = train_dataset.nn

    vocab_size = nb + OPERATOR_OFFSET
    print(f"Vocabulary: nb={nb}, vocab_size={vocab_size}\n")

    # Build bsites for cyclic augmentation
    bsites_arr, nn_sites, nb_bonds = build_bsites(lx, ly)

    cyclic_aug = not args.no_cyclic_aug
    print(f"Cyclic augmentation {'enabled' if cyclic_aug else 'disabled'}: bsites shape={bsites_arr.shape}, nn={nn_sites}, nb={nb_bonds}\n")

    spatial_perm = None
    if args.spatial_aug:
        if lx < 3 or abs(ly) < 3:
            raise ValueError(f"--spatial_aug requires lx>=3 and |ly|>=3, got lx={lx}, ly={ly}")
        spatial_perm = build_spatial_bond_perm(lx, ly, bsites_arr, nb_bonds)
        print(f"Spatial translation augmentation enabled: {spatial_perm.shape[0]} shifts\n")

    pointgroup_perm = None
    if args.pointgroup_aug:
        if lx != abs(ly):
            raise ValueError(f"--pointgroup_aug requires lx=|ly| triangular torus, got lx={lx}, ly={ly}")
        pointgroup_perm = build_pointgroup_bond_perm(lx, ly, bsites_arr, nb_bonds)
        print(f"Point-group (D_6) augmentation enabled: {pointgroup_perm.shape[0]} elements\n")

    # Create collate functions: always pass bsites (needed for V3 + augmentation)
    # ΔK candidates are needed whenever the MLP bond-logit head is active.
    compute_candidates = bool(args.dk_mlp_head)
    if compute_candidates:
        print(f"dk_mlp_head enabled: bond logits via MLP([h ; e(ΔK_b)"
              + (" ; e(b)]" if args.dk_head_bond_emb > 0 else "])")
              + f" (d_k={args.dk_head_dk}, hidden={args.dk_head_hidden}, "
              + f"centering={'on' if args.dk_head_centering else 'off'}, "
              + f"bond_emb={args.dk_head_bond_emb})\n")
    train_collate = functools.partial(
        collate_fn_parity_v2_aug,
        bsites=bsites_arr, nn_sites=nn_sites, nb_bonds=nb_bonds, augment=True,
        cyclic_aug=cyclic_aug,
        spatial_perm=spatial_perm, pointgroup_perm=pointgroup_perm,
        compute_candidates=compute_candidates,
    )
    val_collate = functools.partial(
        collate_fn_parity_v2_aug,
        bsites=bsites_arr, nn_sites=nn_sites, nb_bonds=nb_bonds, augment=False,
        spatial_perm=None, pointgroup_perm=None,
        compute_candidates=compute_candidates,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=val_collate,
        num_workers=args.num_workers
    )

    # Create model
    model = AutoregressiveTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        dk_mlp_head=bool(args.dk_mlp_head),
        dk_head_dk=args.dk_head_dk,
        dk_head_hidden=args.dk_head_hidden,
        dk_head_centering=args.dk_head_centering,
        dk_head_bond_emb=args.dk_head_bond_emb,
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    global_step = 0

    # Flatness loss setup
    logfact_table = precompute_logfact_table(args.max_len + 10)
    flat_kwargs = dict(
        beta=beta, logfact_table=logfact_table, ce_reduction=args.ce_reduction,
        lambda_flat=args.lambda_flat, flat_min_batch=args.flat_min_batch,
    )
    if args.lambda_flat > 0:
        print(f"Flatness loss enabled: lambda_flat={args.lambda_flat}, "
              f"warmup={args.flat_warmup_epochs} epochs, ce_reduction={args.ce_reduction}")
    elif args.ce_reduction != 'token_mean':
        print(f"CE reduction: {args.ce_reduction}")
    print(f"Early stop metric: {args.early_stop_metric}\n")

    # First-batch telescope self-check on the (B,T,V) tensor:
    #   Σ_{t=0..nh-1} tensor[b_idx, t, OPERATOR_OFFSET+b_t] == K - nn.
    if compute_candidates:
        first_batch = next(iter(val_loader))
        _, _, _, _, _, _, dkc, raw = first_batch
        assert dkc is not None, "dk_candidates tensor is None despite --dk_mlp_head=1"
        n_checked, max_abs_err = 0, 0.0
        for b_idx, s in enumerate(raw):
            nh = s['nh']
            if nh <= 0:
                continue
            x = s['x_dense']
            K = s['K']
            bonds = (x // 2).astype(np.int64) - 1
            tele = 0.0
            for t, bt in enumerate(bonds):
                tele += float(dkc[b_idx, t, OPERATOR_OFFSET + int(bt)].item())
            err = abs(tele - float(K - nn_sites))
            max_abs_err = max(max_abs_err, err)
            n_checked += 1
        if max_abs_err > 1e-4:
            raise RuntimeError(
                f"Telescope self-check FAILED on first val batch: "
                f"max|Σcand − (K−nn)| = {max_abs_err} across {n_checked} samples."
            )
        print(f"  ✓ dk_cand telescope self-check passed on {n_checked} samples "
              f"(max err = {max_abs_err:.2e}).\n")

    for epoch in range(args.num_epochs):
        train_dataset._epoch = epoch  # Different shuffle + augmentation each epoch

        tm, global_step = train_epoch(model, train_loader, optimizer, target_parity,
                                      PAD_ID, device, nmin, nmax,
                                      flat_warmup_epochs=args.flat_warmup_epochs,
                                      epoch=epoch, global_step=global_step,
                                      grad_accum_steps=args.grad_accum_steps,
                                      **flat_kwargs)
        vm = evaluate(model, val_loader, target_parity, PAD_ID, device, nmin, nmax,
                      **flat_kwargs)

        train_loss = tm["token_loss"]
        val_loss = vm["token_loss"]

        print(f"Epoch {epoch + 1}/{args.num_epochs}  (global_step={global_step})")
        print(f"  Train CE: {tm['loss_ce']:.4f}  (op_ce={tm['op_ce']:.4f}  eos_ce={tm['eos_ce']:.4f})"
              f"  seq_nll={tm['mean_seq_nll']:.2f}  tok_nll={tm['mean_token_nll']:.4f}")
        print(f"  Val   CE: {vm['loss_ce']:.4f}  (op_ce={vm['op_ce']:.4f}  eos_ce={vm['eos_ce']:.4f})"
              f"  seq_nll={vm['mean_seq_nll']:.2f}  tok_nll={vm['mean_token_nll']:.4f}")
        if logfact_table is not None:
            lambda_eff_t = args.lambda_flat if epoch >= args.flat_warmup_epochs else 0.0
            train_total = tm['loss_ce'] + lambda_eff_t * tm['loss_flat']
            val_total = vm['loss_ce'] + lambda_eff_t * vm['loss_flat']
            print(f"  Train z: mean={tm['z_mean']:.3f}  std={tm['z_std']:.3f}  max={tm['z_max']:.3f}"
                  f"  flat={tm['flat_diag']:.6f}  total={train_total:.4f}")
            print(f"  Val   z: mean={vm['z_mean']:.3f}  std={vm['z_std']:.3f}  max={vm['z_max']:.3f}"
                  f"  flat={vm['flat_diag']:.6f}  total={val_total:.4f}")

        # Early stopping metric (use CE only during warmup)
        if args.early_stop_metric == 'total' and epoch >= args.flat_warmup_epochs:
            es_val = vm["loss_ce"] + args.lambda_flat * vm["loss_flat"]
        else:
            es_val = vm["loss_ce"]

        # Reset early stopping when transitioning from warmup to flat-enabled
        if args.early_stop_metric == 'total' and epoch == args.flat_warmup_epochs:
            best_val_loss = float('inf')
            patience_counter = 0
            print(f"  [warmup ended, early stopping reset to total metric]")

        if es_val < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = es_val
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'parity': args.parity,
                'target_parity': target_parity,
                'vocab_size': vocab_size,
                'nb': nb,
                'lx': lx,
                'ly': ly,
                'nn': nn,
                'beta': beta,
                'nn_sites': nn_sites,
                'bsites': bsites_arr.tolist(),
                'nmin': nmin,
                'nmax': nmax,
                'dk_mlp_head': bool(args.dk_mlp_head),
                'dk_head_dk': args.dk_head_dk,
                'dk_head_hidden': args.dk_head_hidden,
                'version': 2,
                'loss_type': 'nh_weighted',
                'ce_reduction': args.ce_reduction,
                'lambda_flat': args.lambda_flat,
                'flat_warmup_epochs': args.flat_warmup_epochs,
                'flat_min_batch': args.flat_min_batch,
                'early_stop_metric': args.early_stop_metric,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ Best model saved (es_metric={es_val:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{args.early_stopping_patience})")

        if args.save_all_checkpoints:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'nmin': nmin,
                'nmax': nmax,
                'loss_type': 'nh_weighted',
            }, os.path.join(args.output_dir, f'model_{args.parity}_epoch{epoch + 1}.pt'))

        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best epoch: {best_epoch} (es_metric: {best_val_loss:.4f})")
            break
        print()

    print("Training complete!")


if __name__ == '__main__':
    main()

