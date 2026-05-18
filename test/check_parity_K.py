#!/usr/bin/env python3
"""Check stored parity vs Fortran-recomputed values (V4 format: no K in file)."""
import struct, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix

nn, nb = 3, 3
bsites = np.zeros((2, nb), dtype=np.int32, order='F')
bsites[0,0]=1; bsites[1,0]=2
bsites[0,1]=2; bsites[1,1]=3
bsites[0,2]=3; bsites[1,2]=1

path = "/home/user_beiqiao/private/datafile/rsse_data/fortran3_aug/3x1/beta10/train/rsse_L3x1_beta10.000_seed4102_M57.bin"
mp, total = 0, 0

with open(path, "rb") as f:
    magic = f.read(4)
    version = struct.unpack("<i", f.read(4))[0]
    f.read(36)  # lx, ly, nn, nb, mm, beta, surface_n

    if magic == b"RSS4":
        # V4: (nh, parity, opstring)
        for _ in range(50000):
            data = f.read(8)
            if len(data) < 8:
                break
            nh, p_s = struct.unpack("<2i", data)
            if nh > 0:
                ops = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                pp, dkp, K_c = compute_parity_prefix(ops, bsites, nn, nb)
                p_c = int(pp[-1])
                if p_s != p_c:
                    mp += 1
                    if mp <= 5:
                        print(f"  parity mismatch #{mp}: nh={nh}, stored={p_s}, computed={p_c}")
            total += 1
    else:
        # V3: (nh, K, parity, opstring)
        for _ in range(50000):
            data = f.read(12)
            if len(data) < 12:
                break
            nh, K_s, p_s = struct.unpack("<3i", data)
            if nh > 0:
                ops = np.frombuffer(f.read(4*nh), dtype="<i4").copy()
                pp, dkp, K_c = compute_parity_prefix(ops, bsites, nn, nb)
                p_c = int(pp[-1])
                if p_s != p_c:
                    mp += 1
                    if mp <= 5:
                        print(f"  parity mismatch #{mp}: nh={nh}, stored={p_s}, computed={p_c}")
            total += 1

print(f"Total: {total}")
print(f"Parity mismatches: {mp} ({mp/total*100:.2f}%)")
