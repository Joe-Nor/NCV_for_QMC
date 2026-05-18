#!/usr/bin/env python3
"""Check if cyclic rotation preserves total parity and K."""
import struct, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from parity_prefix_wrapper import compute_parity_prefix

nn, nb = 3, 3
bsites = np.zeros((2, nb), dtype=np.int32, order='F')
bsites[0,0]=1; bsites[1,0]=2
bsites[0,1]=2; bsites[1,1]=3
bsites[0,2]=3; bsites[1,2]=1

path = "/home/user_beiqiao/private/datafile/rsse_data/fortran3_aug/3x1/beta10/train/rsse_L3x1_beta10.000_seed4102_M57.bin"
parity_flip = 0
K_change = 0
total = 0

with open(path, "rb") as f:
    magic = f.read(4)
    version = struct.unpack("<i", f.read(4))[0]
    f.read(36)  # lx, ly, nn, nb, mm, beta, surface_n

    for _ in range(10000):
        if magic == b"RSS4":
            data = f.read(8)
            if len(data) < 8:
                break
            nh, p_s = struct.unpack("<2i", data)
        else:
            data = f.read(12)
            if len(data) < 12:
                break
            nh, K_s, p_s = struct.unpack("<3i", data)

        if nh <= 1:
            f.read(4 * nh)
            total += 1
            continue
        ops = np.frombuffer(f.read(4*nh), dtype="<i4").copy()

        # Original
        pp0, dkp0, K0 = compute_parity_prefix(ops, bsites, nn, nb)
        p0 = int(pp0[-1])

        # Check ALL cyclic rotations
        for k in range(1, nh):
            ops_rot = np.roll(ops, -k)
            pp_r, dkp_r, K_r = compute_parity_prefix(ops_rot, bsites, nn, nb)
            p_r = int(pp_r[-1])
            if p_r != p0:
                parity_flip += 1
                if parity_flip <= 5:
                    print(f"PARITY FLIP: sample {total}, nh={nh}, shift={k}, orig_p={p0}, rot_p={p_r}")
                    print(f"  ops_orig: {ops[:10]}...")
                    print(f"  ops_rot:  {ops_rot[:10]}...")
            if K_r != K0:
                K_change += 1
                if K_change <= 5:
                    print(f"K CHANGE: sample {total}, nh={nh}, shift={k}, orig_K={K0}, rot_K={K_r}")

        total += 1

print(f"\nTotal samples: {total}")
print(f"Parity flips across all rotations: {parity_flip}")
print(f"K changes across all rotations: {K_change}")
