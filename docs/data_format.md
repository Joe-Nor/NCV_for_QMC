# Data Format

The training and evaluation scripts stream RSSE samples directly from binary
files produced by the Fortran sampler. The current writer emits the `RSS4`
format. Older `RSSE`/V2 and `RSS3`/V3 files are still accepted by the Python
readers where noted.

All integer fields are little-endian 32-bit signed integers unless stated
otherwise. Floating-point header fields are little-endian 64-bit floats.

## Header

Every supported file starts with a 44-byte header:

```text
char[4]  magic       "RSSE", "RSS3", or "RSS4"
int32    version     2, 3, or 4
int32    lx
int32    ly
int32    nn          number of sites
int32    nb          number of bonds
int32    mm          current SSE cutoff
float64  beta
float64  surface_n   SU(N) parameter used by the sampler
```

The Python readers check that `magic` and `version` agree.

## RSS4 Records

`RSS4` is the current public format written by
`fortran/rsse_update_loops_cursor_optimized_v3.f90`.

Each sample record is:

```text
int32      nh
int32      parity              0 for even, 1 for odd
int32[nh]  opstring_uncolored  entries are 2*b for bond index b
```

`RSS4` does not store `K` or the parity prefix. The Python pipeline recomputes
parity prefixes, delta-K prefixes, and `K` from `opstring_uncolored` using the
Fortran helper libraries in `src/`.

## Older Records

V2 files use magic `RSSE`:

```text
int32      nh
int32      K
int8[nh]   parity_prefix
int32[nh]  opstring_uncolored
```

V3 files use magic `RSS3`:

```text
int32      nh
int32      K
int32      parity
int32[nh]  opstring_uncolored
```

The current training and evaluation scripts accept these files for backward
compatibility.

## Generated Artifacts

Large generated data, checkpoints, logs, and compiled objects are intentionally
not tracked in git. Use external archival storage for paper-scale datasets and
checkpoints, and record the exact commit hash and command line used to produce
them.

Recommended local layout:

```text
data/
  raw/                 RSSE binary files
checkpoints/
  numerator/even/
  numerator/odd/
  denominator/even/
  denominator/odd/
results/
```

## Checkpoints

Training scripts save PyTorch checkpoints containing at least:

```text
model_state_dict
optimizer_state_dict
epoch
vocab_size
nmin, nmax
args
```

The evaluation scripts use the checkpoint `args` dictionary to reconstruct the
model architecture, so keep checkpoint files paired with the code version that
created them.

## Related Files

- [Installation Guide](installation.md)
- [Method Description](method.md)
- [Associated arXiv Paper](https://arxiv.org/abs/2605.26814)
