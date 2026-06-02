# Public Checkpoints

Do not commit checkpoint binaries to the git repository. Keep the repository for
source code, documentation, and manifests, and archive checkpoint bundles as
release assets or on an external service with a persistent DOI.

## Download

The curated public bundle for `v1.0.0` is available from GitHub Releases:

- Release: <https://github.com/Joe-Nor/NCV_for_QMC/releases/tag/v1.0.0>
- Archive: <https://github.com/Joe-Nor/NCV_for_QMC/releases/download/v1.0.0/public_checkpoints_ncv.tar.gz>
- SHA256: <https://github.com/Joe-Nor/NCV_for_QMC/releases/download/v1.0.0/public_checkpoints_ncv.tar.gz.sha256>

Archive checksum:

```text
b9251b3e4ce3ab26021aa9961ce82f3741b5961cdded1faab8cab441b7d4f1d4  public_checkpoints_ncv.tar.gz
```

The `v1.0.0` archive contains 84 inference checkpoints: 11 temperatures for
`2x-2` and 10 temperatures for `3x1`, with numerator/denominator and even/odd
models for each temperature.

## Recommended Scope

Archive only the inference checkpoints used for the reported results:

- one `best_model.pt` per system and temperature;
- four models per system/temperature: numerator-even, numerator-odd,
  denominator-even, denominator-odd;
- final production configuration directories, by default
  `*_200k_Mnmax_seqmean_bondemb`.

Do not archive intermediate epoch checkpoints, Slurm logs, plots, scratch
scripts, or ablation runs unless they are specifically needed for a separate
claim.

## Prepare a Bundle

Run a dry-run first:

```bash
python scripts/prepare_public_checkpoints.py \
  --source-root /path/to/CV_checkpoint3_aug_bias/3x1 \
  --source-root /path/to/CV_checkpoint3_aug_bias/2x-2 \
  --output-dir /path/to/public_checkpoints \
  --dry-run
```

Create an inference bundle with optimizer state removed:

```bash
python scripts/prepare_public_checkpoints.py \
  --source-root /path/to/CV_checkpoint3_aug_bias/3x1 \
  --source-root /path/to/CV_checkpoint3_aug_bias/2x-2 \
  --output-dir /path/to/public_checkpoints \
  --mode inference
```

If PyTorch is unavailable in the current environment, copy the raw `best_model.pt`
files instead:

```bash
python scripts/prepare_public_checkpoints.py \
  --source-root /path/to/CV_checkpoint3_aug_bias/3x1 \
  --source-root /path/to/CV_checkpoint3_aug_bias/2x-2 \
  --output-dir /path/to/public_checkpoints \
  --mode raw
```

The output bundle contains:

```text
public_checkpoints/
  manifest.csv
  README.md
  3x1/
    beta10.0/
      numerator/even/best_model.pt
      numerator/odd/best_model.pt
      denominator/even/best_model.pt
      denominator/odd/best_model.pt
  2x-2/
    beta3.0/
      ...
```

`manifest.csv` records the selected source experiment, public relative path,
file size, and SHA256 checksum.

## Availability Statement

Suggested wording:

> Source code is available on GitHub. The inference checkpoints used for the
> reported neural-control-variate results are archived separately with a
> persistent DOI. Raw Monte Carlo samples and intermediate training checkpoints
> are not included because of storage size, but can be regenerated from the
> sampler, scripts, seeds, and parameters described in the repository.
