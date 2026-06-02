#!/usr/bin/env python3
"""Prepare a curated public checkpoint bundle.

The source training directories can contain every epoch, Slurm logs, plots, and
ablation runs. This script selects only final-configuration best checkpoints,
copies them into a stable public layout, and writes a SHA256 manifest.

By default it selects experiment directories matching
`*_200k_Mnmax_seqmean_bondemb` and expects the following files in each selected
temperature directory:

    numerator/even_NH/best_model.pt
    numerator/odd_NH/best_model.pt
    denominator/even_NH/best_model.pt
    denominator/odd_NH/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


COMPONENTS = ("numerator", "denominator")
PARITIES = ("even", "odd")
DROP_CHECKPOINT_KEYS = {
    "optimizer",
    "optimizer_state",
    "optimizer_state_dict",
    "scheduler",
    "scheduler_state",
    "scheduler_state_dict",
    "lr_scheduler",
    "lr_scheduler_state_dict",
    "scaler",
    "scaler_state",
    "scaler_state_dict",
}


@dataclass(frozen=True)
class CheckpointItem:
    system: str
    beta_label: str
    beta_value: str
    experiment: str
    component: str
    parity: str
    source_root: Path
    source_path: Path

    @property
    def source_rel(self) -> Path:
        return Path(self.system) / self.source_path.relative_to(self.source_root)

    @property
    def dest_rel(self) -> Path:
        return Path(self.system) / f"beta{self.beta_label}" / self.component / self.parity / "best_model.pt"


def parse_beta_label(dirname: str) -> tuple[str, str]:
    match = re.match(r"^beta(?P<beta>[0-9]+(?:\.[0-9]+)?)", dirname)
    if not match:
        raise ValueError(f"Cannot parse beta from directory name: {dirname}")
    beta_label = match.group("beta")
    return beta_label, str(float(beta_label))


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def discover_items(
    source_roots: list[Path],
    include_dir_globs: list[str],
    exclude_dir_globs: list[str],
    allow_missing: bool,
) -> tuple[list[CheckpointItem], list[str]]:
    items: list[CheckpointItem] = []
    warnings: list[str] = []

    for source_root in source_roots:
        source_root = source_root.resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source root is not a directory: {source_root}")

        system = source_root.name
        for experiment_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
            experiment = experiment_dir.name
            if not matches_any(experiment, include_dir_globs):
                continue
            if matches_any(experiment, exclude_dir_globs):
                continue

            beta_label, beta_value = parse_beta_label(experiment)
            missing: list[Path] = []
            for component in COMPONENTS:
                for parity in PARITIES:
                    src = experiment_dir / component / f"{parity}_NH" / "best_model.pt"
                    if not src.is_file():
                        missing.append(src)
                        continue
                    items.append(
                        CheckpointItem(
                            system=system,
                            beta_label=beta_label,
                            beta_value=beta_value,
                            experiment=experiment,
                            component=component,
                            parity=parity,
                            source_root=source_root,
                            source_path=src,
                        )
                    )

            if missing:
                message = f"{experiment_dir}: missing {len(missing)} expected best_model.pt files"
                if allow_missing:
                    warnings.append(message)
                else:
                    details = "\n".join(f"  - {p}" for p in missing)
                    raise FileNotFoundError(f"{message}\n{details}")

    return items, warnings


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def copy_raw(item: CheckpointItem, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination exists, use --overwrite: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.source_path, dest)


def write_inference_checkpoint(item: CheckpointItem, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination exists, use --overwrite: {dest}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "--mode inference requires PyTorch in the current environment. "
            "Use --mode raw to copy checkpoints without stripping optimizer state."
        ) from exc

    ckpt = torch.load(item.source_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        public_ckpt = {k: v for k, v in ckpt.items() if k not in DROP_CHECKPOINT_KEYS}
        public_ckpt["public_checkpoint"] = {
            "system": item.system,
            "beta": item.beta_value,
            "experiment": item.experiment,
            "component": item.component,
            "parity": item.parity,
            "optimizer_state_removed": any(k in ckpt for k in DROP_CHECKPOINT_KEYS),
        }
    else:
        public_ckpt = ckpt

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public_ckpt, dest)


def write_manifest(output_dir: Path, rows: list[dict[str, str]], manifest_name: str) -> Path:
    manifest_path = output_dir / manifest_name
    fieldnames = [
        "system",
        "beta",
        "experiment",
        "component",
        "parity",
        "source_rel",
        "dest_rel",
        "mode",
        "bytes",
        "sha256",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_readme(output_dir: Path, rows: list[dict[str, str]], mode: str) -> Path:
    readme_path = output_dir / "README.md"
    systems = sorted({row["system"] for row in rows})
    betas = sorted({(row["system"], row["beta"]) for row in rows}, key=lambda x: (x[0], float(x[1])))
    with readme_path.open("w") as f:
        f.write("# Public Checkpoints\n\n")
        f.write("Curated inference checkpoints for the NCV for QMC repository.\n\n")
        f.write(f"- Mode: `{mode}`\n")
        f.write(f"- Systems: {', '.join(systems)}\n")
        f.write(f"- Checkpoint files: {len(rows)}\n")
        f.write("- Each system/beta should contain numerator and denominator models for even and odd parity.\n\n")
        f.write("## Contents\n\n")
        current_system = None
        for system, beta in betas:
            if system != current_system:
                current_system = system
                f.write(f"### {system}\n\n")
            f.write(f"- beta = {beta}\n")
        f.write("\n## Integrity\n\n")
        f.write("SHA256 checksums are listed in `manifest.csv`.\n")
    return readme_path


def summarize(items: list[CheckpointItem]) -> None:
    by_system: dict[str, set[str]] = {}
    total_bytes = 0
    for item in items:
        by_system.setdefault(item.system, set()).add(item.beta_value)
        total_bytes += item.source_path.stat().st_size

    print(f"Selected checkpoint files: {len(items)}")
    print(f"Selected source size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    for system in sorted(by_system):
        betas = sorted(by_system[system], key=float)
        print(f"  {system}: {len(betas)} beta values ({', '.join(betas)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public best-checkpoint bundle")
    parser.add_argument("--source-root", action="append", required=True, help="Root such as .../CV_checkpoint3_aug_bias/3x1")
    parser.add_argument("--output-dir", required=True, help="Directory to create the public checkpoint bundle")
    parser.add_argument(
        "--include-dir-glob",
        action="append",
        default=["*_200k_Mnmax_seqmean_bondemb"],
        help="Experiment directory glob to include. May be repeated.",
    )
    parser.add_argument(
        "--exclude-dir-glob",
        action="append",
        default=[],
        help="Experiment directory glob to exclude. May be repeated.",
    )
    parser.add_argument("--mode", choices=["inference", "raw"], default="inference")
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument("--dry-run", action="store_true", help="Only print the selected files")
    parser.add_argument("--allow-missing", action="store_true", help="Warn instead of failing if an included beta is incomplete")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination checkpoints")
    parser.add_argument("--no-readme", action="store_true", help="Do not write README.md in the output bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_roots = [Path(p) for p in args.source_root]
    output_dir = Path(args.output_dir)

    items, warnings = discover_items(
        source_roots=source_roots,
        include_dir_globs=args.include_dir_glob,
        exclude_dir_globs=args.exclude_dir_glob,
        allow_missing=args.allow_missing,
    )
    if not items:
        print("No checkpoints matched the selection.", file=sys.stderr)
        return 1

    summarize(items)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.dry_run:
        for item in items:
            print(f"{item.source_rel} -> {item.dest_rel}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for item in items:
        dest = output_dir / item.dest_rel
        if args.mode == "raw":
            copy_raw(item, dest, overwrite=args.overwrite)
        else:
            write_inference_checkpoint(item, dest, overwrite=args.overwrite)

        rows.append(
            {
                "system": item.system,
                "beta": item.beta_value,
                "experiment": item.experiment,
                "component": item.component,
                "parity": item.parity,
                "source_rel": item.source_rel.as_posix(),
                "dest_rel": item.dest_rel.as_posix(),
                "mode": args.mode,
                "bytes": str(dest.stat().st_size),
                "sha256": sha256_file(dest),
            }
        )

    manifest_path = write_manifest(output_dir, rows, args.manifest_name)
    print(f"Wrote manifest: {manifest_path}")
    if not args.no_readme:
        readme_path = write_readme(output_dir, rows, args.mode)
        print(f"Wrote README: {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
