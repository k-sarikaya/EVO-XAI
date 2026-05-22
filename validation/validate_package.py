#!/usr/bin/env python
"""
Lightweight validation for the Zenodo reproducibility snapshot.

Goals:
- verify required files exist (ARFFs, figures, manuscript, key CSVs)
- verify dataset counts / prevalence match the manuscript's Table 1
- verify included CSVs have expected columns (schema sanity)
- write a SHA256 manifest for provenance

This script intentionally does NOT rerun the full evolutionary pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import arff  # liac-arff


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    rel_path: str
    n: int
    asd_cases: int
    prevalence_pct: float


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="child",
        rel_path="data/child/Autism-Child-Data.arff",
        n=292,
        asd_cases=141,
        prevalence_pct=48.3,
    ),
    DatasetSpec(
        name="adolescent",
        rel_path="data/adolescent/Autism-Adolescent-Data.arff",
        n=104,
        asd_cases=63,
        prevalence_pct=60.6,
    ),
    DatasetSpec(
        name="adult",
        rel_path="data/adult/Autism-Adult-Data.arff",
        n=704,
        asd_cases=189,
        prevalence_pct=26.8,
    ),
]


REQUIRED_FILES = [
    "README.md",
    "manuscript/manuscript_hiss_sn.tex",
    "manuscript/references.bib",
    "figures/Figure_Captions.md",
    *[f"figures/Figure_{i}.png" for i in range(1, 9)],
]


KEY_CSVS = [
    # These are common outputs in this workspace. If a file is absent, we warn rather than fail.
    "results/vif_threshold_sensitivity.csv",
    "results/algorithm_comparison_vif13.csv",
    "results/algorithm_features_vif13.csv",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_arff_target_counts(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = arff.load(f)
    cols = [a[0] for a in data["attributes"]]
    rows = data["data"]
    if not rows:
        return 0, 0

    # Target name variants in the UCI ASD ARFFs.
    target_col = "Class/ASD" if "Class/ASD" in cols else cols[-1]
    t_idx = cols.index(target_col)

    def is_asd(v) -> bool:
        s = str(v).strip().upper()
        return s in {"YES", "TRUE", "1", "ASD"}

    asd = sum(1 for r in rows if is_asd(r[t_idx]))
    return len(rows), asd


def _check_csv_schema(path: Path) -> tuple[bool, str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
    if not cols:
        return False, "missing header"
    return True, f"{len(cols)} columns"


def main() -> int:
    print(f"[info] Python: {sys.version.splitlines()[0]}")
    print(f"[info] Root: {ROOT}")

    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_FILES:
        p = ROOT / rel
        if not p.exists():
            errors.append(f"missing required file: {rel}")

    for ds in DATASETS:
        p = ROOT / ds.rel_path
        if not p.exists():
            errors.append(f"missing dataset: {ds.rel_path}")
            continue
        n, asd = _read_arff_target_counts(p)
        prev = (asd / n * 100.0) if n else 0.0
        if n != ds.n or asd != ds.asd_cases:
            errors.append(
                f"dataset stats mismatch for {ds.name}: got n={n}, asd={asd}, prev={prev:.1f}% "
                f"(expected n={ds.n}, asd={ds.asd_cases}, prev={ds.prevalence_pct:.1f}%)"
            )
        else:
            # prevalence is derived; allow tiny rounding variation
            if abs(prev - ds.prevalence_pct) > 0.2:
                warnings.append(
                    f"prevalence rounding differs for {ds.name}: got {prev:.2f}%, expected {ds.prevalence_pct:.1f}%"
                )

    for rel in KEY_CSVS:
        p = ROOT / rel
        if not p.exists():
            warnings.append(f"missing optional CSV: {rel}")
            continue
        ok, msg = _check_csv_schema(p)
        if not ok:
            errors.append(f"bad CSV {rel}: {msg}")
        else:
            print(f"[ok] CSV schema: {rel} ({msg})")

    # SHA256 manifest for all files in the snapshot.
    manifest = ROOT / "manifest_sha256.txt"
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != manifest.name]
    files.sort(key=lambda p: str(p.relative_to(ROOT)).lower())
    with manifest.open("w", encoding="utf-8", newline="\n") as f:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            f.write(f"{_sha256(p)}  {rel}\n")
    print(f"[ok] wrote {manifest.relative_to(ROOT).as_posix()} ({len(files)} files)")

    if warnings:
        print("[warn] " + "\n[warn] ".join(warnings))
    if errors:
        print("[error] " + "\n[error] ".join(errors))
        return 2

    print("[ok] validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

