#!/usr/bin/env python3
"""
Rename CSV header columns x -> coord_X and y -> coord_Y without loading the whole file:
only the first line is parsed; the rest is copied as a stream.

Edit CSV_PATHS below, then run:
  python3 rename_xy_to_coord_columns.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path


# Fichiers à traiter (petit + gros, ou plus)
CSV_PATHS = [
    Path(
        "/data-master/code/Spatial-Transcriptomics-and-Perturbations-Modeling/data_splits/luna_abc1/output_joined_log2p1_preview100.csv"
    ),
    Path(
        "/data-master/code/Spatial-Transcriptomics-and-Perturbations-Modeling/data_splits/luna_abc1/output_joined_log2p1.csv"
    ),
]

OLD_TO_NEW = {"x": "coord_X", "y": "coord_Y"}


def rename_header_inplace_streaming(path: Path) -> None:
    if not path.is_file():
        print(f"Skip (missing): {path}", file=sys.stderr)
        return

    tmp = path.with_suffix(path.suffix + ".tmp_rename_xy")

    with path.open("r", encoding="utf-8", newline="") as fin, tmp.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        header_line = fin.readline()
        if not header_line:
            raise ValueError(f"Empty file: {path}")

        reader = csv.reader([header_line])
        row = next(reader)

        present = [c for c in OLD_TO_NEW if c in row]
        if not present:
            print(f"Skip (no columns {list(OLD_TO_NEW)} in header): {path}")
            tmp.unlink(missing_ok=True)
            return

        new_row = [OLD_TO_NEW.get(c, c) for c in row]
        writer = csv.writer(fout)
        writer.writerow(new_row)
        shutil.copyfileobj(fin, fout)

    tmp.replace(path)
    print(f"OK: {path} (renamed: {present})")


def main() -> None:
    for p in CSV_PATHS:
        rename_header_inplace_streaming(p)


if __name__ == "__main__":
    main()