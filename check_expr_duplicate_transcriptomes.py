#!/usr/bin/env python3
"""
Quick diagnostic: for MOUSE_ABC1_raw_cell_by_gene.csv, count duplicate cell_id rows
and check whether duplicate rows share the same gene vector (transcriptome).
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path

# Same path as data-setup-ABC.py
EXPRESSION_CSV = Path("/data-master/abel_save/Ricci/data/MERFISH_ABC/exports/MOUSE_ABC1_raw_cell_by_gene.csv")
CELL_ID_COL = "cell_id"
SAMPLE_N = 80  # how many duplicated cell_ids to compare (if that many exist)
PROGRESS_EVERY = 500_000


def _parse_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _vec_equal(a: list[float], b: list[float]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


def _max_abs_diff(ref: list[float], a: list[float]) -> float:
    m = 0.0
    for x, y in zip(ref, a):
        d = abs(x - y)
        if d > m:
            m = d
    return m


def main() -> int:
    if not EXPRESSION_CSV.is_file():
        print(f"Fichier introuvable: {EXPRESSION_CSV}", file=sys.stderr)
        return 1

    print("Passe 1: comptage des cell_id (colonne seulement)...", flush=True)
    with EXPRESSION_CSV.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            id_idx = header.index(CELL_ID_COL)
        except ValueError:
            print(f"Colonne `{CELL_ID_COL}` absente. Colonnes: {header[:20]}...", file=sys.stderr)
            return 1

        cnt: Counter[str] = Counter()
        n = 0
        for row in reader:
            if not row:
                continue
            cnt[row[id_idx]] += 1
            n += 1
            if n % PROGRESS_EVERY == 0:
                print(f"  ... {n:,} lignes lues", flush=True)

    n_unique = len(cnt)
    dup_ids = [k for k, v in cnt.items() if v > 1]
    n_dup_ids = len(dup_ids)
    extra_rows = sum(v - 1 for v in cnt.values() if v > 1)

    print()
    print("Résumé:")
    print(f"  Lignes totales (hors header):     {n:,}")
    print(f"  cell_id distincts:                {n_unique:,}")
    print(f"  cell_id avec >1 occurrence:        {n_dup_ids:,}")
    print(f"  Lignes « en trop » (doublons):    {extra_rows:,}")
    print()

    if n_dup_ids == 0:
        print("Aucun doublon de cell_id — rien à comparer.")
        return 0

    rng = random.Random(42)
    sample = rng.sample(dup_ids, min(SAMPLE_N, n_dup_ids))

    print(
        f"Passe 2: pour {len(sample)} cell_id tirés au hasard parmi les dupliqués, "
        "comparaison des vecteurs gènes (toutes les lignes vs la 1re).",
        flush=True,
    )

    rows_by_id: dict[str, list[list[float]]] = {cid: [] for cid in sample}
    sample_set = set(sample)

    with EXPRESSION_CSV.open(newline="") as f:
        reader = csv.reader(f)
        header2 = next(reader)
        id_idx2 = header2.index(CELL_ID_COL)
        gene_indices = [i for i in range(len(header2)) if i != id_idx2]

        n2 = 0
        for row in reader:
            n2 += 1
            if n2 % PROGRESS_EVERY == 0:
                print(f"  ... {n2:,} lignes relues", flush=True)
            if not row:
                continue
            cid = row[id_idx2]
            if cid not in sample_set:
                continue
            vals = [_parse_float(row[j] if j < len(row) else "") for j in gene_indices]
            rows_by_id[cid].append(vals)

    all_same = 0
    any_diff = 0
    examples_diff: list[tuple[str, int, float]] = []

    for cid in sample:
        arrs = rows_by_id[cid]
        if len(arrs) < 2:
            continue
        ref = arrs[0]
        ok = all(_vec_equal(ref, a) for a in arrs[1:])
        if ok:
            all_same += 1
        else:
            any_diff += 1
            md = max(_max_abs_diff(ref, a) for a in arrs[1:])
            examples_diff.append((cid, len(arrs), md))

    print()
    print("Sur l'échantillon de cell_id dupliqués:")
    print(f"  Même transcritome que la 1re ligne: {all_same} / {len(sample)}")
    print(f"  Au moins une ligne différente:      {any_diff} / {len(sample)}")
    if examples_diff:
        print()
        print("Exemples (cell_id, nb lignes, max |diff| vs 1re ligne):")
        for cid, k, md in examples_diff[:10]:
            print(f"  {cid!r}  n={k}  max_diff={md:g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
