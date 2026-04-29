#!/usr/bin/env python3
"""
Build one CSV row per cell listed in the metadata file (metadata defines order and row count).

Pipeline (no table merge):
1) Load metadata in file order; after strip + dedup on cell_label, each row gets a fixed index
   i in [0, N).
2) Stream the expression matrix once. For each cell_id that appears in metadata and is seen
   for the first time, compute log2(count+1) for genes and write that vector into row i of a
   float32 memmap (staging on disk). Duplicate cell_id rows in the expression file are ignored
   after the first occurrence.
3) Cells never seen in the expression file keep row i as zeros (same as log2(0+1)).
4) Write OUTPUT_CSV by iterating i = 0..N-1: metadata columns for that cell + genes from memmap.

This matches the mental model “for each metadata cell, look up its id in the expression stream;
first hit wins”, while keeping RAM bounded (memmap size ~ N * n_genes * 4 bytes on disk).

Also writes a small preview CSV next to OUTPUT_CSV (first PREVIEW_N_ROWS data rows + header)
during pass 2, so you do not need to rescan the full output file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ----------------------------
# User paths (EDIT THESE)
# ----------------------------
METADATA_CSV = Path("/data-master/abel_save/Ricci/data/MERFISH_ABC/MOUSE_ABC1/cell_metadata_with_cluster_annotation.csv")
EXPRESSION_CSV = Path("/data-master/abel_save/Ricci/data/MERFISH_ABC/exports/MOUSE_ABC1_raw_cell_by_gene.csv")
OUTPUT_CSV = Path("/data-master/code/Spatial-Transcriptomics-and-Perturbations-Modeling/data_splits/luna_abc1/output_joined_log2p1.csv")


# ----------------------------
# Parameters
# ----------------------------
CHUNKSIZE = 10_000  # adjust depending on RAM / file size
CELL_LABEL_COL = "cell_label"
CELL_ID_COL = "cell_id"
X_COL = "x"
Y_COL = "y"
SECTION_SRC_COL = "brain_section_label"
CELL_SECTION_OUT_COL = "cell_section"
CELL_CLASS_SRC_COL = "class"
CELL_CLASS_OUT_COL = "cell_class"

# Staging file for gene matrix (float32); deleted after a successful run.
STAGING_GENES = OUTPUT_CSV.parent / "._staging_genes_memmap.dat"
# Rows per batch when writing the final CSV (adjust if memory is tight).
OUTPUT_BATCH_ROWS = 2_000
# Small CSV written during pass 2 (first batch only): header + first N data rows, no extra scan.
PREVIEW_N_ROWS = 100


def _validate_paths() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    if not EXPRESSION_CSV.exists():
        raise FileNotFoundError(f"Expression CSV not found: {EXPRESSION_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def read_metadata_ordered() -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Returns:
      - meta_df: one row per unique cell_label (first row kept), order preserved, columns include CELL_ID_COL.
      - cell_id_to_row: maps cell_id -> row index i in [0, N) for that ordering.
    """
    usecols = [CELL_LABEL_COL, X_COL, Y_COL, SECTION_SRC_COL, CELL_CLASS_SRC_COL]
    meta = pd.read_csv(METADATA_CSV, usecols=usecols)

    missing = [c for c in usecols if c not in meta.columns]
    if missing:
        raise ValueError(
            "Metadata CSV is missing required columns: "
            + ", ".join(missing)
            + f". Found columns: {list(meta.columns)}"
        )

    if meta[CELL_LABEL_COL].isna().any():
        raise ValueError("Metadata CSV has null `cell_label` values.")

    meta[CELL_LABEL_COL] = meta[CELL_LABEL_COL].astype(str).str.strip()
    meta = meta.drop_duplicates(subset=[CELL_LABEL_COL], keep="first").copy()

    meta = meta.rename(
        columns={
            SECTION_SRC_COL: CELL_SECTION_OUT_COL,
            CELL_CLASS_SRC_COL: CELL_CLASS_OUT_COL,
        }
    )
    meta = meta.rename(columns={CELL_LABEL_COL: CELL_ID_COL})

    if meta[CELL_ID_COL].duplicated().any():
        n_dup = int(meta[CELL_ID_COL].duplicated().sum())
        raise ValueError(
            f"Metadata has {n_dup} duplicate `{CELL_ID_COL}` values after strip/dedup. "
            "Fix the metadata file or adjust deduplication logic."
        )

    meta = meta.reset_index(drop=True)
    ids = meta[CELL_ID_COL].tolist()
    cell_id_to_row = {cid: i for i, cid in enumerate(ids)}
    return meta, cell_id_to_row


def expression_gene_columns() -> list[str]:
    header = pd.read_csv(EXPRESSION_CSV, nrows=0)
    if CELL_ID_COL not in header.columns:
        raise ValueError(
            f"Expression CSV must contain a `{CELL_ID_COL}` column. "
            f"Found columns: {list(header.columns)}"
        )
    genes = [c for c in header.columns if c != CELL_ID_COL]
    if not genes:
        raise ValueError("Expression CSV does not contain any gene columns.")
    return genes


def log2p1_frame(df: pd.DataFrame, gene_cols: Iterable[str]) -> pd.DataFrame:
    gene_df = df.loc[:, list(gene_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    gene_df = np.log2(gene_df.to_numpy(dtype=np.float64) + 1.0)
    return pd.DataFrame(gene_df, columns=list(gene_cols), index=df.index)


def build_dataset() -> None:
    _validate_paths()

    meta_df, cell_id_to_row = read_metadata_ordered()
    n_meta = len(meta_df)
    meta_ids = set(cell_id_to_row.keys())
    genes = expression_gene_columns()
    n_genes = len(genes)

    est_gb = (n_meta * n_genes * 4) / (1024**3)
    print(
        f"Metadata rows (ordered, unique {CELL_ID_COL}): {n_meta:,}",
        flush=True,
    )
    print(
        f"Staging memmap: {n_meta:,} x {n_genes:,} float32 ≈ {est_gb:.2f} GiB on disk → {STAGING_GENES}",
        flush=True,
    )

    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
    preview_csv = OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_preview100.csv")
    if preview_csv.exists():
        preview_csv.unlink()
    if STAGING_GENES.exists():
        STAGING_GENES.unlink()

    # --- Pass 1: stream expression; fill memmap[row_i, :] on first sight of each cell_id ---
    mmap = np.memmap(
        str(STAGING_GENES),
        dtype=np.float32,
        mode="w+",
        shape=(n_meta, n_genes),
    )
    # zeros == log2(0+1) for missing expression rows
    mmap[:] = 0.0

    seen_from_expr: set[str] = set()

    chunk_idx = 0
    try:
        for chunk in pd.read_csv(EXPRESSION_CSV, chunksize=CHUNKSIZE):
            chunk_idx += 1
            if chunk_idx % 10 == 0:
                print(f"{chunk_idx * CHUNKSIZE} expression rows scanned", flush=True)

            if CELL_ID_COL not in chunk.columns:
                raise ValueError(
                    f"Expression CSV chunk missing `{CELL_ID_COL}` column; "
                    "check delimiter/quoting or file integrity."
                )

            chunk[CELL_ID_COL] = chunk[CELL_ID_COL].astype(str).str.strip()
            chunk = chunk[chunk[CELL_ID_COL].isin(meta_ids)]
            if chunk.empty:
                continue

            chunk = chunk.drop_duplicates(subset=[CELL_ID_COL], keep="first")
            chunk = chunk[~chunk[CELL_ID_COL].isin(seen_from_expr)]
            if chunk.empty:
                continue

            pos = chunk[CELL_ID_COL].map(cell_id_to_row)
            if pos.isna().any():
                raise ValueError("Internal error: cell_id not in cell_id_to_row after filter.")
            pos_arr = pos.to_numpy(dtype=np.intp)

            for g in genes:
                if g not in chunk.columns:
                    chunk[g] = 0
            gene_logged = log2p1_frame(chunk, genes)
            gmat = gene_logged.to_numpy(dtype=np.float32)

            mmap[pos_arr, :] = gmat
            seen_from_expr.update(chunk[CELL_ID_COL].tolist())

        mmap.flush()
        del mmap

        n_found = len(seen_from_expr)
        print(
            f"Expression pass done: filled {n_found:,} / {n_meta:,} cells "
            f"({n_meta - n_found:,} left as zeros).",
            flush=True,
        )

        # --- Pass 2: write CSV in metadata order (row 0 .. N-1); no merge ---
        mmap_r = np.memmap(
            str(STAGING_GENES),
            dtype=np.float32,
            mode="r",
            shape=(n_meta, n_genes),
        )

        header_cols = [CELL_ID_COL] + genes + [X_COL, Y_COL, CELL_SECTION_OUT_COL, CELL_CLASS_OUT_COL]

        wrote_header = False
        for start in range(0, n_meta, OUTPUT_BATCH_ROWS):
            end = min(start + OUTPUT_BATCH_ROWS, n_meta)
            sub = meta_df.iloc[start:end]
            gblk = np.asarray(mmap_r[start:end, :], dtype=np.float64)

            out = pd.concat(
                [
                    sub[[CELL_ID_COL]].reset_index(drop=True),
                    pd.DataFrame(gblk, columns=genes),
                    sub[[X_COL, Y_COL, CELL_SECTION_OUT_COL, CELL_CLASS_OUT_COL]].reset_index(
                        drop=True
                    ),
                ],
                axis=1,
            )
            out = out[header_cols]

            if start == 0 and n_meta > 0:
                n_prev = min(PREVIEW_N_ROWS, len(out))
                out.iloc[:n_prev].to_csv(preview_csv, index=False)

            out.to_csv(
                OUTPUT_CSV,
                mode="a",
                index=False,
                header=not wrote_header,
            )
            wrote_header = True

        del mmap_r

    finally:
        if STAGING_GENES.exists():
            STAGING_GENES.unlink(missing_ok=True)

    print(f"Done: wrote {n_meta:,} rows to {OUTPUT_CSV}", flush=True)
    if n_meta > 0:
        print(
            f"Preview ({min(PREVIEW_N_ROWS, n_meta)} rows): {preview_csv}",
            flush=True,
        )


if __name__ == "__main__":
    build_dataset()
