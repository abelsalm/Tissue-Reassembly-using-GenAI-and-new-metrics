"""
Shuffle a (potentially very large) CSV file with low memory usage.

Strategy: two-pass external (disk-based) shuffle
-------------------------------------------------
1. Pass 1 — bucketize: stream the input file line by line and write each row
   to one of ``n_buckets`` temporary files chosen uniformly at random.
2. Pass 2 — in-bucket shuffle: for each bucket (whose size is roughly
   ``total_size / n_buckets``), load it fully, shuffle in memory, and append
   to the output.

Peak RAM is approximately the size of the largest bucket, NOT the whole CSV.
The global result is a fully shuffled CSV (every row equally likely to appear
at any position), within the limit of the random bucketing.

Caveats
-------
- Lines are split on '\\n'. CSVs that contain quoted fields with embedded
  newlines will be corrupted. For typical numeric / gene-expression CSVs
  this is not an issue.
- The temporary buckets are written next to the output file, so the disk
  containing the output must have free space ~ size of the input.

Usage
-----
    python -m utils.data.shuffle input.csv output.csv
    python -m utils.data.shuffle input.csv output.csv --n-buckets 128 --seed 42
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd


def shuffle_csv(
    input_path: str,
    output_path: str,
    n_buckets: int = 128,
    seed: int = 42,
    has_header: bool = True,
    chunk_size: int = None,  # noqa: ARG001  (legacy, unused)
) -> None:
    """Memory-efficient row shuffle of a (possibly huge) CSV via external shuffle.

    Parameters
    ----------
    input_path, output_path
        Source and destination CSV paths.
    n_buckets
        Number of temporary disk buckets. Higher values reduce peak RAM but
        require more open file handles and more random I/O. 128 is a good
        default for tens of GB inputs.
    seed
        RNG seed for reproducibility.
    has_header
        If True (default), the first line of the input is treated as a header
        and copied verbatim to the start of the output.
    chunk_size
        Ignored. Kept only for backwards compatibility with the previous API.
    """
    rng = np.random.default_rng(seed)

    print(f"[shuffle_csv] input  : {input_path}")
    print(f"[shuffle_csv] output : {output_path}")
    print(f"[shuffle_csv] {n_buckets} buckets (peak RAM ~ size_of_largest_bucket)")

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="shuffle_buckets_", dir=out_dir)
    bucket_paths = [os.path.join(tmpdir, f"b_{i:05d}.txt") for i in range(n_buckets)]

    try:
        # ---------- Pass 1: stream input -> random buckets ----------
        t0 = time.time()
        n_rows = 0
        bucket_files = [open(p, "w") for p in bucket_paths]
        try:
            with open(input_path, "r") as f:
                header = f.readline() if has_header else None

                # Generate random bucket assignments in mini-batches to amortise
                # numpy overhead instead of paying it per row.
                BATCH = 1 << 16  # 65 536 rows per draw
                buf = rng.integers(0, n_buckets, size=BATCH).tolist()
                bi = 0
                for line in f:
                    if bi == BATCH:
                        buf = rng.integers(0, n_buckets, size=BATCH).tolist()
                        bi = 0
                    bucket_files[buf[bi]].write(line)
                    bi += 1
                    n_rows += 1
                    if n_rows % 2_000_000 == 0:
                        rate = n_rows / max(time.time() - t0, 1e-9)
                        print(f"  pass 1: {n_rows:>12,} rows  ({rate/1e6:.2f} M rows/s)")
        finally:
            for bf in bucket_files:
                bf.close()
        print(
            f"[shuffle_csv] pass 1 done: {n_rows:,} rows -> {n_buckets} buckets "
            f"in {time.time() - t0:.1f}s"
        )

        # ---------- Pass 2: shuffle each bucket and concatenate ----------
        t1 = time.time()
        # Visit buckets in random order so the macro-structure of the output
        # is not biased by the bucket index.
        order = rng.permutation(n_buckets).tolist()
        total_written = 0
        with open(output_path, "w") as out:
            if header is not None:
                out.write(header)
            for k, b in enumerate(order, start=1):
                with open(bucket_paths[b], "r") as bf:
                    lines = bf.readlines()
                rng.shuffle(lines)
                out.writelines(lines)
                total_written += len(lines)
                if k % 16 == 0 or k == n_buckets:
                    print(f"  pass 2: {k:>4d}/{n_buckets} buckets ({total_written:,} rows written)")

        print(
            f"[shuffle_csv] pass 2 done in {time.time() - t1:.1f}s. "
            f"Total: {total_written:,} rows."
        )

    finally:
        for p in bucket_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def head_csv(input_path: str, output_path: str, n_rows: int = 4) -> None:
    """Copy the first `n_rows` rows of a CSV into a new file without loading the rest."""
    df = pd.read_csv(input_path, nrows=n_rows)
    df.to_csv(output_path, index=True)
    print(f"Saved {len(df)} rows from '{input_path}' to '{output_path}'.")


def csv_same_columns(
    path_a: str,
    path_b: str,
    *,
    same_order: bool = False,
    verbose: bool = False,
) -> bool:
    """Return True if two CSV files share the same columns (only header row is read).

    By default compares column *names* as a set (order may differ). If
    ``same_order`` is True, the column list must match exactly, including order.
    Assumes column names are unique in each file.
    """
    cols_a = list(pd.read_csv(path_a, nrows=0).columns)
    cols_b = list(pd.read_csv(path_b, nrows=0).columns)

    if same_order:
        ok = cols_a == cols_b
        if verbose and not ok:
            print("Column order or names differ between the two files.")
        return ok

    set_a, set_b = set(cols_a), set(cols_b)
    ok = set_a == set_b
    if verbose and not ok:
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)
        if only_a:
            print(f"Only in first file ({path_a}): {only_a}")
        if only_b:
            print(f"Only in second file ({path_b}): {only_b}")
    elif verbose and ok:
        print("Same column names (order may differ).")
    return ok


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly shuffle the rows of a large CSV file (low-memory)."
    )
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("output", help="Path for the shuffled output CSV file.")
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=128,
        metavar="N",
        help="Number of temporary on-disk buckets (default: 128). "
             "Higher = lower peak RAM, more file handles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    '''args = _parse_args()'''
    shuffle_csv(
        input_path= "/data-master/biodata/MERFISH_ABC/MOUSE_ABC1_complete_cell_gene.csv",
        output_path= "/data-master/biodata/MERFISH_ABC/MOUSE_ABC1_complete_cell_gene_shuffled.csv",
        n_buckets=64,
        seed=42,
    )
    shuffle_csv(
        input_path= "/data-master/biodata/MERFISH_ABC/MOUSE_ABC2_complete_cell_gene.csv",
        output_path= "/data-master/biodata/MERFISH_ABC/MOUSE_ABC2_complete_cell_gene_shuffled.csv",
        n_buckets=64,
        seed=42,
    )
    head_csv(
        input_path= "/data-master/biodata/MERFISH_ABC/MOUSE_ABC1_complete_cell_gene_shuffled.csv",
        output_path= "./train_cells_expression_shuffled_head.csv",
        n_rows=4,
    )
    head_csv(
        input_path= "/data-master/biodata/MERFISH_ABC/MOUSE_ABC2_complete_cell_gene_shuffled.csv",
        output_path= "./test_cells_expression_shuffled_head.csv",
        n_rows=4,
    )
    csv_same_columns(
        path_a= "./train_cells_expression_shuffled_head.csv",
        path_b= "./test_cells_expression_shuffled_head.csv",
        same_order=True,
        verbose=True,
    )