"""
Shuffle a (potentially large) CSV file by loading it in random chunks.

Strategy
--------
1. Read the CSV in chunks of `--chunk-size` rows.
2. Shuffle the rows within every chunk immediately after reading.
3. Collect all chunks into a list and shuffle the chunk order.
4. Concatenate and perform one final row-level shuffle across all chunks.
5. Write the result to the output CSV.

This keeps peak memory to one full in-memory copy of the data (unavoidable for
a true global shuffle) while avoiding reading the whole file as a single block.

Usage
-----
    python -m utils.data.shuffle input.csv output.csv
    python -m utils.data.shuffle input.csv output.csv --chunk-size 50000 --seed 42
"""

import argparse
import sys

import numpy as np
import pandas as pd


def shuffle_csv(
    input_path: str,
    output_path: str,
    chunk_size: int = 100_000,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    print(f"Reading '{input_path}' in chunks of {chunk_size:,} rows …")
    chunks: list[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        # Shuffle rows within this chunk right away so the in-memory repr is
        # already partially random before we do the global shuffle at the end.
        chunks.append(chunk.sample(frac=1, random_state=int(rng.integers(2**31))))
        print(f"  chunk {i + 1}: {len(chunk):,} rows", end="\r")

    print(f"\nRead {len(chunks)} chunk(s). Shuffling chunk order …")

    # Shuffle which chunk comes first so rows from the end of the file can end
    # up at the start of the output before the final global shuffle.
    chunk_order = rng.permutation(len(chunks)).tolist()
    chunks = [chunks[j] for j in chunk_order]

    print("Concatenating and applying final global row shuffle …")
    df = pd.concat(chunks, ignore_index=True)
    df = df.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)

    print(f"Writing {len(df):,} rows to '{output_path}' …")
    df.to_csv(output_path, index=True)
    print("Done.")


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
        description="Randomly shuffle the rows of a large CSV file."
    )
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("output", help="Path for the shuffled output CSV file.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of rows to read per chunk (default: 100 000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for reproducibility (default: none).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    '''args = _parse_args()'''
    '''shuffle_csv(
        input_path= "/data-master/biodata/Axolotl/test_cells_expression.csv",
        output_path= "/data-master/biodata/Axolotl/test_cells_expression_shuffled.csv",
        chunk_size=10000,
        seed=42,
    )'''
    head_csv(
        input_path= "/data-master/biodata/Axolotl/train_cells_expression_shuffled.csv",
        output_path= "./train_cells_expression_shuffled_head.csv",
        n_rows=4,
    )
    head_csv(
        input_path= "/data-master/biodata/Axolotl/test_cells_expression_shuffled.csv",
        output_path= "./test_cells_expression_shuffled_head.csv",
        n_rows=4,
    )
    csv_same_columns(
        path_a= "./test_cells_expression_shuffled_head.csv",
        path_b= "./train_cells_expression_shuffled_head.csv",
        same_order=True,
        verbose=True,
    )