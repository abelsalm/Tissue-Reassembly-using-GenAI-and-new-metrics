import numpy as np
import pandas as pd
from pathlib import Path


def create_data_splits(
    n_splits,
    coords_path,
    perturb_path,
    transcriptomics_path,
    output_dir,
    random_seed=None,
):
    """
    Randomly split single-cell spatial transcriptomics data into multiple groups.
    
    For each split, creates three CSV files (without headers):
    - coordinates_{split_id}.csv (cells × 2)
    - perturbations_{split_id}.csv (cells × 77)
    - transcriptomics_{split_id}.csv (cells × 553, i.e., merfishcounttable)
    
    Parameters:
    -----------
    n_splits : int
        Number of random splits to create (e.g., 5, 10)
    coords_path : str
        Path to coordinates.csv (no header, shape: n_cells × 2, columns: x, y)
    perturb_path : str
        Path to allcellsPerturbationTable.csv (no header, shape: n_cells × 77 gRNAs)
    transcriptomics_path : str
        Path to merfishcounttable.csv (no header, shape: n_cells × 553)
        Columns: cell_index, total_barcodes, cell_volume, then 550 gene columns
    output_dir : str
        Directory where to save split files (will be created if it doesn't exist)
    random_seed : int, optional
        Random seed for reproducibility (default: None for non-deterministic)
    
    Returns:
    --------
    dict : Dictionary mapping split_id (0 to n_splits-1) to the list of cell indices in that split
    
    Example:
    --------
    >>> splits = create_data_splits(
    ...     n_splits=5,
    ...     coords_path="C:/path/to/coordinates.csv",
    ...     perturb_path="C:/path/to/allcellsPerturbationTable.csv",
    ...     transcriptomics_path="C:/path/to/merfishcounttable.csv",
    ...     output_dir="./data_splits",
    ...     random_seed=42
    ... )
    >>> print(splits)  # {0: [cell_ids], 1: [cell_ids], ...}
    """
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Load all three data files ---
    print(f"Loading data from {n_splits} splits...")
    coords = pd.read_csv(coords_path, header=None)
    perturb = pd.read_csv(perturb_path, header=None)
    transcriptomics = pd.read_csv(transcriptomics_path, header=None)
    
    n_cells = len(coords)
    
    # Validate that all files have the same number of cells
    assert len(perturb) == n_cells, (
        f"Perturbation table has {len(perturb)} cells but coordinates has {n_cells}"
    )
    assert len(transcriptomics) == n_cells, (
        f"Transcriptomics table has {len(transcriptomics)} cells but coordinates has {n_cells}"
    )
    
    print(f"Total cells: {n_cells:,}")
    print(f"Number of splits: {n_splits}")
    
    # --- Randomly assign cells to splits ---
    split_assignments = np.random.randint(0, n_splits, size=n_cells)
    
    splits_dict = {}
    
    # --- For each split, save the corresponding subset of all three files ---
    for split_id in range(n_splits):
        mask = split_assignments == split_id
        split_indices = np.where(mask)[0]
        splits_dict[split_id] = split_indices.tolist()
        
        n_cells_in_split = len(split_indices)
        
        # Extract subset for this split
        coords_split = coords.iloc[split_indices]
        perturb_split = perturb.iloc[split_indices]
        transcriptomics_split = transcriptomics.iloc[split_indices]
        
        # Save to CSV files (no header, no index)
        coords_output = output_path / f"coordinates_{split_id}.csv"
        perturb_output = output_path / f"perturbations_{split_id}.csv"
        transcriptomics_output = output_path / f"transcriptomics_{split_id}.csv"
        
        coords_split.to_csv(coords_output, header=False, index=False)
        perturb_split.to_csv(perturb_output, header=False, index=False)
        transcriptomics_split.to_csv(transcriptomics_output, header=False, index=False)
        
        print(
            f"Split {split_id}: {n_cells_in_split:,} cells "
            f"({100*n_cells_in_split/n_cells:.1f}%) → "
            f"{coords_output.name}, {perturb_output.name}, {transcriptomics_output.name}"
        )
    
    print(f"\n✓ All {n_splits} splits saved to {output_path.absolute()}")
    return splits_dict


def merge_splits(splits_dir, output_dir, codebook_path, n_splits=None):
    """
    For each split, merge gene expression, perturbation columns, spatial
    coordinates, and the two LUNA-required metadata columns (``cell_section``,
    ``cell_class``) into a single CSV with proper column headers.

    The first 3 columns of each transcriptomics file (cell_index,
    total_barcodes, cell_volume) are dropped; only the 550 gene-expression
    columns are kept.

    Output column order (LUNA-compatible):
        <550 gene names> | <77 gRNA cols> | coord_X | coord_Y
        | cell_section | cell_class

    Parameters
    ----------
    splits_dir : str
        Directory with per-split CSVs from create_data_splits.
    output_dir : str
        Where to save merged CSVs (one per split, with headers + index).
    codebook_path : str
        Path to the MERFISH codebook CSV (must have a ``name`` column with
        550 gene/barcode names).
    n_splits : int, optional
        Number of splits. Auto-detected if None.
    """
    splits_path = Path(splits_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    codebook = pd.read_csv(codebook_path)
    gene_names = codebook["name"].values.tolist()

    if n_splits is None:
        n_splits = len(list(splits_path.glob("transcriptomics_*.csv")))
        print(f"Auto-detected {n_splits} splits in {splits_path}")

    grna_cols = [f"gRNA_{i}" for i in range(77)]

    for split_id in range(n_splits):
        trans   = pd.read_csv(splits_path / f"transcriptomics_{split_id}.csv", header=None)
        perturb = pd.read_csv(splits_path / f"perturbations_{split_id}.csv",  header=None)
        coords  = pd.read_csv(splits_path / f"coordinates_{split_id}.csv",    header=None)

        assert len(trans) == len(perturb) == len(coords), (
            f"Split {split_id}: row-count mismatch "
            f"(trans={len(trans)}, perturb={len(perturb)}, coords={len(coords)})"
        )

        gene_data = trans.iloc[:, 3:].copy()
        gene_data.columns = gene_names

        perturb.columns = grna_cols
        coords.columns = ["coord_X", "coord_Y"]

        merged = pd.concat(
            [gene_data.reset_index(drop=True),
             perturb.reset_index(drop=True),
             coords.reset_index(drop=True)],
            axis=1,
        )

        merged["cell_section"] = "tumor1"
        merged["cell_class"] = _assign_perturbations(perturb.values)

        merged_path = out_path / f"merged_{split_id}.csv"
        merged.to_csv(merged_path)

        print(
            f"Split {split_id}: {merged.shape[0]:,} cells × {merged.shape[1]} cols "
            f"→ {merged_path.name}"
        )

    print(f"\n✓ All {n_splits} merged files saved to {out_path.absolute()}")


# ── perturbation assignment (shared by prepare_luna_files) ──────────────

TARGET_NAMES = [
    "CD14","CHUK","IKBKB","IRAK1","IRAK4","IRF3","IRF5","IRF7","JUN","LBP",
    "LY96","MAP2K1","MAP2K2","MAP2K3","MAP2K4","MAP2K6","MAP2K7","MAP3K7",
    "MAPK14","MYD88","NFKB1","NFKBIA","PELI1","PIK3CA","RELA","RIPK1",
    "TAB1","TAB2","TBK1","TICAM1","TIRAP","TLR4","TRADD","TRAF6","TRAM1",
    "Control",
]


def _assign_perturbations(grna_matrix):
    """Return a perturbation label per cell from the 77-column gRNA matrix.

    Two gRNAs per KO target (35 targets) + 7 control gRNAs → pooled into
    36 groups.  The group with the highest sum wins; cells with zero gRNAs
    are labelled ``unperturbed``.
    """
    vals = grna_matrix.values if hasattr(grna_matrix, "values") else grna_matrix
    pooled = np.zeros((vals.shape[0], 36))
    for t in range(35):
        pooled[:, t] = vals[:, 2 * t] + vals[:, 2 * t + 1]
    pooled[:, 35] = vals[:, 70:].sum(axis=1)

    max_target = pooled.argmax(axis=1)
    max_count  = pooled.max(axis=1)

    return pd.Categorical([
        TARGET_NAMES[t] if c > 0 else "unperturbed"
        for t, c in zip(max_target, max_count)
    ])


# ── LUNA file preparation ───────────────────────────────────────────────

def prepare_luna_files(
    merged_dir,
    output_dir,
    train_split=0,
    test_split=1,
    val_split=None,
    shuffle_seed=42,
):
    """
    Convert merged split CSVs into LUNA-formatted train, (val,) and test files.

    The merged files already contain every column LUNA needs:
        <genes> | <gRNAs> | coord_X | coord_Y | cell_section | cell_class

    For the **test** file ``coord_X`` and ``coord_Y`` are set to zero
    (LUNA convention for inference data without spatial information).
    
    This repository also supports generating *both* variants:
    - ``perturb_fish_test.csv`` with coordinates preserved
    - ``perturb_fish_test_0.csv`` with coordinates zeroed

    Parameters
    ----------
    merged_dir : str
        Directory containing ``merged_{id}.csv`` files from merge_splits.
    output_dir : str
        Where to save the final ``perturb_fish_{train,val,test}.csv``.
    train_split : int
        Split ID to use as training data (default 0).
    test_split : int
        Split ID to use as test data (default 1).
    val_split : int, optional
        Split ID to use as validation data. If None, no validation file
        is created.
    shuffle_seed : int, optional
        Random seed used to shuffle cell order before saving. Set to None
        for non-deterministic shuffling (default: 42).
    """
    merged_path = Path(merged_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    roles = [(train_split, "train"), (test_split, "test")]
    if val_split is not None:
        roles.append((val_split, "val"))

    for split_id, role in roles:
        df = pd.read_csv(merged_path / f"merged_{split_id}.csv", index_col=0)

        # Shuffle cell (row) order for all outputs.
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

        gene_cols = df.columns[:550]
        df[gene_cols] = np.log2(df[gene_cols] + 1)

        if role == "test":
            # Write test with coordinates preserved (default)
            out_file = out_path / "perturb_fish_test.csv"
            df.to_csv(out_file)

            # Write an additional variant with coordinates zeroed (suffix "_0")
            df0 = df.copy()
            df0["coord_X"] = 0.0
            df0["coord_Y"] = 0.0
            out_file0 = out_path / "perturb_fish_test_0.csv"
            df0.to_csv(out_file0)
        else:
            out_file = out_path / f"perturb_fish_{role}.csv"
            df.to_csv(out_file)

        n_features = len(df.columns) - 4  # everything except the 4 metadata cols
        print(
            f"{role.capitalize()}: {df.shape[0]:,} cells, "
            f"{n_features} feature cols (genes+gRNAs), "
            f"{df.shape[1]} total cols → {out_file.name}"
        )
        if role == "test":
            print(f"Test (zeroed): {df.shape[0]:,} cells → {out_file0.name} (coordinates zeroed)")

    print(f"\n✓ LUNA-formatted files saved to {out_path.absolute()}")
    print(f"  gene_columns_start = 0")
    print(f"  gene_columns_end   = {n_features}")


if __name__ == "__main__":
    CODEBOOK  = "/data-master/code/Spatial-Transcriptomics-and-Perturbations-Modeling/data_splits/original/codebook_0_ImmunOncology_0.csv"
    FINALS    = "/data-master/code/Spatial-Transcriptomics-and-Perturbations-Modeling/data_splits/original"

    # 1. Split into train (0), val (1), and test (2)
    splits = create_data_splits(
        n_splits=3,
        coords_path=f"{FINALS}/coordinates.csv",
        perturb_path=f"{FINALS}/allcellsPerturbationTable.csv",
        transcriptomics_path=f"{FINALS}/merfishcounttable.csv",
        output_dir="./data_splits",
        random_seed=42,
    )
    print("\nSplit assignment summary:")
    for split_id, cell_indices in splits.items():
        print(f"  Split {split_id}: {len(cell_indices)} cells")

    # 2. Merge transcriptomics + perturbations + coordinates (intermediate)
    merge_splits(
        splits_dir="./data_splits",
        output_dir="./data_splits/merged_splits",
        codebook_path=CODEBOOK,
    )

    # 3. Produce final LUNA-formatted train / val / test CSVs
    prepare_luna_files(
        merged_dir="./data_splits/merged_splits",
        output_dir="./data_splits/luna",
        train_split=0,
        test_split=2,
        val_split=1,
    )