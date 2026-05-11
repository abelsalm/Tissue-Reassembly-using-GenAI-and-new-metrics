import gc

import numpy as np
import omegaconf
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from utils.data.abstract_datatype import (
    AbstractDataModule,
    AbstractDatasetInfos,
    Statistics,
)
from utils.data.load import (
    character_to_int,
    detect_nan_rows,
    position_normalize,
    standardise_dataframe_colnames
)
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


class Dataset(InMemoryDataset):
    def __init__(
        self,
        split: int,
        input_data: pd.DataFrame,
        root: str = None,
        transform: callable = None,
        pre_transform: callable = None,
        pre_filter: callable = None,
        cfg: omegaconf = None,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.split = split
        self.name = cfg.dataset.dataset_name
        self.input_data = input_data
        self.num_cell_class = len(input_data["cell_class"].unique())
        self.maximum_graph_size = cfg.dataset.maximum_graph_size[split]
        self.cfg = cfg
        
        self._data, self.slices = Data(), {}

        # Dataset processing pipeline
        self.process_data()
        self.process_slices()

    def process_data(self) -> None:
        self.input_data = self.input_data.sort_values("cell_section", ignore_index=False)
        gene_names = self.filter_genes()

        # Normalize coordinates
        self.input_data = position_normalize(self.input_data)
        (
            positions,
            node_features,
            cell_class,
            cell_class_decoder,
        ) = self._convert_data_to_tensors(gene_names)

        # Clean NaN rows (these can appear if coordinates were constant within a section
        # and normalization produced NaNs/inf).
        nan_rows = detect_nan_rows(positions)
        clean_positions, clean_node_features, clean_cell_class = self._clean_data(
            positions, node_features, cell_class, nan_rows
        )
        kept = int((~nan_rows).sum().item())
        total = int(nan_rows.numel())
        if kept == 0:
            raise ValueError(
                f"[{self.name}] Split '{self.split}' became empty after NaN filtering "
                f"(kept={kept}/{total}). Check coord_X/coord_Y and normalization."
            )

        # Store section label for each clean cell (matches tensor row order).
        # Used by rechunk() to restrict shuffling to within-section permutations.
        nan_mask = ~nan_rows
        self._cell_sections_clean = self.input_data["cell_section"].values[nan_mask.numpy()]
        self._cell_ids_clean = self.input_data.index.values[nan_mask.numpy()]

        # Update data attributes
        self._update_data_attributes(
            clean_positions,
            clean_node_features,
            clean_cell_class,
            gene_names,
            cell_class_decoder,
        )

        # Everything we need is now in ``self._data`` (tensors) and in the
        # ``_cell_sections_clean`` / ``_cell_ids_clean`` arrays. The original
        # DataFrame can be very large (~ size_of_csv) and would otherwise stay
        # pinned to this object for the entire run, doubling resident RAM per
        # split. Release it explicitly.
        del self.input_data
        self.input_data = None
        gc.collect()

    def _convert_data_to_tensors(self, gene_names: list):
        positions = torch.tensor(self.input_data[["coord_X", "coord_Y"]].values).float()
        node_features = torch.tensor(self.input_data[gene_names].values).float()
        cell_class = self.input_data["cell_class"]
        unique_class = sorted(list(cell_class.unique()))
        cell_class, cell_class_decoder = character_to_int(
            list(cell_class.values), unique_class
        )
        return positions, node_features, torch.tensor(cell_class), cell_class_decoder

    def _clean_data(self, positions, node_features, cell_class, nan_rows):
        clean_positions = positions[~nan_rows]
        clean_node_features = node_features[~nan_rows]
        clean_cell_class = cell_class[~nan_rows]
        return clean_positions, clean_node_features, clean_cell_class

    def _update_data_attributes(
        self,
        clean_positions,
        clean_node_features,
        clean_cell_class,
        gene_names,
        cell_class_decoder,
    ):
        # IMPORTANT: must match the cleaned tensors length/order (NaN rows removed)
        # Some datasets (e.g. MERFISH ABC) use string cell identifiers as the
        # CSV index, while older ones (e.g. Axolotl) use integers. ``torch.tensor``
        # cannot store strings, so map non-numeric IDs to integer codes here and
        # keep the original labels on the side for debugging / export.
        cell_ids_arr = np.asarray(self._cell_ids_clean)
        if cell_ids_arr.dtype.kind in ("U", "S", "O"):
            codes, uniques = pd.factorize(cell_ids_arr)
            cell_ID = torch.tensor(codes, dtype=torch.long)
            self._cell_ids_original = uniques
        else:
            cell_ID = torch.tensor(cell_ids_arr.astype(np.int64))
            self._cell_ids_original = cell_ids_arr

        self._data.positions = clean_positions
        self._data.node_features = clean_node_features
        self._data.cell_class = clean_cell_class
        self._data.cell_ID = cell_ID

        num_cell_to_region_mapping_dict = self._create_region_mapping_dict()
        self.statistics = Statistics(
            num_cell_class=self.num_cell_class,
            num_genes=len(gene_names),
            cell_class_decoder=cell_class_decoder,
            num_cell_to_region_mapping_dict=num_cell_to_region_mapping_dict,
        )

    def _create_region_mapping_dict(self):
        num_cell_to_region_mapping_dict = (
            self.input_data.groupby("cell_section").size().to_dict()
        )
        return {v: k for k, v in num_cell_to_region_mapping_dict.items()}

    def filter_genes(self) -> list:
        gene_columns_start = self.cfg.dataset.gene_columns_start
        gene_columns_end = self.cfg.dataset.gene_columns_end
        gene_names = list(self.input_data.columns[gene_columns_start:gene_columns_end])
        gene_names.sort()
        return gene_names

    def process_slices(self) -> None:
        slice_indices = self._generate_slice_indices()
        # torch_geometric expects integer slice boundaries
        slice_ = torch.tensor(slice_indices, dtype=torch.long)

        self.slices = {
            k: slice_
            for k in [
                "node_features",
                "positions",
                "cell_class",
                "cell_ID",
            ]
        }

    def rechunk(self, seed=None) -> None:
        """Randomly reshuffle cells within each section, assigning them to new chunks.

        The chunk boundaries (self.slices) stay the same — they define fixed
        windows of size maximum_graph_size into the flat tensor.  What changes
        is the cell order inside that flat tensor: within each section, all
        cells are randomly permuted, so every chunk receives a fresh random
        draw of ~maximum_graph_size cells from the section instead of always
        the same spatial neighbours.

        Has no effect when maximum_graph_size is None.
        """
        if self.maximum_graph_size is None:
            return

        rng = np.random.default_rng(seed)
        total_cells = self._data.positions.shape[0]
        perm = np.arange(total_cells)

        for section in np.unique(self._cell_sections_clean):
            indices = np.where(self._cell_sections_clean == section)[0]
            perm[indices] = rng.permutation(indices)

        perm_t = torch.from_numpy(perm)
        self._data.positions     = self._data.positions[perm_t]
        self._data.node_features = self._data.node_features[perm_t]
        self._data.cell_class    = self._data.cell_class[perm_t]
        self._data.cell_ID       = self._data.cell_ID[perm_t]

    def _generate_slice_indices(self):
        # IMPORTANT: slice boundaries must be computed on the cleaned rows,
        # otherwise they can exceed the length of self._data tensors and crash
        # torch_geometric's InMemoryDataset slicing.
        sections = np.asarray(self._cell_sections_clean)
        if sections.size == 0:
            return np.array([0], dtype=int)

        current_section = sections[0]
        slice_start = 0
        boundaries: list[int] = []

        for i in range(1, sections.size + 1):
            is_end = i == sections.size
            if is_end or sections[i] != current_section:
                slice_end = i

                if self.maximum_graph_size is None:
                    boundaries.extend([slice_start, slice_end])
                else:
                    boundaries.extend(
                        np.arange(slice_start, slice_end, self.maximum_graph_size).astype(int).tolist()
                    )
                    boundaries.append(slice_end)

                if not is_end:
                    current_section = sections[i]
                    slice_start = i

        boundaries = sorted(set(int(b) for b in boundaries))
        if boundaries[0] != 0:
            boundaries = [0] + boundaries
        if boundaries[-1] != sections.size:
            boundaries.append(sections.size)

        # Remove any accidental duplicates / empty ranges (defensive)
        boundaries = [boundaries[0]] + [
            b for i, b in enumerate(boundaries[1:], start=1) if b > boundaries[i - 1]
        ]

        return np.asarray(boundaries, dtype=int)


class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        train_data = self.data_loading(cfg, 'train')
        test_data = self.data_loading(cfg, 'test')
        self.train_dataset = self._initialize_dataset("train", train_data, cfg)
        self.test_dataset = self._initialize_dataset("test", test_data, cfg)

        if cfg.dataset.validation_data_path:
            validation_data = self.data_loading(cfg, 'validation')
            self.validation_dataset = self._initialize_dataset("validation", validation_data, cfg)
        else:
            self.validation_dataset = None

        self.statistics = {
            "train": self.train_dataset.statistics,
            "validation": self.validation_dataset.statistics if self.validation_dataset else None,
            "test": self.test_dataset.statistics,
        }
        super().__init__(
            cfg,
            train_dataset=self.train_dataset,
            val_dataset=self.validation_dataset if self.validation_dataset else None,
            test_dataset=self.test_dataset,
        )

    def _initialize_dataset(self, split, data, cfg):
        return Dataset(split=split, input_data=data, cfg=cfg)

    def collate(self, batch):
        return self._create_batch(batch)

    def _create_batch(self, batch):
        batch_data = Batch()
        batch_data.node_features = torch.cat(
            [data.node_features for data in batch], dim=0
        )
        batch_data.positions = torch.cat([data.positions for data in batch], dim=0)
        batch_data.cell_class = torch.cat([data.cell_class for data in batch], dim=0)
        batch_data.cell_ID = torch.cat([data.cell_ID for data in batch], dim=0)

        batch_data.batch = torch.tensor(
            [
                i
                for i, data in enumerate(batch)
                for _ in range(data.node_features.size(0))
            ],
            dtype=torch.long,
        )
        return batch_data

    def data_loading(self, cfg: omegaconf.DictConfig, split) -> pd.DataFrame:
        if split == 'train':
            data_path = cfg.dataset.train_data_path
        elif split == 'validation':
            data_path = cfg.dataset.validation_data_path
        else:
            data_path = cfg.dataset.test_data_path

        # ── Memory-efficient CSV read for huge datasets ──────────────────
        # 1. Sniff the header so we can identify gene + coordinate columns.
        header_df = pd.read_csv(data_path, nrows=0, index_col=0)
        cols = list(header_df.columns)
        g_start = cfg.dataset.gene_columns_start
        g_end = cfg.dataset.gene_columns_end
        gene_col_names = cols[g_start:g_end]

        # 2. Force float32 for the heavy columns. Pandas defaults to float64
        #    so this halves the peak RAM during parsing.
        dtype_hints = {c: np.float32 for c in gene_col_names}
        for c in ("coord_X", "coord_Y", "x", "y"):
            if c in cols:
                dtype_hints[c] = np.float32

        print(
            f"[DataModule] Reading '{data_path}' (split={split}) "
            f"with float32 dtype on {len(gene_col_names)} gene cols + coord cols …"
        )
        data = pd.read_csv(data_path, index_col=0, dtype=dtype_hints)
        print(f"[DataModule] Loaded {len(data):,} rows for split={split}.")

        # 3. Optional subsampling — set ``dataset.subsample_n`` to cap the
        #    number of rows kept per split. Useful to fit very large datasets
        #    in CPU RAM. ``subsample_per_section: True`` keeps at most N rows
        #    *per section* instead of N rows total.
        subsample_n = getattr(cfg.dataset, "subsample_n", None)
        if subsample_n:
            seed = int(getattr(cfg.general, "seed", 0))
            per_section = bool(getattr(cfg.dataset, "subsample_per_section", False))
            if per_section and "cell_section" in data.columns:
                data = (
                    data.groupby("cell_section", group_keys=False)
                    .apply(lambda g: g.sample(
                        n=min(int(subsample_n), len(g)), random_state=seed
                    ))
                )
            else:
                n = min(int(subsample_n), len(data))
                data = data.sample(n=n, random_state=seed).sort_index()
            print(
                f"[DataModule] Subsampled split={split} to {len(data):,} rows "
                f"(per_section={per_section}, seed={seed})."
            )

        # 4. Standardise column names and validate.
        data = standardise_dataframe_colnames(data)
        assert all(column in data.columns for column in ['coord_X', 'coord_Y', 'cell_section', 'cell_class'])

        return data


class Infos(AbstractDatasetInfos):
    """
    Class for storing information about the MERFISH dataset.

    This class encapsulates various statistics and configurations specific to the
    MERFISH dataset, aiding in dataset handling and model training processes.

    Attributes:
        datamodule: Instance of the data module associated with MERFISH data.
        cfg: Configuration object containing dataset and model parameters.
    """

    def __init__(self, datamodule, cfg):
        self.input_dims = {}
        self.output_dims = {}
        self.name = cfg.dataset.dataset_name
        self.num_cell_class = datamodule.statistics["train"].num_cell_class
        self.num_genes = datamodule.statistics["train"].num_genes
        self.cell_class_decoder = {}
        self.num_cell_to_region_mapping_dict = {}
        self.cell_class_decoder = datamodule.statistics["test"].cell_class_decoder
        self.num_cell_to_region_mapping_dict = datamodule.statistics[
            "test"
        ].num_cell_to_region_mapping_dict
        self.input_dims["node_features_dimensions"] = self.num_genes
        self.input_dims["diffusion_time_dimensions"] = 1
        self.output_dims["node_features_dimensions"] = self.num_genes
        self.output_dims["diffusion_time_dimensions"] = 0
