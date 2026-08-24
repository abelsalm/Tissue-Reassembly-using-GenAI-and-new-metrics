import gc
from typing import Optional

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
    compose_graph_group_labels,
    detect_nan_rows,
    filter_domain_cells,
    graph_group_columns,
    position_normalize,
    resolve_graph_split,
    should_normalize_per_graph,
    standardise_dataframe_colnames,
)
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


def _optional_labels(
    data: pd.DataFrame, column: str, keep_mask: np.ndarray
) -> Optional[np.ndarray]:
    """Return cleaned string labels, preserving missing values as ``None``."""
    if column not in data.columns:
        return None
    values = data[column].to_numpy(dtype=object)[keep_mask]
    return np.asarray(
        [None if pd.isna(value) else str(value) for value in values],
        dtype=object,
    )


def _enabled_embedding(cfg, name: str) -> bool:
    embeddings = getattr(cfg.model, "embeddings", None)
    section = getattr(embeddings, name, None) if embeddings is not None else None
    return bool(getattr(section, "enabled", False)) if section is not None else False


def _build_train_vocabulary(labels: Optional[np.ndarray]) -> tuple[dict, list]:
    """Build a deterministic 1-based vocabulary; index 0 is UNK / pad."""
    if labels is None:
        return {}, []
    names = sorted({str(value) for value in labels if value is not None})
    return {name: i + 1 for i, name in enumerate(names)}, names


def _map_aux_labels(
    labels: Optional[np.ndarray], vocabulary: dict, n_cells: int
) -> torch.Tensor:
    if labels is None:
        return torch.zeros(n_cells, dtype=torch.long)
    return torch.tensor(
        [vocabulary.get(str(value), 0) if value is not None else 0 for value in labels],
        dtype=torch.long,
    )


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
        self.graph_split = resolve_graph_split(cfg)
        self.graph_group_cols = graph_group_columns(cfg)

        self._data, self.slices = Data(), {}

        # Dataset processing pipeline
        self.process_data()
        self.process_slices()

    def process_data(self) -> None:
        self.input_data = filter_domain_cells(self.input_data, self.cfg)
        self.num_cell_class = len(self.input_data["cell_class"].unique())
        self.input_data = self.input_data.sort_values(
            self.graph_group_cols, ignore_index=False
        )
        gene_names = self.filter_genes()
        group_labels = compose_graph_group_labels(
            self.input_data, self.graph_group_cols
        )

        # Normalize coordinates per graph group (domain) or per slice.
        group_by = (
            self.graph_group_cols
            if should_normalize_per_graph(self.cfg)
            else "cell_section"
        )
        self.input_data = position_normalize(self.input_data, group_by=group_by)
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

        # Store grouping labels for each clean cell (matches tensor row order).
        # Used by rechunk() / process_slices() so graphs never cross groups.
        nan_mask = ~nan_rows
        nan_np = nan_mask.numpy()
        self._cell_sections_clean = self.input_data["cell_section"].values[nan_np]
        self._graph_groups_clean = group_labels[nan_np]
        self._cell_ids_clean = self.input_data.index.values[nan_np]
        cell_type_column = str(
            getattr(self.cfg.dataset, "cell_type_column", "cell_class")
        )
        if cell_type_column == "subclass":
            cell_type_column = "cell_class"
        domain_column = str(
            getattr(
                self.cfg.dataset,
                "domain_column",
                "spatial_module_l1_complete",
            )
        )
        self._cell_type_labels_clean = _optional_labels(
            self.input_data, cell_type_column, nan_np
        )
        self._domain_labels_clean = _optional_labels(
            self.input_data, domain_column, nan_np
        )

        (
            clean_positions,
            clean_node_features,
            clean_cell_class,
        ) = self._drop_small_graph_groups(
            clean_positions, clean_node_features, clean_cell_class
        )

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
        # Integer class indices 0 … C-1 (not one-hot); CE expects long targets.
        return (
            positions,
            node_features,
            torch.tensor(cell_class, dtype=torch.long),
            cell_class_decoder,
        )

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
        self._data.cell_type = None
        self._data.domain_id = None
        # Canonical unwarped coordinates; kept in sync with rechunk permutations.
        self._positions_unwarped = clean_positions.clone()

        num_cell_to_region_mapping_dict = self._create_region_mapping_dict()
        self.statistics = Statistics(
            num_cell_class=self.num_cell_class,
            num_genes=len(gene_names),
            cell_class_decoder=cell_class_decoder,
            num_cell_to_region_mapping_dict=num_cell_to_region_mapping_dict,
        )

    def _create_region_mapping_dict(self):
        groups = np.asarray(self._graph_groups_clean)
        names, counts = np.unique(groups, return_counts=True)
        return {int(count): str(name) for name, count in zip(names, counts)}

    def graph_group_count(self) -> int:
        return int(len(np.unique(np.asarray(self._graph_groups_clean))))

    def _drop_small_graph_groups(self, positions, node_features, cell_class):
        min_size = getattr(self.cfg.dataset, "min_graph_size", None)
        if not min_size:
            return positions, node_features, cell_class

        min_size = int(min_size)
        groups = np.asarray(self._graph_groups_clean)
        keep = np.ones(groups.size, dtype=bool)
        dropped_groups = 0
        for group in np.unique(groups):
            idx = groups == group
            if int(idx.sum()) < min_size:
                keep[idx] = False
                dropped_groups += 1
        if dropped_groups == 0:
            return positions, node_features, cell_class

        dropped_cells = int((~keep).sum())
        print(
            f"[{self.name}/{self.split}] Dropped {dropped_groups} graph groups "
            f"with < {min_size} cells ({dropped_cells} cells).",
            flush=True,
        )
        keep_t = torch.from_numpy(keep)
        self._graph_groups_clean = groups[keep]
        self._cell_sections_clean = np.asarray(self._cell_sections_clean)[keep]
        self._cell_ids_clean = np.asarray(self._cell_ids_clean)[keep]
        if self._cell_type_labels_clean is not None:
            self._cell_type_labels_clean = np.asarray(
                self._cell_type_labels_clean, dtype=object
            )[keep]
        if self._domain_labels_clean is not None:
            self._domain_labels_clean = np.asarray(
                self._domain_labels_clean, dtype=object
            )[keep]
        if int(keep_t.sum().item()) == 0:
            raise ValueError(
                f"[{self.name}] Split '{self.split}' became empty after "
                f"min_graph_size={min_size} filtering."
            )
        return positions[keep_t], node_features[keep_t], cell_class[keep_t]

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
        n_graphs = max(int(slice_.numel()) - 1, 0)
        print(
            f"[{self.name}/{self.split}] graph_split={self.graph_split} "
            f"groups={self.graph_group_count()} graphs={n_graphs} "
            f"cells={int(self._data.positions.shape[0])}",
            flush=True,
        )

    def rechunk(self, seed=None) -> None:
        """Randomly reshuffle cells within each graph group.

        Graph groups are whole ``cell_section`` slices, or
        ``(cell_section, domain)`` pairs when ``dataset.graph_split=domain``.
        The chunk boundaries (self.slices) stay the same — they define fixed
        windows of size maximum_graph_size into the flat tensor.  What changes
        is the cell order inside that flat tensor: within each group, all
        cells are randomly permuted, so every chunk receives a fresh random
        draw of ~maximum_graph_size cells from the same slice / domain instead
        of always the same spatial neighbours.

        Has no effect when maximum_graph_size is None.
        """
        if self.maximum_graph_size is None:
            return

        rng = np.random.default_rng(seed)
        total_cells = self._data.positions.shape[0]
        perm = np.arange(total_cells)
        groups = np.asarray(self._graph_groups_clean)

        for group in np.unique(groups):
            indices = np.where(groups == group)[0]
            perm[indices] = rng.permutation(indices)

        perm_t = torch.from_numpy(perm)
        self._data.positions     = self._data.positions[perm_t]
        self._data.node_features = self._data.node_features[perm_t]
        self._data.cell_class    = self._data.cell_class[perm_t]
        self._data.cell_ID       = self._data.cell_ID[perm_t]
        if self._data.cell_type is not None:
            self._data.cell_type = self._data.cell_type[perm_t]
        if self._data.domain_id is not None:
            self._data.domain_id = self._data.domain_id[perm_t]
        if self._cell_type_labels_clean is not None:
            self._cell_type_labels_clean = np.asarray(
                self._cell_type_labels_clean, dtype=object
            )[perm]
        if self._domain_labels_clean is not None:
            self._domain_labels_clean = np.asarray(
                self._domain_labels_clean, dtype=object
            )[perm]
        self._positions_unwarped = self._positions_unwarped[perm_t]

    def set_embedding_ids(
        self,
        *,
        cell_type_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Attach train-vocabulary-aligned auxiliary IDs to this dataset."""
        n_cells = int(self._data.node_features.size(0))
        if cell_type_ids is not None:
            if cell_type_ids.shape != (n_cells,):
                raise ValueError(
                    f"cell_type_ids must be ({n_cells},), "
                    f"got {tuple(cell_type_ids.shape)}"
                )
            self._data.cell_type = cell_type_ids.long()
            self.slices["cell_type"] = self.slices["node_features"]
        if domain_ids is not None:
            if domain_ids.shape != (n_cells,):
                raise ValueError(
                    f"domain_ids must be ({n_cells},), "
                    f"got {tuple(domain_ids.shape)}"
                )
            self._data.domain_id = domain_ids.long()
            self.slices["domain_id"] = self.slices["node_features"]

    def apply_epoch_warp(
        self,
        seed: int,
        *,
        enabled: bool,
        max_displacement: float = 0.01,
        max_angle_span: float = 3.141592653589793 / 2.0,
        grid_size: int = 8,
    ) -> None:
        """Apply (or disable) the global smooth warp for this epoch."""
        from utils.data.warp_augment import apply_warp_field, sample_smooth_warp_field

        if not enabled:
            self._data.positions = self._positions_unwarped.clone()
            return

        field = sample_smooth_warp_field(
            seed,
            grid_size=int(grid_size),
            max_displacement=float(max_displacement),
            max_angle_span=float(max_angle_span),
        )
        self._data.positions = apply_warp_field(self._positions_unwarped, field)

    def _generate_slice_indices(self):
        # IMPORTANT: slice boundaries must be computed on the cleaned rows,
        # otherwise they can exceed the length of self._data tensors and crash
        # torch_geometric's InMemoryDataset slicing.
        # Groups are whole sections, or (section, domain) in domain mode.
        groups = np.asarray(self._graph_groups_clean)
        if groups.size == 0:
            return np.array([0], dtype=int)

        current_group = groups[0]
        slice_start = 0
        boundaries: list[int] = []

        for i in range(1, groups.size + 1):
            is_end = i == groups.size
            if is_end or groups[i] != current_group:
                slice_end = i

                if self.maximum_graph_size is None:
                    boundaries.extend([slice_start, slice_end])
                else:
                    boundaries.extend(
                        np.arange(slice_start, slice_end, self.maximum_graph_size).astype(int).tolist()
                    )
                    boundaries.append(slice_end)

                if not is_end:
                    current_group = groups[i]
                    slice_start = i

        boundaries = sorted(set(int(b) for b in boundaries))
        if boundaries[0] != 0:
            boundaries = [0] + boundaries
        if boundaries[-1] != groups.size:
            boundaries.append(groups.size)

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

        self.cell_type_decoder = []
        self.domain_decoder = []
        self._configure_embedding_metadata(cfg)

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

    def _configure_embedding_metadata(self, cfg) -> None:
        """Align auxiliary categorical IDs to the train split vocabulary."""
        use_cell_type = _enabled_embedding(cfg, "cell_type")
        use_domain = _enabled_embedding(cfg, "domain")

        cell_vocab, self.cell_type_decoder = _build_train_vocabulary(
            self.train_dataset._cell_type_labels_clean
        )
        domain_vocab, self.domain_decoder = _build_train_vocabulary(
            self.train_dataset._domain_labels_clean
        )
        # Build available metadata even when currently disabled. In test-only
        # mode the checkpoint's model config is loaded after DataModule setup,
        # and may enable an embedding that the command-line config did not.
        self.num_cell_types = len(self.cell_type_decoder)
        self.num_domains = len(self.domain_decoder)
        if use_cell_type and self.num_cell_types == 0:
            raise ValueError(
                "Cell-type embedding is enabled but dataset.cell_type_column "
                "was not found or contains no labels."
            )
        if use_domain and self.num_domains == 0:
            raise ValueError(
                "Domain embedding is enabled but dataset.domain_column "
                "was not found or contains no labels."
            )

        datasets = [
            self.train_dataset,
            self.test_dataset,
            self.validation_dataset,
        ]
        for dataset in datasets:
            if dataset is None:
                continue
            n_cells = int(dataset._data.node_features.size(0))
            cell_ids = (
                _map_aux_labels(
                    dataset._cell_type_labels_clean, cell_vocab, n_cells
                )
                if self.num_cell_types > 0
                else None
            )
            domain_ids = (
                _map_aux_labels(
                    dataset._domain_labels_clean, domain_vocab, n_cells
                )
                if self.num_domains > 0
                else None
            )
            dataset.set_embedding_ids(
                cell_type_ids=cell_ids,
                domain_ids=domain_ids,
            )
        print(
            "[DataModule] auxiliary embedding vocabularies: "
            f"cell_types={self.num_cell_types}, domains={self.num_domains} "
            "(0=UNK/pad)",
            flush=True,
        )

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
        if getattr(batch[0], "cell_type", None) is not None:
            batch_data.cell_type = torch.cat(
                [data.cell_type for data in batch], dim=0
            )
        if getattr(batch[0], "domain_id", None) is not None:
            batch_data.domain_id = torch.cat(
                [data.domain_id for data in batch], dim=0
            )

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

        # 3. Standardise column names and validate before grouping / subsample.
        data = standardise_dataframe_colnames(data)
        required = ["coord_X", "coord_Y", "cell_section", "cell_class"]
        if resolve_graph_split(cfg) == "domain":
            required.extend(graph_group_columns(cfg))
        if _enabled_embedding(cfg, "domain"):
            required.append(
                str(
                    getattr(
                        cfg.dataset,
                        "domain_column",
                        "spatial_module_l1_complete",
                    )
                )
            )
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise AssertionError(
                f"CSV is missing required columns {missing}. "
                f"Present: {list(data.columns[:12])}..."
            )

        # 4. Optional subsampling — set ``dataset.subsample_n`` to cap the
        #    number of rows kept per split. Useful to fit very large datasets
        #    in CPU RAM. ``subsample_per_section: True`` keeps at most N rows
        #    *per graph group* (section, or section×domain) instead of N total.
        subsample_n = getattr(cfg.dataset, "subsample_n", None)
        if subsample_n:
            seed = int(getattr(cfg.general, "seed", 0))
            per_section = bool(getattr(cfg.dataset, "subsample_per_section", False))
            group_cols = [c for c in graph_group_columns(cfg) if c in data.columns]
            if per_section and group_cols:
                data = (
                    data.groupby(group_cols, group_keys=False)
                    .apply(lambda g: g.sample(
                        n=min(int(subsample_n), len(g)), random_state=seed
                    ))
                )
            else:
                n = min(int(subsample_n), len(data))
                data = data.sample(n=n, random_state=seed).sort_index()
            print(
                f"[DataModule] Subsampled split={split} to {len(data):,} rows "
                f"(per_group={per_section}, groups={group_cols}, seed={seed})."
            )

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
        self.num_cell_types = int(getattr(datamodule, "num_cell_types", 0))
        self.num_domains = int(getattr(datamodule, "num_domains", 0))
        self.cell_type_decoder = list(
            getattr(datamodule, "cell_type_decoder", [])
        )
        self.domain_decoder = list(getattr(datamodule, "domain_decoder", []))
        train_data = datamodule.train_dataset._data
        self.train_node_features = train_data.node_features
        self.train_cell_type = getattr(train_data, "cell_type", None)
        self.num_cell_to_region_mapping_dict = datamodule.statistics[
            "test"
        ].num_cell_to_region_mapping_dict
        self.input_dims["node_features_dimensions"] = self.num_genes
        self.input_dims["diffusion_time_dimensions"] = 1
        self.output_dims["node_features_dimensions"] = self.num_genes
        self.output_dims["diffusion_time_dimensions"] = 0
