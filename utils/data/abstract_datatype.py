try:
    from torch_geometric.data import LightningDataset
except ImportError:
    from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import DataLoader


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        super().__init__(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=int(getattr(cfg.dataset, "num_workers", 1)),
            pin_memory=bool(getattr(cfg.dataset, "pin_memory", False)),
        )
        self.cfg = cfg

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, self.cfg.train.batch_size)

    def validation_dataloader(self):
        return self._create_dataloader(
            self.validation_dataset, self.cfg.validation.batch_size
        )

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, self.cfg.test.batch_size)

    def _create_dataloader(self, dataset, batch_size):
        # Do not shuffle for validation/test; shuffling can break deterministic evaluation
        # and is explicitly discouraged by Lightning.
        shuffle = dataset is self.train_dataset
        num_workers = int(getattr(self.cfg.dataset, "num_workers", 1))
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            # Avoid oversubscribing CPUs on shared/HPC nodes; allow override via cfg.dataset.num_workers
            num_workers=num_workers,
            pin_memory=bool(getattr(self.cfg.dataset, "pin_memory", True)),
            collate_fn=self.collate,
        )
        # ``multiprocessing_context`` is only valid when workers are actually
        # spawned. Setting it with num_workers=0 raises a ValueError in PyTorch.
        if num_workers > 0:
            kwargs["multiprocessing_context"] = "fork"
        return DataLoader(dataset, **kwargs)


class AbstractDatasetInfos:
    def __init__(self, cfg):
        self.cfg = cfg
        self.statistics = None
        self.num_cell_class = None
        self.cell_class_decoder = None
        self.num_genes = None


class Statistics:
    def __init__(
        self,
        num_cell_class,
        num_genes,
        num_cell_to_region_mapping_dict,
        cell_class_decoder,
    ):
        self.num_cell_class = num_cell_class
        self.cell_class_decoder = cell_class_decoder
        self.num_cell_to_region_mapping_dict = num_cell_to_region_mapping_dict
        self.num_genes = num_genes
