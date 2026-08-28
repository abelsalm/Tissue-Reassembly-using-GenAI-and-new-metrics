import pandas as pd
import torch
from omegaconf import OmegaConf

from datasets.data_module import (
    DataModule,
    Dataset,
    _build_train_vocabulary,
    _map_aux_labels,
)
from utils.data.misc import to_batch, to_domain_context_batch


def _frame():
    rows = []
    for section in ("s1", "s2"):
        for domain, count in (("target", 3), ("context", 2)):
            for index in range(count):
                rows.append(
                    {
                        "g0": float(index + 1),
                        "g1": float(index + 2),
                        "coord_X": float(index),
                        "coord_Y": float(index * index + (domain == "context")),
                        "cell_section": section,
                        "cell_class": "a",
                        "cell_type": "type_a",
                        "spatial_module_l1_complete": domain,
                    }
                )
    result = pd.DataFrame(rows)
    result.index = range(100, 100 + len(result))
    return result


def _cfg(graph_split, maximum_graph_size=None):
    return OmegaConf.create(
        {
            "dataset": {
                "dataset_name": "toy",
                "maximum_graph_size": {"train": maximum_graph_size},
                "graph_split": graph_split,
                "domain_column": "spatial_module_l1_complete",
                "target_domain_values": ["target"],
                "min_target_cells": 2,
                "min_graph_size": None,
                "normalize_positions_per_graph": True,
                "gene_columns_start": 0,
                "gene_columns_end": 2,
                "cell_type_column": "cell_type",
            },
            "model": {
                "embeddings": {
                    "cell_type": {"enabled": True},
                    "domain": {"enabled": True},
                }
            },
        }
    )


def _attach_ids(dataset):
    domain_vocab, _ = _build_train_vocabulary(dataset._domain_labels_clean)
    dataset.set_embedding_ids(
        cell_type_ids=_map_aux_labels(
            dataset._cell_type_labels_clean,
            {"type_a": 1},
            len(dataset._cell_type_labels_clean),
        ),
        domain_ids=_map_aux_labels(
            dataset._domain_labels_clean,
            domain_vocab,
            len(dataset._domain_labels_clean),
        ),
    )
    dataset.configure_context_targets(domain_vocab)
    return domain_vocab


def test_one_target_domain_keeps_other_domains_as_feature_only_context():
    dataset = Dataset("train", _frame(), cfg=_cfg("domain_with_context"))
    domain_vocab = _attach_ids(dataset)

    assert len(dataset) == 2
    item = dataset[0]
    assert item.target_domain_id.item() == domain_vocab["target"]
    assert item.target_membership.sum().item() == 3
    assert set(item.domain_id.tolist()) == set(domain_vocab.values())

    sparse = DataModule._create_batch(None, [dataset[0], dataset[1]])
    batch = to_domain_context_batch(sparse)
    assert batch.context.positions is None
    assert batch.context.node_features.shape[:2] == (2, 5)
    assert batch.target.node_features.shape[:2] == (2, 3)
    assert torch.all(batch.target.domain_id[batch.target.node_mask] == domain_vocab["target"])


def test_legacy_section_batch_contract_is_unchanged():
    dataset = Dataset("train", _frame(), cfg=_cfg("section"))
    _attach_ids(dataset)
    sparse = DataModule._create_batch(None, [dataset[0]])
    dense = to_batch(sparse)

    assert not hasattr(sparse, "target_membership")
    assert dense.positions is not None
    assert dense.node_features.shape[:2] == (1, 5)


def test_maximum_graph_size_subsamples_section_before_target_selection():
    dataset = Dataset(
        "train",
        _frame(),
        cfg=_cfg("domain_with_context", maximum_graph_size=3),
    )
    domain_vocab = _attach_ids(dataset)
    dataset.rechunk(seed=4)

    assert all(
        (end - start) <= 3
        for _, start, end, _ in dataset._context_examples
    )
    for index in range(len(dataset)):
        item = dataset[index]
        assert item.node_features.size(0) <= 3
        assert item.target_membership.any()
        assert torch.equal(
            item.domain_id.eq(domain_vocab["target"]),
            item.target_membership,
        )

    # Ordinary section mode retains its existing section-level chunk count.
    legacy = Dataset(
        "train", _frame(), cfg=_cfg("section", maximum_graph_size=3)
    )
    _attach_ids(legacy)
    assert len(legacy) == 4


if __name__ == "__main__":
    test_one_target_domain_keeps_other_domains_as_feature_only_context()
    test_legacy_section_batch_contract_is_unchanged()
    test_maximum_graph_size_subsamples_section_before_target_selection()
