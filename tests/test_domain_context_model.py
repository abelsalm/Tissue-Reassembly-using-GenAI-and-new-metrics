import torch
from omegaconf import OmegaConf
from types import SimpleNamespace

from diffusion_model import FullDenoisingDiffusion
from models.domain_context import DomainContextModel
from models.model import Model
from utils.data.dataholder import DataHolder, DomainContextBatch
from utils.diffusion_model.sample.sample import sample_from_single_graph
from utils.diffusion_model.train.train import training_step_func


INPUT_DIMS = {
    "node_features_dimensions": 4,
    "diffusion_time_dimensions": 1,
}
OUTPUT_DIMS = {
    "node_features_dimensions": 4,
    "diffusion_time_dimensions": 1,
}
HIDDEN_MLP = {"X": 16, "y": 8, "pos": 8}
HIDDEN = {
    "dx": 8,
    "dy": 1,
    "num_heads": 2,
    "dim_ffX": 16,
    "dim_ffy": 8,
    "dd": 8,
    "output_features_to_pos_dims": 4,
}


def _holder(features, mask, domain_ids=None, positions=None, cell_type=None):
    return DataHolder(
        node_features=features,
        positions=positions,
        node_mask=mask,
        domain_id=domain_ids,
        cell_type=cell_type,
        diffusion_time=torch.full((features.size(0), 1), 0.5),
    )


def _model(attention_type="linear"):
    return DomainContextModel(
        input_dims=INPUT_DIMS,
        output_dims=OUTPUT_DIMS,
        n_layers=2,
        hidden_mlp_dims=HIDDEN_MLP,
        hidden_dims=HIDDEN,
        num_domains=3,
        context_cfg={
            "n_layers": 1,
            "num_heads": 2,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "encoder_attention_type": attention_type,
            "cross_attention_type": attention_type,
            "cross_attention_every": 1,
            "cross_attention_dropout": 0.0,
        },
    ).eval()


def _inputs():
    torch.manual_seed(3)
    context_features = torch.randn(2, 5, 4)
    context_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    domains = torch.tensor([[1, 2, 1, 3, 2], [2, 1, 2, 3, 0]])
    requested = torch.tensor([1, 2])
    membership = domains.eq(requested[:, None]) & context_mask
    mappings = torch.tensor([[0, 2], [0, 2]])
    target_features = context_features.gather(
        1, mappings.unsqueeze(-1).expand(-1, -1, 4)
    )
    target = _holder(
        target_features,
        torch.ones(2, 2, dtype=torch.bool),
        domain_ids=requested[:, None, None].expand(-1, 2, 1),
        positions=torch.randn(2, 2, 2),
        cell_type=torch.ones(2, 2, 1, dtype=torch.long),
    )
    context = _holder(
        context_features,
        context_mask,
        domain_ids=domains.unsqueeze(-1),
        cell_type=torch.ones(2, 5, 1, dtype=torch.long),
    )
    return context, target, membership, requested, mappings


def test_linear_and_multihead_conditional_forward_are_finite():
    for attention_type in ("linear", "multihead"):
        model = _model(attention_type)
        context, target, membership, requested, mappings = _inputs()
        memory = model.encode_context(context, requested, membership)
        output = model.decode_target(target, memory, mappings)
        assert output.positions.shape == target.positions.shape
        assert torch.isfinite(output.positions).all()


def test_context_padding_is_masked_and_cell_permutation_is_equivariant():
    model = _model("linear")
    context, target, membership, requested, mappings = _inputs()
    baseline = model.decode_target(
        target, model.encode_context(context, requested, membership), mappings
    ).positions

    # Padded content cannot affect the second sample.
    padded = context.copy()
    padded.node_features[1, 4] = 1e6
    padded_result = model.decode_target(
        target, model.encode_context(padded, requested, membership), mappings
    ).positions
    assert torch.allclose(baseline[1], padded_result[1], atol=1e-5, rtol=1e-5)

    permutation = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted_context = _holder(
        context.node_features[:, permutation],
        context.node_mask[:, permutation],
        domain_ids=context.domain_id[:, permutation],
        cell_type=context.cell_type[:, permutation],
    )
    permuted_membership = membership[:, permutation]
    permuted_mappings = inverse[mappings]
    permuted = model.decode_target(
        target,
        model.encode_context(
            permuted_context, requested, permuted_membership
        ),
        permuted_mappings,
    ).positions
    assert torch.allclose(baseline, permuted, atol=1e-4, rtol=1e-4)


def test_context_encoder_receives_gradients_and_legacy_model_still_runs():
    model = _model("linear").train()
    context, target, membership, requested, mappings = _inputs()
    output = model.decode_target(
        target, model.encode_context(context, requested, membership), mappings
    )
    output.positions.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.context_encoder.parameters()
    )

    legacy = Model(
        input_dims=INPUT_DIMS,
        output_dims=OUTPUT_DIMS,
        n_layers=2,
        hidden_mlp_dims=HIDDEN_MLP,
        hidden_dims=HIDDEN,
    ).eval()
    legacy_output = legacy(target)
    assert legacy_output.positions.shape == target.positions.shape
    assert torch.isfinite(legacy_output.positions).all()
    conditional = _model("linear")
    report = conditional.load_legacy_decoder_state_dict(legacy.state_dict())
    assert len(report["loaded"]) == len(legacy.state_dict())
    assert not report["skipped"]


def test_diffusion_wrapper_noises_and_decodes_target_rows_only():
    model_cfg = OmegaConf.load("configs/model/default.yaml")
    model_cfg.n_layers = 2
    model_cfg.diffusion_steps = 5
    model_cfg.hidden_mlp_dims = HIDDEN_MLP
    model_cfg.hidden_dims = HIDDEN
    model_cfg.domain_context.enabled = True
    model_cfg.domain_context.n_layers = 1
    model_cfg.domain_context.num_heads = 2
    model_cfg.domain_context.dim_feedforward = 16
    model_cfg.domain_context.cross_attention_dim_feedforward = 16
    model_cfg.domain_context.dropout = 0.0
    model_cfg.domain_context.cross_attention_dropout = 0.0
    model_cfg.embeddings.cell_type.enabled = True
    model_cfg.embeddings.cell_type.dim = 2
    model_cfg.embeddings.cell_type.pca_max_cells = None
    model_cfg.embeddings.domain.enabled = True
    model_cfg.embeddings.domain.dim = 2
    model_cfg.embeddings.domain.pca_max_cells = None
    train_cfg = OmegaConf.load("configs/train/default.yaml")
    train_cfg.cell_type_mds_weight = 0.0
    cfg = OmegaConf.create(
        {
            "general": {"name": "test"},
            "dataset": {"graph_split": "domain_with_context"},
            "model": OmegaConf.to_container(model_cfg, resolve=True),
            "train": OmegaConf.to_container(train_cfg, resolve=True),
        }
    )
    infos = SimpleNamespace(
        input_dims=INPUT_DIMS,
        output_dims=OUTPUT_DIMS,
        num_cell_types=2,
        num_domains=3,
        train_node_features=torch.randn(12, 4),
        train_cell_type=torch.tensor([1, 2] * 6),
    )
    module = FullDenoisingDiffusion(cfg, infos)
    context, target, membership, requested, mappings = _inputs()
    conditional = DomainContextBatch(
        context=context,
        target=target,
        target_membership=membership,
        target_to_context=mappings,
        target_domain_id=requested,
    )
    noisy_target = module.noise_model.apply_noise(target)
    assert noisy_target.positions.shape[1] == mappings.size(1)
    memory = module.prepare_domain_context(
        conditional, apply_dropout=False
    )
    prediction = module.forward(
        noisy_target,
        conditional_batch=conditional,
        context_memory=memory,
    )
    assert prediction.positions.shape == target.positions.shape
    prediction.positions.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in module.model.context_encoder.parameters()
    )

    full_positions = torch.randn(2, 5, 2)
    full_positions.scatter_(
        1, mappings.unsqueeze(-1).expand(-1, -1, 2), target.positions
    )
    sparse_rows = context.node_mask
    sparse = SimpleNamespace(
        node_features=context.node_features[sparse_rows],
        positions=full_positions[sparse_rows],
        cell_class=torch.zeros(int(sparse_rows.sum()), dtype=torch.long),
        cell_ID=torch.arange(int(sparse_rows.sum())),
        cell_type=context.cell_type.squeeze(-1)[sparse_rows],
        domain_id=context.domain_id.squeeze(-1)[sparse_rows],
        target_membership=membership[sparse_rows],
        target_domain_id=requested,
        section_id=torch.arange(2),
        batch=torch.arange(2)[:, None].expand_as(sparse_rows)[sparse_rows],
    )
    module.log_dict = lambda *args, **kwargs: None
    step_loss = training_step_func(module, sparse, 0)
    assert torch.isfinite(step_loss)

    calls = {"count": 0}
    original_prepare = module.prepare_domain_context

    def counted_prepare(*args, **kwargs):
        calls["count"] += 1
        return original_prepare(*args, **kwargs)

    module.prepare_domain_context = counted_prepare
    sampled = sample_from_single_graph(module, batch=conditional)
    assert sampled.shape == target.positions.shape
    assert calls["count"] == 1


def test_full_diffusion_keeps_legacy_section_model_path():
    model_cfg = OmegaConf.load("configs/model/default.yaml")
    model_cfg.n_layers = 2
    model_cfg.diffusion_steps = 5
    model_cfg.hidden_mlp_dims = HIDDEN_MLP
    model_cfg.hidden_dims = HIDDEN
    model_cfg.domain_context.enabled = False
    train_cfg = OmegaConf.load("configs/train/default.yaml")
    cfg = OmegaConf.create(
        {
            "general": {"name": "legacy-test"},
            "dataset": {"graph_split": "section"},
            "model": OmegaConf.to_container(model_cfg, resolve=True),
            "train": OmegaConf.to_container(train_cfg, resolve=True),
        }
    )
    infos = SimpleNamespace(
        input_dims=INPUT_DIMS,
        output_dims=OUTPUT_DIMS,
        num_cell_types=0,
        num_domains=0,
        train_node_features=None,
        train_cell_type=None,
    )
    module = FullDenoisingDiffusion(cfg, infos)
    assert type(module.model) is Model
    _, target, _, _, _ = _inputs()
    noisy = module.noise_model.apply_noise(target)
    output = module(noisy)
    assert output.positions.shape == target.positions.shape


def test_overfit_uses_correct_context_better_than_dropped_or_shuffled():
    torch.manual_seed(17)
    model = _model("linear").train()
    # Target genes/noisy positions are identical across the two examples.
    # Only their non-target slice context distinguishes the desired layouts.
    target_features = torch.zeros(2, 2, 4)
    noisy_positions = torch.tensor(
        [[[-0.3, 0.0], [0.3, 0.0]], [[-0.3, 0.0], [0.3, 0.0]]]
    )
    target = _holder(
        target_features,
        torch.ones(2, 2, dtype=torch.bool),
        domain_ids=torch.ones(2, 2, 1, dtype=torch.long),
        positions=noisy_positions,
        cell_type=torch.ones(2, 2, 1, dtype=torch.long),
    )
    context_features = torch.zeros(2, 4, 4)
    context_features[0, 2:] = 4.0
    context_features[1, 2:] = -4.0
    context = _holder(
        context_features,
        torch.ones(2, 4, dtype=torch.bool),
        domain_ids=torch.tensor(
            [[[1], [1], [2], [2]], [[1], [1], [2], [2]]]
        ),
        cell_type=torch.ones(2, 4, 1, dtype=torch.long),
    )
    membership = torch.tensor(
        [[True, True, False, False], [True, True, False, False]]
    )
    requested = torch.ones(2, dtype=torch.long)
    mappings = torch.tensor([[0, 1], [0, 1]])
    truth = torch.tensor(
        [[[-0.15, 0.0], [0.15, 0.0]], [[-0.48, 0.0], [0.48, 0.0]]]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(80):
        optimizer.zero_grad()
        encoded = model.encode_context(context, requested, membership)
        prediction = model.decode_target(target, encoded, mappings)
        loss = (prediction.positions - truth).square().mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        encoded = model.encode_context(context, requested, membership)
        correct = model.decode_target(target, encoded, mappings).positions
        dropped = model.decode_target(
            target,
            model.modify_context(
                encoded, drop_mask=torch.ones(2, dtype=torch.bool)
            ),
            mappings,
        ).positions
        torch.manual_seed(5)
        shuffled = model.decode_target(
            target,
            model.modify_context(encoded, shuffle_cells=True),
            mappings,
        ).positions
        correct_loss = (correct - truth).square().mean()
        dropped_loss = (dropped - truth).square().mean()
        shuffled_loss = (shuffled - truth).square().mean()
    assert correct_loss < dropped_loss
    assert correct_loss < shuffled_loss


if __name__ == "__main__":
    test_linear_and_multihead_conditional_forward_are_finite()
    test_context_padding_is_masked_and_cell_permutation_is_equivariant()
    test_context_encoder_receives_gradients_and_legacy_model_still_runs()
    test_diffusion_wrapper_noises_and_decodes_target_rows_only()
    test_full_diffusion_keeps_legacy_section_model_path()
    test_overfit_uses_correct_context_better_than_dropped_or_shuffled()
