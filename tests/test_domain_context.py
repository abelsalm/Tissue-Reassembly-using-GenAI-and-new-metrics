import torch

from models.domain_context import (
    DomainContextModel,
    MaskedLinearCrossAttention,
    SliceContextEncoder,
)
from utils.data.dataholder import DataHolder


def test_slice_context_masks_padding_and_uses_role():
    torch.manual_seed(0)
    encoder = SliceContextEncoder(
        hidden_dim=8,
        num_domains=3,
        n_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
    ).eval()
    cells = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, True, False]])
    target = torch.tensor([[True, False, False, False]])
    domain = torch.tensor([1])

    encoded = encoder(cells, mask, target, domain)
    cells_with_changed_pad = cells.clone()
    cells_with_changed_pad[:, -1] = 1e6
    changed_pad = encoder(cells_with_changed_pad, mask, target, domain)
    assert torch.allclose(
        encoded.memory[:, :5], changed_pad.memory[:, :5], atol=1e-5
    )
    assert torch.count_nonzero(encoded.cell_states[:, -1]) == 0

    changed_role = encoder(
        cells,
        mask,
        torch.tensor([[False, True, False, False]]),
        domain,
    )
    assert not torch.allclose(encoded.cell_states, changed_role.cell_states)


def test_masked_linear_cross_attention_ignores_masked_memory():
    torch.manual_seed(1)
    block = MaskedLinearCrossAttention(
        hidden_dim=8,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
    ).eval()
    target = torch.randn(2, 3, 8)
    memory = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]]
    )
    first = block(target, memory, mask)
    changed = memory.clone()
    changed[~mask] = 1e6
    second = block(target, changed, mask)
    assert torch.allclose(first, second, atol=1e-5)


def test_domain_context_encode_once_decode_target_backward():
    torch.manual_seed(2)
    model = DomainContextModel(
        input_dims={
            "node_features_dimensions": 5,
            "diffusion_time_dimensions": 1,
        },
        n_layers=2,
        hidden_mlp_dims={"X": 12, "y": 8, "pos": 8},
        hidden_dims={
            "dx": 8,
            "dy": 1,
            "dd": 8,
            "num_heads": 2,
            "dim_ffX": 16,
            "dim_ffy": 8,
            "output_features_to_pos_dims": 4,
        },
        output_dims={
            "node_features_dimensions": 4,
            "diffusion_time_dimensions": 1,
        },
        context_cfg={
            "n_layers": 1,
            "num_heads": 2,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "cross_attention_every": 1,
            "cross_attention_dropout": 0.0,
            "cross_attention_dim_feedforward": 16,
        },
        num_domains=3,
    )
    model.train()
    full_features = torch.randn(1, 5, 5)
    full_domains = torch.tensor([[1, 2, 1, 3, 0]])
    full_mask = torch.tensor([[True, True, True, True, False]])
    # Coordinates and time are intentionally absent from full-slice context.
    full_slice = DataHolder(
        positions=None,
        node_features=full_features,
        diffusion_time=None,
        domain_id=full_domains,
        node_mask=full_mask,
    )
    cached = model.encode_context(full_slice, torch.tensor([1]))

    target_indices = torch.tensor([[0, 2]])
    target = DataHolder(
        positions=torch.randn(1, 2, 2),
        node_features=full_features[:, [0, 2]].clone(),
        diffusion_time=torch.rand(1, 1),
        domain_id=torch.ones(1, 2, dtype=torch.long),
        node_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    output = model.decode_target(target, cached, target_indices)
    assert output.positions.shape == (1, 2, 2)
    assert output.node_features.shape == (1, 2, 4)
    loss = output.positions.square().mean() + output.node_features.square().mean()
    loss.backward()
    assert model.context_encoder.role_embedding.weight.grad is not None
    assert torch.isfinite(model.context_encoder.role_embedding.weight.grad).all()


def test_legacy_forward_is_unchanged_by_wrapper():
    torch.manual_seed(3)
    kwargs = dict(
        input_dims={
            "node_features_dimensions": 5,
            "diffusion_time_dimensions": 1,
        },
        n_layers=1,
        hidden_mlp_dims={"X": 12, "y": 8, "pos": 8},
        hidden_dims={
            "dx": 8,
            "dy": 1,
            "dd": 8,
            "num_heads": 2,
            "dim_ffX": 16,
            "dim_ffy": 8,
            "output_features_to_pos_dims": 4,
        },
        output_dims={
            "node_features_dimensions": 4,
            "diffusion_time_dimensions": 1,
        },
        num_domains=3,
    )
    wrapper = DomainContextModel(**kwargs).eval()
    data = DataHolder(
        positions=torch.randn(1, 3, 2),
        node_features=torch.randn(1, 3, 5),
        diffusion_time=torch.rand(1, 1),
        domain_id=torch.tensor([[1, 1, 1]]),
        node_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    with torch.no_grad():
        direct = wrapper.target_decoder(data.copy())
        wrapped = wrapper(data.copy())
    assert torch.equal(direct.node_features, wrapped.node_features)
    assert torch.equal(direct.positions, wrapped.positions)
