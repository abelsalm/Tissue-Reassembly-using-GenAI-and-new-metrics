import math

import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from torch_geometric.utils import to_dense_batch

from utils.data.dataholder import DataHolder, DomainContextBatch


def to_batch(data: DataHolder, device=None) -> DataHolder:
    """
    Convert data to dense representation.
    I.e. it converts the node features and positions to a batch of tensors of
    same dimensions. (i.e. it pads the data to the maximum number of nodes in the batch).

    Args:
        data: Input data.
        device (torch.device, optional): Device for the dense data. Defaults to None.

    Returns:
        DataHolder: Batch representation of the input data.
    """

    node_features, node_mask = to_dense_batch(x=data.node_features, batch=data.batch)
    pos, _ = to_dense_batch(x=data.positions, batch=data.batch)
    cell_class, _ = to_dense_batch(x=data.cell_class, batch=data.batch)
    if cell_class.dim() == 2:
        cell_class = cell_class.unsqueeze(-1)
    elif cell_class.dim() == 3:
        pass
    else:
        raise ValueError("cell_class has wrong dimensionality")
    try:
        cell_ID, _ = to_dense_batch(x=data.cell_ID, batch=data.batch)
        if cell_ID.dim() == 2:
            cell_ID = cell_ID.unsqueeze(-1)
        elif cell_ID.dim() == 3:
            pass
        else:
            raise ValueError("cell_ID has wrong dimensionality")
    except AttributeError:
        cell_ID = None
    cell_type = _optional_dense_attribute(data, "cell_type")
    domain_id = _optional_dense_attribute(data, "domain_id")
    pos = pos.float()

    if device is not None:
        node_features = node_features.to(device)
        pos = pos.to(device)
        node_mask = node_mask.to(device)
        cell_class = cell_class.to(device)
        cell_ID = cell_ID.to(device) if cell_ID is not None else None
        cell_type = cell_type.to(device) if cell_type is not None else None
        domain_id = domain_id.to(device) if domain_id is not None else None

    data = DataHolder(
        node_features=node_features,
        positions=pos,
        node_mask=node_mask,
        cell_class=cell_class,
        cell_ID=cell_ID,
        cell_type=cell_type,
        domain_id=domain_id,
        diffusion_time=None,
    ).mask()

    return data


def _optional_dense_attribute(data, name):
    """Densify an optional per-node integer attribute as ``(B, N, 1)``."""
    value = getattr(data, name, None)
    if value is None:
        return None
    dense, _ = to_dense_batch(x=value, batch=data.batch)
    if dense.dim() == 2:
        dense = dense.unsqueeze(-1)
    elif dense.dim() != 3:
        raise ValueError(f"{name} has wrong dimensionality: {tuple(dense.shape)}")
    return dense


def _gather_dense_rows(
    value: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Gather padded row indices from a dense ``(B, N, ...)`` tensor."""
    if value is None:
        return None
    safe = indices.clamp_min(0)
    gather_index = safe
    while gather_index.dim() < value.dim():
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(
        *safe.shape, *value.shape[2:]
    )
    gathered = torch.gather(value, dim=1, index=gather_index)
    return apply_target_mask(gathered, mask)


def apply_target_mask(value, mask):
    if value is None:
        return None
    expanded = mask
    while expanded.dim() < value.dim():
        expanded = expanded.unsqueeze(-1)
    return value * expanded.to(dtype=value.dtype)


def to_domain_context_batch(data, device=None) -> DomainContextBatch:
    """Create a feature-only full-slice context and target-only DDPM holder."""
    if getattr(data, "target_membership", None) is None:
        raise ValueError("Batch is missing target_membership")
    if getattr(data, "target_domain_id", None) is None:
        raise ValueError("Batch is missing target_domain_id")

    node_features, node_mask = to_dense_batch(data.node_features, data.batch)
    positions, _ = to_dense_batch(data.positions, data.batch)
    cell_class = _optional_dense_attribute(data, "cell_class")
    cell_ID = _optional_dense_attribute(data, "cell_ID")
    cell_type = _optional_dense_attribute(data, "cell_type")
    domain_id = _optional_dense_attribute(data, "domain_id")
    membership, _ = to_dense_batch(
        data.target_membership.bool(), data.batch, fill_value=False
    )
    membership &= node_mask

    target_domain_id = data.target_domain_id.long().reshape(-1)
    batch_size = int(node_mask.size(0))
    if target_domain_id.numel() != batch_size:
        raise ValueError(
            "Expected exactly one target_domain_id per slice, got "
            f"{target_domain_id.numel()} IDs for batch size {batch_size}"
        )
    if domain_id is None:
        raise ValueError("domain_with_context requires per-cell domain_id")

    counts = membership.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every conditional sample must contain target cells")
    selected_domain = domain_id.squeeze(-1)
    expected = target_domain_id[:, None].expand_as(selected_domain)
    if not torch.equal(membership, selected_domain.eq(expected) & node_mask):
        raise ValueError(
            "target_membership must identify exactly the requested domain "
            "inside each parent slice"
        )

    max_targets = int(counts.max().item())
    target_to_context = torch.full(
        (batch_size, max_targets),
        -1,
        dtype=torch.long,
        device=node_features.device,
    )
    target_node_mask = torch.zeros(
        (batch_size, max_targets),
        dtype=torch.bool,
        device=node_features.device,
    )
    for batch_index in range(batch_size):
        rows = torch.nonzero(membership[batch_index], as_tuple=False).flatten()
        n_rows = int(rows.numel())
        target_to_context[batch_index, :n_rows] = rows
        target_node_mask[batch_index, :n_rows] = True

    target_features = _gather_dense_rows(
        node_features, target_to_context, target_node_mask
    )
    target_positions = _gather_dense_rows(
        positions.float(), target_to_context, target_node_mask
    )
    target = DataHolder(
        node_features=target_features,
        positions=target_positions,
        node_mask=target_node_mask,
        cell_class=_gather_dense_rows(
            cell_class, target_to_context, target_node_mask
        ),
        cell_ID=_gather_dense_rows(cell_ID, target_to_context, target_node_mask),
        cell_type=_gather_dense_rows(
            cell_type, target_to_context, target_node_mask
        ),
        domain_id=_gather_dense_rows(
            domain_id, target_to_context, target_node_mask
        ),
        diffusion_time=None,
    ).mask()
    context = DataHolder(
        node_features=node_features,
        positions=None,
        node_mask=node_mask,
        cell_class=cell_class,
        cell_ID=cell_ID,
        cell_type=cell_type,
        domain_id=domain_id,
        diffusion_time=None,
    ).mask()
    result = DomainContextBatch(
        context=context,
        target=target,
        target_membership=membership,
        target_to_context=target_to_context,
        target_domain_id=target_domain_id,
        section_id=getattr(data, "section_id", None),
    )
    if device is not None:
        result.device_as(node_features.to(device))
    return result


def setup_wandb(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """
    Initializes the Weights & Biases (wandb) environment for experiment tracking.

    This function converts the OmegaConf configuration object to a dictionary, sets up the
    wandb environment with specified settings (including project name, configuration, and other
    wandb settings), and then initializes wandb. It also saves any .txt files to the wandb dashboard.

    Parameters:
    cfg (OmegaConf): An OmegaConf configuration object containing the setup parameters for wandb.

    Returns:
    OmegaConf: The configuration object (unchanged).
    """
    # Convert OmegaConf configuration to a dictionary
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    # Setup wandb initialization arguments
    kwargs = {
        "name": cfg.general.name,
        "project": f'MolDiffusion_{cfg.dataset["dataset_name"]}',
        "config": config_dict,
        "reinit": True,
        "mode": cfg.general.wandb,
    }

    # Initialize wandb
    wandb.init(**kwargs)

    # Save .txt files to wandb
    wandb.save("*.txt")

    return cfg


class GradientMagnitudeCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        """
        Callback function executed after the backward pass to log gradient magnitudes and statistics.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer object.
            pl_module (Module): PyTorch Lightning Module object.

        Raises:
            ValueError: If gradient magnitude is NaN.
        """
        list_of_nans = []  # List to store parameters with NaN gradients
        check_nan_global = False  # Flag to check if any gradient magnitude is NaN
        for name, param in pl_module.named_parameters():
            check_nan = False
            if param.grad is not None:
                # Calculate gradient magnitude and check for NaN
                gradient_magnitude = param.grad.abs().mean().item()
                check_nan = math.isnan(gradient_magnitude)
                if check_nan:
                    list_of_nans.append(name)
                    check_nan_global = True
                else:
                    wandb.log({f"gradient_norm/{name}": gradient_magnitude})

        if not check_nan_global:
            # If no NaN gradients, log statistics of all gradients
            all_gradients = torch.cat(
                [
                    param.grad.view(-1)
                    for param in pl_module.parameters()
                    if param.grad is not None
                ]
            )
            wandb.log(
                {
                    "cumulative_gradient_norms/cumulative_gradient_histogram": wandb.Histogram(
                        all_gradients.cpu().detach().numpy()
                    )
                }
            )
            wandb.log(
                {
                    "cumulative_gradient_norms/cumulative_gradient_statistics": {
                        "mean": all_gradients.mean().item(),
                        "std": all_gradients.std().item(),
                        "min": all_gradients.min().item(),
                        "max": all_gradients.max().item(),
                    }
                }
            )
        else:
            # If any NaN gradients, log the parameters with NaN gradients
            nan_data = {
                "list_of_nans": list_of_nans,
                "sl_no": list(range(len(list_of_nans))),
            }
            df = pd.DataFrame(nan_data)
            wandb.log({"Nan_layers": wandb.Table(dataframe=df)})
            raise ValueError("Gradient magnitude is NaN")
