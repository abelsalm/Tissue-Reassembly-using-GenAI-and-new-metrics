import numpy as np
import torch
from utils.data.dataholder import DataHolder, DomainContextBatch


@torch.no_grad()
def sample_noise(self, batch: DataHolder) -> torch.Tensor:
    """
    Samples noise z_t from the noise model based on the node features and other batch information.

    Parameters:
        batch (DataHolder): Batch data containing node features, cell class, and other related information.
        seed (int): Seed for random noise generation.

    Returns:
        torch.Tensor: Sampled noise z_t.
    """
    node_features = batch.node_features
    cell_class = batch.cell_class
    cell_ID = batch.cell_ID
    node_mask = batch.node_mask

    z_t = self.noise_model.sample_limit_dist(
        node_features=node_features,
        cell_class=cell_class,
        cell_ID=cell_ID,
        cell_type=batch.cell_type,
        domain_id=batch.domain_id,
        node_mask=node_mask,
    )

    return z_t.device_as(node_features)


def iterate_sampling(
    self,
    z_t: torch.Tensor,
    batch: DataHolder,
    conditional_batch: DomainContextBatch = None,
) -> torch.Tensor:
    """
    Iteratively sample p(z_s | z_t) over the diffusion steps.

    Parameters:
        z_t (torch.Tensor): The initial sampled noise.
        batch (DataHolder): The batch data containing node features.

    Returns:
        torch.Tensor: The final sampled graph after diffusion.
    """
    sample_interval = 1  # Sample interval for the diffusion process

    # Iteratively sample z_s from z_t for each diffusion step
    batch_size = batch.node_features.size(0)
    context_memory = None
    unconditional_memory = None
    guidance_scale = 1.0
    if conditional_batch is not None:
        context_memory = self.prepare_domain_context(
            conditional_batch, apply_dropout=False
        )
        guidance_scale = float(
            getattr(
                getattr(self.cfg.model, "domain_context", None),
                "guidance_scale",
                1.0,
            )
        )
        if guidance_scale != 1.0:
            unconditional_memory = self.drop_domain_context(context_memory)
    for s_int in reversed(range(0, self.max_diffusion_steps, sample_interval)):
        s_array = torch.full(
            (batch_size, 1),
            s_int,
            dtype=torch.long,
            device=batch.node_features.device,
        )
        z_s = sample_zs_from_zt(
            self,
            z_t,
            s_array,
            conditional_batch=conditional_batch,
            context_memory=context_memory,
            unconditional_memory=unconditional_memory,
            guidance_scale=guidance_scale,
        )
        z_t = z_s

    return z_t


@torch.no_grad()
def sample_from_single_graph(
    self, test: bool = True, batch: DataHolder = None
) -> torch.Tensor:
    """
    Samples a batch with specified number of nodes for each graph.

    Parameters:
        test (bool, optional): If True, sampling is done in test mode. Defaults to True.
        seed (int, optional): Seed for random noise generation. Defaults to 0.
        batch (DataHolder, optional): DataHolder containing the data. Defaults to None.

    Returns:
        torch.Tensor: Sampled graph positions.
    """
    conditional_batch = (
        batch if isinstance(batch, DomainContextBatch) else None
    )
    target_batch = batch.target if conditional_batch is not None else batch
    num_node = target_batch.positions[target_batch.node_mask].shape[0]
    print(f"Sampling. The number of nodes to sample is {num_node}.")

    # Sample noise z_t from the batch
    z_t = sample_noise(self, target_batch)

    # Perform iterative sampling over diffusion steps
    sampled_graph = iterate_sampling(
        self, z_t, target_batch, conditional_batch=conditional_batch
    )

    return sampled_graph.positions


def sample_zs_from_zt(
    self,
    z_t: torch.Tensor,
    s_int: torch.Tensor,
    conditional_batch: DomainContextBatch = None,
    context_memory=None,
    unconditional_memory=None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Samples zs ~ p(zs | zt) for the denoising process.

    Parameters:
        z_t (torch.Tensor): The tensor representing z_t in the diffusion process.
        s_int (torch.Tensor): The tensor representing the integer time step s.

    Returns:
        torch.Tensor: The sampled zs tensor.
    """
    pred = self.forward(
        z_t,
        conditional_batch=conditional_batch,
        context_memory=context_memory,
    )
    if unconditional_memory is not None:
        unconditional_pred = self.forward(
            z_t,
            conditional_batch=conditional_batch,
            context_memory=unconditional_memory,
        )
        pred.positions = unconditional_pred.positions + float(guidance_scale) * (
            pred.positions - unconditional_pred.positions
        )
    z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_int)
    return z_s


@torch.no_grad()
def sample_graphs(self, batch: DataHolder, test: bool) -> list[torch.Tensor]:
    """
    Samples multiple graphs from the given batch data.

    Parameters:
        batch (DataHolder): The batch of data to sample from.
        samples_to_generate (int): Number of graph samples to generate.
        test (bool): If True, sampling is done in test mode.

    Returns:
        list[torch.Tensor]: A list of sampled graph positions.
    """
    samples = []

    sample = sample_from_single_graph(self, test=test, batch=batch)
    samples.extend(sample)

    return samples
