from math import ceil

import torch
import wandb

from utils.data.dataholder import DataHolder
from utils.data.misc import to_batch


def _compute_ch_weight(cfg, current_epoch: int) -> float:
    """Step the CH loss weight from zero to its final value after warmup."""
    total_epochs = int(cfg.train.n_epochs)
    warmup_epochs = int(total_epochs * float(cfg.train.ch_warmup_fraction))
    final_weight = float(cfg.train.ch_final_weight)
    ramp_every = max(1, int(cfg.train.ch_ramp_every_n_epochs))

    if current_epoch < warmup_epochs or final_weight == 0.0:
        return 0.0

    ramp_epochs = max(1, total_epochs - warmup_epochs)
    ramp_steps = max(1, ceil(ramp_epochs / ramp_every))
    current_step = min(
        ((current_epoch - warmup_epochs) // ramp_every) + 1,
        ramp_steps,
    )
    return final_weight * current_step / ramp_steps


def training_step_func(self, data: DataHolder, i: int) -> torch.Tensor:
    """
    Training step for a single batch.

    Parameters:
    - data: Batch of input data.
    - i: Index of the current batch.

    Returns:
    - torch.Tensor: Loss for the current batch.
    """
    # Get the current learning rate and log it if using WandB
    lr = self.optimizers().param_groups[0]["lr"]
    if wandb.run:
        wandb.log({"LR": lr}, commit=False)

    # Set the model to train mode
    self.model.train()

    # Preprocess the input data
    batched_data = to_batch(data)
    z_t = self.noise_model.apply_noise(batched_data)

    # Forward pass through the model

    pred = self.forward(z_t)

    # Compute the training loss
    loss, tl_log_dict = self.train_loss(
        masked_pred=pred, masked_true=batched_data, log=i % self.log_every_steps == 0
    )
    loss = loss

    # Log the training loss and metrics if available
    if tl_log_dict is not None:
        self.log_dict(tl_log_dict, batch_size=self.BS)

    # Log epoch metrics for training loss
    tle_log = self.train_loss.log_epoch_metrics()
    self.log_dict(tle_log, batch_size=self.BS)

    # Log the epoch number if using WandB
    if wandb.run:
        wandb.log({"epoch": self.current_epoch}, commit=False)
    return loss


def on_train_epoch_end_func(self) -> None:
    """
    Callback function called at the end of each training epoch.

    Returns:
    - None
    """
    epoch_loss = self.trainer.callback_metrics.get("train_epoch/position_mse")
    if epoch_loss is not None:
        print(f"[Epoch {self.current_epoch}] Loss: {epoch_loss:.6f}")


def on_train_epoch_start_func(self) -> None:
    """
    Callback function called at the start of each training epoch.

    Returns:
    - None
    """

    ch_weight = None
    if hasattr(self.train_loss, "ch_weight"):
        ch_weight = _compute_ch_weight(self.cfg, self.current_epoch)
        self.train_loss.ch_weight = ch_weight

    # Reset training loss and metrics for the new epoch
    self.train_loss.reset()

    if ch_weight is not None:
        self.log("train_loss/ch_weight", ch_weight, on_epoch=True, sync_dist=True)
        if wandb.run:
            wandb.log({"train_loss/ch_weight": ch_weight}, commit=False)

    # Re-randomise chunk boundaries every N epochs to prevent the model from
    # overfitting to fixed local cell neighbourhoods.
    rechunk_every = getattr(self.cfg.train, "rechunk_every_n_epochs", 0)
    if rechunk_every > 0 and self.current_epoch % rechunk_every == 0:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "train_dataset"):
            datamodule.train_dataset.rechunk(seed=self.current_epoch)
            if wandb.run:
                wandb.log({"rechunk_epoch": self.current_epoch}, commit=False)

    # Debug: print where a fixed cell_ID ended up after shuffling.
    dbg_every = getattr(self.cfg.train, "debug_print_shuffle_every_n_epochs", 0)
    if dbg_every > 0 and self.current_epoch % dbg_every == 0:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "train_dataset"):
            ds = datamodule.train_dataset
            row = int(getattr(self.cfg.train, "debug_shuffle_row_index", 0))
            n = int(ds._data.positions.shape[0])
            if row < 0 or row >= n:
                print(f"[Epoch {self.current_epoch}] shuffle-canary: row_index={row} out_of_range (n_cells={n})")
            else:
                cell_id = int(ds._data.cell_ID[row].item())
                x, y = ds._data.positions[row].tolist()
                print(
                    f"[Epoch {self.current_epoch}] shuffle-canary: "
                    f"row_index={row} cell_ID={cell_id} coord=({x:.4f}, {y:.4f})"
                )
