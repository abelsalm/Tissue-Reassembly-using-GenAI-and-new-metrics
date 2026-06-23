import torch
import wandb

from utils.data.dataholder import DataHolder
from utils.data.misc import to_batch


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

    loss, tl_log_dict = self.train_loss(
        masked_pred=pred,
        masked_true=batched_data,
        log=i % self.log_every_steps == 0,
        batch_idx=i,
    )

    # Log the training loss and metrics if available
    if tl_log_dict is not None:
        self.log_dict(tl_log_dict, batch_size=self.BS)

    # Log epoch metrics for training loss. ``on_epoch=True`` is required so
    # the key is guaranteed to be in ``trainer.callback_metrics`` at the end
    # of the epoch (otherwise the per-epoch print below silently no-ops).
    tle_log = self.train_loss.log_epoch_metrics()
    self.log_dict(tle_log, batch_size=self.BS, on_step=False, on_epoch=True)

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
    cm = self.trainer.callback_metrics
    # Pick the first key Lightning actually produced for this run. The
    # epoch suffix is appended automatically when ``on_epoch=True`` is
    # used in ``log_dict``. Multi-radius-slide keys come first (the
    # combined loss), then the plain multi-radius keys.
    epoch_loss = (
        cm.get("train_epoch/combined")
        or cm.get("train_epoch/combined_epoch")
        or cm.get("train_epoch/neighborhood_multi_radius_slide")
        or cm.get("train_epoch/neighborhood_multi_radius")
    )
    if epoch_loss is not None:
        try:
            print(f"[Epoch {self.current_epoch}] Loss: {float(epoch_loss):.6f}", flush=True)
        except (TypeError, ValueError):
            print(f"[Epoch {self.current_epoch}] Loss: {epoch_loss}", flush=True)
    else:
        print(f"[Epoch {self.current_epoch}] done (no loss in callback_metrics)", flush=True)


def on_train_epoch_start_func(self) -> None:
    """
    Callback function called at the start of each training epoch.

    Returns:
    - None
    """

    # Tell the loss which epoch we're on (used e.g. for the transcriptome
    # tolerance-band warmup). Defensive ``hasattr`` so swapping in a plain
    # ``LossFunction`` keeps working.
    if hasattr(self.train_loss, "set_current_epoch"):
        self.train_loss.set_current_epoch(self.current_epoch)

    # Reset training loss and metrics for the new epoch
    self.train_loss.reset()

    # Re-randomise chunk boundaries every N epochs to prevent the model from
    # overfitting to fixed local cell neighbourhoods.
    rechunk_every = getattr(self.cfg.train, "rechunk_every_n_epochs", 0)
    datamodule = getattr(self.trainer, "datamodule", None)
    if datamodule is not None and hasattr(datamodule, "train_dataset"):
        train_ds = datamodule.train_dataset
        if rechunk_every > 0 and self.current_epoch % rechunk_every == 0:
            train_ds.rechunk(seed=self.current_epoch)
            if wandb.run:
                wandb.log({"rechunk_epoch": self.current_epoch}, commit=False)

        if getattr(self.cfg.train, "position_warp_augment", False):
            train_ds.apply_epoch_warp(
                seed=self.current_epoch,
                enabled=True,
                max_displacement=float(
                    getattr(self.cfg.train, "position_warp_max_displacement", 0.01)
                ),
                max_angle_span=float(
                    getattr(
                        self.cfg.train,
                        "position_warp_max_angle_span",
                        3.141592653589793 / 2.0,
                    )
                ),
                grid_size=int(getattr(self.cfg.train, "position_warp_grid_size", 8)),
            )
        elif hasattr(train_ds, "apply_epoch_warp"):
            train_ds.apply_epoch_warp(seed=self.current_epoch, enabled=False)

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
