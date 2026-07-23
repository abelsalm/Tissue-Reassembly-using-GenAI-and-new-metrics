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
    # Set the model to train mode
    self.model.train()

    # Preprocess the input data
    batched_data = to_batch(data)
    z_t = self.noise_model.apply_noise(batched_data)

    # Forward pass through the model
    pred = self.forward(z_t)

    min_snr_weight = None
    if getattr(self.cfg.train, "min_snr_weighting", False):
        min_snr_weight = self.noise_model.get_min_snr_weight(
            t_int=z_t.t_int,
            gamma=float(getattr(self.cfg.train, "min_snr_gamma", 5.0)),
            key="p",
        )

    # ``log=False``: do not emit per-batch ``train_loss/*`` metrics (those
    # create spiky WandB curves). Epoch aggregates are logged below.
    loss, _ = self.train_loss(
        masked_pred=pred,
        masked_true=batched_data,
        log=False,
        batch_idx=i,
        min_snr_weight=min_snr_weight,
    )

    # Feed last-step scalars into Lightning every batch; ``on_epoch=True``
    # averages them into a single point per epoch.
    tle_log = self.train_loss.log_epoch_metrics()
    self.log_dict(tle_log, batch_size=self.BS, on_step=False, on_epoch=True)

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

    # Push epoch-averaged metrics (and LR) to WandB once per epoch.
    if wandb.run:
        wandb_log = {"epoch": self.current_epoch}
        try:
            wandb_log["LR"] = float(self.optimizers().param_groups[0]["lr"])
        except Exception:
            pass
        for key, value in cm.items():
            key_str = str(key)
            if key_str.startswith("train_epoch/"):
                try:
                    wandb_log[key_str] = float(value)
                except (TypeError, ValueError):
                    continue
        wandb.log(wandb_log, commit=False)


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

    # Drop GT caches after warp/rechunk so losses recompute from updated GT coords.
    if hasattr(self.train_loss, "clear_gt_cache"):
        self.train_loss.clear_gt_cache()

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
