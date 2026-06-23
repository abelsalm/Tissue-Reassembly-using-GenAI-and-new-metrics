import torch
import wandb

from utils.data.dataholder import DataHolder
from utils.data.misc import to_batch


def _validation_batch_size(self) -> int:
    return int(getattr(self.cfg.validation, "batch_size", self.BS))


def on_validation_epoch_start_func(self) -> None:
    """Reset validation losses and sync epoch-dependent loss state."""
    if hasattr(self.train_loss, "set_current_epoch"):
        self.train_loss.set_current_epoch(self.current_epoch)
    self.train_loss.reset()
    self.vanilla_val_loss.reset()


def validation_step_func(self, data: DataHolder, i: int) -> torch.Tensor:
    """Validation step: combined training loss + vanilla position MSE on the side."""
    self.model.eval()
    batch_size = _validation_batch_size(self)
    should_log = i % self.log_every_steps == 0

    with torch.no_grad():
        batched_data = to_batch(data)
        z_t = self.noise_model.apply_noise(batched_data, train_flag=False)
        pred = self.forward(z_t)

        loss, val_log_dict = self.train_loss(
            masked_pred=pred,
            masked_true=batched_data,
            train_stage=False,
            log=should_log,
            batch_idx=i,
        )

        _, vanilla_log_dict = self.vanilla_val_loss(
            masked_pred=pred,
            masked_true=batched_data,
            train_stage=False,
            log=should_log,
        )

    if val_log_dict is not None:
        self.log_dict(val_log_dict, batch_size=batch_size)

    if vanilla_log_dict is not None:
        self.log_dict(vanilla_log_dict, batch_size=batch_size)

    val_epoch_log = self.train_loss.log_epoch_metrics(train_stage=False)
    self.log_dict(val_epoch_log, batch_size=batch_size, on_step=False, on_epoch=True)

    vanilla_epoch_log = self.vanilla_val_loss.log_epoch_metrics(train_stage=False)
    self.log_dict(vanilla_epoch_log, batch_size=batch_size, on_step=False, on_epoch=True)

    return loss


def on_validation_epoch_end_func(self) -> None:
    """Print validation summary and push epoch aggregates to WandB."""
    cm = self.trainer.callback_metrics
    epoch_loss = (
        cm.get("val_epoch/combined")
        or cm.get("val_epoch/combined_epoch")
        or cm.get("val_loss/combined")
        or cm.get("val_loss/combined_epoch")
    )
    if epoch_loss is not None:
        try:
            print(
                f"[Val epoch {self.current_epoch}] Loss: {float(epoch_loss):.6f}",
                flush=True,
            )
        except (TypeError, ValueError):
            print(
                f"[Val epoch {self.current_epoch}] Loss: {epoch_loss}",
                flush=True,
            )

    if wandb.run:
        wandb_log = {}
        for key, value in cm.items():
            key_str = str(key)
            if key_str.startswith("val_loss/") or key_str.startswith("val_epoch/"):
                try:
                    wandb_log[key_str] = float(value)
                except (TypeError, ValueError):
                    continue
        if wandb_log:
            wandb.log(wandb_log, commit=False)
