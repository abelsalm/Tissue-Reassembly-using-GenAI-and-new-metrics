import math

import pytorch_lightning as pl
import torch
import wandb
from metrics.train_loss import CombinedTrainLoss
from metrics.test_vanilla_loss import LossFunction
from models.model import Model
from utils.data.dataholder import DataHolder
from utils.data.misc import setup_wandb
from utils.diffusion_model.diffusion.noise_model import NoiseModel

from utils.diffusion_model.test.test import (
    on_test_epoch_end_func,
    on_test_epoch_start_func,
    test_step_func,
)
from utils.diffusion_model.train.train import (
    on_train_epoch_end_func,
    on_train_epoch_start_func,
    training_step_func,
)
from utils.diffusion_model.validation.val import (
    on_validation_epoch_end_func,
    on_validation_epoch_start_func,
    validation_step_func,
)


class LogCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    """Cosine warm-restarts with log-space interpolation between ``eta_max`` and ``eta_min``.

    Standard ``CosineAnnealingWarmRestarts`` interpolates LR linearly between
    the peak and the floor. This variant interpolates ``log(lr)`` instead, so
    the schedule is geometric in LR space::

        log(η_t) = log(η_min) + ½ (log(η_max) - log(η_min)) (1 + cos(π T_cur / T_i))

    Requires ``eta_min > 0`` and positive base learning rates.
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if float(eta_min) <= 0.0:
            raise ValueError(
                "LogCosineAnnealingWarmRestarts requires eta_min > 0 "
                f"(got {eta_min})."
            )
        super().__init__(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings

            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        cos_term = (1.0 + math.cos(math.pi * self.T_cur / self.T_i)) / 2.0
        log_min = math.log(float(self.eta_min))
        return [
            math.exp(log_min + (math.log(float(base_lr)) - log_min) * cos_term)
            for base_lr in self.base_lrs
        ]


class FullDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None

    def __init__(self, cfg, dataset_infos):
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name
        self.max_diffusion_steps = cfg.model.diffusion_steps
        self.log_every_steps = True

        self.dataset_infos = dataset_infos
        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.train_loss = CombinedTrainLoss(cfg.train)
        self.vanilla_val_loss = LossFunction()

        self.model = Model(
            input_dims=self.input_dims,
            n_layers=cfg.model.n_layers,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
        )

        self.noise_model = NoiseModel(cfg)

    def on_train_epoch_start(self) -> None:
        anomaly_from_epoch = int(
            getattr(self.cfg.train, "detect_anomaly_from_epoch", -1)
        )
        anomaly_enabled = (
            anomaly_from_epoch >= 0 and self.current_epoch >= anomaly_from_epoch
        )
        torch.autograd.set_detect_anomaly(anomaly_enabled, check_nan=True)
        if (
            anomaly_enabled
            and self.current_epoch == anomaly_from_epoch
            and self.trainer.is_global_zero
        ):
            print(
                f"[Epoch {self.current_epoch}] Autograd anomaly detection enabled.",
                flush=True,
            )
        on_train_epoch_start_func(self)

    def training_step(self, data, i) -> torch.Tensor:
        loss = training_step_func(self, data, i)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the global pre-clipping gradient norm and reject non-finite gradients."""
        squared_norm = torch.zeros((), device=self.device, dtype=torch.float64)
        max_abs = torch.zeros((), device=self.device)
        nonfinite_tensors = []

        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                nonfinite_tensors.append(name)
                continue
            grad_norm = torch.linalg.vector_norm(grad, ord=2, dtype=torch.float64)
            squared_norm += grad_norm.square()
            max_abs = torch.maximum(max_abs, grad.abs().max().to(max_abs.dtype))

        global_norm = squared_norm.sqrt()
        global_norm_value = float(global_norm.item())
        max_abs_value = float(max_abs.item())
        clip_val = float(getattr(self.cfg.train, "gradient_clip_val", 1.0))

        self.log(
            "train_step/grad_global_l2_norm_pre_clip",
            global_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        if self.trainer.is_global_zero and wandb.run:
            wandb.log(
                {
                    "grad/global_l2_norm_pre_clip": global_norm_value,
                    "grad/max_abs_pre_clip": max_abs_value,
                    "grad/clip_threshold": clip_val,
                    "grad/nonfinite_tensor_count": len(nonfinite_tensors),
                    "trainer/global_step": int(self.global_step),
                },
                commit=True,
            )

        if nonfinite_tensors or not math.isfinite(global_norm_value):
            names = ", ".join(nonfinite_tensors[:10])
            suffix = " ..." if len(nonfinite_tensors) > 10 else ""
            raise FloatingPointError(
                "Non-finite gradients detected before the optimizer step"
                + (f" in: {names}{suffix}" if names else "")
            )

    def on_train_epoch_end(self) -> None:
        on_train_epoch_end_func(self)

    def on_validation_epoch_start(self) -> None:
        on_validation_epoch_start_func(self=self)

    def validation_step(self, data: DataHolder, i: int) -> torch.Tensor:
        loss = validation_step_func(self, data, i)
        return loss

    def on_validation_epoch_end(self):
        on_validation_epoch_end_func(self=self)

    def on_test_epoch_start(self):
        on_test_epoch_start_func(self=self)

    def test_step(self, data: DataHolder, i: int):
        test_step_func(self, data, i)

    def on_test_epoch_end(self) -> None:
        """Measure likelihood on a test set and compute stability metrics."""
        on_test_epoch_end_func(self=self)

    def forward(self, z_t: DataHolder) -> DataHolder:
        assert z_t.node_mask is not None
        model_input = z_t.copy()
        return self.model(model_input)

    def on_fit_start(self) -> None:
        self.train_iterations = 100
        if self.local_rank == 0:
            setup_wandb(self.cfg)

    @property
    def BS(self) -> int:
        return self.cfg.train.batch_size

    def configure_optimizers(self):
        train = self.cfg.train
        base_lr = float(train.lr)
        n_warmup = int(getattr(train, "lr_warmup_epochs", 0))
        use_cosine = bool(getattr(train, "lr_cosine_warm_restarts", False))
        constant_after = int(
            getattr(train, "lr_cosine_constant_at_min_after_epochs", 0) or 0
        )

        if use_cosine:
            lr_max = float(getattr(train, "lr_cosine_max", None) or base_lr)
            lr_min = float(getattr(train, "lr_cosine_min", 1e-6))
            period = int(getattr(train, "lr_cosine_period_epochs", 50))
            period_mult = int(
                getattr(train, "lr_cosine_subsequent_cycle_mult", None)
                or getattr(train, "lr_cosine_period_mult", 1)
            )
            log_scale = bool(getattr(train, "lr_cosine_log_scale", False))
            if period < 1:
                raise ValueError("lr_cosine_period_epochs must be >= 1.")
            if period_mult < 1:
                raise ValueError(
                    "lr_cosine_subsequent_cycle_mult must be >= 1."
                )
            if constant_after < 0:
                raise ValueError(
                    "lr_cosine_constant_at_min_after_epochs must be >= 0."
                )
            if log_scale and lr_min <= 0.0:
                raise ValueError(
                    "lr_cosine_log_scale=True requires lr_cosine_min > 0 "
                    f"(got {lr_min})."
                )
            # LinearLR warmup scales from lr_max / n_warmup up to lr_max.
            opt_lr = lr_max
        else:
            opt_lr = base_lr / n_warmup if n_warmup > 0 else base_lr
            if constant_after > 0:
                raise ValueError(
                    "lr_cosine_constant_at_min_after_epochs requires "
                    "lr_cosine_warm_restarts=True."
                )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_lr,
            amsgrad=True,
            weight_decay=train.weight_decay,
        )

        if use_cosine:
            cosine_cls = (
                LogCosineAnnealingWarmRestarts
                if log_scale
                else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            )
            cosine = cosine_cls(
                optimizer,
                T_0=period,
                T_mult=period_mult,
                eta_min=lr_min,
            )
            schedulers = []
            milestones = []
            if n_warmup > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0 / float(n_warmup),
                    end_factor=1.0,
                    total_iters=n_warmup,
                )
                schedulers.append(warmup)
                milestones.append(n_warmup)
            schedulers.append(cosine)
            if constant_after > 0:
                # Hold exactly at lr_min (relative to optimizer base_lrs = opt_lr).
                constant = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda _: lr_min / opt_lr,
                )
                milestones.append(n_warmup + constant_after)
                schedulers.append(constant)
            if len(schedulers) == 1:
                scheduler = schedulers[0]
            else:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=schedulers,
                    milestones=milestones,
                )
        elif n_warmup > 0:

            def lr_lambda(epoch: int) -> float:
                # Optimizer init lr = base_lr / n. Multiplier ramps 1 -> n over
                # epochs 0 .. n-1 so training uses lr/n on epoch 0 and lr from
                # epoch n onward.
                if n_warmup <= 1:
                    return float(n_warmup)
                if epoch >= n_warmup - 1:
                    return float(n_warmup)
                return 1.0 + (float(epoch) / float(n_warmup - 1)) * (
                    float(n_warmup) - 1.0
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
        else:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
