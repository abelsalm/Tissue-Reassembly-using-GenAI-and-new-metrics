import pytorch_lightning as pl
import torch
## HERE ADD NEW LOSS
from metrics.loss_function_plus import (
    MultiRadiusNeighborhoodLoss,
    MultiRadiusSlideCombinedLoss,
    SlidePointCloudMetricLoss,
)
from metrics.loss_function import LossFunction
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
        if cfg.train.loss_type == "neighborhood_multi_radius":
            # Multi-radius neighborhood loss combining (a) the
            # transcriptome RMSE evaluated at each radius in
            # ``multi_radius_radii`` and (b) the log-density difference at
            # each radius. ``density_weight`` rescales the density term
            # relative to the transcriptome term; see
            # ``MultiRadiusNeighborhoodLoss`` for the exact aggregation.
            self.train_loss = MultiRadiusNeighborhoodLoss(
                radii=list(cfg.train.multi_radius_radii),
                density_weight=getattr(cfg.train, "multi_radius_density_weight", 1.0),
                global_transcriptome_weight=getattr(
                    cfg.train, "multi_radius_global_transcriptome_weight", 0.0
                ),
                loss_radius_scale=getattr(
                    cfg.train, "multi_radius_loss_radius_scale", 512.0
                ),
                transcriptome_tolerance=getattr(
                    cfg.train, "multi_radius_transcriptome_tolerance", 0.05
                ),
                transcriptome_tolerance_gate_beta=getattr(
                    cfg.train, "multi_radius_transcriptome_tolerance_soft_beta", 256.0
                ),
                transcriptome_tolerance_warmup_epochs=int(
                    getattr(
                        cfg.train,
                        "multi_radius_transcriptome_tolerance_warmup_epochs",
                        100,
                    )
                ),
                soft_beta=getattr(cfg.train, "multi_radius_soft_beta", None),
                eps=getattr(cfg.train, "multi_radius_eps", 1e-6),
                include_self=getattr(cfg.train, "multi_radius_include_self", True),
            )
        elif cfg.train.loss_type == "neighborhood_multi_radius_slide":
            neighborhood = MultiRadiusNeighborhoodLoss(
                radii=list(cfg.train.multi_radius_radii),
                density_weight=getattr(cfg.train, "multi_radius_density_weight", 1.0),
                global_transcriptome_weight=getattr(
                    cfg.train, "multi_radius_global_transcriptome_weight", 0.0
                ),
                loss_radius_scale=getattr(
                    cfg.train, "multi_radius_loss_radius_scale", 512.0
                ),
                transcriptome_tolerance=getattr(
                    cfg.train, "multi_radius_transcriptome_tolerance", 0.05
                ),
                transcriptome_tolerance_gate_beta=getattr(
                    cfg.train, "multi_radius_transcriptome_tolerance_soft_beta", 256.0
                ),
                transcriptome_tolerance_warmup_epochs=int(
                    getattr(
                        cfg.train,
                        "multi_radius_transcriptome_tolerance_warmup_epochs",
                        100,
                    )
                ),
                soft_beta=getattr(cfg.train, "multi_radius_soft_beta", None),
                eps=getattr(cfg.train, "multi_radius_eps", 1e-6),
                include_self=getattr(cfg.train, "multi_radius_include_self", True),
            )
            slide = SlidePointCloudMetricLoss(
                ch_auc_weight=getattr(cfg.train, "slide_ch_auc_weight", 1.0),
                anisotropy_weight=getattr(
                    cfg.train, "slide_pca_anisotropy_weight", 1.0
                ),
                omnivariance_weight=getattr(
                    cfg.train, "slide_pca_omnivariance_weight", 1.0
                ),
                linearity_weight=getattr(
                    cfg.train, "slide_pca_linearity_weight", 1.0
                ),
                radii=list(cfg.train.slide_ch_radii),
                grid_resolution=int(
                    getattr(cfg.train, "slide_ch_grid_resolution", 512)
                ),
                kappa=getattr(cfg.train, "slide_ch_kappa", 0.5),
                soft_max_beta=getattr(cfg.train, "slide_ch_soft_max_beta", 32.0),
                support_factor=getattr(cfg.train, "slide_ch_support_factor", 8),
                landscape_chunk_size=getattr(
                    cfg.train, "slide_ch_landscape_chunk_size", 128
                ),
                square_bbox=getattr(cfg.train, "slide_ch_square_bbox", True),
                margin=getattr(cfg.train, "slide_ch_margin", 0.01),
                eps=getattr(cfg.train, "slide_ch_eps", 1e-6),
                min_cells=int(getattr(cfg.train, "slide_min_cells", 10)),
                cache_gt=getattr(cfg.train, "slide_ch_cache_gt", True),
            )
            self.train_loss = MultiRadiusSlideCombinedLoss(
                neighborhood=neighborhood,
                slide=slide,
                neighborhood_weight=getattr(
                    cfg.train, "slide_neighborhood_weight", 1.0
                ),
            )
        else:
            raise ValueError(f"Unsupported train loss_type: {cfg.train.loss_type}")
        self.val_loss = LossFunction()

        self.model = Model(
            input_dims=self.input_dims,
            n_layers=cfg.model.n_layers,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
        )

        self.noise_model = NoiseModel(cfg)

    def on_train_epoch_start(self) -> None:
        on_train_epoch_start_func(self)

    def training_step(self, data, i) -> torch.Tensor:
        loss = training_step_func(self, data, i)
        return loss

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
        base_lr = float(self.cfg.train.lr)
        n_warmup = int(getattr(self.cfg.train, "lr_warmup_epochs", 0))
        start_lr = base_lr / n_warmup if n_warmup > 0 else base_lr
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=start_lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )
        if n_warmup <= 0:
            return {"optimizer": optimizer}

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

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
