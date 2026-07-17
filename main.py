import hydra
import random
import numpy as np
import torch
import os
import pathlib
import wandb
from omegaconf import DictConfig
from yaml import safe_load
import pytorch_lightning as pl
from utils.diffusion_model.setup.setup import (
    setup_callbacks,
    setup_dataset,
    setup_model,
    setup_trainer,
)

@hydra.main(version_base="1.3", config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    set_seed(cfg.general.seed)
    # Set output path for local saving
    cfg.general.local_saved_path = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    # Set up the dataset
    datamodule, dataset_infos = setup_dataset(cfg)

    # Run training or testing based on mode
    if cfg.general.mode == "train_and_test":
        train_model(cfg, datamodule, dataset_infos)
        if wandb.run is not None:
            wandb.finish()
        test_model(cfg, datamodule, dataset_infos)
    elif cfg.general.mode == "test_only":
        test_model(cfg, datamodule, dataset_infos)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(cfg: DictConfig, datamodule, dataset_infos):
    """Train the model from scratch."""
    model = setup_model(cfg, dataset_infos)
    callbacks = setup_callbacks(cfg, datamodule)
    trainer = setup_trainer(cfg, callbacks)

    trainer.fit(model, datamodule=datamodule)

    checkpoints_parent_dir = os.path.join(os.getcwd(), "checkpoints")
    cfg.test.checkpoints_parent_dir = checkpoints_parent_dir
    return checkpoints_parent_dir


def test_model(cfg: DictConfig, datamodule, dataset_infos):
    """Test the model using saved checkpoints."""
    # Supported patterns (checked in this order):
    # 1) test.checkpoint_paths (list of full paths, may span different
    #    training sessions / directories)
    # 2) test.checkpoints_parent_dir + test.checkpoints_name_list (directory-based)
    # 3) test.checkpoint_path (single-checkpoint)
    checkpoint_paths = get_checkpoint_paths(cfg)

    dataloaders_test = datamodule.test_dataloader()

    for checkpoint_path in checkpoint_paths:
        test_single_checkpoint(
            cfg, datamodule, dataset_infos, checkpoint_path, dataloaders_test
        )


def _is_set(value) -> bool:
    """True if a Hydra/OmegaConf value is present and non-empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except TypeError:
        return True


def get_checkpoint_paths(cfg: DictConfig):
    """Resolve the list of checkpoint paths to test.

    Priority:
      1) test.checkpoint_paths (optional list of full .ckpt paths)
      2) test.checkpoints_parent_dir + test.checkpoints_name_list
      3) test.checkpoint_path (single file)
    """
    # 1) Explicit list of full paths (may come from different training sessions).
    explicit_paths = getattr(cfg.test, "checkpoint_paths", None)
    if _is_set(explicit_paths):
        if isinstance(explicit_paths, str):
            explicit_paths = [explicit_paths]
        resolved = [str(pathlib.Path(p)) for p in explicit_paths]
        for p in resolved:
            if not pathlib.Path(p).exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
        return resolved

    # 2) Directory-based: checkpoints_parent_dir + checkpoints_name_list
    # 3) Single checkpoint: checkpoint_path
    parent_dir = getattr(cfg.test, "checkpoints_parent_dir", None)
    if not _is_set(parent_dir):
        single = getattr(cfg.test, "checkpoint_path", None)
        if _is_set(single):
            ckpt = pathlib.Path(single)
            cfg.test.checkpoints_parent_dir = str(ckpt.parent)
            cfg.test.checkpoints_name_list = [ckpt.name]
        else:
            raise ValueError(
                "For test_only you must set one of: "
                "`test.checkpoint_paths=[/full/path/a.ckpt, /full/path/b.ckpt]`, "
                "`test.checkpoint_path=/path/to/epoch=....ckpt`, "
                "or `test.checkpoints_parent_dir=/path/to/checkpoints` "
                "(with optional `test.checkpoints_name_list='all'`)."
            )

    checkpoints_parent_dir = pathlib.Path(cfg.test.checkpoints_parent_dir)
    print("Directory:", checkpoints_parent_dir)

    checkpoints_name_list = get_checkpoints_list(cfg, checkpoints_parent_dir)
    if not checkpoints_name_list:
        raise FileNotFoundError(
            f"No checkpoints found under {checkpoints_parent_dir}"
        )
    return [os.path.join(checkpoints_parent_dir, item) for item in checkpoints_name_list]


def get_checkpoints_list(cfg: DictConfig, checkpoints_parent_dir: pathlib.Path):
    """Get the list of checkpoints to test."""
    if not checkpoints_parent_dir.exists():
        return []
    if cfg.test.checkpoints_name_list == "all":
        checkpoints_name_list = os.listdir(checkpoints_parent_dir)
        # Keep deterministic order and only checkpoints.
        checkpoints_name_list = sorted([n for n in checkpoints_name_list if n.endswith(".ckpt")])
    else:
        checkpoints_name_list = cfg.test.checkpoints_name_list
    return checkpoints_name_list


def test_single_checkpoint(
    cfg: DictConfig, datamodule, dataset_infos, checkpoint_path, dataloader_test
):
    """Test the model using a single checkpoint."""
    cfg.test.checkpoint_path = checkpoint_path
    cfg.test.test_save_parent_path = os.path.join(cfg.test.save_dir, cfg.general.name)

    print("Testing checkpoint:", checkpoint_path)
    try:
        # Supports both styles:
        # - .../epoch=199.ckpt  -> epoch_index=199
        # - .../last_epoch.ckpt -> epoch_index=-1
        if os.path.basename(checkpoint_path) == "last_epoch.ckpt":
            cfg.test.epoch_index = -1
        else:
            cfg.test.epoch_index = int(checkpoint_path.split("=")[-1].split(".")[0])
    except ValueError:
        return
    print("Epoch index:", cfg.test.epoch_index)

    if cfg.general.mode == "test_only":
        load_model_config(cfg, checkpoint_path)

    model = setup_model(cfg, dataset_infos, checkpoint_path=checkpoint_path)
    callbacks = setup_callbacks(cfg, datamodule)
    trainer = setup_trainer(cfg, callbacks)

    trainer.test(model, ckpt_path=checkpoint_path, dataloaders=dataloader_test)


def load_model_config(cfg: DictConfig, checkpoint_path: str):
    """Load model configuration from a previous training session."""
    config_file = "/".join(checkpoint_path.split("/")[:-2])
    loading_model_cfg = safe_load(open(f"{config_file}/.hydra/config.yaml"))
    cfg["model"] = loading_model_cfg["model"]


if __name__ == "__main__":
    main()