import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..core import flatten_dict
from ..import_utils import is_bitsandbytes_available, is_torchvision_available


@dataclass
class DiffusionDPOConfig:
    """
    Configuration class for DDPOTrainer
    """

    # common parameters
    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    """the name of this experiment (by default is the file name without the extension name)"""
    run_name: Optional[str] = ""
    """Run name for wandb logging and checkpoint saving."""
    seed: int = 0
    """Seed value for random generations"""
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    """Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"""
    tracker_kwargs: dict = field(default_factory=dict)
    """Keyword arguments for the tracker (e.g. wandb_project)"""
    accelerator_kwargs: dict = field(default_factory=dict)
    """Keyword arguments for the accelerator"""
    project_kwargs: dict = field(default_factory=dict)
    """Keyword arguments for the accelerator project config (e.g. `logging_dir`)"""
    tracker_project_name: str = "trl"
    """Name of project to use for tracking"""
    logdir: str = "logs"
    """Top-level logging directory for checkpoint saving."""

    # hyperparameters
    save_freq: int = 500
    """Number of steps between saving model checkpoints."""
    num_checkpoint_limit: int = 5
    """Number of checkpoints to keep before overwriting old ones."""
    mixed_precision: str = "no"
    """Mixed precision training."""
    allow_tf32: bool = False
    """Allow tf32 on Ampere GPUs."""
    resume_from: Optional[str] = ""
    """Resume training from a checkpoint."""
    sample_num_steps: int = 50
    """Number of sampler inference steps."""
    sample_eta: float = 1.0
    """Eta parameter for the DDIM sampler."""
    sample_guidance_scale: float = 5.0
    """Classifier-free guidance weight."""
    sample_batch_size: int = 1
    """Batch size (per GPU!) to use for sampling."""
    num_training_steps: int = 2000
    """Number of batches to sample in training."""
    train_batch_size: int = 1
    """Batch size (per GPU!) to use for training."""
    train_use_8bit_adam: bool = False
    """Whether to use the 8bit Adam optimizer from bitsandbytes."""
    train_use_adafactor: bool = False
    """Whether to use the Adafactor for training (saves memory)"""
    train_learning_rate: float = 2.5e-7
    """Learning rate."""
    train_adam_beta1: float = 0.9
    """Adam beta1."""
    train_adam_beta2: float = 0.999
    """Adam beta2."""
    train_adam_weight_decay: float = 1e-2
    """Adam weight decay."""
    train_adam_epsilon: float = 1e-8
    """Adam epsilon."""
    train_gradient_accumulation_steps: int = 16  # 128 in orig paper
    """Number of gradient accumulation steps."""
    train_max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping."""
    train_cfg: bool = True
    """Whether or not to use classifier-free guidance during training."""
    train_timestep_fraction: float = 1.0
    """The fraction of timesteps to train on."""
    negative_prompts: Optional[str] = ""
    """Comma-separated list of prompts to use as negative examples."""

    proportion_empty_prompts: float = 0.2
    """The fraction of prompts to be null"""
    beta_dpo: float = 5000
    """DPO KL Divergence penalty"""
    dataset_name: str = "yuvalkirstain/pickapic_v2"
    """Name of HF dataset to use (only pickapics supported for now"""
    dataset_cache_dir: str = ""
    """Where HF dataset gets/is cached"""
    max_train_samples: int = 0
    """Maximum number of samples to train on (randomly sampled)"""
    dataloader_num_workers: int = 16
    """Number of dataloader workers"""
    resolution: int = 16
    """Image resolution"""
    random_crop: bool = False
    """Random vs center cropping"""
    no_hflip: bool = False
    """Hflip or not"""
    lr_scheduler: str = "constant_with_warmup"
    """Learning rate scheduler"""
    lr_warmup_steps: int = 500
    """Num LR warm-up steps"""

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.log_with not in ["wandb", "tensorboard"]:
            warnings.warn(
                ("Accelerator tracking only supports image logging if `log_with` is set to 'wandb' or 'tensorboard'.")
            )

        if self.log_with == "wandb" and not is_torchvision_available():
            warnings.warn("Wandb image logging requires torchvision to be installed")

        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
