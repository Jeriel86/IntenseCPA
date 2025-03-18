from scvi.train._trainrunner import TrainRunner as SCVITrainRunner
from typing import Optional, Union, List
import pytorch_lightning as pl
import torch
from scvi.model.base import BaseModelClass
from scvi.dataloaders import DataSplitter, SemiSupervisedDataSplitter
import jax

class CPATrainRunner(SCVITrainRunner):
    """Custom TrainRunner for CPA to support multi-GPU training."""
    def __init__(
        self,
        model: BaseModelClass,
        training_plan: pl.LightningModule,
        data_splitter: Union[SemiSupervisedDataSplitter, DataSplitter],
        max_epochs: int,
        use_gpu: Optional[Union[str, int, bool, List[int]]] = None,
        num_gpus: Optional[int] = None,
        **trainer_kwargs,
    ):
        # Determine accelerator, devices, and strategy based on inputs
        if num_gpus is not None and isinstance(num_gpus, int) and num_gpus > 1:
            accelerator = "gpu"
            devices = num_gpus
            strategy = "ddp"
        elif isinstance(use_gpu, list):
            accelerator = "gpu"
            devices = use_gpu
            strategy = "ddp"
        elif use_gpu is True or (isinstance(use_gpu, str) and "cuda" in use_gpu.lower()):
            accelerator = "gpu"
            devices = 1
            strategy = "auto"
        elif isinstance(use_gpu, int):
            accelerator = "gpu"
            devices = [use_gpu]
            strategy = "auto"
        else:
            accelerator = "cpu"
            devices = None
            strategy = "auto"

        # Update trainer_kwargs with our settings
        trainer_kwargs["accelerator"] = accelerator
        trainer_kwargs["devices"] = devices
        if strategy != "auto":
            trainer_kwargs["strategy"] = strategy

        # Initialize attributes required by SCVITrainRunner
        self.model = model
        self.training_plan = training_plan
        self.data_splitter = data_splitter

        # Set up the Trainer directly, avoiding parent class conflicts
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            **trainer_kwargs,
        )
        devices = trainer_kwargs.get("devices", 1)
        self.is_multi_gpu = (isinstance(devices, list) and len(devices) > 1) or \
                            (isinstance(devices, int) and devices > 1)

        # Set device attribute for compatibility (used in SCVITrainRunner.__call__)
        # For DDP, each process handles its own device, so this is a fallback
        self.device = torch.device("cuda:0") if accelerator == "gpu" and devices else torch.device("cpu")

    def __call__(self):
        if hasattr(self.data_splitter, "n_train"):
            self.training_plan.n_obs_training = self.data_splitter.n_train
        if hasattr(self.data_splitter, "n_val"):
            self.training_plan.n_obs_validation = self.data_splitter.n_val
        self.trainer.fit(self.training_plan, self.data_splitter)
        # Only move model to device if not using multiple GPUs
        if not self.is_multi_gpu:
            self.model.to_device(self.device)