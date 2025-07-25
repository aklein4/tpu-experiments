"""Base trainer module for TPU-based model training using PyTorch/XLA.

This script provides a `Trainer` class that sets up model sharding, activation checkpointing,
optimization, and the training loop with XLA-specific configurations. It is designed to work with
distributed TPU training and includes utilities for metrics logging and MFU computation.

Typical usage example:

    trainer = Trainer(model, config, train_dataset)
    trainer.train_loop(metrics_logger)
"""

import logging
import math
import os
from timeit import default_timer as timer
import shutil

import torch
import torch.nn.utils as nn_utils
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

from omegaconf import DictConfig, OmegaConf

from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
from transformers import (
    get_scheduler,
)
from transformers.optimization import Adafactor

from torchprime.torch_xla_models.model_rewriting.assume_pure import (
    mark_pure_modules,
)
from torchprime.torch_xla_models.model_rewriting.auto_trace import auto_trace
from torchprime.torch_xla_models.model_rewriting.rematerialization_utils import (
    add_activation_checkpointing_and_scan,
    add_optimization_barriers,
)
from torchprime.torch_xla_models.model_rewriting.sharding_initialization import (
    setup_sharding_and_mesh,
)
from torchprime.utils.parallelism_utils import lb_cp_enabled, reorder_sequence
from torchprime.utils.profiling import ensure_profile_end_step

import wandb
import huggingface_hub as hf

from models.xla import BaseXLAModel
from utils.import_utils import import_class
from utils import constants


logger = logging.getLogger(__name__)


def get_model_dtype(module: nn.Module) -> torch.dtype:
    dtypes = {param.dtype for param in module.parameters()}
    if len(dtypes) != 1:
        raise ValueError(f"Inconsistent dtypes found: {dtypes}")
    return dtypes.pop()


_ADAFACTOR = "adafactor"
_ADAMW = "adamw"


class BaseTrainer:
    """Trainer class for TPU-accelerated model training using PyTorch/XLA.

    This class encapsulates model preparation, optimizer configuration, data loading,
    and the training loop. It is designed to handle distributed training across TPU cores,
    enabling features like SPMD sharding, activation checkpointing, and profiling.

    Args:
        model: The model to train.
        config: Configuration object containing training hyperparameters and setup.
        train_dataset: Dataset used for training.
    """

    minibatch: bool

    def __init__(
        self,
        model: BaseXLAModel,
        config: DictConfig,
        train_dataset: Dataset | IterableDataset | None,
    ):
        self.config = config
        ensure_profile_end_step(config)
        self.device = xm.xla_device()
        self.global_batch_size = self.config.trainer.global_batch_size
        self.train_dataset = train_dataset

        # -- Model transformations --
        # Recursively replace `nn.Linear` layers with einsum operations in the model.
        # Without this patch, an `nn.Linear` module will flatten non-contracting dimensions
        # (e.g. batch and sequence), thus destroying the sharding constraints on those dimensions.
        model = apply_xla_patch_to_nn_linear(model)

        # Add `xp.Trace` to linear layers in the module tree (just for profiling?).
        model = auto_trace(model)

        # Setup SPMD mesh and shard the model.
        model, self.input_sharding_spec, self.minibatch = setup_sharding_and_mesh(
            model, config
        )
        model = mark_pure_modules(model, config)
        model = add_activation_checkpointing_and_scan(model, config)
        model = add_optimization_barriers(model, config)
        self.model = model

        # create optimizer and learning rate scheduler
        self.optimizer = type(self)._create_optimizer(config, model.parameters())
        self.lr_scheduler = get_scheduler(
            name=self.config.trainer.lr_scheduler.type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.trainer.lr_scheduler.warmup_steps,
            num_training_steps=self.config.trainer.lr_scheduler.training_steps,
        )

        # create the local data path
        if not self.config.debug and constants.PROCESS_IS_MAIN():
            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

            # create the huggingface save repo
            self.repo_name = f"{constants.HF_ID}/{self.config.project}_{self.config.name}"

            hf.create_repo(
                self.repo_name, private=True, exist_ok=True, token=constants.HF_TOKEN
            )

            # create the wandb project
            wandb.init(
                project=self.config.project,
                name=self.config.name,
                notes=self.config.notes,
            )

        # Execute all initialization work queued so far before starting training.
        torch_xla.sync()


    @staticmethod
    def _create_optimizer(config, model_parameters) -> torch.optim.Optimizer:
        """Helper for optimizer initialization."""
        if config.trainer.optimizer.type not in (_ADAFACTOR, _ADAMW):
            raise ValueError(
                f"Supported optimizers are {[_ADAFACTOR, _ADAMW]}, "
                f"but got {config.trainer.optimizer.type}"
            )

        if config.trainer.optimizer.type == _ADAMW:
            optimizer = torch.optim.AdamW(
                params=model_parameters,
                lr=config.trainer.optimizer.learning_rate,
                weight_decay=config.trainer.optimizer.weight_decay,
                betas=(
                    config.trainer.optimizer.beta1,
                    config.trainer.optimizer.beta2,
                ),
            )

        elif config.trainer.optimizer.type == _ADAFACTOR:
            # Adafactor optimizer does not support weight decay.
            if "weight_decay" in config.trainer.optimizer:
                raise ValueError("Adafactor does not support weight decay.")

            optimizer = Adafactor(
                params=model_parameters,
                lr=config.trainer.optimizer.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )

        else:
            raise AssertionError(f"Invalid optimizer type: {config.trainer.optimizer.type}")

        return optimizer


    def _get_train_dataloader(self) -> pl.MpDeviceLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        num_replicas = xr.process_count()
        logger.info("Num replicas: %d", num_replicas)

        # if self.minibatch:
        #     sampler = torch.utils.data.DistributedSampler(
        #         self.train_dataset,
        #         num_replicas=num_replicas,
        #         rank=xr.process_index(),
        #         shuffle=False,
        #         drop_last=True,
        #     )
        # else:
        #     # Without minibatch, every process loads the global batch the same way.
        #     sampler = torch.utils.data.DistributedSampler(
        #         self.train_dataset,
        #         num_replicas=1,
        #         rank=0,
        #         shuffle=False,
        #         drop_last=True,
        #     )

        assert self.global_batch_size is not None
        if self.minibatch:
            # Each process loads the per-host batch size.
            batch_size = self.global_batch_size // num_replicas
        else:
            # Each process will load the global batch, then discard the unneeded parts.
            batch_size = self.global_batch_size

        # handle the collator
        collator_cls = import_class(self.config.data.collator_class, constants.COLLATOR_MODULE)
        collator = collator_cls(**self.config.data.collator_kwargs)

        dataloader = DataLoader(
            self.train_dataset,
            collate_fn=collator,
            batch_size=batch_size,
            # sampler=sampler,
            shuffle=False,
            drop_last=True,
        )
        loader = pl.MpDeviceLoader(
            dataloader, self.device, input_sharding=self.input_sharding_spec
        )
        return loader
    

    def save_checkpoint(
        self,
        step: int,
    ):
        logger.info("[SAVING] Starting distributed checkpoint...")

        save_path = os.path.join(
            constants.LOCAL_DATA_PATH,
            "tmp_checkpoint",
        )

        self.model._maybe_save_checkpoint(save_path, convert_to_safetensors=False)
        logger.info(f"Saved checkpoint to {save_path} at step {step}")

        if constants.PROCESS_IS_MAIN(): 

            api = hf.HfApi()
            out_path = f"{step:012d}"

            api.upload_folder(
                repo_id=self.repo_name,
                folder_path=save_path,
                path_in_repo=out_path,
                repo_type="model",
                token=constants.HF_TOKEN,
            )
            logger.info(f"Uploaded checkpoint to {self.repo_name}/{out_path}")

        shutil.rmtree(save_path, ignore_errors=True)
        
        xm.rendezvous(f"checkpoint_saved")
        logger.info("[SAVING] Finished distributed checkpoint.")      
    

    def train_loop(self) -> None:
        self.model.train()
        self.model.zero_grad()

        # For now we assume that we will never train for more than one epoch
        max_step = self.config.trainer.max_steps
        train_loader = self._get_train_dataloader()
        steps_per_epoch = max_step
        train_iterator = iter(train_loader)

        logger.info("Starting training")
        logger.info("    Max step: %d", max_step)
        logger.info("    Global batch size: %d", self.global_batch_size)

        epoch = 0
        for step in range(max_step):
            try:
                batch = next(train_iterator)
            except StopIteration:
                logger.warning("DataLoader exhausted at step %d, reset iterator", step)
                epoch += 1
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            # when context parallel and load balance context parallel is enabled,
            # we will reorder the sequence here for each batch
            if lb_cp_enabled(self.config):
                return {
                    key: reorder_sequence(
                        tensor=value,
                        cp_size=self.config.ici_mesh.context,
                        seq_dim=1,
                        to_contiguous=False,
                    )
                    for key, value in batch.items()
                }

            trace_start_time = timer()
            loss = self.train_step(batch)
            aux = {}
            trace_end_time = timer()

            def step_closure(
                epoch, step, loss, aux, trace_start_time, trace_end_time, lr
            ):
                loss = loss.detach().item()

                logger.info(
                    "Epoch: %.4f, step: %d, loss: %.4f, lr: %.2e, trace time: %.2f ms",
                    step / steps_per_epoch,
                    step,
                    loss,
                    lr,
                    (trace_end_time - trace_start_time) * 1000,
                )

                to_wandb = {}
                for k, v in aux.items():
                    if isinstance(v, torch.Tensor):
                        to_wandb[k] = v.detach().item()
                    else:
                        to_wandb[k] = v
                to_wandb["loss"] = loss
                to_wandb["lr"] = lr
                to_wandb["epoch"] = epoch
                to_wandb["examples_seen"] = (step + 1) * self.global_batch_size

                if not self.config.debug and constants.PROCESS_IS_MAIN():
                    wandb.log(to_wandb)

                # if math.isnan(loss):
                #     raise ValueError(f"Loss is NaN at step {step}")
                
            xm.add_step_closure(
                step_closure,
                args=(
                    epoch,
                    step,
                    loss,
                    aux,
                    trace_start_time,
                    trace_end_time,
                    self.lr_scheduler.get_last_lr()[0],
                ),
                run_async=True,
            )
        
            # # Start profiler trace at the configured step
            # if step == self.config.profile_start_step:
            #     # Wait until device execution catches up to tracing before triggering the profile.
            #     # This will interrupt training slightly on the hosts which are capturing, but by waiting
            #     # after tracing for the step, the interruption will be minimal.
            #     xm.wait_device_ops()

            #     if os.path.exists(self.config.profile_dir):
            #         shutil.rmtree(self.config.profile_dir)
            #     os.makedirs(self.config.profile_dir)

            #     xp.start_trace(self.config.profile_dir)

            # # Stop profiler trace at the configured step
            # if step == self.config.profile_end_step:
            #     xm.wait_device_ops()
            #     xp.stop_trace()

            if (step+1) % self.config.trainer.checkpoint_interval == 0:    
                self.save_checkpoint(step+1)

        xm.wait_device_ops()
        logger.info("Finished training run")


    @torch_xla.compile(full_graph=True)
    def train_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        
        loss = self.forward(batch)
        
        loss.backward()
        
        # self.clip_gradients()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.zero_grad()

        return loss


    def forward(self, batch: dict):
        raise NotImplementedError(
            "The forward method should be implemented in the derived class."
        )


    def clip_gradients(self):
        """Clip gradients by the specified max norm and/or max absolute value."""
        max_grad_norm = self.config.trainer.max_grad_norm
        if max_grad_norm is None or max_grad_norm <= 0:
            grad_norm = nn_utils.get_total_norm(self.model.parameters(), norm_type=2)
        else:
            grad_norm = nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_grad_norm, norm_type=2
            )
        max_grad_value = self.config.trainer.max_grad_value
        if max_grad_value is not None and max_grad_value > 0:
            nn_utils.clip_grad_value_(self.model.parameters(), clip_value=max_grad_value)
        return grad_norm
