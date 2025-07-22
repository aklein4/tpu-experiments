"""Sharding initialization module for TPU-based training using PyTorch/XLA SPMD APIs.

This file defines logic for setting up device mesh topology, determining minibatch support,
and applying weight/activation sharding annotations to the model. It ensures model layers like
`nn.Linear` are patched for SPMD compatibility and leverages user-provided configuration
(OmegaConf) to control sharding behavior.
"""

import torch.nn as nn
import torch_xla.distributed.spmd as xs
from omegaconf import DictConfig, OmegaConf
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear

from torchprime.sharding.shard_model import shard_torch_xla_model_from_config
from torchprime.topology import get_mesh, is_1d_sharding

from utils.logging_utils import log_master_print


def setup_sharding_and_mesh(
  model: nn.Module, config: DictConfig
) -> tuple[nn.Module, xs.ShardingSpec, bool]:
  """Sets up XLA mesh topology and applies SPMD sharding annotations to the model.

  This function:
    - Initializes the global device mesh based on `ici_mesh` in the config.
    - Determines whether minibatch dataloading can be used (only valid for 1D sharding).
    - Creates an input sharding spec indicating how input tensors should be partitioned.
    - Applies a patch to replace `nn.Linear` with einsum-backed versions to preserve dimension semantics.
    - Annotates model weights and intermediate activations with sharding specs.

  Args:
    model: The model to be sharded.
    config: Configuration object specifying mesh and sharding.

  Returns:
    A tuple containing:
      - The sharded model.
      - The input `ShardingSpec` for dataloader inputs.
      - A boolean indicating whether minibatch sharding is supported.
  """
  mesh = get_mesh(config)
  xs.set_global_mesh(mesh)
  log_master_print(f"Logical mesh shape: {mesh.shape()}")
  log_master_print(f"Logical mesh device assignments: {mesh.device_ids}")

  # TODO(https://github.com/pytorch/xla/issues/8696): Minibatch only works in 1D sharding.
  minibatch = is_1d_sharding(tuple(config.ici_mesh.values()))
  log_master_print(f"Minibatch dataloading: {minibatch}")

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/66): Test this for multislice
  input_sharding_spec = xs.ShardingSpec(
    mesh, (("data", "fsdp"), None), minibatch=minibatch
  )

  # Recursively replace `nn.Linear` layers with einsum operations in the model.
  # Without this patch, an `nn.Linear` module will flatten non-contracting dimensions
  # (e.g. batch and sequence), thus destroying the sharding constraints on those dimensions.
  model = apply_xla_patch_to_nn_linear(model)

  # Annotate model weights and activations with sharding constraints to distribute
  # the training across devices following the SPMD paradigm.
  sharding_config = OmegaConf.to_container(config.model.sharding, resolve=True)
  assert isinstance(sharding_config, dict)
  model = shard_torch_xla_model_from_config(model, config=sharding_config)

  return model, input_sharding_spec, minibatch
