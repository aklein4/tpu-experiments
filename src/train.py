"""Train script for LLMs using PyTorch/XLA with some torchax for lowering."""

import os
os.environ['PJRT_DEVICE'] = 'TPU'

import logging
import sys

import datasets
import hydra
import omegaconf
import torch
import torch_xla
import torch_xla.runtime as xr
import transformers

from torchprime.torch_xla_models.model import model_utils
from torchprime.torch_xla_models.utils.config_utils import config_vaidator

from data.datasets import get_dataset
from utils import constants
from utils.import_utils import import_class

transformers.utils.check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    mess = " ===== DEVICE INFO ===== \n"
    mess += f"XLA_DEVICE: {constants.XLA_DEVICE()}\n"
    mess += f"XLA_LOCAL_RANK: {constants.XLA_LOCAL_RANK()}\n"
    mess += f"XLA_RANK: {constants.XLA_RANK()}\n"
    mess += f"XLA_LOCAL_MAIN: {constants.XLA_LOCAL_MAIN()}\n"
    mess += f"XLA_MAIN: {constants.XLA_MAIN()}\n"
    mess += f"NUM_XLA_DEVICES: {constants.NUM_XLA_DEVICES()}\n"
    mess += f"NUM_PROCESSES: {xr.process_count()}\n"
    mess += f"PROCCESS_INDEX: {xr.process_index()}\n"
    mess += " ======================= "
    print(mess, flush=True)

    # Validate the config to avoid misuse and feature combination
    # Adding any new feature should update the config validator to
    # ensure different features can be combined together
    # config_vaidator(config)

    # Print the config for debugging
    print("\n ===== Configuration ===== \n", flush=True)
    print(omegaconf.OmegaConf.to_yaml(config), flush=True)
    print("\n ========================= \n", flush=True)

    # set up logging
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    print(f"Logging level set to: {log_level}", flush=True)

    # set training seeds
    transformers.set_seed(config.seed)
    torch_xla.manual_seed(config.seed)
    print(f"Set training seed to: {config.seed}", flush=True)

    # Set the model dtype to bfloat16, and set the default device to the XLA device.
    # This will capture the model constructor into a graph so that we can add
    # sharding annotations to the weights later, and run the constructor on the XLA device.
    assert config.torch_dtype == "bfloat16", "Currently only bfloat16 is supported"
    model_dtype = getattr(torch, config.torch_dtype)
    with model_utils.set_default_dtype(model_dtype), torch_xla.device():
        model_cls = import_class(config.model.model_class, constants.MODEL_MODULE)
        model = model_cls(config.model)
    print(f"Model class initialized: {config.model.model_class}", flush=True)

    # print model information
    model_utils.log_parameter_breakdown(model, logger, simple=True)
    print("Model parameter breakdown logged.", flush=True)

    # Create the dataset
    data = get_dataset(**config.data.dataset)
    print(f"Dataset loaded: {config.data.dataset.name}", flush=True)

    # initialize the trainer
    trainer_cls = import_class(config.trainer.trainer_class, constants.TRAINER_MODULE)
    trainer = trainer_cls(
        model=model,
        config=config,
        train_dataset=data,
    )
    print(f"Trainer initialized: {config.trainer.trainer_class}", flush=True)

    # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
    print("Starting training loop...", flush=True)
    with torch_xla._internal.jax_workarounds.jax_env_context():
        trainer.train_loop()

    return 0


if __name__ == "__main__":
    print("Starting training script...", flush=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    print("Running main function...", flush=True)
    sys.exit(main())
