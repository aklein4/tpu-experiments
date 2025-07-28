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
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp

import transformers

from torchprime.torch_xla_models.model import model_utils
from torchprime.torch_xla_models.utils.config_utils import config_vaidator

from data.datasets import get_dataset
from utils import constants
from utils.import_utils import import_class

transformers.utils.check_min_version("4.39.3")
logger = logging.getLogger(__name__)


def _mp_fn(index, config: omegaconf.DictConfig):

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Validate the config to avoid misuse and feature combination
    # Adding any new feature should update the config validator to
    # ensure different features can be combined together
    config_vaidator(config)

    mess = "\n ========= INFO ========= \n"
    mess += f"device_type: {xr.device_type()}\n"
    mess += f"process_index: {xr.process_index()}\n"
    mess += f"local_process_count: {xr.local_process_count()}\n"
    mess += f"local_device_count: {xr.local_device_count()}\n"
    mess += f"addressable_device_count: {xr.addressable_device_count()}\n"
    mess += f"glocal_device_count: {xr.global_device_count()}\n"
    mess += f"global_runtime_device_count: {xr.global_runtime_device_count()}\n"
    mess += f"world_size: {xr.world_size()}\n"
    mess += f"global_ordinal: {xr.global_ordinal()}\n"
    mess += f"local_ordinal: {xr.local_ordinal()}\n"
    mess += f"is_master_ordinal (local): {torch_xla.core.xla_model.is_master_ordinal(local=True)}\n"
    mess += f"is_master_ordinal (global): {torch_xla.core.xla_model.is_master_ordinal(local=False)}\n"
    mess += " ========================= "
    print(mess, flush=True)

    # Print the config for debugging
    if constants.PROCESS_IS_MAIN():
        print("\n ===== Configuration ===== \n", flush=True)
        print(omegaconf.OmegaConf.to_yaml(config), flush=True)
        print("\n ========================= \n", flush=True)

    # set up logging
    logger.setLevel(logging.INFO)
    if constants.PROCESS_IS_MAIN() or True:
        verbosity = logging.INFO 
    else:
        logging.disable(logging.CRITICAL)
        verbosity = logging.CRITICAL
    datasets.utils.logging.set_verbosity(verbosity)
    transformers.utils.logging.set_verbosity(verbosity)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # set training seeds
    transformers.set_seed(config.seed)
    torch_xla.manual_seed(config.seed)

    # Set the model dtype to bfloat16, and set the default device to the XLA device.
    # This will capture the model constructor into a graph so that we can add
    # sharding annotations to the weights later, and run the constructor on the XLA device.
    # assert config.torch_dtype == "bfloat16", "Currently only bfloat16 is supported"
    torch.set_default_dtype(torch.float32)
    model_dtype = getattr(torch, config.torch_dtype)
    with model_utils.set_default_dtype(model_dtype), torch_xla.device():
        model_cls = import_class(config.model.model_class, constants.MODEL_MODULE)
        model = model_cls(config.model)

    # print model information
    model_utils.log_parameter_breakdown(model, logger)
    logger.info(f"Model initialized: {config.model.model_class}")

    # sync the model to the XLA device
    logger.info("Syncing model to XLA device...")
    model = model.to(constants.XLA_DEVICE())
    if not config.debug:
        xm.broadcast_master_param(model)
    logger.info("Model synced to XLA device!")

    # Create the dataset
    data = get_dataset(**config.data.dataset)
    logger.info(f"Dataset loaded: {config.data.dataset.name}")

    # initialize the trainer
    trainer_cls = import_class(config.trainer.trainer_class, constants.TRAINER_MODULE)
    trainer = trainer_cls(
        model=model,
        config=config,
        train_dataset=data,
    )
    logger.info(f"Trainer initialized: {config.trainer.trainer_class}")

    # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
    with torch_xla._internal.jax_workarounds.jax_env_context():
        trainer.train_loop()

    return 0


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    xmp.spawn(_mp_fn, args=(config,))


if __name__ == "__main__":
    main()


