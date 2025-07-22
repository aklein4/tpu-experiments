import torch
import torch_xla
import torch_xla.runtime as xr
from torch_xla._internal.jax_workarounds import jax_env_context

from contextlib import contextmanager
import hydra
from omegaconf import DictConfig, OmegaConf

import transformers

from utils.logging_utils import log_print, log_master_print
from utils.import_utils import import_class
from utils import constants


def initialize_model_class(model_config):
    model_str = model_config.model_class
    
    model_class = import_class(model_str, constants.MODEL_MODULE)

    return model_class(model_config)


@contextmanager
def set_default_dtype(dtype):

    # Get the current default dtype
    previous_dtype = torch.get_default_dtype()

    # Set the new default dtype
    torch.set_default_dtype(dtype)
    
    try:
        yield
    finally:
        # Revert to the original default dtype
        torch.set_default_dtype(previous_dtype)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig):

    # print config for debugging
    log_master_print(OmegaConf.to_yaml(config))

    transformers.set_seed(config.seed)
    torch_xla.manual_seed(config.seed)

    # get dtype
    assert config.torch_dtype == "bfloat16", "Currently only bfloat16 is supported"
    model_dtype = getattr(torch, config.torch_dtype)

    # Set the model dtype to bfloat16, and set the default device to the XLA device.
    # This will capture the model constructor into a graph so that we can add
    # sharding annotations to the weights later, and run the constructor on the XLA device.
    with set_default_dtype(model_dtype), torch_xla.device():
        model = initialize_model_class(config.model)

    n_params = sum([p.numel() for p in model.parameters()])
    log_master_print(f"Training new model from scratch - Total size={n_params:_} params")

    # Downloading and loading a dataset from the hub.
    data = retry(
        lambda: make_huggingface_dataset(
            name=config.dataset_name,
            config_name=config.dataset_config_name,
            split="train",
            cache_dir=config.cache_dir,
            tokenizer=tokenizer,
            block_size=config.block_size,
        )
    )

    trainer_class = import_class(config.trainer_class, constants.TRAINER_MODULE)
    trainer = trainer_class(
        model=model,
        config=config,
        train_dataset=data,
    )

    # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
    with jax_env_context():
        trainer.train_loop()


if __name__ == "__main__":
    
    transformers.utils.check_min_version("4.39.3")

    xr.use_spmd()
    assert xr.is_spmd() is True

    main()
