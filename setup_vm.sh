#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Huggingface token
    $2: wandb token

Example:
    . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>
'

# # upgrade and update pip
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools

# install torch
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# install torch_xla for TPU VM
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp310-cp310-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

# update path(?)
export PATH="/home/$USER/.local/bin:$PATH"

# install extras
pip install transformers datasets matplotlib huggingface_hub wandb hydra-core omegaconf

# login to huggingface
# huggingface-cli login --token $1 --add-to-git-credential

# login to wandb
python -m wandb login $2
