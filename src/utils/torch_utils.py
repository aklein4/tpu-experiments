import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_gradient(x, scale):
    if isinstance(scale, (torch.Tensor, nn.Parameter)):
        scale = scale.detach()
    return x.detach() + (x - x.detach()) * scale


def unsqueeze_to_batch(x, target):
    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(x, target):
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.expand(
        *([target.shape[i] for i in range(num_unsqueeze)] + list(og_shape))
    )

    return x
