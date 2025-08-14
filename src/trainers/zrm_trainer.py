import torch
import torch.nn.functional as F

import numpy as np

from models.zrm import ZRMModel
from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils
from utils.torch_utils import scale_gradient


def kl_div(a, b):
    return (a - b).pow(2).sum(dim=-1) / 2


def per_token(x, labels, pad_token_id):
    return x.sum() / ((labels != pad_token_id).float().sum() + 1)


def get_w_kl(kl):
    og_total = kl.sum()

    w = kl.mean(0, keepdim=True)
    new_total = (kl * w).sum()

    w = w * og_total / new_total
    return w.detach()


def effective_parties(x):
    p = x / (x.sum() + 1e-5)
    parties = 1 / p.pow(2).sum()
    return parties / x.numel()


def cosine_schedule(
    step,
    wait_steps,
    warmup_steps,
    up=True
):
    t = torch.clip(
        (step.float() - wait_steps) / warmup_steps,
        0.0, 1.0
    )
    if up:
        return 0.5 * (1 - torch.cos(np.pi * t))
    return 0.5 * (1 + torch.cos(np.pi * t))


class ZRMTrainer(BaseTrainer):

    model: ZRMModel


    def forward(self, batch):
        pad_token_id = self.model.config.pad_token_id
        labels = batch['output_ids']

        if not hasattr(self, 'threshold_step'):
            self.threshold_step = torch.zeros_like(labels.view(-1).long()).sum() + self.config.trainer.init_threshold_step
        if not hasattr(self, 'activated'):
            self.activated = torch.zeros_like(self.threshold_step.bool()).any() | self.config.trainer.init_activated

        gen_grad_scale = cosine_schedule(
            self.threshold_step, self.config.trainer.gen_grad_wait, self.config.trainer.gen_grad_warmup, up=True
        )
        dec_grad_scale = {}

        out = self.model(
            input_ids=batch['input_ids'],
            output_ids=batch['output_ids'],
            gen_grad_scale=gen_grad_scale,
            dec_grad_scale=dec_grad_scale,
        )

        # handle LM
        lm_losses = loss_utils.fast_lm_loss(
            out['output_logits'],
            labels,
            ignore_index=pad_token_id,
            shift_labels=False,
            shift_logits=False,
            loss_threshold_lower=None,
            loss_threshold_upper=None
        )

        self.activated = (
            self.activated | (lm_losses['loss'] <= self.config.trainer.lm_loss_threshold_upper).any()
        )
        self.threshold_step += self.activated.long().sum()

        aux = {
            'lm_loss': lm_losses['loss'],
            'acc': lm_losses['acc'],
            'pcorr': lm_losses['pcorr'],
            'loss_threshold_perc': lm_losses['loss_threshold_perc'],
            'gen_grad_scale': gen_grad_scale,

            'threshold_step': self.threshold_step,
            'activated': self.activated.long(),
        }

        dec_grad_scale['value'] = torch.clip(
            (
                (lm_losses['loss'] - self.config.trainer.lm_loss_threshold_lower) /
                (self.config.trainer.lm_loss_threshold_upper - self.config.trainer.lm_loss_threshold_lower)
            ),
            0.0, 1.0
        ).detach()
        aux['dec_grad_scale'] = dec_grad_scale['value']

        # true kl
        kl_true = kl_div(
            out['encoder_mu'], out['generator_mu']
        )
        aux['true_kl_per_token'] = per_token(
            kl_true, labels, pad_token_id
        )
        aux['true_kl_parties'] = effective_parties(kl_true.mean(0))
        aux['elbo'] = aux['lm_loss'] + aux['true_kl_per_token']

        # base kl
        kl_base = kl_div(
            scale_gradient(out['encoder_mu'], gen_grad_scale),
            out['generator_mu']
        )
        w_kl = get_w_kl(kl_base)
        aux['base_kl_per_token'] = per_token(
            kl_base * w_kl, labels, pad_token_id
        )
        aux['base_kl_parties'] = effective_parties(kl_base.mean(0))
        
        # mean kls
        kl_base_mean = kl_div(
            out['encoder_mu'], out['encoder_mu'].mean(dim=0, keepdim=True)
        )
        aux["mean_base_kl_per_token"] = per_token(kl_base_mean, labels, pad_token_id)
        aux["mean_base_kl_parties"] = effective_parties(kl_base_mean.mean(0))
        
        # the loss
        loss = (
            aux['lm_loss'] +
            self.config.trainer.kl_weight * aux['base_kl_per_token']
        )

        # check for NaNs
        aux["nan_loss"] = (~torch.isfinite(loss)).any().float()

        # count the number of tokens
        aux["atom_count"] = (
            (batch['output_ids'] != pad_token_id).long().sum()
        )

        return loss, aux
    