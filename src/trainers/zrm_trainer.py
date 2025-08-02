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
            self.threshold_step = torch.zeros_like(labels.view(-1).long()).sum()
        if not hasattr(self, 'activated'):
            self.activated = torch.zeros_like(self.threshold_step.bool()).any()

        alpha = cosine_schedule(
            self.threshold_step, self.config.trainer.alpha_wait, self.config.trainer.alpha_warmup, up=False
        ) * np.sqrt(2 * self.config.trainer.alpha_scale / self.model.z_size)
        noise_scale = cosine_schedule(
            self.threshold_step, self.config.trainer.noise_wait, self.config.trainer.noise_warmup
        )

        out = self.model(
            input_ids=batch['input_ids'],
            output_ids=batch['output_ids'],
            alpha=alpha,
            noise_scale=noise_scale,
        )

        # handle LM
        lm_losses = loss_utils.fast_lm_loss(
            out['output_logits'],
            labels,
            ignore_index=pad_token_id,
            shift_labels=False,
            shift_logits=False
        )
        self.activated = (
            self.activated | (lm_losses['acc'] >= self.config.trainer.acc_threshold).any()
        )
        self.threshold_step += self.activated.long().sum()
        aux = {
            'lm_loss': lm_losses['loss'],
            'acc': lm_losses['acc'],
            'pcorr': lm_losses['pcorr'],
        
            'alpha': alpha,
            'noise_scale': noise_scale,
            'z_scale': out['z_scale'],

            'threshold_step': self.threshold_step,
            'activated': self.activated.long(),
        }

        # handle input logits
        input_losses = loss_utils.fast_lm_loss(
            out['input_logits'],
            batch['input_ids'],
            ignore_index=pad_token_id,
            shift_labels=True,
            shift_logits=False
        )
        aux['input_lm_loss'] = input_losses['loss']
        aux['input_acc'] = input_losses['acc']
        aux['input_pcorr'] = input_losses['pcorr']

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
        base_check = kl_div(out['encoder_mu_base'], out['generator_mu'])
        w_kl = get_w_kl(base_check)
        kl_base = kl_div(
            scale_gradient(out['encoder_mu_base'], self.config.trainer.kl_weight * w_kl),
            out['generator_mu']
        )
        aux['base_kl_per_token'] = per_token(
            kl_base, labels, pad_token_id
        )
        aux['base_kl_parties'] = effective_parties(kl_base.mean(0))
        
        # mean kls
        kl_base_mean = kl_div(
            out['encoder_mu_base'], out['encoder_mu_base'].mean(dim=0, keepdim=True)
        )
        aux["mean_base_kl_per_token"] = per_token(kl_base_mean, labels, pad_token_id)
        aux["mean_base_kl_parties"] = effective_parties(kl_base_mean.mean(0))
        
        kl_extra_mean = kl_div(
            out['encoder_mu_extra'] * alpha,
            out['encoder_mu_extra'].mean(dim=0, keepdim=True) * alpha
        )
        aux["mean_extra_kl_per_token"] = per_token(kl_extra_mean, labels, pad_token_id)
        aux["mean_extra_kl_parties"] = effective_parties(kl_extra_mean.mean(0))

        kl_true_mean = kl_div(
            out['encoder_mu'], out['encoder_mu'].mean(dim=0, keepdim=True)
        )
        aux["mean_true_kl_per_token"] = per_token(kl_true_mean, labels, pad_token_id)
        aux["mean_true_kl_parties"] = effective_parties(kl_true_mean.mean(0))

        # the loss
        loss = (
            aux['lm_loss'] +
            aux['input_lm_loss'] +
            aux['base_kl_per_token']
        )

        # check for NaNs
        aux["nan_loss"] = (~torch.isfinite(loss)).any().float()

        # count the number of tokens
        aux["atom_count"] = (
            (batch['input_ids'] != pad_token_id).long().sum() +
            (batch['output_ids'] != pad_token_id).long().sum()
        )

        return loss, aux
    