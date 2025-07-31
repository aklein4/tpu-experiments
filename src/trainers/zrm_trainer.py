import torch
import torch.nn.functional as F

import numpy as np

from models.zrm import ZRMModel
from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils
from utils.torch_utils import scale_gradient


def kl_div(a, b):
    return (a - b).pow(2).sum(dim=-1) / 2


def get_w_kl(kl):
    og_total = kl.sum()

    w = kl.mean(0, keepdim=True)
    new_total = (kl * w).sum()

    w = w * og_total / new_total
    return w.detach()


def per_token(x, labels, pad_token_id):
    return x.sum() / ((labels != pad_token_id).float().sum() + 1)


def effective_parties(x):
    p = x / (x.sum() + 1e-5)
    parties = 1 / p.pow(2).sum()
    return parties / x.numel()


def cosine_schedule(
    step,
    wait_steps,
    warmup_steps,
):
    t = torch.clip(
        (step.float() - wait_steps) / warmup_steps,
        0.0, 1.0
    )
    return 0.5 * (1 - torch.cos(np.pi * t))


class ZRMTrainer(BaseTrainer):

    model: ZRMModel

    def forward(self, batch):
        pad_token_id = self.model.config.pad_token_id
        labels = batch['output_ids']

        if not hasattr(self, 'acc_step'):
            self.acc_step = torch.zeros_like(labels.view(-1).long()).sum()
        if not hasattr(self, 'activated'):
            self.activated = torch.zeros_like(self.acc_step.bool()).any()

        gen_grad_scale = cosine_schedule(
            self.acc_step, self.config.trainer.gen_grad_wait, self.config.trainer.gen_grad_warmup
        )
        noise_scale = cosine_schedule(
            self.acc_step, self.config.trainer.noise_wait, self.config.trainer.noise_warmup
        )

        out = self.model(
            input_ids=batch['input_ids'],
            output_ids=batch['output_ids'],
            gen_grad_scale=gen_grad_scale,
            noise_scale=noise_scale,
        )

        # handle LM
        lm_losses = loss_utils.fast_lm_loss(
            out['lm_logits'],
            labels,
            ignore_index=pad_token_id,
            shift_labels=False,
            shift_logits=False
        )
        self.activated = (
            self.activated | (lm_losses['acc'] >= self.config.trainer.acc_threshold).any()
        )
        self.acc_step += self.activated.long().sum()
        aux = {
            'lm_loss': lm_losses['loss'],
            'acc': lm_losses['acc'],
            'pcorr': lm_losses['pcorr'],
        
            'alpha': out['alpha'],
            'z_scale': out['z_scale'],

            'acc_step': self.acc_step,
            'activated': self.activated.long(),

            'gen_grad_scale': gen_grad_scale,
            'noise_scale': noise_scale,

            'frac_labelled': (labels != pad_token_id).float().mean(),
        }

        # get basic KL stuff
        kl = kl_div(
            out['encoder_mu'],
            out['generator_mu']
        )
        aux['kl_per_token'] = per_token(kl, labels, pad_token_id)
        aux['elbo'] = aux['lm_loss'] + aux['kl_per_token']
        aux['kl_parties'] = effective_parties(kl.mean(0))
        w_kl = get_w_kl(kl)

        # kl with respect to the encoder
        aux['enc_kl_scale'] = cosine_schedule(
            self.acc_step, self.config.trainer.enc_kl_wait, self.config.trainer.enc_kl_warmup
        )
        kl_enc = kl_div(
            out['alpha'].detach() * scale_gradient(out['encoder_mu_raw'], aux['enc_kl_scale']),
            out['generator_mu'].detach()
        ) * w_kl
        aux["enc_kl_per_token"] = per_token(kl_enc, labels, pad_token_id)

        # kl with respect to the generator
        kl_gen = kl_div(
            out['encoder_mu'].detach(),
            out['alpha'].detach() * out['generator_mu_raw']
        )
        aux["gen_kl_per_token"] = per_token(kl_gen, labels, pad_token_id)

        # kl with respect to alpha
        kl_alpha = kl_div(
            out['alpha'] * out['encoder_mu_raw'].detach(),
            out['alpha'] * out['generator_mu_raw'].detach()
        )
        aux["alpha_kl_per_token"] = per_token(kl_alpha, labels, pad_token_id)

        # kl with respect to the mean of the encoder mu
        kl_mean = kl_div(
            out['encoder_mu'],
            out['encoder_mu'].mean(dim=0, keepdim=True)
        )
        aux["mean_kl_per_token"] = per_token(kl_mean, labels, pad_token_id)
        aux["mean_kl_parties"] = effective_parties(kl_mean.mean(0))

        # uniformity loss
        # aux['uniformity_weight_scaled'] = self.config.trainer.uniformity_weight * (
        #     1 - np.clip(
        #         self.step / self.config.trainer.enc_kl_start,
        #         0.0, 1.0
        #     )
        # )
        # mu_norm = out['encoder_mu_raw'] / out['encoder_mu_raw'].norm(dim=-1, keepdim=True)
        # dists = torch.cdist(
        #     mu_norm.permute(1, 0),
        #     mu_norm.permute(1, 0),
        #     p=2
        # )
        # dists = torch.masked_fill(
        #     dists,
        #     dists < 1e-5,
        #     10.0
        # )
        # aux["uniformity_loss"] = torch.logsumexp(
        #     -(dists ** 2) * self.config.trainer.uniformity_temp,
        #     dim=-1
        # ).mean()

        # the loss
        kl_loss = (
            self.config.trainer.kl_weight * aux["enc_kl_per_token"] +
            self.config.trainer.kl_weight * aux["alpha_kl_per_token"] +
            aux["gen_kl_per_token"]
            # + aux['uniformity_weight_scaled'] * aux["uniformity_loss"]
        )
        loss = aux['lm_loss'] + kl_loss

        # check for NaNs
        aux["nan_loss"] = (~torch.isfinite(loss)).any().float()

        return loss, aux
    