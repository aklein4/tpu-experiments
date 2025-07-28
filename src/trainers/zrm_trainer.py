import torch

import numpy as np

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


class ZRMTrainer(BaseTrainer):

    def forward(self, batch):
        pad_token_id = self.model.config.pad_token_id
        labels = batch['output_ids']

        out = self.model(
            input_ids=batch['input_ids'],
            output_ids=batch['output_ids'],
        )

        # handle LM
        lm_losses = loss_utils.fast_lm_loss(
            out['lm_logits'],
            labels,
            ignore_index=pad_token_id,
            shift_labels=False,
            shift_logits=False
        )
        aux = {
            'lm_loss': lm_losses['loss'],
            'acc': lm_losses['acc'],
            'pcorr': lm_losses['pcorr'],
        
            'alpha': out['alpha'],
        }

        # get basic KL stuff
        kl = kl_div(
            out['encoder_mu'],
            out['generator_mu']
        )
        aux['kl_per_token'] = per_token(kl, labels, pad_token_id)
        aux['elbo'] = aux['lm_loss'] + aux['kl_per_token']
        w_kl = get_w_kl(kl)

        # kl with respect to the encoder and alpha
        aux['enc_kl_scale'] = np.clip(
            (self.step - self.config.trainer.enc_kl_start) / self.config.trainer.enc_kl_warmup,
            0.0, 1.0
        )
        # this will trigger a recompile, but that's fine because it's only once (we do it this way because of floating point precision issues)
        if self.step > self.config.trainer.enc_kl_start:
            enc_mu = out['alpha'] * scale_gradient(
                out['encoder_mu_raw'], aux['enc_kl_scale']
            )
        else:
            enc_mu = out['alpha'] * out['encoder_mu_raw'].detach()
        kl_enc = kl_div(
            enc_mu,
            out['generator_mu'].detach()
        ) * w_kl
        aux["enc_kl_per_token"] = per_token(kl_enc, labels, pad_token_id)

        # kl with respect to the generator
        kl_gen = kl_div(
            out['encoder_mu'].detach(),
            out['alpha'].detach() * out['generator_mu_raw']
        ) * w_kl
        aux["gen_kl_per_token"] = per_token(kl_gen, labels, pad_token_id)

        # extra kl stuff
        kl_mean = kl_div(
            out['encoder_mu'],
            out['encoder_mu'].mean(dim=0, keepdim=True)
        )
        aux["mean_kl_per_token"] = per_token(kl_mean, labels, pad_token_id)

        # the loss
        kl_loss = (
            self.config.trainer.kl_weight * aux["enc_kl_per_token"] +
            aux["gen_kl_per_token"]
        )
        loss = aux['lm_loss'] + kl_loss

        # check for NaNs
        aux["nan_loss"] = (~torch.isfinite(loss)).any().float()

        return loss, aux
    