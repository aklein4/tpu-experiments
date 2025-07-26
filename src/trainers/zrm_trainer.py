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

        out = self.model(
            input_ids=batch['input_ids'],
            output_ids=batch['output_ids'],
        )

        # handle LM
        logits, labels = out['lm_logits'], batch['output_ids']
        aux = {
            'lm_loss': loss_utils.cross_entropy_loss(logits, labels, pad_token_id, shifted=True),
            'acc': loss_utils.accuracy(logits, labels, pad_token_id, shifted=True),
            'pcorr': loss_utils.pcorr(logits, labels, pad_token_id, shifted=True),
        
            'alpha': out['alpha'],
        }

        # get basic KL stuff
        kl = kl_div(
            out['encoder_mu'],
            out['generator_mu']
        )
        aux['kl_per_token'] = per_token(kl, labels, pad_token_id)
        w_kl = get_w_kl(kl)

        # kl with respect to the encoder and alpha
        aux['enc_kl_scle'] = np.clip(
            (self.step - self.config.trainer.enc_kl_start) / self.config.trainer.enc_kl_warmup,
            0.0, 1.0
        )
        # enc_mu = out['alpha'] * scale_gradient(
        #     out['encoder_mu_raw'], aux['enc_kl_scle']
        # )
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

        kl_loss = (
            self.config.trainer.kl_weight * aux["enc_kl_per_token"] +
            aux["gen_kl_per_token"]
        )
        loss = aux['lm_loss'] + kl_loss

        return loss, aux
    