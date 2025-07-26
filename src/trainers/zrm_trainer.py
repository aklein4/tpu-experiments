import torch

import numpy as np

from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils
from utils.torch_utils import scale_gradient


def kl_div(a, b):
    return (a - b).pow(2).sum(dim=-1) / 2


class ZRMTrainer(BaseTrainer):

    def forward(self, batch):
        pad_token_id = self.model.config.pad_token_id

        out = self.model(
            input_ids=batch['input_ids'],
            output_ids=batch['output_ids'],
        )

        logits, labels = out['lm_logits'], batch['output_ids']
        aux = {
            'lm_loss': loss_utils.cross_entropy_loss(logits, labels, pad_token_id, shifted=True),
            'acc': loss_utils.accuracy(logits, labels, pad_token_id, shifted=True),
            'pcorr': loss_utils.pcorr(logits, labels, pad_token_id, shifted=True),
        
            'alpha': out['alpha'],
        }

        aux['enc_kl_scle'] = np.clip(
            (self.step - self.config.trainer.enc_kl_start) / self.config.trainer.enc_kl_warmup,
            0.0, 1.0
        )
        # enc_mu = out['alpha'] * scale_gradient(
        #     out['encoder_mu_raw'], aux['enc_kl_scle']
        # )
        enc_mu = out['alpha'] * out['encoder_mu_raw'].detach()
        
        kl = kl_div(enc_mu, out['generator_mu'])
        og_kl = kl.sum().detach()

        kl_scaled = kl * kl.detach()
        kl_scaled = kl_scaled * og_kl / kl_scaled.sum().detach()
        
        aux["num_output_tokens"] = (labels != pad_token_id).sum().float()
        aux['kl_per_token'] = kl_scaled.sum() / (aux['num_output_tokens'] + 1)

        loss = aux['lm_loss'] + self.config.trainer.kl_weight * aux['kl_per_token']

        return loss, aux
    