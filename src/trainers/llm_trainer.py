import torch

from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils

class LLMTrainer(BaseTrainer):

    def forward(self, input_ids):
        pad_token_id = self.model.config.pad_token_id

        hidden_states = self.model(
            input_ids=input_ids,
            hidden_states_only=True
        )

        losses = loss_utils.fast_lm_loss(
            hidden_states=hidden_states,
            lm_head=self.model.lm_head,
            labels=input_ids,
            ignore_index=pad_token_id,
            shift=True
        )

        loss = losses["loss"]
        aux = {
            "acc": losses["acc"],
            "pcorr": losses["pcorr"]
        }

        return loss, aux
    