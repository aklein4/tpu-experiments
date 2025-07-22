

from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils


class LLMTrainer(BaseTrainer):

    def forward(self, input_ids):
        pad_token_id = self.config.pad_token_id
        v_size = self.config.vocab_size

        logits, _ = self.model(
            input_ids=input_ids,
        )

        loss = loss_utils.cross_entropy_loss(
            logits, input_ids, v_size, pad_token_id
        )
        aux = {
            'acc': loss_utils.accuracy(logits, input_ids, pad_token_id),
            'pcorr': loss_utils.pcorr(logits, input_ids, pad_token_id)
        }

        return loss, aux
    