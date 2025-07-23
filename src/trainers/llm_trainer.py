

from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils


class LLMTrainer(BaseTrainer):

    def forward(self, input_ids):
        pad_token_id = self.model.config.pad_token_id
        v_size = self.model.config.vocab_size

        logits, _ = self.model(
            input_ids=input_ids,
        )
        shift_logits, shift_labels = loss_utils.shift_tokens(logits, input_ids)

        loss = loss_utils.cross_entropy_loss(
            shift_logits, shift_labels,
            v_size, pad_token_id,
            shifted=True
        )
        aux = {
            'acc': loss_utils.accuracy(shift_logits, shift_labels, pad_token_id, shifted=True),
            'pcorr': loss_utils.pcorr(shift_logits, shift_labels, pad_token_id, shifted=True)
        }

        return loss, aux
    