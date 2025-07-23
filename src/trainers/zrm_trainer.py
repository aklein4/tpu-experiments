

from trainers.base_trainer import BaseTrainer
from utils import loss as loss_utils


def kl_div(a, b):
    return (a - b).pow(2).sum(-1) / 2


class LLMTrainer(BaseTrainer):

    def forward(self, input_ids):

        out = self.model(
            input_ids=input_ids, labels=input_ids,
        )

        return loss, aux
    