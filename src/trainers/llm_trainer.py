

from trainers.base_trainer import BaseTrainer


class LLMTrainer(BaseTrainer):

    def forward(self, batch):

        _logits, loss = self.model(input_ids=batch["input_ids"])

        return loss, {}