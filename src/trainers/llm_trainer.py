

from trainers.base_trainer import BaseTrainer


class LLMTrainer(BaseTrainer):

    def forward(self, input_ids, mask):
        print(f" ==== {input_ids.shape} ==== ", flush=True)

        _logits, loss = self.model(
            input_ids=input_ids,
            labels=input_ids,
        )

        return loss, {}