import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SingleSequenceCollator:

    def __init__(
        self,
        sequence_length: int,
        pad_token_id: int,
        **kwargs,
    ):
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id

    
    def __call__(
        self,
        batch,
    ):
        
        input_ids = []
        for x in batch:

            in_ids = torch.from_numpy(x["input_ids"].astype(np.int32)).long()
            out_ids = torch.from_numpy(x["output_ids"].astype(np.int32)).long()

            input_ids.append(torch.cat([in_ids, out_ids], dim=-1))
        
        # pad to length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,   
        )
        input_ids = input_ids[:, :self.sequence_length]
        input_ids = torch.cat(
            [
                input_ids,
                torch.full(
                    (input_ids.shape[0], self.sequence_length - input_ids.shape[1]),
                    self.pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device
                )
            ]
        )

        mask = input_ids != self.pad_token_id

        return {
            "input_ids": input_ids,
            "mask": mask
        }
    