from typing import Iterator

import torch
from torch.utils.data import Dataset

from utils import constants


class StreamingDataset(Dataset):

    iter: Iterator

    def __init__(self, hf_ds, fake_length=1_000_000):
        self.hf_ds = hf_ds
        self.fake_length = fake_length

        self.iter = self._init_iter()

    
    def __len__(self):
        return self.fake_length
    

    def _init_iter(self):
        self.iter = iter(self.hf_ds)

        for _ in range(constants.PROCESS_INDEX()+1):
            out = next(self.iter)

        return out


    def _next_iter(self):
        try:
            for _ in range(constants.PROCESS_COUNT()):
                out = next(self.iter)

        except StopIteration:
            out = self._init_iter()

        return out


    def __getitem__(self, idx):
        return self._next_iter()