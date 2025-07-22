import torch
from torch.utils.data import Dataset

from utils import constants


class StreamingDataset(Dataset):

    def __init__(self, hf_ds, fake_length=1_000_000):
        self.hf_ds = hf_ds

        self.fake_length = self.fake_length

        self.iter = iter(self.hf_ds)
        self.seen_ids = set()

    
    def __len__(self):
        return self.fake_length
    

    def __getitem__(self, idx):
        print(f"[{constants.PROCESS_INDEX()}] Accessing index {idx} in StreamingDataset", flush=True)

        # if idx in self.seen_ids:
        #     raise ValueError(f"Index {idx} has already been seen. This dataset does not support repeated access to the same index.")

        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.hf_ds)
            return next(self.iter)