import torch
from torch.utils.data import Dataset


class StreamingDataset(Dataset):

    def __init__(self, hf_ds, fake_length=None):
        self.hf_ds = hf_ds