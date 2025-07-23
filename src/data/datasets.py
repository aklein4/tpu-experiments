
import datasets
import os

from torchprime.utils.retry import retry
from utils import constants


class RetryIterableDataset(datasets.IterableDataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        it = iter(self.dataset)
        try:
            yield retry(next(it))
        except StopIteration:
            raise StopIteration


def get_dataset(name: str, **kwargs) -> datasets.Dataset:
    """
    Get a dataset by name.
    
    Args:
        name (str): The name of the dataset to retrieve.
    
    Returns:
        datasets.Dataset: The requested dataset.
    """
    ds = datasets.load_dataset(name, **kwargs, token=constants.HF_TOKEN)

    if "streaming" in kwargs.keys() and kwargs["streaming"]:
        ds = ds.shard(num_shards=(constants.PROCESS_COUNT()+10), index=(constants.PROCESS_INDEX()+10))

    return ds

