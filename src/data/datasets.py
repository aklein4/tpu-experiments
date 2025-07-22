
import datasets
import os

from utils import constants


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
        ds = ds.shard(num_shards=constants.PROCESS_COUNT(), index=constants.PROCESS_INDEX())

    return ds

