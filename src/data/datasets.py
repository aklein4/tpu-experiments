

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
    return datasets.load_dataset(name, **kwargs, token=constants.HF_TOKEN)


class Streaming