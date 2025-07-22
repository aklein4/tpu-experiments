

import datasets
import os


def get_dataset(name: str, **kwargs) -> datasets.Dataset:
    """
    Get a dataset by name.
    
    Args:
        name (str): The name of the dataset to retrieve.
    
    Returns:
        datasets.Dataset: The requested dataset.
    """
    return datasets.load_dataset(name, **kwargs, token=os.environ['HF_TOKEN'])
