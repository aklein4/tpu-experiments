
import os

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    XLA_AVAILABLE = True
except ImportError:
    print("Warning: torch_xla not found", flush=True)
    XLA_AVAILABLE = False

# get the base path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)


XLA_DEVICE = lambda: xm.xla_device()

PROCESS_COUNT = lambda: xr.process_count()

PROCESS_INDEX = lambda: xr.process_index()
PROCESS_IS_MAIN = lambda: xm.is_master_ordinal(local=False)


# local data path
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# paths to config files
CONFIG_PATH = os.path.join(BASE_PATH, "configs")

# modules for classes
MODEL_MODULE = "models"
TRAINER_MODULE = "trainers"
COLLATOR_MODULE = "collators"

# huggingface login id
HF_ID = "aklein4"

# token for huggingface
HF_TOKEN = os.getenv("HF_TOKEN")
