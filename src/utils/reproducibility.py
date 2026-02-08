"""Set global random seeds for reproducibility across numpy, tensorflow, and python."""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all frameworks."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
