"""
Deterministic seed utilities (§2.3 code quality requirements).

Call set_seed() at the start of every script that touches randomness.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all RNG seeds for full reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed. Use the value from configs/train.yaml `training.seed`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # benchmark=True breaks determinism
