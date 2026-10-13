import os
import random
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = DATA_DIR / "articles"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ========================================
# MODELS AND DATA CONFIGURATION
# ========================================
RANDOM_SEED = 42

SEQ_LENGTH = 20

TEST_SIZE_DAYS = 252

RAW_COLUMNS = ["date", "openPrice", "highPrice", "lowPrice", "closePrice", "volume"]

TARGET_COL = "target_return"

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False