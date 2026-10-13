import numpy as np
import pandas as pd
import torch
from typing import Tuple
from config import SEQ_LENGTH, TEST_SIZE_DAYS, TARGET_COL

def split_time_series(df: pd.DataFrame, val_size_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split into train, validation and test sets.
    - Test set: strictly out-of-sample data at the end (e.g., last 1 year).
    - Validation set: used for hyperparameter tuning / early stopping.
    - Training set: everything before validation.
    """
    test_start = len(df) - TEST_SIZE_DAYS
    val_start = test_start - val_size_days

    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    return train_df, val_df, test_df

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates 3D sliding windows required by LSTM and GRU.
    Converts 2D arrays (samples, features) to 3D tensors (sample, seq_length, features).
    """
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        window = X[i:(i + seq_length)]
        target = y[i + seq_length]

        xs.append(window)
        ys.append(target)

    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)

def prepare_tensors(df: pd.DataFrame, features: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper method to extract arrays and build sequences for a given dataframe."""
    X_arr = df[features].values
    y_arr = df[TARGET_COL].values
    return create_sequences(X_arr, y_arr, SEQ_LENGTH)
