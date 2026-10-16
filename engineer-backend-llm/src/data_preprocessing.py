import joblib
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler

from config import SEQ_LENGTH, TEST_SIZE_DAYS, TARGET_COL, ARTIFACTS_DIR


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

def scale_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fits StandardScaler strictly on training data to prevent data leakage,
    transforms all splits, and saves the scaler artifact.
    """
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Fit ONLY on train_df, then transform all sets
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # Save scaler for production inference in FastAPI
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

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
