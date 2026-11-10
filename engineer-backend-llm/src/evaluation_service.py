"""
Rolling-validation evaluation service for the FastAPI serving layer.

Reuses the exact feature engineering and model definitions from the
training pipeline (src/feature_engineering.py, src/models.py) and the
artifacts saved under data/articles (scaler.pkl, rf_model*.pkl,
lstm_model*.pt, gru_model*.pt).

Semantics (matching the training pipeline):
  - the target is the NEXT day's log return: target_return[r] = log(close[r+1]/close[r])
  - features of a sliding window predict target_return of the row right
    after the window, so a prediction for row r maps to:
        prev_close = close[r], actual = close[r+1], date = date[r+1]
  - model outputs are raw (unscaled) log returns, price conversion:
        predicted_price = prev_close * exp(predicted_log_return)

Known limitation (documented on purpose):
  the artifacts were trained once with a scaler fitted on the training
  part of the final fold. In production all folds are evaluated with this
  single production scaler - consistent with the /predict endpoint - and
  NO per-fold refit happens (retraining per fold would be too expensive
  for an interactive demo).
"""

import sys
from pathlib import Path
from typing import Dict, List

# The sibling pipeline modules (data_preprocessing, models, ...) import each
# other with top-level imports; make this package importable when running
# `main.py` from the repo root (same trick run_pipeline.py relies on).
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import math
import numpy as np
import pandas as pd
import torch

from config import ARTIFACTS_DIR, FEATURE_COLUMNS, SEQ_LENGTH
from data_preprocessing import create_sequences
from feature_engineering import prepare_dataset
from models import GRUPredictor, LSTMPredictor

# Hyperparameters of the saved NN checkpoints are inferred from the state
# dict at load time (see _infer_nn_architecture), so retraining with new
# Optuna-best params does not require touching this file.

ML_MODELS_SERVED: List[str] = ["LSTM", "GRU", "Random Forest"]
BASELINE_MODELS_SERVED: List[str] = ["Zero Return", "Mean Return"]

_pack: Dict = {}
_pack_loaded = False


def _infer_nn_architecture(state_dict: Dict) -> Dict[str, int]:
    """
    Infers hidden_size / num_layers from the saved state dict, so the serving
    layer does not need to duplicate the Optuna best-params for each model.
    Saved keys are prefixed with the module name (e.g. "lstm.weight_ih_l0");
    weight_ih_l0 has shape [num_directions * H, input] - 4*H for LSTM,
    3*H for GRU; every `_l<i>` suffix is one layer.
    """
    layer_keys = [k for k in state_dict if k.endswith("weight_ih_l0")]
    prefix = next(iter(layer_keys)).rsplit(".", 1)[0]  # "lstm" or "gru"
    multiplier = 4 if prefix == "lstm" else 3

    num_layers = sum(1 for k in state_dict if k.startswith(f"{prefix}.weight_ih_l"))
    weight_ih_l0 = state_dict[f"{prefix}.weight_ih_l0"]
    input_size = int(weight_ih_l0.shape[1])
    hidden_size = int(weight_ih_l0.shape[0]) // multiplier
    return {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers}


def _load_pack() -> None:
    """Loads all serving artifacts once (lazy singleton). Raises on failure."""
    global _pack, _pack_loaded

    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    rf_path = ARTIFACTS_DIR / "rf_model_optimized.pkl"
    if not rf_path.exists():
        rf_path = ARTIFACTS_DIR / "rf_model.pkl"
    lstm_path = ARTIFACTS_DIR / "lstm_model_optimized.pt"
    gru_path = ARTIFACTS_DIR / "gru_model_optimized.pt"

    missing = [str(p) for p in [scaler_path, rf_path, lstm_path, gru_path] if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing model artifacts: {', '.join(missing)}")

    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)

    arch = _infer_nn_architecture(torch.load(lstm_path, map_location="cpu"))
    lstm_model = LSTMPredictor(**arch)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    lstm_model.eval()

    arch = _infer_nn_architecture(torch.load(gru_path, map_location="cpu"))
    gru_model = GRUPredictor(**arch)
    gru_model.load_state_dict(torch.load(gru_path, map_location="cpu"))
    gru_model.eval()

    _pack = {"scaler": scaler, "rf": rf_model, "lstm": lstm_model, "gru": gru_model}
    _pack_loaded = True


def _infer_seq_windows(test_feats: np.ndarray, test_targets: np.ndarray) -> torch.Tensor:
    """Builds the sliding-window tensor for the test slice only (pipeline semantics)."""
    X_windows, _ = create_sequences(test_feats, test_targets, SEQ_LENGTH)
    return X_windows


def evaluate_history(df_raw: pd.DataFrame, n_folds: int = 3, test_days: int = 60) -> Dict:
    """
    Runs rolling validation over the provided price history and returns,
    for every fold, the CS arrays the frontend needs to compute metrics:
    dates, actual close, previous close and one price prediction array
    per model (LSTM / GRU / RF + two naive baselines).

    Also returns "next_day": the forward forecast for the NEXT trading day
    after the last available close (main-chart overlay + headline metric).

    Window semantics (both for folds and the forward forecast):
    a sliding window of SEQ_LENGTH feature rows predicts target_return of
    the row right AFTER the window, i.e. log(close[r+1]/close[r]) — so:
        prev_close = close[r], actual = close[r+1],
        predicted price = prev_close * exp(predicted_log_return).
    """
    global _pack_loaded
    if not _pack_loaded:
        _load_pack()

    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    df = prepare_dataset(df_raw.copy())  # adds log_return, indicators, target_return, dropna
    n_total = len(df)

    # prepare_dataset drops the indicator warm-up rows at the head and the
    # trailing row (NaN target after shift). Map df-row -> raw-frame-row:
    # df index i corresponds to raw index i + warmup.
    close_raw = df_raw["closePrice"].values
    dates_raw = df_raw["date"].values
    last_raw_idx = len(df_raw) - 1
    warmup = int(df_raw.index[df_raw["date"] == df["date"].iloc[0]][0])
    if warmup + n_total != last_raw_idx:
        raise ValueError("Unexpected data gaps: history may contain missing rows.")

    fold_results = []
    for fold in range(n_folds):
        # Windows split so that the LAST fold's predictions reach the very
        # last available close (test_end == n_total -> anchor row n_total-1).
        test_start = n_total - (n_folds - fold) * test_days
        test_end = test_start + test_days
        if test_start <= SEQ_LENGTH + 30:
            raise ValueError(
                "Not enough history data: need roughly "
                f"{(n_folds + 1) * test_days + SEQ_LENGTH + 60} rows, got {len(df_raw)}."
            )
        rows_predicted = test_days - SEQ_LENGTH
        if rows_predicted <= 0:
            raise ValueError(f"test_days must exceed seq_length ({SEQ_LENGTH}).")

        train_slice = df.iloc[:test_start]
        test_slice = df.iloc[test_start:test_end]
        mean_train_return = float(train_slice["target_return"].mean())

        # Production scaler (fitted once during training) for both splits.
        train_feats = _pack["scaler"].transform(train_slice[FEATURE_COLUMNS])
        test_feats = _pack["scaler"].transform(test_slice[FEATURE_COLUMNS])
        X_windows = _infer_seq_windows(test_feats, test_slice["target_return"].values)

        # ML predictions in RETURN space (raw log return, as trained).
        with torch.no_grad():
            lstm_returns = _pack["lstm"](X_windows).numpy().flatten()
            gru_returns = _pack["gru"](X_windows).numpy().flatten()
        rf_returns = _pack["rf"].predict(X_windows.view(len(X_windows), -1).numpy())

        # df rows involved in the predictions. Window j in the test slice
        # covers rows [test_start+j, test_start+j+SEQ) and predicts the
        # return from close[test_start+j+SEQ] to the NEXT session close.
        row_idx = np.arange(test_start + SEQ_LENGTH, test_end)  # df indices r
        prev_close = close_raw[warmup + row_idx]                # close[r]
        actual_close = close_raw[warmup + row_idx + 1]          # close[r+1]
        dates = dates_raw[warmup + row_idx + 1]                 # day of the actual close

        # Convert returns to PRICES, matching the frontend metric contract.
        def to_prices(pred_returns: np.ndarray) -> np.ndarray:
            return prev_close * np.exp(np.asarray(pred_returns, dtype=float))

        predictions = {
            "LSTM": to_prices(lstm_returns),
            "GRU": to_prices(gru_returns),
            "Random Forest": to_prices(rf_returns),
            # Baselines in price space (like the training pipeline):
            "Zero Return": prev_close.copy(),
            "Mean Return": prev_close * np.exp(mean_train_return),
        }

        fold_results.append({
            "fold": fold + 1,
            "dates": [str(pd.Timestamp(d).date()) for d in dates],
            "actual": actual_close.tolist(),
            "prev_close": prev_close.tolist(),
            "predictions": {model: preds.tolist() for model, preds in predictions.items()},
        })

    # ------------------------------------------------------------------
    # Forward forecast for the NEXT trading day after the last REAL close.
    # Trick: append one dummy calendar-day row whose close equals the last
    # real close. prepare_dataset keeps the real last row (its target is
    # then ~0, non-NaN), so its window predicts the return to the dummy
    # day - which is exactly the next-session return. The dummy's own NaN
    # target is dropped, keeping feature windows aligned with training.
    next_session = pd.bdate_range(
        start=pd.Timestamp(df_raw["date"].iloc[-1]) + pd.Timedelta(days=1), periods=1
    )[0]
    forward_raw = pd.concat([
        df_raw,
        pd.DataFrame({
            "date": [next_session],
            "closePrice": [float(df_raw["closePrice"].iloc[-1])],
            "volume": [int(df_raw["volume"].iloc[-1])],
        }),
    ], ignore_index=True)
    fwd = prepare_dataset(forward_raw)
    fwd_feats = _pack["scaler"].transform(fwd[FEATURE_COLUMNS])
    # The last window ends exactly at the real last session (its target is
    # the return to the dummy row - the value we do NOT use).
    X_fwd, _ = create_sequences(fwd_feats, np.zeros(len(fwd)), SEQ_LENGTH)
    fwd_window = X_fwd[-1:]  # window ending exactly at the real last session
    with torch.no_grad():
        lstm_ret = float(_pack["lstm"](fwd_window).numpy().flatten()[0])
        gru_ret = float(_pack["gru"](fwd_window).numpy().flatten()[0])
    rf_ret = float(_pack["rf"].predict(fwd_window.view(len(fwd_window), -1).numpy())[0])
    last_close = float(close_raw[last_raw_idx])
    history_mean_return = float(df["target_return"].mean())

    next_day = {
        "date": str(pd.Timestamp(next_session).date()),
        "prev_close": last_close,
        "predictions": {
            "LSTM": float(last_close * math.exp(lstm_ret)),
            "GRU": float(last_close * math.exp(gru_ret)),
            "Random Forest": float(last_close * math.exp(rf_ret)),
            "Zero Return": last_close,
            "Mean Return": float(last_close * math.exp(history_mean_return)),
        },
    }

    return {
        "seq_length": SEQ_LENGTH,
        "n_folds": n_folds,
        "test_days": test_days,
        "folds": fold_results,
        "next_day": next_day,
    }
