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
  the artifacts are trained once per ticker (static weights, no walk-
  forward retraining) and every fold is evaluated with this single
  pack - consistent with the /predict endpoint. Retraining per fold
  would be too expensive for an interactive demo.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from predict_kalman import KalmanFilterModel, tune_kalman_params

# Hyperparameters of the saved NN checkpoints are inferred from the state
# dict at load time (see _infer_nn_architecture), so retraining with new
# Optuna-best params does not require touching this file.

# Kalman filter hyperparameters are NOT hardcoded any more: both Q and R
# are estimated from the training segment only (see tune_kalman_params) -
# R from the variance of training returns, Q by a grid search on the last
# 20% of the training data (same protocol as the RF/NN tuning).

ML_MODELS_SERVED: List[str] = ["LSTM", "GRU", "Random Forest", "Kalman Filter"]
BASELINE_MODELS_SERVED: List[str] = ["Zero Return", "Mean Return"]

# Artifact packs keyed by ticker ("" = legacy shared artifacts).
_packs: Dict[str, Dict] = {}


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


def _resolve_paths(ticker: Optional[str]) -> Dict[str, Path]:
    """
    Strict per-ticker artifact resolution: {ticker}_scaler.pkl,
    {ticker}_rf_model_optimized.pkl, ... The legacy shared-artifact
    fallback is DISABLED on purpose - serving wrong (another ticker's)
    weights silently would be worse than failing loudly.
    """
    if not ticker:
        raise RuntimeError(
            "Missing 'ticker' in the request - every evaluation must name "
            "the instrument whose artifacts should be served."
        )
    prefix = ticker.lower()
    paths = {
        "scaler": ARTIFACTS_DIR / f"{prefix}_scaler.pkl",
        "rf": ARTIFACTS_DIR / f"{prefix}_rf_model_optimized.pkl",
        "lstm": ARTIFACTS_DIR / f"{prefix}_lstm_model_optimized.pt",
        "gru": ARTIFACTS_DIR / f"{prefix}_gru_model_optimized.pt",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Missing per-ticker artifacts for {ticker}: {', '.join(missing)}. "
            "Retrain this instrument first (python run_pipeline.py "
            f"{ticker})."
        )
    return paths


def _load_pack(ticker: Optional[str] = None) -> Dict:
    """Loads serving artifacts for the given ticker (lazy per-ticker cache)."""
    cache_key = (ticker or "").lower()
    if cache_key in _packs:
        return _packs[cache_key]

    paths = _resolve_paths(ticker)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing model artifacts: {', '.join(missing)}")

    scaler = joblib.load(paths["scaler"])
    rf_model = joblib.load(paths["rf"])

    # Optional: standardized-target scaler from training (only present for
    # per-ticker packs retrained with target standardization). When missing,
    # NN outputs are assumed to be in raw log-return space (legacy artifacts).
    yscaler_path = ARTIFACTS_DIR / f"{cache_key}_yscaler.pkl"
    y_scaler = joblib.load(yscaler_path) if yscaler_path.exists() else None

    arch = _infer_nn_architecture(torch.load(paths["lstm"], map_location="cpu"))
    lstm_model = LSTMPredictor(**arch)
    lstm_model.load_state_dict(torch.load(paths["lstm"], map_location="cpu"))
    lstm_model.eval()

    arch = _infer_nn_architecture(torch.load(paths["gru"], map_location="cpu"))
    gru_model = GRUPredictor(**arch)
    gru_model.load_state_dict(torch.load(paths["gru"], map_location="cpu"))
    gru_model.eval()

    pack = {"scaler": scaler, "rf": rf_model, "lstm": lstm_model, "gru": gru_model,
            "yscaler": y_scaler,
            }
    _packs[cache_key] = pack
    return pack


def evaluate_history(df_raw: pd.DataFrame, n_folds: int = 3, test_days: int = 60,
                     ticker: Optional[str] = None) -> Dict:
    """
    Runs rolling validation over the provided price history and returns,
    for every fold, the CS arrays the frontend needs to compute metrics:
    dates, actual close, previous close and one price prediction array
    per model (LSTM / GRU / RF + two naive baselines).

    Also returns "next_day": the forward forecast for the NEXT trading day
    after the last available close (main-chart overlay + headline metric).

    Window semantics (both for folds and the forward forecast):
    a sliding window of SEQ_LENGTH feature rows predicts target_return of
    the row right AFTER the window, i.e. log(close[r+1]/close[r]) â€” so:
        prev_close = close[r], actual = close[r+1],
        predicted price = prev_close * exp(predicted_log_return).
    """
    pack = _load_pack(ticker)

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
        # Feature rows BEFORE test_start are already known at evaluation time,
        # so they are used as sliding-window context: every one of the 60 test
        # sessions is evaluated (nothing is lost to the SEQ_LENGTH warm-up).
        # No leakage: every window ends before its own anchor day.
        rows_predicted = test_days

        # Anchor r (df index) predicts the return close[r] -> close[r+1];
        # we evaluate r in [test_start-1, test_end-2] so all actual closes
        # fall inside the test window.
        row_idx = np.arange(test_start - 1, test_end - 1)  # df indices r

        # Only returns up to close[test_start-1] (target rows ..test_start-2)
        # are "past" relative to the first anchor - the baseline constant and
        # the Kalman fit must not see the first evaluated day's return.
        past_returns = df["target_return"].values[: test_start - 1]
        mean_train_return = float(past_returns.mean())

        # Sliding windows over ALL feature rows up to test_end; window i
        # covers rows [i, i+SEQ) and anchors r = i+SEQ, so we take windows
        # i in [test_start-1-SEQ, test_end-2-SEQ] -> exactly test_days windows.
        ctx_feats = pack["scaler"].transform(df.iloc[:test_end][FEATURE_COLUMNS])
        X_all, _ = create_sequences(
            ctx_feats, df["target_return"].values, SEQ_LENGTH
        )
        i0 = test_start - 1 - SEQ_LENGTH
        X_windows = X_all[i0: i0 + test_days]

        # ML predictions in RETURN space (raw log return, as trained).
        with torch.no_grad():
            lstm_returns = pack["lstm"](X_windows).numpy().flatten()
            gru_returns = pack["gru"](X_windows).numpy().flatten()
        rf_returns = pack["rf"].predict(X_windows.view(len(X_windows), -1).numpy())

        # The saved NNs were trained on a STANDARDIZED target when a
        # {ticker}_yscaler.pkl artifact exists - map their output back to
        # raw log-return space before any metric or price conversion.
        if pack.get("yscaler") is not None:
            lstm_returns = pack["yscaler"].inverse_transform(
                lstm_returns.reshape(-1, 1)
            ).ravel()
            gru_returns = pack["yscaler"].inverse_transform(
                gru_returns.reshape(-1, 1)
            ).ravel()

        # Kalman filter: hyperparameters estimated from the PAST only
        # (R = variance of past returns, Q = grid search on the last 20%
        # of the past segment), then the filter is fit on those same past
        # returns and predicts ONLINE over the test anchors - prediction i
        # is emitted BEFORE the filter sees the i-th actual return, so
        # there is no lookahead. Refitting per fold on the (expanding)
        # past keeps the state warm: by fold 2 and 3 the filter has
        # already digested every earlier test session's returns.
        kalman_q, kalman_r = tune_kalman_params(past_returns)
        kalman = KalmanFilterModel(
            process_var=kalman_q, measurement_var=kalman_r
        ).fit(past_returns)
        anchor_actual_returns = df["target_return"].values[test_start - 1: test_end - 1]
        kalman_returns = kalman.predict(anchor_actual_returns)
        print(f"[fold {fold + 1}] Kalman params (train-only): Q={kalman_q:g}, R={kalman_r:g}")

        # df rows involved in the predictions. Window j ends at row
        # test_start-1+j and predicts the return from close[r] to close[r+1].
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
            "Kalman Filter": to_prices(kalman_returns),
            # Baselines in price space (like the training pipeline):
            "Zero Return": prev_close.copy(),
            "Mean Return": prev_close * np.exp(mean_train_return),
        }

        fold_results.append({
            "fold": fold + 1,
            "kalman_params": {"Q": kalman_q, "R": kalman_r},
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
    fwd_feats = pack["scaler"].transform(fwd[FEATURE_COLUMNS])
    # The last window ends exactly at the real last session (its target is
    # the return to the dummy row - the value we do NOT use).
    X_fwd, _ = create_sequences(fwd_feats, np.zeros(len(fwd)), SEQ_LENGTH)
    fwd_window = X_fwd[-1:]  # window ending exactly at the real last session
    with torch.no_grad():
        lstm_ret = float(pack["lstm"](fwd_window).numpy().flatten()[0])
        gru_ret = float(pack["gru"](fwd_window).numpy().flatten()[0])
    rf_ret = float(pack["rf"].predict(fwd_window.view(len(fwd_window), -1).numpy())[0])
    # Kalman forecast for the next session: hyperparameters again estimated
    # from the observed history only; fit on ALL observed log returns and
    # take the posterior mean (no future measurement to update with).
    fwd_q, fwd_r = tune_kalman_params(df["target_return"].values)
    kalman_global = KalmanFilterModel(
        process_var=fwd_q, measurement_var=fwd_r
    ).fit(df["target_return"].values)
    kalman_ret = float(kalman_global.post_mean)
    last_close = float(close_raw[last_raw_idx])
    history_mean_return = float(df["target_return"].mean())

    next_day = {
        "date": str(pd.Timestamp(next_session).date()),
        "kalman_params": {"Q": fwd_q, "R": fwd_r},
        "prev_close": last_close,
        "predictions": {
            "LSTM": float(last_close * math.exp(lstm_ret)),
            "GRU": float(last_close * math.exp(gru_ret)),
            "Random Forest": float(last_close * math.exp(rf_ret)),
            "Kalman Filter": float(last_close * math.exp(kalman_ret)),
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
