import pandas as pd
import numpy as np
import torch
import joblib
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config import DATA_DIR, ARTIFACTS_DIR, FEATURE_COLUMNS, set_seed
from feature_engineering import prepare_dataset
from data_preprocessing import scale_datasets, prepare_tensors
from tune_rf import tune_and_save_rf
from tune_nn import tune_and_save_nn

def directional_accuracy(y_true, y_pred):
    """Calculates Directional Accuracy in percentage."""
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

def evaluate_model(y_true, y_pred, model_name):
    """Calculates and prints the updated evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    da = directional_accuracy(y_true, y_pred)

    print(f"--- Results for {model_name} ---")
    print(f"MAE:  ${mae:.5f}")
    print(f"RMSE: ${rmse:.5f}")
    print(f"DA:   {da:.2f}%")
    return mae, rmse, da

def dm_test(actuals, pred1, pred2):
    """
    Diebold-Mariano test for squared errors (MSE).
    H0: Expected difference in loss between the two models is 0.
    """
    e1 = actuals - pred1
    e2 = actuals - pred2

    d = (e1**2) - (e2**2)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    if var_d == 0:
        return 0.0, 1.0

    stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * stats.norm.sf(abs(stat))
    return stat, p_value

def process_etf(ticker):
    """Processes the full rolling validation pipeline for a single ETF."""
    data_path = DATA_DIR / f"{ticker}.csv"
    if not data_path.exists():
        print(f"\n[WARNING] Data file missing for {ticker}! Expected path: {data_path}. Skipping.")
        return

    print("1. Performing Feature Engineering...")
    df_raw = pd.read_csv(data_path)
    df_features = prepare_dataset(df_raw)

    print(f"2. Starting Rolling Validation (TimeSeriesSplit) for {ticker}...")
    tscv = TimeSeriesSplit(n_splits=3)

    for fold, (train_val_idx, test_idx) in enumerate(tscv.split(df_features)):
        print(f"\n" + "-"*15 + f" {ticker} - FOLD {fold+1}/3 " + "-"*15)

        train_val_df = df_features.iloc[train_val_idx].copy()
        test_df = df_features.iloc[test_idx].copy()

        val_size = int(len(train_val_df) * 0.2)
        train_df = train_val_df.iloc[:-val_size].copy()
        val_df = train_val_df.iloc[-val_size:].copy()

        train_df, val_df, test_df = scale_datasets(
            train_df, val_df, test_df, FEATURE_COLUMNS
        )

        X_train, y_train = prepare_tensors(train_df, FEATURE_COLUMNS)
        X_val, y_val = prepare_tensors(val_df, FEATURE_COLUMNS)
        X_test, y_test = prepare_tensors(test_df, FEATURE_COLUMNS)

        print(f"\n--- Tuning Deep Learning models for {ticker} (Fold {fold+1}) ---")
        # Save models with ticker prefix to avoid overwriting
        lstm_model = tune_and_save_nn(f"LSTM_{ticker}", X_train, y_train, X_val, y_val, n_trials=5)
        gru_model = tune_and_save_nn(f"GRU_{ticker}", X_train, y_train, X_val, y_val, n_trials=5)

        print(f"\n--- Tuning Baseline model (Random Forest) for {ticker} (Fold {fold+1}) ---")
        X_train_rf = X_train.view(X_train.size(0), -1).numpy()
        y_train_rf = y_train.view(-1).numpy()
        X_val_rf = X_val.view(X_val.size(0), -1).numpy()
        y_val_rf = y_val.view(-1).numpy()
        X_test_rf = X_test.view(X_test.size(0), -1).numpy()

        rf_model = tune_and_save_rf(X_train_rf, y_train_rf, X_val_rf, y_val_rf, n_trials=10)

        print(f"\n--- Final Evaluation for {ticker} - Fold {fold+1} ---")
        actuals = y_test.numpy().flatten()

        lstm_model.eval()
        gru_model.eval()
        with torch.no_grad():
            lstm_preds = lstm_model(X_test).numpy().flatten()
            gru_preds = gru_model(X_test).numpy().flatten()

        rf_preds = rf_model.predict(X_test_rf)

        baseline_zero = np.zeros_like(actuals)
        baseline_mean = np.full_like(actuals, y_train_rf.mean())

        print("\nBaselines:")
        evaluate_model(actuals, baseline_zero, "Baseline (Zero Return)")
        evaluate_model(actuals, baseline_mean, "Baseline (Mean Return)")

        print("\nML Models:")
        evaluate_model(actuals, lstm_preds, "LSTM")
        evaluate_model(actuals, gru_preds, "GRU")
        evaluate_model(actuals, rf_preds, "Random Forest")

        print("\nDiebold-Mariano Tests (p-value < 0.05 indicates statistically significant difference):")
        _, p_val_rf_lstm = dm_test(actuals, rf_preds, lstm_preds)
        _, p_val_rf_gru = dm_test(actuals, rf_preds, gru_preds)
        print(f"RF vs LSTM (p-value): {p_val_rf_lstm:.5f}")
        print(f"RF vs GRU  (p-value): {p_val_rf_gru:.5f}")

def main():
    set_seed()

    # Target list of instruments for the experiment
    etfs = ["NASDAQ", "SPY500", "WIG20", "INVESCO", "GLD"]

    for ticker in etfs:
        print(f"\n" + "="*50)
        print(f"STARTING ANALYSIS FOR INSTRUMENT: {ticker}")
        print("="*50)
        process_etf(ticker)

if __name__ == "__main__":
    main()