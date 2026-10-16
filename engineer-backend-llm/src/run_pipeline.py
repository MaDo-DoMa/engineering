import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DATA_DIR, ARTIFACTS_DIR, FEATURE_COLUMNS, set_seed
from feature_engineering import prepare_dataset
from data_preprocessing import split_time_series, scale_datasets, prepare_tensors
from models import LSTMPredictor, GRUPredictor
from train_rf import train_and_save_rf
from train_nn import train_pytorch_model


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE safely, avoiding division by zero errors."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def evaluate_model(y_true, y_pred, model_name):
    """Calculates and prints evaluation metrics required for the thesis."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"--- Results for {model_name} ---")
    print(f"MAE:  {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAPE: {mape:.2f}%\n")


def main():
    set_seed()

    # 1. Load data
    data_path = DATA_DIR / "sample_etf.csv"
    if not data_path.exists():
        print(f"File not found! Please place historical data in: {data_path}")
        print("Required columns: date, openPrice, highPrice, lowPrice, closePrice, volume")
        return

    print("1. Performing Feature Engineering...")
    df_raw = pd.read_csv(data_path)
    df_features = prepare_dataset(df_raw)

    print("2. Splitting datasets chronologically...")
    train_df, val_df, test_df = split_time_series(df_features)

    print("3. Scaling features (fitting strictly on training set)...")
    train_df, val_df, test_df = scale_datasets(
        train_df, val_df, test_df, FEATURE_COLUMNS
    )

    input_size = len(FEATURE_COLUMNS)

    print("4. Generating 3D Tensors (Sliding Windows)...")
    X_train, y_train = prepare_tensors(train_df, FEATURE_COLUMNS)
    X_val, y_val = prepare_tensors(val_df, FEATURE_COLUMNS)
    X_test, y_test = prepare_tensors(test_df, FEATURE_COLUMNS)

    print("\n5. Training Deep Learning models (PyTorch)...")
    lstm_model = LSTMPredictor(input_size=input_size)
    train_pytorch_model(lstm_model, X_train, y_train, X_val, y_val, "lstm_model", epochs=50)

    gru_model = GRUPredictor(input_size=input_size)
    train_pytorch_model(gru_model, X_train, y_train, X_val, y_val, "gru_model", epochs=50)

    print("\n6. Training Baseline model (Random Forest)...")
    X_train_rf = X_train.view(X_train.size(0), -1).numpy()
    y_train_rf = y_train.view(-1).numpy()
    X_test_rf = X_test.view(X_test.size(0), -1).numpy()

    train_and_save_rf(X_train_rf, y_train_rf)

    print("\n7. Final Evaluation on Test Set (Out-of-Sample)...")
    actuals = y_test.numpy().flatten()

    lstm_model.eval()
    gru_model.eval()
    with torch.no_grad():
        lstm_preds = lstm_model(X_test).numpy().flatten()
        gru_preds = gru_model(X_test).numpy().flatten()

    rf_model = joblib.load(ARTIFACTS_DIR / "rf_model.pkl")
    rf_preds = rf_model.predict(X_test_rf)

    evaluate_model(actuals, lstm_preds, "LSTM")
    evaluate_model(actuals, gru_preds, "GRU")
    evaluate_model(actuals, rf_preds, "Random Forest")


if __name__ == "__main__":
    main()