import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from config import ARTIFACTS_DIR
from models import LSTMPredictor, GRUPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial, model_type, X_train, y_train, X_val, y_val):
    #Define hyperparameter search type
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    input_size = X_train.shape[2]

    #Initialize model dynamically
    if model_type.upper()=="LSTM":
        model = LSTMPredictor(input_size, hidden_size, num_layers).to(DEVICE)
    elif model_type.upper()=="GRU":
        model = GRUPredictor(input_size, hidden_size, num_layers).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    #DataLoaders from current batch_size
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #Training loop with internal early stopping
    epochs = 30
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improvement = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
            val_loss /= len(val_loader)

            #early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= patience:
                    break

            #Optuna pruning: stop unpromising trials early
            trial.report(best_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_val_loss

def tune_and_save_nn(model_type, X_train, y_train, X_val, y_val, n_trials=10):
    print(f"\n--- Starting Optuna Optimization for {model_type} ---")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_val, y_val), n_trials=n_trials)

    print(f"\nOptimization for {model_type} finished!")
    print(f"Best parameters: {study.best_params}")
    print(f"Best validation MSE: {study.best_value:.7f}")

    #retrain model on best parameters
    print(f"Training final optimized {model_type} model...")
    best_params = study.best_params

    input_size = X_train.shape[2]
    if model_type.upper()=="LSTM":
        best_model = LSTMPredictor(input_size, best_params['hidden_size'], best_params['num_layers']).to(DEVICE)
    else:
        best_model = GRUPredictor(input_size, best_params['hidden_size'], best_params['num_layers']).to(DEVICE)

    optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False)

    best_model.train()
    for epoch in range(25):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = best_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model_path = ARTIFACTS_DIR / f"{model_type.lower()}_model_optimized.pt"
    torch.save(best_model.state_dict(), model_path)
    print(f"Optimized {model_type} saved at {model_path}")

    return best_model
