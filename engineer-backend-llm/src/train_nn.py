import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from pathlib import Path
from config import ARTIFACTS_DIR

def train_pytorch_model(
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        model_name: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 15,
):
    """Trains PyTorch model with Early Stopping and saves the best weights"""
    #Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #early stopping
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"Starting training for {model_name}")

    for epoch in range(epochs):
        # ---Training phase ----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                prediction = model(X_batch)
                loss = criterion(prediction, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        # --- Early stopping check  ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train loss: {train_loss:.5f} | Validation loss: {val_loss:.5f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Validation loss: {best_val_loss:.5f}")
            break


    # Load best weights and save to disk
    model.load_state_dict(best_model_weights)
    save_path = ARTIFACTS_DIR / f"{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Best {model_name} model saved to {save_path}")

    return model
