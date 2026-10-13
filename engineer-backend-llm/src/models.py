import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int,  hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)

        last_time_step_out = out[:, -1, :]

        prediction = self.fc(last_time_step_out)
        return prediction

class GRUPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        last_time_step_out = out[:, -1, :]
        prediction = self.fc(last_time_step_out)
        return prediction
