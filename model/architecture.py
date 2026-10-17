import torch
import torch.nn as nn

class DisruptionLSTM(nn.Module):
    """
    TorchScript-compatible LSTM for predicting plasma disruption probabilities.
    Input expects a batch of sequences: (batch_size, sequence_length, features)
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(DisruptionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # batch_first=True aligns with our input tensor shape (1, 10, 5)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Classifier head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
        
        # Sigmoid squashes the output into a probability score [0.0, 1.0]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch automatically initializes hidden states to zero if not provided,
        # which is ideal for stateless TorchScript tracing.
        lstm_out, _ = self.lstm(x)
        
        # Extract the hidden state from the final time step of the sequence
        last_step_out = lstm_out[:, -1, :]
        
        out = self.fc1(last_step_out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out