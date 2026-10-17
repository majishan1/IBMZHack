import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import the architecture defined above
from architecture import DisruptionLSTM

def generate_mock_historical_data(samples=5000, seq_len=10, features=5):
    """
    Generates synthetic telemetry data for the hackathon prototype.
    Person A should replace this with the actual MDSplus/HDF5 dataset loader.
    """
    # Random normalized sensor data
    X = np.random.randn(samples, seq_len, features).astype(np.float32)
    # Binary targets: 1 = disruption imminent, 0 = stable nominal plasma
    y = np.random.randint(0, 2, size=(samples, 1)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

def run_training_pipeline():
    print("[1] Loading historical Tokamak data...")
    X_train, y_train = generate_mock_historical_data()
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize the architecture, loss function, and optimizer
    model = DisruptionLSTM()
    criterion = nn.BCELoss()  # Binary Cross Entropy for probability outputs
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print("[2] Starting training loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"    Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save standard PyTorch weights (useful if you need to resume training later)
    torch.save(model.state_dict(), "checkpoints/lstm_weights.pth")
    print("\n[3] Standard weights saved to checkpoints/lstm_weights.pth")
    
    # ==============================================================================
    # EXPORT FOR IBM Z PRODUCTION INFERENCE
    # ==============================================================================
    print("[4] Compiling model to TorchScript for sub-millisecond execution...")
    model.eval()
    
    # Create a dummy tensor that exactly matches the expected live inference shape
    example_input = torch.randn(1, 10, 5)
    
    # Trace the execution graph and freeze it
    traced_model = torch.jit.trace(model, example_input)
    frozen_model = torch.jit.freeze(traced_model)
    
    # Save the compiled C++ representation
    frozen_model.save("checkpoints/compiled_lstm.pt")
    print("[5] Success: TorchScript model saved to checkpoints/compiled_lstm.pt")

if __name__ == "__main__":
    run_training_pipeline()