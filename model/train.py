import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from data_pipeline.normalizer import SensorNormalizer
    from .architecture import DisruptionLSTM
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.normalizer import SensorNormalizer
    from model.architecture import DisruptionLSTM


SEQUENCE_LENGTH = 10
FEATURE_COUNT = 5


def load_historical_data(dataset_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load raw sensor windows and binary future-disruption labels from an NPZ file."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {dataset_path}")
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("Training data must be an .npz file containing arrays 'X' and 'y'.")

    with np.load(dataset_path, allow_pickle=False) as dataset:
        missing = {"X", "y"} - set(dataset.files)
        if missing:
            raise ValueError(f"Training dataset is missing array(s): {', '.join(sorted(missing))}")
        raw_features = np.asarray(dataset["X"], dtype=np.float32)
        labels = np.asarray(dataset["y"], dtype=np.float32)

    if raw_features.ndim != 3 or raw_features.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"'X' must have shape (samples, {SEQUENCE_LENGTH}, {FEATURE_COUNT}); "
            f"received {raw_features.shape}."
        )
    if labels.shape not in ((raw_features.shape[0],), (raw_features.shape[0], 1)):
        raise ValueError(
            f"'y' must have shape (samples,) or (samples, 1); received {labels.shape}."
        )
    if not np.isfinite(raw_features).all() or not np.isfinite(labels).all():
        raise ValueError("Training features and labels must contain only finite values.")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("Disruption labels must be binary: 0 (stable) or 1 (disruption).")

    labels = labels.reshape(-1, 1)
    if raw_features.shape[0] < 4 or np.unique(labels).size != 2:
        raise ValueError(
            "Training data must contain at least four samples and both label classes."
        )

    features = SensorNormalizer().process_raw_batch(raw_features)
    return features, torch.from_numpy(labels)


def split_dataset(
    features: torch.Tensor, labels: torch.Tensor, seed: int
) -> Tuple[TensorDataset, TensorDataset]:
    """Create a deterministic stratified split so both classes appear in each set."""
    rng = np.random.default_rng(seed)
    label_values = labels.numpy().reshape(-1)
    train_indices = []
    validation_indices = []

    for label in (0.0, 1.0):
        class_indices = np.flatnonzero(label_values == label)
        if class_indices.size < 2:
            raise ValueError(
                "Need at least two samples from each label class for a train/validation split."
            )
        class_indices = rng.permutation(class_indices)
        validation_count = min(
            class_indices.size - 1, max(1, round(class_indices.size * 0.2))
        )
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    train_indices = rng.permutation(train_indices)
    validation_indices = rng.permutation(validation_indices)
    return (
        TensorDataset(features[train_indices], labels[train_indices]),
        TensorDataset(features[validation_indices], labels[validation_indices]),
    )


def run_training_pipeline(
    data_path: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
    num_threads: int = 4,
) -> None:
    if epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("epochs, batch_size, and num_threads must be positive integers.")

    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[1] Loading labeled historical data from {data_path}...")
    features, labels = load_historical_data(data_path)
    train_data, validation_data = split_dataset(features, labels, seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    device = torch.device("cpu")
    model = DisruptionLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(
        f"[2] Training on {device} ({features.shape[0]} samples; "
        f"{len(train_data)} train / {len(validation_data)} validation; "
        f"{epochs} epochs)..."
    )
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.shape[0]

        model.eval()
        validation_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for batch_features, batch_labels in validation_loader:
                predictions = model(batch_features.to(device))
                validation_loss += criterion(
                    predictions, batch_labels.to(device)
                ).item() * batch_features.shape[0]
                correct += (
                    (predictions >= 0.5).cpu() == batch_labels
                ).sum().item()

        print(
            f"    Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {train_loss / len(train_data):.4f} | "
            f"validation loss: {validation_loss / len(validation_data):.4f} | "
            f"validation accuracy: {correct / len(validation_data):.3f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "lstm_weights.pth"
    compiled_path = output_dir / "compiled_lstm.pt"
    model.eval()
    torch.save(model.state_dict(), weights_path)
    traced_model = torch.jit.trace(model, features[:1].to(device))
    torch.jit.freeze(traced_model).save(str(compiled_path))
    print(f"[3] PyTorch weights saved to {weights_path}")
    print(f"[4] TorchScript inference model saved to {compiled_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the disruption LSTM on labeled historical sensor windows."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="NPZ file containing raw sensor array X and binary future-event labels y.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
        help="Directory for trained weights and the TorchScript model.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    run_training_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
