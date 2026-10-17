from pathlib import Path
import torch
from typing import Tuple

# Keep CPU inference bounded on IBM Z LinuxONE and other CPU-only installations.
torch.set_num_threads(4)

# ==============================================================================
# 2. INFERENCE ENGINE CLASS
# ==============================================================================
class OptimizedInferenceEngine:
    """PyTorch inference wrapper for the 20ms tokamak telemetry loop.

    Uses the trained TorchScript model on CPU; this code does not enable or verify
    a Telum Integrated Accelerator runtime.
    """

    def __init__(self, model_path: str = "checkpoints/compiled_lstm.pt", input_shape: Tuple[int, int, int] = (1, 10, 5)):
        self.input_shape = input_shape
        self.device = torch.device("cpu")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = Path(__file__).resolve().parents[1] / resolved_model_path
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Trained TorchScript model not found: {resolved_model_path}. "
                "Train a model first with `python3 -m model.train --data <dataset.npz>`."
            )

        print(f"[Inference Engine] Loading trained model from {resolved_model_path}")
        self.model = torch.jit.load(str(resolved_model_path), map_location=self.device)
        self.model.eval()
        self.model = torch.jit.freeze(self.model)

        # Warm up the CPU model before processing the telemetry loop.
        with torch.inference_mode():
            self.model(torch.randn(*self.input_shape, device=self.device))
        print("[Inference Engine] Ready for CPU inference.")

    def predict_disruption_risk(self, normalized_sensor_tensor: torch.Tensor) -> float:
        """Executes zero-overhead forward pass.
        
        Args:
            normalized_sensor_tensor: Input tensor shaped (1, sequence_length, features)
            
        Returns:
            Disruption probability score between 0.0 and 1.0
        """
        # torch.inference_mode() is faster than torch.no_grad() by stripping all tracking metadata
        with torch.inference_mode():
            # Direct execution via TorchScript graph engine
            prediction = self.model(normalized_sensor_tensor)
            return prediction.item()

# ==============================================================================
# 3. SINGLETON INSTANTIATION FOR FASTAPI CONSUMPTION
# ==============================================================================
# Instantiate the engine exactly once when this module is imported by backend/main.py
# This ensures the model weights remain locked in memory for the continuous loop.
engine = OptimizedInferenceEngine()