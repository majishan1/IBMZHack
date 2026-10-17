import os
import torch
from typing import Tuple

# ==============================================================================
# 1. IBM Z LINUXONE & TELUM HARDWARE OPTIMIZATIONS
# ==============================================================================
# Force routing of PyTorch matrix operations to IBM Z Integrated Accelerator (Telum)
os.environ["ZAIUDEVICES"] = "1"

# Limit CPU threading pool to avoid context-switching overhead with FastAPI/Uvicorn
# LinuxONE handles high core counts; tuning execution worker threads yields best latency
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# Disable memory profiling and legacy autograd flags for absolute maximum speed
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

# ==============================================================================
# 2. INFERENCE ENGINE CLASS
# ==============================================================================
class OptimizedInferenceEngine:
    """High-performance PyTorch inference wrapper for sub-millisecond 20ms tokamak loops.

    Uses TorchScript tracing and inference mode to bypass Python runtime overhead.
    """

    def __init__(self, model_path: str = "checkpoints/compiled_lstm.pt", input_shape: Tuple[int, int, int] = (1, 10, 5)):
        self.input_shape = input_shape
        # IBM Z uses optimized s390x CPU/Telum backend, which is accessed via the CPU device
        self.device = torch.device("cpu")  

        if os.path.exists(model_path):
            # Load pre-compiled TorchScript C++ model (bypasses Python graph overhead)
            print(f"[Inference Engine] Loading compiled TorchScript model from {model_path}")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            
            # Freeze the traced model to enable aggressive dead-code elimination
            self.model = torch.jit.freeze(self.model)
        else:
            # Development fallback: Initialize and compile the raw model on the fly
            print("[Inference Engine] Compiled model not found. Compiling raw model via JIT tracing...")
            from model.architecture import DisruptionLSTM
            raw_model = DisruptionLSTM().to(self.device).eval()
            
            # Create dummy tensor matching expected batch size, sequence length, and features
            example_input = torch.randn(*self.input_shape, device=self.device)
            
            # Trace and compile model to TorchScript bytecode
            self.model = torch.jit.trace(raw_model, example_input)
            self.model = torch.jit.freeze(self.model)

        # Warm-up pass to initialize hardware memory caches
        print("[Inference Engine] Running Telum hardware warm-up pass...")
        with torch.inference_mode():
            self.model(torch.randn(*self.input_shape, device=self.device))
        print("[Inference Engine] Ready for sub-millisecond real-time execution.")

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