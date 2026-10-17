import numpy as np
import torch
from typing import Union, List

# Standard physical bounds for normalization across core tokamak diagnostics
# [Min, Max] arrays corresponding to 5 channels:
# [0]: Plasma Current Ip (MA)
# [1]: Electron Temp Te (keV)
# [2]: Magnetic Fluctuations n=1 (Tesla)
# [3]: Electron Density Ne (10^20 m^-3)
# [4]: Radiated Power Prad (MW)
DEFAULT_FEATURE_MINS = np.array([0.0,  0.1, -0.05, 0.1, 0.0], dtype=np.float32)
DEFAULT_FEATURE_MAXS = np.array([2.5, 10.0,  0.05, 5.0, 8.0], dtype=np.float32)


class SensorNormalizer:
    """Fast, vectorised normaliser converting raw sensor windows into PyTorch tensors."""

    def __init__(
        self,
        feature_mins: np.ndarray = DEFAULT_FEATURE_MINS,
        feature_maxs: np.ndarray = DEFAULT_FEATURE_MAXS,
    ):
        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs
        self.feature_range = self.feature_maxs - self.feature_mins
        
        # Prevent division-by-zero on flat channels
        self.feature_range[self.feature_range == 0.0] = 1.0

    def process_raw_batch(
        self, raw_data: Union[np.ndarray, List]
    ) -> torch.Tensor:
        """Normalises raw input using Min-Max scaling to [0.0, 1.0] and returns a PyTorch Tensor.

        Args:
            raw_data: Array or list shaped (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: Normalized tensor shaped (batch_size, seq_len, num_features)
        """
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data, dtype=np.float32)
            
        # Min-Max Scaling: (X - X_min) / (X_max - X_min)
        normalized = (raw_data - self.feature_mins) / self.feature_range
        
        # Clip values to ensure strictly valid bounded inputs [0.0, 1.0]
        normalized = np.clip(normalized, 0.0, 1.0)

        # Convert zero-copy to FloatTensor on s390x CPU
        return torch.from_numpy(normalized.astype(np.float32))