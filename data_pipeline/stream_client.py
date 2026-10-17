import numpy as np
import MDSplus  # Requires: pip install mdsplus-3

class RemoteTokamakStreamer:
    """Streams data slice-by-slice from remote MDSplus server into RAM."""
    def __init__(self, server_address: str = "alcdata.psfc.mit.edu", shot_number: int = 1120106012):
        # Open remote connection without local file downloading
        self.conn = MDSplus.Connection(server_address)
        self.conn.openTree("cmod", shot_number)
        self.current_time_ms = 0

    def fetch_next_20ms_window(self) -> np.ndarray:
        """Queries 10 time steps (2ms sampling) across 5 core channels directly from remote RAM."""
        # TDI Expression to grab slice between t_start and t_end
        t_start = self.current_time_ms / 1000.0
        t_end = (self.current_time_ms + 20) / 1000.0
        
        # TDI query pulls signal slice remotely over TCP
        ip = self.conn.get(f'\\ip[{t_start}:{t_end}]').data()
        te = self.conn.get(f'\\te[{t_start}:{t_end}]').data()
        
        # Increment sequence time step
        self.current_time_ms += 20
        
        # Stack features into expected shape (1, sequence_length=10, features=5)
        raw_window = np.column_stack([ip[:10], te[:10], ...])
        return np.expand_dims(raw_window, axis=0)