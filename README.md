# IBMZHack: Tokamak Disruption Predictor

An LSTM-based research prototype for estimating tokamak disruption risk from
diagnostic signals. The dashboard can run in synthetic demo mode without a
trained model or tokamak connection.

## Quick start: synthetic dashboard

### Requirements

- Python 3.10 or newer
- pip
- A modern web browser

From the repository root, create and activate a virtual environment in
PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the following in that terminal and then
activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Open **two PowerShell terminals** in the repository root and activate `.venv`
in both. In the first terminal, start the backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In the second terminal, serve the dashboard:

```powershell
python -m http.server 3000 --directory frontend
```

Open [http://localhost:3000](http://localhost:3000). The dashboard connects to
the backend WebSocket at `ws://localhost:8000/ws/telemetry`. Stop each server
with **Ctrl+C** in its terminal.

On Linux or macOS, the same two commands work with `python3` instead of
`python`. The optional `start.sh` script starts both servers together in demo
mode:

```bash
./start.sh
```

It prints the dashboard URL when the servers start; press **Ctrl+C** to stop
them.

### Demo controls and safety

Demo mode is on by default (`TOKAMAK_DEMO_MODE=1`). It generates synthetic,
bounded sensor values and cycles through nominal, tearing, critical, and
recovery states about every 10 seconds at 1x speed. The dashboard provides
pause/resume, 0.5x–2x playback speed, and graphs for
plasma current, electron temperature, n=1 magnetic fluctuation, electron
density, and radiated power. Critical mitigation suggestions and recovery
events appear in the action log.

The demo is illustrative only: its risk scores are not model predictions.
Mitigation suggestions are not safety-qualified and must not be connected to
tokamak actuators.

## Data and model

Training requires labeled historical tokamak data; this repository does not
download or create a training dataset. Prepare a NumPy `.npz` file with:

- `X`: raw sensor windows shaped `(samples, 10, 5)`.
- `y`: binary disruption labels shaped `(samples,)` or `(samples, 1)`. A label
  is `1` when a disruption follows the corresponding window within the chosen
  forecast horizon, and `0` otherwise. The intended forecast objective is
  200–300 ms.

The five feature columns in `X` must be in this order, with the same physical
units used in `data_pipeline/normalizer.py`:

| Feature | Units | Normalization range |
| --- | --- | --- |
| Plasma current (`Ip`) | MA | 0–2.5 |
| Electron temperature (`Te`) | keV | 0.1–10 |
| n=1 magnetic fluctuation | T | -0.05–0.05 |
| Electron density (`Ne`) | 10^20 m^-3 | 0.1–5 |
| Radiated power (`Prad`) | MW | 0–8 |

The trainer applies min-max normalization, validates the input arrays, and
creates a deterministic stratified train/validation split. Both label classes
must have at least two samples.

Train from the repository root:

```powershell
python -m model.train --data C:\path\to\labeled_tokamak_windows.npz
```

The equivalent Linux/macOS command uses `python3`. Optional arguments are
`--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--num-threads`, and
`--output-dir`. By default, training writes `checkpoints/lstm_weights.pth` and
`checkpoints/compiled_lstm.pt`.

The model is a two-layer LSTM that consumes 10 time steps of five features and
returns a disruption-risk score from 0 to 1. In live inference, the mitigation
rules map that score to actions:

| Risk score | State | Suggested response |
| --- | --- | --- |
| Below 0.40 | Nominal | Maintain state |
| 0.40 to below 0.75 | Warning | Steer ECH gyrotrons to the Q2 surface |
| 0.75 to below 0.90 | Critical | Fast plasma-current rampdown; maximize ECH power |
| 0.90 or higher | Abort | Massive gas injection; fire shattered pellet |

## Live-data status

Live MDSplus/model mode is not ready for operation. The remote streamer in
`data_pipeline/stream_client.py` currently fetches only two channels and has a
placeholder where the remaining three features should be assembled. The live
path therefore cannot provide the complete five-feature model input. Do not
set `TOKAMAK_DEMO_MODE=0` expecting a working live prediction stream.

For a research deployment on IBM Z LinuxONE, use a CPU PyTorch build compatible
with the target `s390x` environment. This project does not require CUDA and
does not enable or verify a Telum Integrated Accelerator. Before any real
tokamak use, the data pipeline, model, thresholds, and responses require
independent physics and safety validation.
