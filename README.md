# IBMZHack Project 2026 - Tokamak Prediction Model

## Project overview

This prototype trains an LSTM to classify whether a disruption is expected from a
window of tokamak diagnostic signals. Training requires labeled historical data;
the repository does not download or create a real training dataset.

## Training on IBM Z LinuxONE

Use a Python/PyTorch installation built for the LinuxONE system architecture
(normally `s390x`). The training code explicitly uses PyTorch's CPU device and
does not require CUDA. Setting `ZAIUDEVICES=1` does not enable or verify a Telum
Integrated Accelerator, so this project does not set or report that variable.

Prepare a NumPy `.npz` file with:

- `X`: raw sensor windows shaped `(samples, 10, 5)`. The five feature columns
  must be in the same order and physical units used by
  `data_pipeline/normalizer.py` (plasma current, electron temperature, magnetic
  fluctuation, electron density, radiated power).
- `y`: binary labels shaped `(samples,)` or `(samples, 1)`. A label is `1` when
  the corresponding window is followed by a disruption in the prediction
  horizon, and `0` otherwise. Build that horizon consistently with the intended
  200-300 ms forecast objective.

The trainer applies the same min-max normalizer as live inference, rejects
malformed/non-finite data, and creates a deterministic stratified train and
validation split. Each class must have at least two labeled samples.

```bash
python3 -m model.train --data /path/to/labeled_tokamak_windows.npz
```

Optional arguments include `--epochs`, `--batch-size`, `--learning-rate`,
`--seed`, `--num-threads`, and `--output-dir`. The default output directory is
`checkpoints/`, which contains `lstm_weights.pth` and the TorchScript
`compiled_lstm.pt` used by inference.

To start the backend and dashboard, train first, then run `./start.sh`. If the
checkpoint is absent, set `TOKAMAK_TRAINING_DATA` to the `.npz` dataset path and
the startup script will train before launching the services.

This is a research prototype: validation metrics are not a substitute for
physics review, independent evaluation, or a safety-qualified control system.
The mitigation rules are suggestions and must not be connected directly to
tokamak actuators.