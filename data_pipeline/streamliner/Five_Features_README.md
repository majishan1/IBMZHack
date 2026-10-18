# Five MAST measurements for PyTorch

Open `Five_Features_Start_Here.ipynb` in the IBM server's JupyterLab. Keep all
five files in this bundle in one folder. Run cells from top to bottom with
Shift + Enter, using an IBM-provided PyTorch environment.

The Python modules and every notebook code cell now have beginner-focused
comments explaining the data flow, shapes, units, timing, missing values,
normalization, matrix calculations, and actual model-weight updates.

## Why the previous files were included

The previous bundle reused `MemoryChunkCache` from `tokamak_stream.py` and
`official_splits` from `tokamak_training_data.py`. Their reusable functions were
needed, but importing the complete old examples made the folder confusing.

Those helpers now live in **`fusion_data_utils.py`**. The current five-file
bundle is self-contained and leaves both previous example files out. The
five-feature reader and notebook import the new utility module directly.
Start with the notebook; read `mast_five_features.py` for the measurement
pipeline and `fusion_data_utils.py` for RAM caching and shot assignments.

You can commit these small code files and pull the repository on LinuxONE.
Measurements are fetched from the remote archive into the machine executing
Python's RAM. This code does not write measurement files or a disk cache.

## Why the source changed

All five requested quantities are not in the curated 39-signal TokaMark subset.
This extension explicitly uses the wider **UKAEA FAIR MAST level2 archive**,
with all five quantities coming from the same MAST shot. No C-Mod connection,
credentials, or mixed-experiment rows are used. It is not the official TokaMark
five-channel benchmark or a reproduction of its fourteen tasks.

| Input order | Meaning | Source | Output unit |
|---|---|---|---|
| 0 | Plasma current | `magnetics/ip` | MA |
| 1 | Core electron temperature | `thomson_scattering/t_e_core` | keV |
| 2 | Estimated n=1 saddle-field amplitude | `magnetics/b_field_tor_probe_saddle_field` and midplane geometry | T |
| 3 | Line-average electron density | `summary/line_average_n_e` | 10^20 m^-3 |
| 4 | Total radiated power | `bolometer/power_radiated_total` | MW |

The scalar temperature and density choices do not retain every spatial profile
position. They select core temperature and line-average density explicitly.

## What the magnetic input means

There is no direct precomputed n=1 channel in the selected release. At each
magnetic timestamp, fit B(phi) = offset + a*cos(phi) + b*sin(phi) across the
available calibrated midplane saddle fields, and return sqrt(a^2+b^2).
Match coil labels to archived geometry, use the centers of the toroidal
extents, and require at least six finite coils with a well-conditioned fit.
The output is nonnegative and has the field's source unit, tesla.

This is a **derived coil-averaged n=1 harmonic estimate**. It includes external
field pickup, has no vacuum-field subtraction, does not resolve a frequency
band, and is not the n1rms feature or a validated plasma-only instability
diagnostic. Higher harmonics, missing coils, coil aperture, and calibration
affect the estimate. The geometry metadata has development status. Review this
definition with a fusion specialist before scientific claims or comparing it
with another machine's n=1 diagnostic. If your model requires a specific
facility-approved n=1 signal, this estimate must not be silently substituted.

## Timing and missing data

Each raw batch is `(1,10,5)`: ten steps at 2 ms spacing over a nominal 20 ms
window. Features use only finite readings at or before each requested step.
Carry-forward limits are 2 ms (Ip), 20 ms (Te), 0.2 ms (magnetic estimate),
3 ms (Ne), and 2 ms (Prad). These configurable limits are preprocessing choices,
not measured hardware rates. Inspect `window.sample_age_s`; repeating a slow
measurement does not create new observations or higher temporal resolution.

Unavailable diagnostics, stale measurements, unrecognized geometry, wrong
units, or insufficient coils fail explicitly. The training iterator skips
incomplete windows and can explicitly skip unavailable shots with warnings.
It never inserts fake values or silently switches experiments. Reading a
window fetches intersecting compressed chunks, potentially larger than that
window; it does not guarantee 20 ms network latency. Timestamp arrays, geometry,
compressed cache, decoded chunks, arrays, and tensors all consume RAM.

## Normalization and training

`SensorNormalizer` validates exactly five features and returns native float32
CPU PyTorch tensors. It takes explicit min/max arrays. The example constants
mirror the original proposal and are not universal physical limits. Clipping
defaults off; `clip=True` explicitly bounds results. Training fits bounds on
training data only, then reuses them for validation/test/inference. A constant
training feature receives a positive scale, not a divide-by-zero.

The notebook trains a small MLP accepting `[batch,10,5]` to forecast an actual
later current reading. A nominal 10 ms target uses the latest recorded current
at or before the requested target time, within the current age limit, and must
occur after the last input. This is not a disruption classifier; that requires
a defined event and verified labels. Evaluation includes a persistence baseline.

The notebook uses pinned official TokaMark shot assignments for disjoint
train/validation/test membership, while reading the wider FAIR MAST diagnostics.
This remains a separate task and archive version, not an official benchmark
result. The explicitly capped example windows remain in RAM for training.
Increase `LIMIT_PER_SPLIT` and `EXAMPLES_PER_SHOT` for more complete examples.
Not every shot has all five diagnostics, so actual counts can be lower.

Plain PyTorch CPU execution does not prove Telum use. Tests in this workspace
use local x86 CPU; the notebook must still be run in IBM's LinuxONE environment.

## Files

| File | Purpose |
|---|---|
| `Five_Features_Start_Here.ipynb` | The commented walkthrough: read, inspect, normalize, train, evaluate. |
| `mast_five_features.py` | The commented five-feature reader, magnetic estimate, normalizer, and training-pair iterator. |
| `fusion_data_utils.py` | The commented RAM cache and pinned official shot-split functions. |
| `requirements-five-features.txt` | Commented data dependencies; use the IBM-provided PyTorch build. |
| `Five_Features_README.md` | This guide, exact feature definitions, and verification details. |

Source and attribution: UKAEA FAIR MAST, https://www.ukaea.org/service/fair-mast/
and https://s3.echo.stfc.ac.uk/mast/level2/shots/. Source diagnostic metadata
retains provenance; underlying data are CC BY-SA 4.0 unless otherwise noted.
TokaMark shot metadata: UKAEA / IBM / STFC,
https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark, pinned by
`fusion_data_utils.REFERENCE_COMMIT`.

## Verification in this workspace

Real shot 30421 produced finite raw and PyTorch arrays of shape `[1,10,5]`.
The default example used eleven finite saddle coils. Tests recovered a known
synthetic n=1 harmonic with a missing coil, rejected insufficient coils, future
or stale readings, wrong feature counts and NaNs, and checked explicit clipping
and native float32 byte order. These math/input checks do not scientifically
validate the derived magnetic diagnostic.

Every Python code cell was executed locally (the install cell was skipped
because dependencies were already installed). Three training epochs used eight
complete windows from shot 21719; validation used seven from 16431; test used
eight from 12860. Missing radiated power on shots 29562/24883 and missing saddle
field on 20509 were explicitly reported and skipped. Training MSE decreased
from 0.804162 to 0.628962 MA^2, but test MSE was 1.10840 MA^2 versus persistence
0.0000299654 MA^2. The tiny network is much worse than persistence and is only
a pipeline demonstration, not a useful trained predictor.

Environment: Python 3.12.14, NumPy 2.5.4, Zarr 3.4.1, PyTorch 2.14.1+cpu on
x86_64. LinuxONE/s390x execution and accelerator use have not been tested here.

The commented edition also checks executable Python syntax trees against the
previous edition. The intended code changes are only the two imports that now
point to `fusion_data_utils` and the renamed installation requirements file.
Comments and docstrings explain behavior without changing the data calculations.
