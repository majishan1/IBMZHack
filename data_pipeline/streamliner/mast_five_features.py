"""Five real MAST features for PyTorch, fetched into RAM from FAIR MAST.

This extends beyond the 39-signal TokaMark v1 subset. Keep fusion_data_utils.py
beside this file: it supplies the bounded RAM cache used for remote reads.

Feature order and units:
    Ip: magnetics/ip, A -> MA
    Te: thomson_scattering/t_e_core, eV -> keV (core temperature)
    Bn1: derived n=1 amplitude of calibrated midplane saddle fields, T
    Ne: summary/line_average_n_e, m^-3 -> 10^20 m^-3 (line average)
    Prad: bolometer/power_radiated_total, W -> MW

Bn1 is a least-squares n=0,1 toroidal harmonic estimate of coil-averaged
field, using archived coil geometry. It includes external-field pickup;
it is NOT a validated plasma-only instability measurement, a frequency-
resolved n1rms signal, or a replacement for a facility-calibrated n=1 product.
Geometry metadata currently carries development status. Review the feature
definition with a fusion specialist before scientific use.

All features use past-or-present readings only. Slow diagnostics are held
between measurements, with explicit age limits. A 2 ms output grid does not
create new independent temperature measurements. Missing/stale inputs fail;
no zeros, random numbers, or other experiments are substituted.

Source: UKAEA FAIR MAST; source attributes retain diagnostic provenance.
    https://www.ukaea.org/service/fair-mast/
    https://s3.echo.stfc.ac.uk/mast/level2/shots/30421.zarr/zarr.json
Underlying data: CC BY-SA 4.0 unless otherwise noted by the source.
"""

# Postpone evaluating type annotations until needed (useful for type hints).
from __future__ import annotations

# dataclass generates the constructor for the window's collection of fields.
from dataclasses import dataclass
# math checks that scalar parameters are ordinary finite numbers.
import math
# re matches coil names to the corresponding geometry labels.
import re

# aiohttp supplies HTTP connection options, including a finite timeout.
import aiohttp
# NumPy stores arrays and performs the vector/matrix calculations.
import numpy as np
# Zarr reads selected chunks from the remotely stored diagnostic arrays.
import zarr
# FsspecStore gives Zarr read-only access to an HTTPS-backed remote store.
from zarr.storage import FsspecStore

# The shared helper retains recently fetched compressed chunks in RAM only.
from fusion_data_utils import MemoryChunkCache

# These names define the meaning and order of the five tensor columns.
FEATURE_NAMES = ("Ip", "Te_core", "Bn1_saddle_estimate", "Ne_line_average", "Prad")
# Each output column uses the corresponding unit in this tuple.
FEATURE_UNITS = ("MA", "keV", "T", "10^20 m^-3", "MW")
# Example bounds from the proposed normalizer, NOT universal physical limits.
DEFAULT_FEATURE_MINS = np.array([0.0, 0.1, -0.05, 0.1, 0.0], dtype=np.float32)
DEFAULT_FEATURE_MAXS = np.array([2.5, 10.0, 0.05, 5.0, 8.0], dtype=np.float32)


class MissingMeasurementError(ValueError):
    """A requested window cannot provide all five verified feature values."""


class SensorNormalizer:
    """Validate [batch,time,5] and apply supplied fixed min/max scaling.

    Fit or choose these bounds using training data only, then reuse unchanged
    for validation, test, and inference. Clipping is explicit and defaults off
    so unusual measurements are not silently flattened. No training occurs.
    """

    def __init__(self, feature_mins, feature_maxs, *, clip=False):
        """Copy five supplied bounds; * makes clip a named-only argument."""
        # Independent float32 copies prevent later edits to the caller's arrays
        # from unexpectedly changing the model's normalization.
        self.feature_mins = np.array(feature_mins, dtype=np.float32, copy=True)
        self.feature_maxs = np.array(feature_maxs, dtype=np.float32, copy=True)
        # One lower and upper bound is required for each of the five features.
        if self.feature_mins.shape != (5,) or self.feature_maxs.shape != (5,):
            raise ValueError("Supply one min and max for each of the five features")
        # NaN and infinity would make all subsequent scaled values unreliable.
        if not np.isfinite(self.feature_mins).all() or not np.isfinite(self.feature_maxs).all():
            raise ValueError("Scaling bounds must be finite")
        # Compute five scaling divisors once, rather than for every batch.
        self.feature_range = self.feature_maxs - self.feature_mins
        # Flat or reversed bounds cannot define a valid min/max scale.
        if not np.isfinite(self.feature_range).all() or np.any(self.feature_range <= 0):
            raise ValueError("Each feature max must be strictly greater than its min")
        # Remember whether the caller explicitly wants values clipped to 0..1.
        self.clip = bool(clip)

    def process_raw_batch(self, raw_data):
        """Turn finite [batch,time,5] measurements into a float32 CPU tensor."""
        # Import PyTorch here so basic archive access can also run without it.
        import torch

        # Convert Python lists or arrays to native-endian 32-bit floating point.
        raw = np.asarray(raw_data, dtype=np.float32)
        # Checking the last axis prevents a one-feature array from being
        # accidentally broadcast into five misleading copies.
        if raw.ndim != 3 or raw.shape[-1] != 5 or 0 in raw.shape:
            raise ValueError("Expected nonempty [batch,time,5]; refusing broadcasting")
        # Missing measurements must be handled by the reader, not hidden here.
        if not np.isfinite(raw).all():
            raise ValueError("Missing or nonfinite measurements cannot be normalized")
        # NumPy broadcasts the five bounds across every batch item/time step.
        # Each feature becomes (measurement - minimum) / (maximum - minimum).
        normalized = (raw - self.feature_mins) / self.feature_range
        # Even finite inputs could overflow if a supplied scale were too small.
        if not np.isfinite(normalized).all():
            raise ValueError("Scaling overflowed")
        # Clipping discards how far a value lies outside the chosen bounds;
        # leave it off unless that behavior is explicitly wanted.
        if self.clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        # Native float32 byte order works on s390x too. Only from_numpy shares
        # memory; normalization above allocates a new array. This is a CPU tensor.
        return torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))


# This is a simple data container: @dataclass supplies its initialization method.
@dataclass
class FiveFeatureWindow:
    """Measurements and quality information for one aligned input window."""
    shot: int  # The identifier of the experimental pulse supplying every input.
    time_s: np.ndarray  # Requested grid times, in seconds, shaped [time].
    values: np.ndarray  # [time,5], in FEATURE_UNITS, before normalization
    sample_age_s: np.ndarray  # [time,5]; slow channels may repeat older readings
    source_url: str  # Remote archive location, recorded for traceability.
    minimum_magnetic_coils: int  # Fewest finite coils among valid fitted samples.
    magnetic_fit_rms_T: float  # RMS mismatch of the fitted coil fields, in tesla.
    feature_names: tuple = FEATURE_NAMES  # Names of the five value columns.
    feature_units: tuple = FEATURE_UNITS  # Physical units before normalization.
    dataset: str = "FAIR MAST level2 (five-feature extension beyond TokaMark)"

    # A property is accessed as window.raw_batch rather than raw_batch().
    @property
    def raw_batch(self):
        """Add one batch dimension: [time,5] becomes [1,time,5]."""
        # np.newaxis creates an extra axis without copying the measurements.
        return self.values[np.newaxis, :, :]

    def as_tensor(self, normalizer):
        """Apply the supplied training-compatible preprocessing to this window."""
        # Reuse one normalizer so training and inference apply the same bounds.
        return normalizer.process_raw_batch(self.raw_batch)


class MASTFiveFeatureStreamer:
    """One shot, five simultaneous feature definitions, no measurement disk cache.

    Defaults: 10 time steps on a 2 ms grid. Finite readings are carried forward
    for at most [2,20,0.2,3,2] ms respectively. These are explicit preprocessing
    choices, not hardware sampling rates. Inspect sample_age_s in each window.
    """

    # These four diagnostics are direct scalar records. Each tuple gives the
    # archive path, required source unit, and multiplier to the output unit.
    # The magnetic fifth feature is derived separately from several coils.
    DIRECT = {
        "Ip": ("magnetics/ip", "A", 1e-6),
        "Te": ("thomson_scattering/t_e_core", "eV", 1e-3),
        "Ne": ("summary/line_average_n_e", "1 / m ** 3", 1e-20),
        "Prad": ("bolometer/power_radiated_total", "W", 1e-6),
    }

    def __init__(self, shot=30421, *, start_s=0.1, step_s=0.002, samples=10,
                 max_age_s=(0.002, 0.020, 0.0002, 0.003, 0.002),
                 endpoint="https://s3.echo.stfc.ac.uk", cache_bytes=16*1024*1024):
        """Open one pulse and prepare its diagnostic/time/geometry metadata.

        start_s: initial requested time in seconds.
        step_s: spacing of the output grid, not a hardware sampling rate.
        samples: number of time steps in each returned window.
        max_age_s: oldest allowed input in each FEATURE_NAMES position.
        cache_bytes: compressed payload RAM budget (decoded arrays use more).
        """
        # bool is an int subclass in Python, so reject True/False explicitly.
        if isinstance(shot, bool) or not isinstance(shot, int) or shot <= 0:
            raise ValueError("shot must be a positive integer")
        # A window must contain at least one actual requested time step.
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError("samples must be a positive integer")
        # A positive step ensures the requested timeline always moves forward.
        if not math.isfinite(start_s) or not math.isfinite(step_s) or step_s <= 0:
            raise ValueError("Need finite start_s and positive step_s")
        # Use float64 for time calculations to reduce accumulated rounding error.
        ages = np.asarray(max_age_s, dtype=np.float64)
        # Each of the five inputs needs its own finite positive age limit.
        if ages.shape != (5,) or not np.isfinite(ages).all() or np.any(ages <= 0):
            raise ValueError("max_age_s must contain five finite positive values")
        # Save the shot ID and the next sequential window's starting time.
        self.shot, self.current_time_s = shot, float(start_s)
        # Save the window length, grid spacing, and staleness limits.
        self.samples, self.step_s, self.max_age_s = samples, float(step_s), ages
        # All diagnostics are read from this same shot in the wider MAST archive.
        self.source_url = f"{endpoint.rstrip('/')}/mast/level2/shots/{shot}.zarr"
        # read_only prevents writes to the archive. trust_env respects the
        # machine's proxy settings; each HTTP request has a 30-second timeout.
        remote = FsspecStore.from_url(self.source_url, read_only=True,
            storage_options={"client_kwargs": {"trust_env": True,
                "timeout": aiohttp.ClientTimeout(total=30)}})
        # Place our bounded compressed-byte RAM cache in front of remote reads.
        self.store = MemoryChunkCache(remote, max_bytes=cache_bytes)
        # This level2 release uses Zarr v3. Consolidated metadata describes all
        # arrays together, reducing separate metadata requests.
        self.root = zarr.open_group(self.store, mode="r", zarr_format=3, use_consolidated=True)
        # Map each diagnostic path to its array, time-axis number, and timestamps.
        self._signals = {}
        # Validate required diagnostic metadata before returning a usable reader.
        try:
            # _register checks each source unit and identifies the time axis.
            for path, units, _ in self.DIRECT.values():
                self._register(path, units, scalar=True)
            # Saddle fields have both a time axis and a coil axis.
            self._register("magnetics/b_field_tor_probe_saddle_field", "T", scalar=False)
            # Match coil names to physical toroidal positions for the n=1 fit.
            self._prepare_geometry()
        # A missing signal or unsupported metadata makes this pulse unusable.
        except (KeyError, ValueError) as error:
            # Release the store, and attach the shot ID to the explanatory error.
            self.close()
            raise MissingMeasurementError(f"Shot {shot}: {error}") from error

    def _register(self, path, units, *, scalar):
        """Validate one source array and retain its own timestamp vector."""
        # Opening the array reads its metadata, not all measurement samples.
        array = self.root[path]
        # Reject unexpected units rather than applying an incorrect conversion.
        if array.attrs.get("units") != units:
            raise ValueError(f"{path}: expected {units!r}, found {array.attrs.get('units')!r}")
        # Zarr names the axes; time is not assumed to be the first axis.
        dims = array.metadata.dimension_names
        # Enumerate returns both each axis number and its recorded name.
        axes = [i for i, dim in enumerate(dims or ()) if str(dim).startswith("time")]
        # Exactly one time dimension is required for this kind of diagnostic.
        if len(axes) != 1:
            raise ValueError(f"{path}: cannot identify a unique time axis")
        # Remember which dimension must be sliced when requesting a time window.
        axis = axes[0]
        # The timestamp array lives inside the same diagnostic group.
        times_array = self.root[f"{path.split('/')[0]}/{dims[axis]}"]
        # All time parameters in this project are seconds.
        if str(times_array.attrs.get("units", "")).lower() not in {"s", "sec", "seconds"}:
            raise ValueError(f"{path}: timestamps must be in seconds")
        # Read only the full timestamp vector into RAM, allowing fast searches.
        times = np.asarray(times_array[:], dtype=np.float64)
        # Values and timestamps must agree; searchsorted also requires strictly
        # increasing finite times without duplicated entries.
        if (times.ndim != 1 or len(times) < 2 or len(times) != array.shape[axis]
                or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)):
            raise ValueError(f"{path}: invalid timestamp vector")
        # Scalar diagnostics should provide one value at each timestamp.
        if scalar and (array.ndim != 1):
            raise ValueError(f"{path}: expected one scalar diagnostic")
        # A coil diagnostic should contain only time and coil dimensions.
        if not scalar and (array.ndim != 2):
            raise ValueError(f"{path}: expected time and coil axes")
        # Retain metadata and timestamps for repeated window requests.
        self._signals[path] = (array, axis, times)

    def _prepare_geometry(self):
        """Build columns [1,cos(phi),sin(phi)] in the field-channel order."""
        # Group the magnetic arrays under their shared archive directory.
        group = self.root["magnetics"]
        # Field labels use names such as M01/M010; geometry labels use
        # sad_out_m01/sad_out_m10. Match their numbers instead of array positions.
        fields = [str(x) for x in group["b_field_tor_probe_saddle_field_channel"][:]]
        labels = [str(x) for x in group["b_field_tor_probe_saddle_m_geometry_channel"][:]]
        # Midplane geometry supplies each saddle coil's toroidal outline.
        geometry = group["b_field_tor_probe_saddle_m_phi"]
        # The subsequent conversion to radians assumes recorded degrees.
        if "degrees" not in geometry.attrs.get("units", ""):
            raise ValueError("Midplane saddle geometry must provide toroidal angles in degrees")
        # Geometry is small and can be kept entirely in RAM.
        phis = np.asarray(geometry[:], dtype=np.float64)
        def channel_id(label, pattern):
            """Convert a recognized field/geometry label into its coil number."""
            # fullmatch rejects labels that only partly match the expected form.
            found = re.fullmatch(pattern, label.lower())
            # Guessing an unknown label could assign a coil the wrong angle.
            if found is None:
                raise ValueError(f"Unrecognized midplane coil label {label!r}")
            # Extract the captured digits; leading zeros do not change the ID.
            return int(found.group(1))
        # Parse geometry IDs and signal IDs using their respective name formats.
        ids = [channel_id(x, r"sad_out_m0*(\d+)") for x in labels]
        field_ids = [channel_id(x, r"m0*(\d+)") for x in fields]
        # Each ID must identify a unique coil in both arrays.
        if len(set(ids)) != len(ids) or len(set(field_ids)) != len(field_ids):
            raise ValueError("Duplicate coil labels")
        # Every field channel needs one matching geometry outline.
        if phis.shape[0] != len(ids) or not set(field_ids).issubset(ids):
            raise ValueError("Coil signals and geometry cannot be matched by label")
        # Use the center of each toroidal extent, not a vertex-weighted mean.
        # Geometry vertices may cross 0/360, so unwrap before finding bounds.
        angles = []
        # Preserve field-channel order when constructing the fitting matrix.
        for coil in field_ids:
            # Select this coil's geometry by its matched identifier.
            row = phis[ids.index(coil)]
            # Missing geometry cannot define a reliable fitting angle.
            if not np.isfinite(row).all():
                raise ValueError("Nonfinite coil geometry")
            # Trigonometric functions use radians; unwrap avoids a false jump
            # when an outline crosses from 359 degrees to 0 degrees.
            radians = np.unwrap(np.deg2rad(row))
            # The midpoint of the angular extent represents the coil center.
            angles.append((radians.min() + radians.max()) / 2)
        # Store one toroidal angle per magnetic field channel.
        phi = np.asarray(angles)
        # The fit is B(phi)=offset+a*cos(phi)+b*sin(phi). Its three columns
        # respectively represent the offset, cosine coefficient, and sine coefficient.
        self._design = np.column_stack([np.ones(len(phi)), np.cos(phi), np.sin(phi)])
        # Keep human-readable labels and source geometry status for inspection.
        self.coil_labels = tuple(fields)
        self.geometry_status = geometry.attrs.get("status", "unknown")

    def _slice(self, path, start, end):
        """Fetch a source time range and return values with time as the first axis."""
        # Recover the array, its time-axis location, and its sorted time vector.
        array, axis, times = self._signals[path]
        # Find the first timestamp at or after the requested beginning.
        lo = int(np.searchsorted(times, start, side="left"))
        # Include timestamps equal to the end so causal selection can use them.
        hi = int(np.searchsorted(times, end, side="right"))
        # Initially select every item on every axis, including all coils.
        selection = [slice(None)] * array.ndim
        # Narrow only the time dimension to the required source range.
        selection[axis] = slice(lo, hi)
        # Zarr fetches intersecting compressed chunks. Move time to axis zero
        # so scalar output is [time] and magnetic output is [time,coils].
        values = np.moveaxis(np.asarray(array[tuple(selection)], dtype=np.float64), axis, 0)
        # The returned timestamps describe exactly the selected value rows.
        return times[lo:hi], values

    # This helper needs only its arguments, not any state stored on self.
    @staticmethod
    def _causal_hold(times, values, grid, max_age, name):
        """Use the most recent finite reading at or before each output time."""
        # Exclude missing source samples before searching the remaining times.
        finite = np.isfinite(values)
        times, values = times[finite], values[finite]
        # There is no defensible replacement if this entire source range is missing.
        if not len(times):
            raise MissingMeasurementError(f"{name}: no finite measurements in requested region")
        # Never take a reading after the requested grid time (no look-ahead).
        indices = np.searchsorted(times, grid, side="right") - 1
        # A negative index means there is no reading yet for an output time.
        if np.any(indices < 0):
            raise MissingMeasurementError(f"{name}: no earlier measurement for part of the window")
        # Calculate how long each selected measurement has been held forward.
        ages = grid - times[indices]
        # A tiny tolerance accounts only for floating-point rounding of seconds.
        if np.any(ages > max_age + 1e-12):
            raise MissingMeasurementError(f"{name}: measurement is older than {max_age*1000:g} ms")
        # Return both the held measurements and their ages for quality inspection.
        return values[indices], ages

    def _n1(self, values):
        """Estimate n=1 amplitude from [time,coils] field measurements in tesla."""
        # Each value column must match one column in the geometry design matrix.
        if values.ndim != 2 or values.shape[1] != len(self.coil_labels):
            raise MissingMeasurementError("Saddle field axes do not match coil labels")
        # A Boolean mask marks working finite coil readings at every timestamp.
        valid = np.isfinite(values)
        # Group timestamps that have the same available coils, so a matrix
        # inverse can be reused across all rows sharing that pattern.
        masks, groups = np.unique(valid, axis=0, return_inverse=True)
        # Initialize results as missing; unsupported fits remain missing.
        amplitude = np.full(len(values), np.nan)
        residual = np.full(len(values), np.nan)
        # Sum True values to count the available coils at each timestamp.
        counts = valid.sum(axis=1)
        # Fit each distinct pattern of working coil channels separately.
        for i, mask in enumerate(masks):
            # Select geometry rows only for the coils available in this pattern.
            design = self._design[mask]
            # Require enough observations, three independent fitting columns,
            # and a condition number small enough to avoid an unstable inverse.
            if mask.sum() < 6 or np.linalg.matrix_rank(design) != 3 or np.linalg.cond(design) > 10:
                continue  # Leave these unsupported timestamps as NaN.
            # Find every time row that shares the current availability pattern.
            rows = np.flatnonzero(groups == i)
            # Extract all selected times and working coils as one rectangular block.
            field = values[np.ix_(rows, np.flatnonzero(mask))]
            # @ is matrix multiplication; pinv computes a least-squares inverse.
            # coeff has [offset,a,b] for each selected timestamp.
            coeff = field @ np.linalg.pinv(design).T
            # The nonnegative n=1 amplitude is sqrt(a^2+b^2), still in tesla.
            amplitude[rows] = np.hypot(coeff[:, 1], coeff[:, 2])
            # Quantify how closely the offset+n=1 fit reproduces the coil data.
            residual[rows] = np.sqrt(np.mean((field - coeff @ design.T)**2, axis=1))
        # The residual and coil counts accompany the feature as quality information.
        return amplitude, residual, counts

    def read_window(self, start_s=None):
        """Read one aligned [time,5] window without advancing the stream cursor."""
        # A supplied start overrides the cursor for this request only.
        start = self.current_time_s if start_s is None else float(start_s)
        # NaN or infinite times cannot identify a source interval.
        if not math.isfinite(start):
            raise ValueError("start_s must be finite")
        # Example: 0.100, 0.102, ... 0.118 seconds for ten samples at 2 ms.
        grid = start + np.arange(self.samples, dtype=np.float64) * self.step_s
        # Collect (feature-column-number,array) pairs to restore the chosen order.
        values, ages = [], []
        # Read the four direct diagnostics; column 2 is reserved for derived n=1.
        for name, index in [("Ip", 0), ("Te", 1), ("Ne", 3), ("Prad", 4)]:
            # _ ignores the already checked source-unit string in this tuple.
            path, _, scale = self.DIRECT[name]
            # Include earlier samples up to the permitted age of this feature.
            t, raw = self._slice(path, grid[0] - self.max_age_s[index], grid[-1])
            # Align by causal hold, then convert to the model's physical unit.
            v, age = self._causal_hold(t, raw, grid, self.max_age_s[index], name)
            values.append((index, v * scale))
            # Retain staleness information independently of the measurements.
            ages.append((index, age))
        # Read the simultaneous midplane coil fields on their own native time base.
        t, field = self._slice("magnetics/b_field_tor_probe_saddle_field",
                               grid[0] - self.max_age_s[2], grid[-1])
        # Calculate a magnetic amplitude before aligning it to the common grid.
        amplitude, residual, counts = self._n1(field)
        # Apply the magnetic channel's tighter staleness limit.
        n1, age = self._causal_hold(t, amplitude, grid, self.max_age_s[2], "Bn1")
        # Insert the derived feature into its reserved third tensor column.
        values.append((2, n1)); ages.append((2, age))
        # Ignore unsupported fitting timestamps when summarizing fit quality.
        good = np.isfinite(amplitude)
        # Sorting restores Ip,Te,Bn1,Ne,Prad order; stack them into [time,5].
        values = np.column_stack([v for _, v in sorted(values)])
        ages = np.column_stack([v for _, v in sorted(ages)])
        values = np.asarray(values, dtype=np.float32)  # Native-endian on LinuxONE.
        # Reject conversion overflow instead of passing invalid values to PyTorch.
        if not np.isfinite(values).all():
            raise MissingMeasurementError("Unit conversion produced nonfinite measurements")
        # Return measurements together with their timing, provenance, and fit quality.
        return FiveFeatureWindow(self.shot, grid, values, ages, self.source_url,
            int(counts[good].min()), float(np.sqrt(np.mean(residual[good]**2))))

    def fetch_next_20ms_window(self):
        """Drop-in raw-batch method for the pasted normalizer, fixed to 10 x 2 ms."""
        # Enforce the exact format implied by this convenience method's name.
        if self.samples != 10 or not np.isclose(self.step_s, 0.002):
            raise ValueError("This method requires samples=10 and step_s=0.002")
        # Fetch the current 20 ms window; errors leave the cursor unchanged.
        window = self.read_window()
        self.current_time_s += 0.020  # Only advance after successful retrieval.
        # Return NumPy [1,10,5]; normalization and model execution are separate.
        return window.raw_batch

    def future_current(self, target_time_s):
        """Actual later Ip measurement, MA, with its recorded timestamp."""
        # Use the same maximum current-sample age as the input channel.
        age = self.max_age_s[0]
        # Request a narrow range ending at the desired prediction-target time.
        t, raw = self._slice("magnetics/ip", target_time_s-age, target_time_s)
        # Choose an actual recorded reading, not an invented/interpolated target.
        value, ages = self._causal_hold(t, raw, np.array([target_time_s]), age, "target Ip")
        # Return a one-element MA target plus its actual recorded time in seconds.
        return np.asarray([value[0]*1e-6], dtype=np.float32), float(target_time_s-ages[0])

    def close(self):
        """Close the underlying remote store when this pulse is finished."""
        self.store.close()

    def __enter__(self):
        """Support `with MASTFiveFeatureStreamer(...) as stream:`."""
        return self

    def __exit__(self, *exc):
        """Close the store on leaving a with block, including after an error."""
        self.close()


def iter_five_feature_pairs(shots, *, start_s=0.1, stop_s=0.5,
                            lead_s=0.010, examples_per_shot=32,
                            skip_bad_shots=False):
    """Yield raw [10,5] histories and actual future current [1], shot by shot.

    This is a current-forecasting exercise, not a disruption classifier.
    Use disjoint shot-level training/validation/test lists. Missing windows are
    skipped; unavailable shots raise unless explicitly allowed to be skipped.
    """
    # Warnings make explicitly skipped shots visible in the notebook's output.
    import warnings
    # The iterator needs a valid time region and a target later than the input.
    if not all(map(math.isfinite, [start_s, stop_s, lead_s])) or stop_s <= start_s or lead_s <= 0:
        raise ValueError("Need finite stop > start and positive forecast lead")
    # None permits all complete windows in the selected interval; otherwise cap
    # the number of examples supplied by one experimental pulse.
    if (examples_per_shot is not None and (isinstance(examples_per_shot, bool)
            or not isinstance(examples_per_shot, int) or examples_per_shot <= 0)):
        raise ValueError("examples_per_shot must be a positive integer or None")
    # Track whether any requested shot produced a usable training example.
    total = 0
    # dict.fromkeys removes duplicate IDs while preserving their supplied order.
    for shot in dict.fromkeys(int(x) for x in shots):
        # Opening or processing one shot may fail because a diagnostic is missing.
        try:
            # Count complete windows from this pulse separately from the total.
            produced = 0
            # Automatically close each shot's store before moving to the next.
            with MASTFiveFeatureStreamer(shot, start_s=start_s) as stream:
                # Windows begin 20 ms apart. Ten input steps span 18 ms from
                # first to last timestamp; leave room for the future target too.
                for start in np.arange(start_s, stop_s - 0.018 - lead_s, 0.020):
                    # Missing input windows are skipped rather than fabricated.
                    try:
                        # The returned values contain all five aligned inputs.
                        window = stream.read_window(float(start))
                        # The label is a later recorded current measurement.
                        target, target_time = stream.future_current(window.time_s[-1] + lead_s)
                        # Check that rounding or sparse readings did not select
                        # a label from the input's own time range.
                        if target_time <= window.time_s[-1]:
                            raise MissingMeasurementError("Target must occur after the input")
                    # Continue trying later windows if this one is incomplete.
                    except MissingMeasurementError:
                        continue
                    # yield supplies one pair at a time: x=[10,5], y=[1].
                    # Merely yielding a pair does not update any model weights.
                    yield window.values, target
                    # Increment counts only after supplying a complete example.
                    produced += 1; total += 1
                    # Respect the configured per-pulse demonstration limit.
                    if examples_per_shot is not None and produced >= examples_per_shot:
                        break
            # Report a pulse whose entire chosen interval lacked complete inputs.
            if not produced:
                raise MissingMeasurementError(f"Shot {shot}: no complete five-feature windows")
        # Distinguish explicit permission to skip a bad shot from the default
        # behavior of surfacing its error to the caller.
        except (KeyError, ValueError, OSError, RuntimeError) as error:
            if not skip_bad_shots:
                raise  # Re-raise the original error with its useful traceback.
            # Keep a readable record of every unavailable pulse that was omitted.
            warnings.warn(f"Skipping shot {shot}: {error}", stacklevel=2)
    # Never allow an empty training source to appear to have run successfully.
    if not total:
        raise MissingMeasurementError("No complete examples; inspect diagnostics and shot availability")
