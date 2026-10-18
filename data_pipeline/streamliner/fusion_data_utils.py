"""Small shared utilities used by the five-feature MAST project.

This file replaces the need to import the previous one-feature example files.
It contains two independent helpers:

1. MemoryChunkCache: keep recently fetched compressed Zarr chunks in RAM.
2. official_splits: obtain the published train/validation/test shot lists.

The shot lists come from TokaMark metadata, but the five diagnostic inputs
come from the wider FAIR MAST archive. Reusing the lists does not make this
expanded forecasting task an official TokaMark benchmark result.
"""

# OrderedDict remembers insertion order, allowing us to discard the oldest
# cache entry when the RAM budget is full.
from collections import OrderedDict
# csv reads the small shot-assignment table without requiring pandas.
import csv
# functools provides a decorator that remembers parsed metadata in RAM.
from functools import lru_cache
# hashlib verifies the exact contents of the downloaded split table.
import hashlib
# io lets the CSV reader treat an in-memory string like an open text file.
import io
# random shuffles a subset reproducibly without moving shots between splits.
import random
# urllib fetches the metadata over HTTPS; it does not save a local CSV file.
import urllib.request

# WrapperStore lets us place a cache in front of Zarr's existing remote store.
from zarr.storage import WrapperStore


class MemoryChunkCache(WrapperStore):
    """Read-only compressed-byte cache with a fixed RAM budget.

    LRU means least recently used: when space runs out, discard the chunk
    that has gone longest without being read. Decoded arrays, timestamps,
    and PyTorch tensors consume extra RAM outside this compressed-byte budget.
    """

    def __init__(self, store, max_bytes=8 * 1024 * 1024):
        """Wrap a remote store; the default cache budget is 8 MiB."""
        # Initialize Zarr's wrapper with the store that actually reads the URL.
        super().__init__(store)
        # Refuse invalid budgets before fetching any data.
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        # The limit applies to the compressed payloads retained by this cache.
        self.max_bytes = max_bytes
        # Count how many compressed bytes are currently retained.
        self.cached_bytes = 0
        # Count fetched payload bytes, including metadata, but not HTTP overhead.
        self.fetched_bytes = 0
        # Count reads satisfied from RAM without another remote fetch.
        self.cache_hits = 0
        # Entries are ordered from least recently used to most recently used.
        self._items = OrderedDict()

    async def get(self, key, prototype, byte_range=None):
        """Return one stored object or byte range, consulting the RAM cache first.

        Zarr uses asynchronous I/O internally. The notebook-facing reader is
        synchronous: its cell waits until the requested array slice arrives.
        """
        # Distinguish requests for different byte ranges of the same object.
        cache_key = (key, repr(byte_range))
        # An existing entry can be returned entirely from local RAM.
        if cache_key in self._items:
            # Remove and reinsert it so it becomes the newest entry.
            raw = self._items.pop(cache_key)
            self._items[cache_key] = raw
            # Update the diagnostic counter; this does not affect the data.
            self.cache_hits += 1
            # Recreate the buffer type requested by this Zarr read.
            return prototype.buffer.from_bytes(raw)

        # await allows Zarr to perform the network read through its async API.
        value = await self._store.get(key, prototype, byte_range)
        # A missing remote object is returned as missing, never as fake data.
        if value is None:
            return None
        # Keep an immutable compressed byte payload for possible reuse.
        raw = value.to_bytes()
        # Record the received payload size, even if it is too large to cache.
        self.fetched_bytes += len(raw)

        # Oversized objects can still be read, but are not retained in the cache.
        if len(raw) <= self.max_bytes:
            # Remove a previous entry for the same request, if one exists.
            old = self._items.pop(cache_key, None)
            # Subtract the bytes occupied by any replaced entry.
            if old is not None:
                self.cached_bytes -= len(old)
            # Free enough space by evicting the least recently used entries.
            while self._items and self.cached_bytes + len(raw) > self.max_bytes:
                # last=False removes the oldest item in this OrderedDict.
                _, evicted = self._items.popitem(last=False)
                # Keep the retained-byte counter accurate after each eviction.
                self.cached_bytes -= len(evicted)
            # Save the new payload as the most recently used entry.
            self._items[cache_key] = raw
            # Account for the bytes now retained in RAM.
            self.cached_bytes += len(raw)
        # Return the original Zarr buffer to the caller that requested it.
        return value


# Pin the source revision so future repository changes do not change splits.
REFERENCE_COMMIT = "1a200f4588addaad2ae3d53d438d9e1946c09294"
# This URL prefix is incomplete on its own: the revision and filename are
# appended below before any network request is made.
REFERENCE_BASE = (
    "https://raw.githubusercontent.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark/"
    + REFERENCE_COMMIT + "/src/tokamark/"
)
# For each strategy, record the relative CSV path and expected SHA-256 digest.
# A checksum detects changed or corrupted content; it is not a model metric.
SPLIT_FILES = {
    "random": (
        "metadata/TokaMark_data_splits.csv",
        "d8e63b3bfba50f9292ed307e2afa0eb0ec2dfac861ceaa649c114999a9b8ef60",
    ),
    "temporal": (
        "metadata/TokaMark_temporal_data_splits.csv",
        "8d188293cb9c0fde187b183a89165c933cd3c3d81cbe01cdc17d23bfb361432b",
    ),
}


def _read_url(url, max_bytes=3_000_000):
    """Fetch a small metadata document into RAM, with a size limit and timeout."""
    # Close the HTTP response automatically when the with block finishes.
    with urllib.request.urlopen(url, timeout=30) as response:
        # Read one extra byte so an oversized response can be detected.
        payload = response.read(max_bytes + 1)
    # Refuse a response larger than the intended small-metadata budget.
    if len(payload) > max_bytes:
        raise ValueError("Remote metadata exceeded the size limit")
    # Return raw bytes; the CSV parser below decodes them as UTF-8 text.
    return payload


# Cache at most two parsed tables in RAM: one random and one temporal split.
@lru_cache(maxsize=2)
def _official_table(strategy):
    """Validate published shot assignments before making them available."""
    # Only the two published strategies have known paths and checksums.
    if strategy not in SPLIT_FILES:
        raise ValueError("strategy must be 'random' or 'temporal'")
    # Choose the filename and expected digest for the requested strategy.
    relative_path, expected_sha = SPLIT_FILES[strategy]
    # Complete the raw-file URL and fetch the small CSV into RAM.
    payload = _read_url(REFERENCE_BASE + relative_path)
    # Refuse unexpected table contents rather than silently changing splits.
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Published split metadata failed its checksum")
    # Dictionary order defines the returned order: training, validation, test.
    groups = {"train": [], "val": [], "test": []}
    # Track shot IDs so a duplicated row cannot enter the result.
    seen = set()
    # DictReader maps each CSV header to its value in the current row.
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8"))):
        # A shot identifies one entire experimental plasma pulse.
        shot = int(row["shot_id"])
        # A duplicate shot makes the metadata ambiguous and is rejected.
        if shot in seen:
            raise ValueError("Duplicate shot in published split metadata")
        # Remember this shot before processing the following rows.
        seen.add(shot)
        # Collect which published split, if any, contains this shot.
        memberships = []
        # Inspect each train/val/test Boolean flag in the row.
        for name in groups:
            # Accept capitalization/whitespace differences in Boolean text.
            flag = row[name].strip().lower()
            # Reject other text instead of guessing its meaning.
            if flag not in {"true", "false"}:
                raise ValueError("Unrecognised split flag")
            # True means the published table assigns this shot to this split.
            if flag == "true":
                memberships.append(name)
        # Sharing a shot between splits would leak experimental information.
        if len(memberships) > 1:
            raise ValueError("A published shot appears in multiple splits")
        # Keep assigned shots; an all-False row is intentionally excluded.
        if memberships:
            groups[memberships[0]].append(shot)
    # Tuples protect the cached assignments against accidental modification.
    return {name: tuple(ids) for name, ids in groups.items()}


def official_splits(*, strategy="random", limit_per_split=None, seed=None):
    """Return (training_shots, validation_shots, test_shots), in that order.

    limit_per_split selects a smaller demonstration subset. seed optionally
    shuffles IDs inside each published group; it never reassigns membership.
    All windows from a shot therefore remain together in one split.
    """
    # None means all assigned shots; otherwise require a positive integer.
    if limit_per_split is not None and (
        isinstance(limit_per_split, bool)
        or not isinstance(limit_per_split, int) or limit_per_split <= 0
    ):
        raise ValueError("limit_per_split must be positive or None")
    # Accumulate three separate lists in the published dictionary's order.
    result = []
    # Use an independent generator so this helper does not change global RNG state.
    generator = random.Random(seed)
    # Read only assignments; this does not fetch diagnostic measurement arrays.
    for name, ids in _official_table(strategy).items():
        # Copy the cached tuple into a list that can safely be shuffled.
        selected = list(ids)
        # A supplied seed makes subset selection reproducible.
        if seed is not None:
            generator.shuffle(selected)
        # Keep only the requested number of candidate shots in this split.
        if limit_per_split is not None:
            selected = selected[:limit_per_split]
        # Add this split's candidate list to the ordered return value.
        result.append(selected)
    # The caller unpacks the tuple or names its three lists explicitly.
    return tuple(result)
