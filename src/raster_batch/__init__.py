"""Parallel batch processing for raster datasets.

The public surface is the :func:`raster_batch.engine.run_batch` engine plus the
operations in :mod:`raster_batch.ops`; the CLI in :mod:`raster_batch.cli` is a thin
layer over both.
"""

from __future__ import annotations

from raster_batch.engine import BatchReport, ItemResult, OnError, Task, run_batch
from raster_batch.errors import ItemError, RasterBatchError, UsageError

__version__ = "0.1.0"

__all__ = [
    "BatchReport",
    "ItemError",
    "ItemResult",
    "OnError",
    "RasterBatchError",
    "Task",
    "UsageError",
    "__version__",
    "run_batch",
]
