"""Choosing read/write windows so peak memory is a knob, not a surprise.

Every operation in this package streams rasters in windows rather than calling
``dataset.read()``. That single decision is what lets a 40 GB mosaic tile go through
a machine with 4 GB of RAM, and it is what makes the memory arithmetic in the README
possible:

    peak RSS  ~  workers x window_bytes x 2   (one source window, one destination)

plus a fixed GDAL overhead of a few tens of MB per process. Because the multiplier is
the *worker count*, raising ``--workers`` on a memory-tight box is exactly as
dangerous as raising ``--window-mb``, which is not obvious until it is written down.

Background: `Streaming Raster Windows to Cap Memory in Mosaics
<https://www.batch-processing.com/spatial-batch-processing-async-workflows/memory-management-for-large-datasets/streaming-raster-windows-to-cap-memory-in-mosaics/>`_.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rasterio.windows import Window

DEFAULT_WINDOW_MB = 64
"""Default per-worker window budget in mebibytes."""


def bytes_per_row(width: int, count: int, dtype: str) -> int:
    """Return the in-memory size of one full raster row across all bands."""
    return width * count * int(np.dtype(dtype).itemsize)


def rows_per_window(width: int, count: int, dtype: str, budget_bytes: int) -> int:
    """How many rows fit in ``budget_bytes``, never fewer than one.

    Returning zero would produce empty windows and an infinite loop, so a single row
    is the floor even when one row exceeds the budget. A raster whose single row does
    not fit in the budget is pathological (a 500k-pixel-wide float64 image); we
    process it and let the caller feel the memory rather than failing the job.
    """
    per_row = bytes_per_row(width, count, dtype)
    if per_row <= 0:
        return 1
    return max(1, budget_bytes // per_row)


def iter_row_windows(
    width: int,
    height: int,
    count: int,
    dtype: str,
    budget_bytes: int,
) -> Iterator[Window]:
    """Yield full-width row-stripe windows covering the raster exactly once.

    Stripes rather than square blocks: GeoTIFF data is stored either in strips or in
    tiles, and a full-width stripe read is sequential on disk for both layouts, which
    matters far more than block alignment for a one-pass copy.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        count: Band count.
        dtype: NumPy dtype name of the band data.
        budget_bytes: Memory budget for one window.

    Yields:
        ``rasterio.windows.Window`` objects, top to bottom, with the final window
        clipped to the raster height.
    """
    step = rows_per_window(width, count, dtype, budget_bytes)
    for row_off in range(0, height, step):
        yield Window(0, row_off, width, min(step, height - row_off))


def iter_tile_windows(width: int, height: int, size: int) -> Iterator[tuple[int, int, Window]]:
    """Yield ``(col_index, row_index, window)`` for a fixed-size tile grid.

    Edge tiles are clipped rather than padded, so a 1000x1000 raster split at 512
    yields four tiles of 512x512, 488x512, 512x488 and 488x488. Padding would invent
    pixels that were never measured, which is the wrong default for scientific data.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        size: Tile edge length in pixels.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError("tile size must be >= 1")
    for row_index, row_off in enumerate(range(0, height, size)):
        for col_index, col_off in enumerate(range(0, width, size)):
            yield (
                col_index,
                row_index,
                Window(col_off, row_off, min(size, width - col_off), min(size, height - row_off)),
            )
