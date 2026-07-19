"""Shared fixtures: synthetic GeoTIFFs written into ``tmp_path``.

Nothing here touches the network or a checked-in binary. Every raster is a few
kilobytes of generated data, which keeps the multiprocessing tests fast enough to run
on every commit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

RasterFactory = Callable[..., Path]


def _write_raster(
    path: Path,
    *,
    crs: str = "EPSG:4326",
    width: int = 64,
    height: int = 48,
    count: int = 1,
    dtype: str = "uint8",
    nodata: float | None = None,
    origin: tuple[float, float] = (-120.0, 38.0),
    res: float = 0.001,
) -> Path:
    """Write one deterministic synthetic raster and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(origin[0], origin[1], res, res)
    rng = np.random.default_rng(seed=abs(hash(path.name)) % (2**32))
    if np.issubdtype(np.dtype(dtype), np.integer):
        data = rng.integers(0, 200, size=(count, height, width)).astype(dtype)
    else:
        data = rng.random(size=(count, height, width)).astype(dtype)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data)
    return path


@pytest.fixture
def make_raster(tmp_path: Path) -> RasterFactory:
    """Return a factory writing synthetic GeoTIFFs under ``tmp_path``."""

    def factory(name: str = "raster.tif", **kwargs: object) -> Path:
        return _write_raster(tmp_path / name, **kwargs)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def wgs84_raster(make_raster: RasterFactory) -> Path:
    """A single-band uint8 EPSG:4326 raster with a nodata value."""
    return make_raster("wgs84.tif", crs="EPSG:4326", nodata=0)


@pytest.fixture
def utm_raster(make_raster: RasterFactory) -> Path:
    """A three-band float32 UTM raster (EPSG:32610) in metre units."""
    return make_raster(
        "utm.tif",
        crs="EPSG:32610",
        count=3,
        dtype="float32",
        origin=(500000.0, 4200000.0),
        res=30.0,
    )


@pytest.fixture
def raster_dir(make_raster: RasterFactory, tmp_path: Path) -> Path:
    """A directory of five small EPSG:4326 rasters."""
    for index in range(5):
        make_raster(f"batch/scene{index}.tif", crs="EPSG:4326")
    return tmp_path / "batch"


@pytest.fixture
def corrupt_raster(tmp_path: Path) -> Path:
    """A file with a .tif extension that is not a raster at all."""
    path = tmp_path / "corrupt.tif"
    path.write_bytes(b"II*\x00this is not a tiff at all, really")
    return path
