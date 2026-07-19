"""The raster operations themselves: reproject, clip, tile, convert and describe.

Each operation is split into a picklable payload dataclass and a module-level worker
function taking that payload. That shape is not stylistic — ``ProcessPoolExecutor``
pickles both the callable and its argument, so closures, lambdas and bound methods
cannot cross the process boundary. Keeping the payloads to plain strings, ints and
tuples also means a task is cheap to send and can be serialised into a checkpoint
manifest unchanged.

Every worker streams through :mod:`raster_batch.windows` instead of reading whole
arrays, so a single task's memory is bounded by its window budget regardless of how
large the source raster is.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import rasterio
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from raster_batch.errors import ItemError, UsageError
from raster_batch.windows import DEFAULT_WINDOW_MB, iter_row_windows

__all__ = [
    "ClipPayload",
    "ConvertPayload",
    "OutputFormat",
    "RasterInfo",
    "ReprojectPayload",
    "ResamplingName",
    "TilePayload",
    "clip_one",
    "convert_one",
    "describe",
    "parse_bounds",
    "parse_crs",
    "reproject_one",
    "tile_one",
]


class ResamplingName(StrEnum):
    """Resampling algorithms exposed on the CLI.

    A deliberately short list: these six cover categorical data (``nearest``,
    ``mode``), continuous data (``bilinear``, ``cubic``, ``lanczos``) and downsampling
    (``average``). Exposing all of GDAL's twelve invites people to pick ``lanczos``
    for a land-cover class raster and quietly invent categories that do not exist.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    AVERAGE = "average"
    LANCZOS = "lanczos"
    MODE = "mode"

    def to_rasterio(self) -> Resampling:
        """Map to the rasterio enum member of the same name."""
        return Resampling[self.value]


class OutputFormat(StrEnum):
    """Supported output drivers."""

    GTIFF = "GTiff"
    COG = "COG"


@dataclass(frozen=True, slots=True)
class ReprojectPayload:
    """Everything one reprojection needs, picklable and self-contained."""

    src: str
    dst: str
    dst_crs: str
    resampling: ResamplingName = ResamplingName.NEAREST
    window_mb: int = DEFAULT_WINDOW_MB
    compress: str = "deflate"
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class ClipPayload:
    """One clip: a source, a destination and bounds in a stated CRS."""

    src: str
    dst: str
    bounds: tuple[float, float, float, float]
    bounds_crs: str | None = None
    window_mb: int = DEFAULT_WINDOW_MB
    compress: str = "deflate"
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class TilePayload:
    """One output tile, addressed by its grid position and pixel window."""

    src: str
    dst: str
    col_off: int
    row_off: int
    width: int
    height: int
    compress: str = "deflate"
    overwrite: bool = False

    def window(self) -> Window:
        """Rebuild the rasterio window from the stored integers."""
        return Window(self.col_off, self.row_off, self.width, self.height)


@dataclass(frozen=True, slots=True)
class ConvertPayload:
    """One format/profile conversion."""

    src: str
    dst: str
    driver: OutputFormat = OutputFormat.GTIFF
    compress: str = "deflate"
    window_mb: int = DEFAULT_WINDOW_MB
    blocksize: int = 512
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class RasterInfo:
    """A flat, printable summary of a raster's header."""

    path: str
    driver: str
    crs: str
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | None
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]


def parse_crs(value: str) -> CRS:
    """Parse a CRS string, raising a usage error with a readable message.

    Accepts anything rasterio accepts (``EPSG:3857``, a PROJ string, WKT). Bad codes
    are caught here, in the parent process, so a 100k-file job fails in a tenth of a
    second instead of after spawning a pool.

    Raises:
        UsageError: If the value is not a CRS rasterio can construct.
    """
    try:
        return CRS.from_user_input(value)
    except (RasterioError, ValueError, TypeError) as exc:
        raise UsageError(f"Not a usable CRS: {value!r} ({exc})") from exc


def parse_bounds(value: str) -> tuple[float, float, float, float]:
    """Parse ``minx,miny,maxx,maxy`` into a validated tuple.

    Raises:
        UsageError: If there are not four numbers, or the box is empty/inverted.
    """
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise UsageError(f"Bounds need four comma-separated numbers, got {value!r}")
    try:
        minx, miny, maxx, maxy = (float(part) for part in parts)
    except ValueError as exc:
        raise UsageError(f"Bounds must be numbers, got {value!r}") from exc
    if minx >= maxx or miny >= maxy:
        raise UsageError(f"Bounds must satisfy minx<maxx and miny<maxy, got {value!r}")
    return (minx, miny, maxx, maxy)


def _budget_bytes(window_mb: int) -> int:
    return max(1, window_mb) * 1024 * 1024


def _prepare_destination(dst: str, overwrite: bool) -> Path:
    """Validate the destination path and make its parent directory.

    Raises:
        ItemError: If the file exists and ``overwrite`` is False, or the directory
            cannot be created.
    """
    path = Path(dst)
    if path.exists() and not overwrite:
        raise ItemError(f"{path} already exists (pass --overwrite to replace it)")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ItemError(f"Cannot create output directory {path.parent}: {exc}") from exc
    return path


def _clamp_window(window: Window, width: int, height: int) -> Window:
    """Intersect a window with the raster extent, returning an empty window if disjoint.

    ``Window.intersection`` raises when there is no overlap, which would surface as an
    opaque rasterio error; a clip box that misses the raster deserves its own message.
    """
    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    col_end = min(width, int(window.col_off) + int(window.width))
    row_end = min(height, int(window.row_off) + int(window.height))
    return Window(col_off, row_off, max(0, col_end - col_off), max(0, row_end - row_off))


def _open_source(src: str) -> Any:
    """Open a raster, turning rasterio's errors into an :class:`ItemError`.

    Raises:
        ItemError: If the file is missing, unreadable or not a raster.
    """
    if not Path(src).exists():
        raise ItemError(f"Source raster does not exist: {src}")
    try:
        return rasterio.open(src)
    except (RasterioError, OSError) as exc:
        raise ItemError(f"Cannot open {src} as a raster: {exc}") from exc


def _gtiff_profile(base: dict[str, Any], compress: str, blocksize: int = 512) -> dict[str, Any]:
    """Return a tiled, compressed GeoTIFF creation profile derived from ``base``."""
    profile = dict(base)
    profile.update(
        driver="GTiff",
        compress=compress,
        tiled=True,
        blockxsize=blocksize,
        blockysize=blocksize,
        BIGTIFF="IF_SAFER",
    )
    for key in ("blockysize", "blockxsize"):
        if profile[key] % 16:
            profile[key] = 512
    return profile


def _copy_windowed(source: Any, dst_path: Path, profile: dict[str, Any], budget: int) -> None:
    """Stream ``source`` into a new dataset one row-stripe at a time."""
    with rasterio.open(dst_path, "w", **profile) as dst_ds:
        for window in iter_row_windows(
            source.width, source.height, source.count, source.dtypes[0], budget
        ):
            dst_ds.write(source.read(window=window), window=window)


def reproject_one(payload: ReprojectPayload) -> str:
    """Warp one raster into ``dst_crs``, streaming windows through a ``WarpedVRT``.

    A ``WarpedVRT`` is used rather than ``rasterio.warp.reproject`` on full arrays
    because the VRT computes the destination grid up front and then warps lazily per
    window — so the memory cost is one window, not one whole reprojected image. The
    destination grid (transform, width, height) is GDAL's default for the target CRS,
    which is what ``gdalwarp`` would also choose.

    Returns:
        A short human-readable description of what was written.

    Raises:
        ItemError: If the source cannot be read, the target CRS cannot be applied to
            it, or the destination cannot be written.
    """
    dst_path = _prepare_destination(payload.dst, payload.overwrite)
    budget = _budget_bytes(payload.window_mb)
    try:
        with _open_source(payload.src) as src:
            if src.crs is None:
                raise ItemError(f"{payload.src} has no CRS; cannot reproject it")
            with WarpedVRT(
                src,
                crs=CRS.from_user_input(payload.dst_crs),
                resampling=payload.resampling.to_rasterio(),
            ) as vrt:
                profile = _gtiff_profile(vrt.profile, payload.compress)
                _copy_windowed(vrt, dst_path, profile, budget)
                return f"{vrt.width}x{vrt.height} -> {payload.dst_crs}"
    except ItemError:
        raise
    except (RasterioError, OSError, ValueError) as exc:
        raise ItemError(f"Reprojecting {payload.src} failed: {exc}") from exc


def clip_one(payload: ClipPayload) -> str:
    """Clip one raster to ``bounds``, reading only the intersecting window.

    The bounds are transformed into the source CRS first when ``bounds_crs`` differs,
    so you can clip a UTM scene with WGS84 coordinates without pre-converting. The
    window is rounded outward to whole pixels: rounding inward would shave a partial
    pixel off each edge and slowly erode a raster across repeated clips.

    Returns:
        The clipped size and the realised bounds.

    Raises:
        ItemError: If the clip box does not intersect the raster, or I/O fails.
    """
    dst_path = _prepare_destination(payload.dst, payload.overwrite)
    budget = _budget_bytes(payload.window_mb)
    try:
        with _open_source(payload.src) as src:
            bounds = payload.bounds
            if payload.bounds_crs is not None and src.crs is not None:
                src_bounds_crs = CRS.from_user_input(payload.bounds_crs)
                if src_bounds_crs != src.crs:
                    bounds = transform_bounds(src_bounds_crs, src.crs, *bounds)
            requested = from_bounds(*bounds, transform=src.transform)
            requested = requested.round_offsets(op="floor").round_lengths(op="ceil")
            window = _clamp_window(requested, src.width, src.height)
            if window.width < 1 or window.height < 1:
                raise ItemError(f"Clip bounds do not intersect {payload.src}")
            profile = _gtiff_profile(src.profile, payload.compress)
            profile.update(
                width=int(window.width),
                height=int(window.height),
                transform=src.window_transform(window),
            )
            with rasterio.open(dst_path, "w", **profile) as dst_ds:
                for stripe in iter_row_windows(
                    int(window.width),
                    int(window.height),
                    src.count,
                    src.dtypes[0],
                    budget,
                ):
                    read_window = Window(
                        window.col_off + stripe.col_off,
                        window.row_off + stripe.row_off,
                        stripe.width,
                        stripe.height,
                    )
                    dst_ds.write(src.read(window=read_window), window=stripe)
            return f"{int(window.width)}x{int(window.height)} from {Path(payload.src).name}"
    except ItemError:
        raise
    except (RasterioError, OSError, ValueError) as exc:
        raise ItemError(f"Clipping {payload.src} failed: {exc}") from exc


def tile_one(payload: TilePayload) -> str:
    """Write one tile of a source raster with its own correct geotransform.

    Each tile gets ``src.window_transform(window)``, which shifts the origin by the
    tile's pixel offset while keeping the pixel size. Copying the parent transform
    unchanged — a common bug — georeferences every tile to the same top-left corner.

    Returns:
        The tile's pixel size.

    Raises:
        ItemError: If the source cannot be read or the tile cannot be written.
    """
    dst_path = _prepare_destination(payload.dst, payload.overwrite)
    window = payload.window()
    try:
        with _open_source(payload.src) as src:
            profile = _gtiff_profile(src.profile, payload.compress, blocksize=256)
            profile.update(
                width=payload.width,
                height=payload.height,
                transform=src.window_transform(window),
            )
            with rasterio.open(dst_path, "w", **profile) as dst_ds:
                dst_ds.write(src.read(window=window))
            return f"tile {payload.width}x{payload.height} at ({payload.col_off},{payload.row_off})"
    except ItemError:
        raise
    except (RasterioError, OSError, ValueError) as exc:
        raise ItemError(f"Tiling {payload.src} failed: {exc}") from exc


def convert_one(payload: ConvertPayload) -> str:
    """Rewrite a raster with a different driver, compression or block layout.

    ``COG`` is produced in two steps: a windowed copy to a temporary tiled GeoTIFF,
    then a driver-level copy that builds the overviews and lays out the IFDs in
    cloud-optimised order. The COG driver wants to see the whole image to do that, so
    handing it a temporary file is how the operation stays windowed on *our* side
    rather than materialising the array in Python.

    Returns:
        The driver and compression that were written.

    Raises:
        ItemError: If the source cannot be read or the destination cannot be written.
    """
    dst_path = _prepare_destination(payload.dst, payload.overwrite)
    budget = _budget_bytes(payload.window_mb)
    try:
        with _open_source(payload.src) as src:
            profile = _gtiff_profile(src.profile, payload.compress, payload.blocksize)
            if payload.driver is OutputFormat.GTIFF:
                _copy_windowed(src, dst_path, profile, budget)
            else:
                with tempfile.TemporaryDirectory(dir=str(dst_path.parent)) as tmp:
                    staged = Path(tmp) / "staged.tif"
                    _copy_windowed(src, staged, profile, budget)
                    rasterio.shutil.copy(
                        str(staged),
                        str(dst_path),
                        driver="COG",
                        compress=payload.compress,
                        blocksize=payload.blocksize,
                        overwrite=True,
                    )
            return f"{payload.driver.value} compress={payload.compress}"
    except ItemError:
        raise
    except (RasterioError, OSError, ValueError) as exc:
        raise ItemError(f"Converting {payload.src} failed: {exc}") from exc


def describe(src: str) -> RasterInfo:
    """Read a raster's header into a :class:`RasterInfo`.

    Header-only: no pixels are touched, so this is fast even on remote or very large
    datasets.

    Raises:
        ItemError: If the file cannot be opened as a raster.
    """
    with _open_source(src) as dataset:
        return RasterInfo(
            path=src,
            driver=str(dataset.driver),
            crs=dataset.crs.to_string() if dataset.crs else "(none)",
            width=int(dataset.width),
            height=int(dataset.height),
            count=int(dataset.count),
            dtype=str(dataset.dtypes[0]),
            nodata=dataset.nodata,
            bounds=tuple(float(value) for value in dataset.bounds),  # type: ignore[arg-type]
            resolution=(float(dataset.res[0]), float(dataset.res[1])),
        )
