"""The ``raster-batch`` command line interface.

Subcommands are thin: they validate arguments in the parent process, build a lazy
generator of :class:`~raster_batch.engine.Task` objects, and hand it to
:func:`~raster_batch.engine.run_batch`. Validation happens before any pool is spawned
so that a typo in an EPSG code costs milliseconds rather than the startup cost of
eight worker processes.

The batch flags (``--workers``, ``--on-error``, ``--resume`` …) are defined once as
reusable ``Annotated`` aliases and shared by every subcommand, so they behave and
document identically everywhere.

Background: `Structuring a Multi-Command GDAL CLI with Typer Sub-Apps
<https://www.batch-processing.com/cli-architecture-design-patterns/cli-subcommand-organization/structuring-a-multi-command-gdal-cli-with-typer-sub-apps/>`_.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, Any

import rasterio
import typer
from rich.console import Console
from rich.table import Table

from raster_batch.checkpoint import Checkpoint
from raster_batch.engine import BatchReport, OnError, Task, run_batch
from raster_batch.errors import EXIT_OK, EXIT_PARTIAL_FAILURE, EXIT_USAGE, ItemError, UsageError
from raster_batch.logging_setup import configure_logging
from raster_batch.ops import (
    ClipPayload,
    ConvertPayload,
    OutputFormat,
    ReprojectPayload,
    ResamplingName,
    TilePayload,
    clip_one,
    convert_one,
    describe,
    parse_bounds,
    parse_crs,
    reproject_one,
    tile_one,
)
from raster_batch.progress import make_reporter
from raster_batch.windows import DEFAULT_WINDOW_MB, iter_tile_windows

app = typer.Typer(
    name="raster-batch",
    help="Apply raster operations to many files in parallel, with resumable progress.",
    no_args_is_help=True,
    add_completion=True,
)

_out_console = Console()
_err_console = Console(stderr=True)
_log = logging.getLogger("raster_batch")

SrcArg = Annotated[
    list[Path],
    typer.Argument(help="Source rasters. Shell globs work: data/*.tif", show_default=False),
]
OutOpt = Annotated[
    Path,
    typer.Option("--out", "-o", help="Output directory (created if missing).", show_default=False),
]
WorkersOpt = Annotated[
    int | None,
    typer.Option("--workers", "-j", help="Worker processes. Default: min(cpu_count, 8)."),
]
OnErrorOpt = Annotated[
    OnError,
    typer.Option(
        "--on-error",
        help="stop: abort at first failure. skip: continue, count only. "
        "collect: continue and report every failure.",
    ),
]
InFlightOpt = Annotated[
    int | None,
    typer.Option(
        "--max-in-flight",
        help="Cap on simultaneously submitted tasks. Default: workers * 2.",
    ),
]
WindowOpt = Annotated[
    int,
    typer.Option("--window-mb", help="Per-worker read window budget, in MiB."),
]
CompressOpt = Annotated[
    str,
    typer.Option("--compress", help="GeoTIFF compression: deflate, lzw, zstd, none."),
]
CheckpointOpt = Annotated[
    Path | None,
    typer.Option("--checkpoint", help="JSONL manifest to write item outcomes to."),
]
ResumeOpt = Annotated[
    bool,
    typer.Option("--resume", help="Skip items already marked done in --checkpoint."),
]
ProgressOpt = Annotated[
    bool,
    typer.Option("--progress/--no-progress", help="Show the live progress display."),
]
LogJsonOpt = Annotated[
    bool,
    typer.Option("--log-json", help="Emit logs as one JSON object per line."),
]
OverwriteOpt = Annotated[
    bool,
    typer.Option("--overwrite", help="Replace existing output files."),
]


def _fail(message: str) -> typer.Exit:
    """Print a usage error to stderr and return the exit to raise."""
    _err_console.print(f"error: {message}", style="red")
    return typer.Exit(EXIT_USAGE)


def _validate_sources(sources: Sequence[Path]) -> list[Path]:
    """Ensure every source exists before starting a pool.

    Raises:
        UsageError: If the list is empty or any path is missing.
    """
    if not sources:
        raise UsageError("No source rasters given.")
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise UsageError(f"Source raster(s) not found: {', '.join(missing)}")
    return list(sources)


def _prepare_out_dir(out: Path) -> Path:
    """Create the output directory, or explain why we cannot.

    Raises:
        UsageError: If the path exists as a file or cannot be created.
    """
    if out.exists() and not out.is_dir():
        raise UsageError(f"Output path is not a directory: {out}")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UsageError(f"Cannot create output directory {out}: {exc}") from exc
    return out


def _finish(report: BatchReport, on_error: OnError) -> None:
    """Print any collected failures and exit with the right code.

    Raises:
        typer.Exit: Always — 0 when clean, 2 when any item failed.
    """
    if report.failures and on_error is not OnError.SKIP:
        table = Table(title="Failures", show_lines=False)
        table.add_column("item", overflow="fold")
        table.add_column("error")
        table.add_column("digest", style="dim")
        for failure in report.failures[:50]:
            table.add_row(
                failure.key,
                f"{failure.error_type}: {failure.error_message}",
                failure.traceback_digest,
            )
        _err_console.print(table)
    _log.info(
        "batch finished",
        extra={
            "succeeded": report.succeeded,
            "failed": report.failed,
            "skipped": report.skipped,
            "elapsed_s": round(report.elapsed_s, 3),
        },
    )
    raise typer.Exit(EXIT_OK if report.ok else EXIT_PARTIAL_FAILURE)


def _run(
    tasks: Iterator[Task],
    worker: Any,
    *,
    total: int | None,
    workers: int | None,
    on_error: OnError,
    max_in_flight: int | None,
    checkpoint_path: Path | None,
    resume: bool,
    progress: bool,
    log_json: bool,
) -> None:
    """Shared driver for every batch subcommand.

    Raises:
        typer.Exit: With the process exit code for the run.
    """
    configure_logging(json_output=log_json)
    if resume and checkpoint_path is None:
        raise _fail("--resume needs --checkpoint to know what was already done")
    checkpoint = Checkpoint(checkpoint_path, resume=resume) if checkpoint_path is not None else None
    reporter = make_reporter(_err_console, enabled=progress)
    try:
        report = run_batch(
            tasks,
            worker,
            workers=workers,
            on_error=on_error,
            max_in_flight=max_in_flight,
            checkpoint=checkpoint,
            reporter=reporter,
            total=total,
        )
    finally:
        if checkpoint is not None:
            checkpoint.close()
    _finish(report, on_error)


def _dst_for(src: Path, out: Path, suffix: str = ".tif") -> Path:
    """Map a source file to its output path inside ``out``."""
    return out / (src.stem + suffix)


@app.command()
def reproject(
    src: SrcArg,
    dst_crs: Annotated[
        str,
        typer.Option("--dst-crs", help="Target CRS, e.g. EPSG:3857.", show_default=False),
    ],
    out: OutOpt,
    resampling: Annotated[
        ResamplingName, typer.Option("--resampling", help="Resampling algorithm.")
    ] = ResamplingName.NEAREST,
    workers: WorkersOpt = None,
    on_error: OnErrorOpt = OnError.SKIP,
    max_in_flight: InFlightOpt = None,
    window_mb: WindowOpt = DEFAULT_WINDOW_MB,
    compress: CompressOpt = "deflate",
    checkpoint: CheckpointOpt = None,
    resume: ResumeOpt = False,
    progress: ProgressOpt = True,
    log_json: LogJsonOpt = False,
    overwrite: OverwriteOpt = False,
) -> None:
    """Warp rasters into a target CRS, one process per file."""
    try:
        sources = _validate_sources(src)
        out_dir = _prepare_out_dir(out)
        crs = parse_crs(dst_crs)
    except UsageError as exc:
        raise _fail(str(exc)) from exc
    tasks = (
        Task(
            key=str(path),
            payload=ReprojectPayload(
                src=str(path),
                dst=str(_dst_for(path, out_dir)),
                dst_crs=crs.to_string(),
                resampling=resampling,
                window_mb=window_mb,
                compress=compress,
                overwrite=overwrite,
            ),
        )
        for path in sources
    )
    _run(
        tasks,
        reproject_one,
        total=len(sources),
        workers=workers,
        on_error=on_error,
        max_in_flight=max_in_flight,
        checkpoint_path=checkpoint,
        resume=resume,
        progress=progress,
        log_json=log_json,
    )


@app.command()
def clip(
    src: SrcArg,
    bounds: Annotated[
        str,
        typer.Option("--bounds", help="minx,miny,maxx,maxy", show_default=False),
    ],
    out: OutOpt,
    bounds_crs: Annotated[
        str | None,
        typer.Option("--bounds-crs", help="CRS of --bounds; defaults to each raster's own CRS."),
    ] = None,
    workers: WorkersOpt = None,
    on_error: OnErrorOpt = OnError.SKIP,
    max_in_flight: InFlightOpt = None,
    window_mb: WindowOpt = DEFAULT_WINDOW_MB,
    compress: CompressOpt = "deflate",
    checkpoint: CheckpointOpt = None,
    resume: ResumeOpt = False,
    progress: ProgressOpt = True,
    log_json: LogJsonOpt = False,
    overwrite: OverwriteOpt = False,
) -> None:
    """Clip rasters to a bounding box, reading only the intersecting window."""
    try:
        sources = _validate_sources(src)
        out_dir = _prepare_out_dir(out)
        box = parse_bounds(bounds)
        crs_text = parse_crs(bounds_crs).to_string() if bounds_crs else None
    except UsageError as exc:
        raise _fail(str(exc)) from exc
    tasks = (
        Task(
            key=str(path),
            payload=ClipPayload(
                src=str(path),
                dst=str(_dst_for(path, out_dir)),
                bounds=box,
                bounds_crs=crs_text,
                window_mb=window_mb,
                compress=compress,
                overwrite=overwrite,
            ),
        )
        for path in sources
    )
    _run(
        tasks,
        clip_one,
        total=len(sources),
        workers=workers,
        on_error=on_error,
        max_in_flight=max_in_flight,
        checkpoint_path=checkpoint,
        resume=resume,
        progress=progress,
        log_json=log_json,
    )


@app.command()
def tile(
    src: Annotated[Path, typer.Argument(help="Raster to split.", show_default=False)],
    out: OutOpt,
    size: Annotated[int, typer.Option("--size", help="Tile edge length in pixels.")] = 512,
    workers: WorkersOpt = None,
    on_error: OnErrorOpt = OnError.SKIP,
    max_in_flight: InFlightOpt = None,
    compress: CompressOpt = "deflate",
    checkpoint: CheckpointOpt = None,
    resume: ResumeOpt = False,
    progress: ProgressOpt = True,
    log_json: LogJsonOpt = False,
    overwrite: OverwriteOpt = False,
) -> None:
    """Split one raster into a grid of tiles, each with its own transform."""
    try:
        _validate_sources([src])
        out_dir = _prepare_out_dir(out)
        if size < 1:
            raise UsageError("--size must be at least 1 pixel")
        with rasterio.open(src) as dataset:
            width, height = int(dataset.width), int(dataset.height)
    except UsageError as exc:
        raise _fail(str(exc)) from exc
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise _fail(f"Cannot open {src} as a raster: {exc}") from exc

    grid = list(iter_tile_windows(width, height, size))
    tasks = (
        Task(
            key=f"{src}#{col}_{row}",
            payload=TilePayload(
                src=str(src),
                dst=str(out_dir / f"{src.stem}_{row:04d}_{col:04d}.tif"),
                col_off=int(window.col_off),
                row_off=int(window.row_off),
                width=int(window.width),
                height=int(window.height),
                compress=compress,
                overwrite=overwrite,
            ),
        )
        for col, row, window in grid
    )
    _run(
        tasks,
        tile_one,
        total=len(grid),
        workers=workers,
        on_error=on_error,
        max_in_flight=max_in_flight,
        checkpoint_path=checkpoint,
        resume=resume,
        progress=progress,
        log_json=log_json,
    )


@app.command()
def convert(
    src: SrcArg,
    out: OutOpt,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output driver.")
    ] = OutputFormat.GTIFF,
    compress: CompressOpt = "deflate",
    blocksize: Annotated[int, typer.Option("--blocksize", help="Internal tile size.")] = 512,
    workers: WorkersOpt = None,
    on_error: OnErrorOpt = OnError.SKIP,
    max_in_flight: InFlightOpt = None,
    window_mb: WindowOpt = DEFAULT_WINDOW_MB,
    checkpoint: CheckpointOpt = None,
    resume: ResumeOpt = False,
    progress: ProgressOpt = True,
    log_json: LogJsonOpt = False,
    overwrite: OverwriteOpt = False,
) -> None:
    """Rewrite rasters with a different driver, compression or block layout."""
    try:
        sources = _validate_sources(src)
        out_dir = _prepare_out_dir(out)
    except UsageError as exc:
        raise _fail(str(exc)) from exc
    tasks = (
        Task(
            key=str(path),
            payload=ConvertPayload(
                src=str(path),
                dst=str(_dst_for(path, out_dir)),
                driver=output_format,
                compress=compress,
                window_mb=window_mb,
                blocksize=blocksize,
                overwrite=overwrite,
            ),
        )
        for path in sources
    )
    _run(
        tasks,
        convert_one,
        total=len(sources),
        workers=workers,
        on_error=on_error,
        max_in_flight=max_in_flight,
        checkpoint_path=checkpoint,
        resume=resume,
        progress=progress,
        log_json=log_json,
    )


@app.command()
def info(src: SrcArg) -> None:
    """Print a table of CRS, size, dtype, band count and nodata for each raster.

    Reads headers only, in the parent process: there is no per-file work worth the
    cost of a process pool here.
    """
    table = Table(title="Raster info")
    for column in ("file", "driver", "crs", "size", "bands", "dtype", "nodata", "resolution"):
        table.add_column(column, overflow="fold")
    failures = 0
    for path in src:
        try:
            summary = describe(str(path))
        except ItemError as exc:
            failures += 1
            _err_console.print(f"error: {exc}", style="red")
            continue
        table.add_row(
            Path(summary.path).name,
            summary.driver,
            summary.crs,
            f"{summary.width}x{summary.height}",
            str(summary.count),
            summary.dtype,
            "none" if summary.nodata is None else f"{summary.nodata:g}",
            f"{summary.resolution[0]:g}, {summary.resolution[1]:g}",
        )
    if table.row_count:
        _out_console.print(table)
    if failures:
        raise typer.Exit(EXIT_PARTIAL_FAILURE if table.row_count else EXIT_USAGE)


def main() -> None:  # pragma: no cover - exercised via the console script
    """Console-script entry point."""
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
