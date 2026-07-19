<h1>raster-batch</h1>

A command-line tool that applies a raster operation — reproject, clip, tile, convert —
to thousands of files in parallel, with bounded memory, resumable checkpoints and a
live progress display.

Documentation and background articles: [batch-processing.com](https://www.batch-processing.com).

## The problem

Batch raster work usually starts as a shell loop around `gdalwarp` and stays there
until it hurts. The loop is serial, so it wastes seven of your eight cores. It has no
memory ceiling, so one unexpectedly large scene takes the machine down. It has no
memory of what it already did, so an interrupted run at hour six starts from zero. And
because a single bad file aborts the loop, one corrupt GeoTIFF in forty thousand means
babysitting the job.

The obvious Python rewrite tends to reproduce two of those problems and add a third:

```python
files = list(Path("data").glob("*.tif"))
with ProcessPoolExecutor() as pool:
    futures = [pool.submit(warp, f) for f in files]   # 100k live futures
    for fut in as_completed(futures):
        fut.result()                                  # first failure kills the batch
```

`raster-batch` is what that script looks like once the operational problems are taken
seriously: a bounded submission window, per-item error isolation, windowed I/O, a
JSONL checkpoint, and a dead-letter file you can re-drive.

## Install

Not published to an index — clone and run it:

```bash
git clone https://github.com/batch-processing-geospatial-cli-tools/raster-batch.git
cd raster-batch
uv sync
uv run raster-batch --help
```

Or without `uv`:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/raster-batch --help
```

Python 3.11, 3.12 and 3.13 are supported. Every dependency (`rasterio`, `numpy`,
`pyproj`, `typer`, `rich`) ships manylinux wheels, so there is no GDAL source build.

## Usage

### Reproject a directory

```bash
uv run raster-batch reproject data/*.tif \
    --dst-crs EPSG:3857 \
    --resampling bilinear \
    --out out/webmerc \
    --workers 8
```

```
⠙ processing ━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━ 2841/4096 21.4/s 0:02:12 0:00:58
  working data/n38w120_0184.tif
  working data/n38w120_0185.tif
  working data/n38w120_0186.tif
  3 failed so far
```

### Clip everything to a box in a different CRS

```bash
uv run raster-batch clip data/*.tif \
    --bounds -122.5,37.7,-122.3,37.9 \
    --bounds-crs EPSG:4326 \
    --out out/sf
```

The bounds are transformed into each raster's own CRS before the window is computed,
so a folder of mixed-UTM-zone scenes can be clipped with one WGS84 box.

### Split one large raster into tiles

```bash
uv run raster-batch tile mosaic.tif --size 512 --out out/tiles
```

Output files are named `mosaic_<row>_<col>.tif`, and each tile carries its own
geotransform — the origin is shifted by the tile's pixel offset, not copied from the
parent.

### Convert to Cloud-Optimized GeoTIFF

```bash
uv run raster-batch convert data/*.tif --format COG --compress deflate --out out/cog
```

### Inspect headers

```bash
uv run raster-batch info out/cog/*.tif
```

```
                                 Raster info
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ file       ┃ driver ┃ crs       ┃ size      ┃ bands ┃ dtype ┃ nodata ┃ resolution ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ scene1.tif │ GTiff  │ EPSG:4326 │ 3601x3601 │ 1     │ int16 │ -32768 │ 0.000278…  │
│ scene2.tif │ GTiff  │ EPSG:326… │ 4096x4096 │ 3     │ uint8 │ none   │ 30, 30     │
└────────────┴────────┴───────────┴───────────┴───────┴───────┴────────┴────────────┘
```

### Resume an interrupted run

```bash
uv run raster-batch reproject data/*.tif --dst-crs EPSG:3857 --out out \
    --checkpoint state/run.jsonl

# ... killed at item 38000 ...

uv run raster-batch reproject data/*.tif --dst-crs EPSG:3857 --out out \
    --checkpoint state/run.jsonl --resume
```

## How it works

### Streaming submission, not a list of futures

`run_batch` in `raster_batch/engine.py` keeps a bounded set of in-flight futures and
pulls from the task iterator only when a slot frees up:

```python
while not exhausted and len(in_flight) < max_in_flight:
    task = next(source, None)
    ...
done, in_flight = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
```

`max_in_flight` defaults to `workers * 2` — enough to keep every worker fed with one
queued item, and nothing more. Because the task source is a generator, a 100,000-file
job never materialises 100,000 task payloads, 100,000 `Future` objects, or 100,000
results. The engine's memory is flat in the size of the job.

The engine is deliberately raster-free. It takes `Task(key, payload)` objects and a
picklable worker function; everything geospatial lives in `raster_batch/ops.py`. That
separation is what lets the concurrency tests run against a three-line pure function
instead of against GDAL.

### The memory arithmetic

No operation calls `dataset.read()` on a whole raster. Everything moves through
row-stripe windows sized from a byte budget (`--window-mb`, default 64):

```
rows_per_window = window_bytes // (width × bands × dtype_itemsize)
peak RSS ≈ workers × window_bytes × 2  +  workers × GDAL_overhead
```

The ×2 is the source window and the destination window held at the same time; the GDAL
overhead is roughly 30–60 MB per process, including the block cache. So the defaults —
8 workers, 64 MiB windows — land at about 1.4 GB peak, and it stays there whether the
inputs are 50 MB scenes or 40 GB mosaics.

The important consequence is that `--workers` is a memory knob just as much as
`--window-mb` is. Doubling worker count on a memory-tight machine doubles peak RSS.
If a job is being OOM-killed, halve one of the two.

### Failure policy

`--on-error` picks what happens when an item raises:

| Policy    | Remaining work | Failure records | Exit code |
|-----------|----------------|-----------------|-----------|
| `stop`    | cancelled      | kept in memory  | 2         |
| `skip`    | continues      | counted only    | 2         |
| `collect` | continues      | kept in memory  | 2         |

`skip` is the default precisely because `collect` is unbounded: on a job where a large
fraction of items fail, retaining every failure record is how a "robust" pipeline runs
the parent process out of memory. Either way the full record — exception type, message
and a 12-character digest of the traceback — is appended to the dead-letter file, so
nothing is actually lost by choosing the cheap policy.

Workers never let an exception escape. `execute_task` catches everything except
`KeyboardInterrupt`/`SystemExit` and converts it into an `ItemResult`, because an
exception raised out of a child process surfaces as `BrokenProcessPool` and takes the
whole batch with it. GDAL-backed exceptions are also not reliably picklable, which is
the second reason failures cross the pipe as text.

### Checkpoints and the dead-letter file

The checkpoint is JSON Lines, appended and flushed per item. That format is chosen for
one property: it survives `kill -9`. A truncated final line is discarded on read and
every complete line before it is still valid, so there is no repair step.

```jsonl
{"key": "data/a.tif", "status": "done", "detail": "4096x4096 -> EPSG:3857", "duration_s": 1.83}
{"key": "data/b.tif", "status": "failed", "error_type": "ItemError", "error_message": "...", "traceback_digest": "9f31c0aa41de"}
```

Failures are written to the manifest *and* to `<manifest>.failed.jsonl`. They are
recorded in the manifest as `failed`, and `--resume` only skips `done` entries — so a
resumed run retries them. Marking them done would hide real errors; not recording them
at all would lose the audit trail.

### Progress output

On a TTY you get a Rich live display: a bar with ETA and throughput, plus a capped tail
of the items currently in flight. When stdout is not a terminal — a CI log, a pipe, a
`nohup` — the reporter degrades to one plain line per completed item on stderr, because
ANSI cursor movement written into a log file produces megabytes of escape sequences.
`--no-progress` silences it entirely, and `--log-json` switches the log stream to one
JSON object per line for ingestion.

### Exit codes

| Code | Meaning |
|------|---------|
| 0    | every item succeeded |
| 1    | usage or configuration error; no work was attempted |
| 2    | the batch ran, but at least one item failed |

The distinction between 1 and 2 matters in a pipeline: "you called me wrong" and "3 of
40,000 rasters were corrupt" call for very different responses, and a single non-zero
code cannot express both.

## Command reference

Shared batch options, available on `reproject`, `clip`, `tile` and `convert`:

| Option | Default | Meaning |
|--------|---------|---------|
| `--workers`, `-j` | `min(cpu_count, 8)` | Worker processes. `1` runs in-process. |
| `--max-in-flight` | `workers × 2` | Cap on simultaneously submitted tasks. |
| `--window-mb` | `64` | Per-worker read window budget in MiB. |
| `--on-error` | `skip` | `stop`, `skip` or `collect`. |
| `--checkpoint` | none | JSONL manifest path. |
| `--resume` | off | Skip items already marked done. Requires `--checkpoint`. |
| `--progress/--no-progress` | on | Live display. |
| `--log-json` | off | Structured log output. |
| `--compress` | `deflate` | GeoTIFF compression. |
| `--overwrite` | off | Replace existing outputs. |

Command-specific options:

| Command | Option | Meaning |
|---------|--------|---------|
| `reproject` | `--dst-crs` | Target CRS, e.g. `EPSG:3857`. Required. |
| `reproject` | `--resampling` | `nearest`, `bilinear`, `cubic`, `average`, `lanczos`, `mode`. |
| `clip` | `--bounds` | `minx,miny,maxx,maxy`. Required. |
| `clip` | `--bounds-crs` | CRS of the box; defaults to each raster's own. |
| `tile` | `--size` | Tile edge in pixels. Default 512. |
| `convert` | `--format` | `GTiff` or `COG`. |
| `convert` | `--blocksize` | Internal tile size. Default 512. |

### Using the engine directly

The batch engine works on anything, not just rasters:

```python
from raster_batch import OnError, Task, run_batch

def work(payload: str) -> str:
    return payload.upper()

report = run_batch(
    (Task(key=name, payload=name) for name in names),
    work,
    workers=4,
    on_error=OnError.COLLECT,
)
print(report.succeeded, report.failed)
```

The worker must be importable in a child process — a module-level function, not a
lambda or a closure. Results are not ordered; sort by key at the call site if you need
determinism.

## Development

```bash
uv sync
uv run pytest                       # test suite, no network, fixtures in tmp_path
uv run pytest --cov=raster_batch    # with coverage
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The test suite generates every fixture raster with rasterio inside `tmp_path` — a few
CRSs, nodata values, multiband, `uint8` and `float32`. Multiprocessing tests use two
workers and rasters small enough that the whole suite stays fast.

## Further reading

Longer write-ups of the design decisions above:

- [Choosing Chunk Size for Multiprocessing Raster Warps](https://www.batch-processing.com/spatial-batch-processing-async-workflows/multiprocessing-geospatial-tasks/choosing-chunk-size-for-multiprocessing-raster-warps/) — how `--workers` and window size interact.
- [Streaming Raster Windows to Cap Memory in Mosaics](https://www.batch-processing.com/spatial-batch-processing-async-workflows/memory-management-for-large-datasets/streaming-raster-windows-to-cap-memory-in-mosaics/) — the windowed I/O approach used by every operation here.
- [Checkpointing for Interrupted Spatial Batch Jobs](https://www.batch-processing.com/spatial-batch-processing-async-workflows/progress-tracking-in-batch-jobs/implementing-checkpointing-for-interrupted-spatial-batches/) — why the manifest is append-only JSONL.
- [Building a Dead-Letter Queue for Failed Geometry Transforms](https://www.batch-processing.com/spatial-batch-processing-async-workflows/error-handling-in-spatial-pipelines/building-a-dead-letter-queue-for-failed-geometry-transforms/) — the failed-item pattern behind `--on-error`.
- [Detecting Non-TTY Output and Disabling Rich Color](https://www.batch-processing.com/cli-architecture-design-patterns/rich-console-output-progress-bars/detecting-non-tty-output-and-disabling-rich-color/) — the progress auto-degradation rule.
- [Multiprocessing vs Asyncio for Raster Batch Jobs](https://www.batch-processing.com/spatial-batch-processing-async-workflows/async-io-for-raster-processing/multiprocessing-vs-asyncio-for-raster-batch-jobs/) — why this tool uses processes rather than an event loop.

## License

MIT — see [LICENSE](LICENSE).
