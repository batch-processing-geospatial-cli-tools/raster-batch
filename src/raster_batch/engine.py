"""A streaming, bounded, resumable batch execution engine.

The whole point of this module is that it never materialises the job. A naive
``ProcessPoolExecutor`` batch looks like::

    futures = [pool.submit(work, t) for t in tasks]      # 100k futures, 100k tasks
    for f in as_completed(futures): ...

which pins every task payload *and* every result in RAM before the first item is
even done. :func:`run_batch` instead keeps a bounded window of in-flight futures,
pulling from the task iterator only when a slot frees up, so memory is a function
of ``max_in_flight`` rather than of job size.

The engine is deliberately generic: it knows nothing about rasters. Everything
geospatial lives in :mod:`raster_batch.ops`, which keeps the concurrency logic
testable with trivial pure-function workers.

Design background: `Choosing Chunk Size for Multiprocessing Raster Warps
<https://www.batch-processing.com/spatial-batch-processing-async-workflows/multiprocessing-geospatial-tasks/choosing-chunk-size-for-multiprocessing-raster-warps/>`_.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from raster_batch.checkpoint import Checkpoint

T = TypeVar("T")


class OnError(StrEnum):
    """What to do when an individual item raises.

    ``STOP`` aborts the run at the first failure without submitting further work.
    ``SKIP`` and ``COLLECT`` both run the batch to completion; they differ in
    memory behaviour — ``COLLECT`` retains every failure record in the returned
    report so the caller can print or re-drive them, while ``SKIP`` only counts
    them (the dead-letter file still receives the full record either way). On a
    job with a large failing fraction, ``COLLECT`` is the thing that will run you
    out of memory, so it is not the default.
    """

    STOP = "stop"
    SKIP = "skip"
    COLLECT = "collect"


@dataclass(frozen=True, slots=True)
class Task:
    """One unit of work.

    Attributes:
        key: Stable identity used for checkpointing and de-duplication. Two runs
            over the same inputs must produce the same keys or ``--resume`` will
            silently redo work.
        payload: Anything picklable that the worker understands.
    """

    key: str
    payload: Any


@dataclass(frozen=True, slots=True)
class ItemResult:
    """The outcome of one :class:`Task`, safe to send back across a pipe."""

    key: str
    ok: bool
    detail: str = ""
    error_type: str = ""
    error_message: str = ""
    traceback_digest: str = ""
    duration_s: float = 0.0


@dataclass(slots=True)
class BatchReport:
    """Aggregate outcome of a :func:`run_batch` call."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    stopped_early: bool = False
    failures: list[ItemResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        """True when nothing failed and the run was not aborted."""
        return self.failed == 0 and not self.stopped_early


class ProgressReporter(Protocol):
    """Minimal surface the engine needs from a progress display."""

    def on_start(self, total: int | None) -> None: ...

    def on_submit(self, key: str) -> None: ...

    def on_result(self, result: ItemResult) -> None: ...

    def on_finish(self, report: BatchReport) -> None: ...


class NullReporter:
    """A reporter that does nothing; used when progress output is suppressed."""

    def on_start(self, total: int | None) -> None:
        """Ignore the start of the run."""

    def on_submit(self, key: str) -> None:
        """Ignore a submission."""

    def on_result(self, result: ItemResult) -> None:
        """Ignore an item result."""

    def on_finish(self, report: BatchReport) -> None:
        """Ignore the end of the run."""


def _digest(tb: str) -> str:
    """Hash a traceback so identical failures can be grouped without storing text.

    Storing full tracebacks for every one of ten thousand failures is wasteful and
    unreadable; a short digest lets you count distinct failure modes at a glance.
    """
    return hashlib.sha256(tb.encode("utf-8", "replace")).hexdigest()[:12]


def execute_task(worker: Callable[[Any], str], task: Task) -> ItemResult:
    """Run ``worker`` against one task, converting any exception into a record.

    This runs inside the child process. It must never raise: a raising child would
    surface as a bare ``BrokenProcessPool`` and take the whole batch down, which is
    exactly the fragility per-item isolation exists to prevent.
    """
    started = time.perf_counter()
    try:
        detail = worker(task.payload)
    except BaseException as exc:  # deliberate isolation boundary
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        tb = "".join(traceback.format_exception(exc))
        return ItemResult(
            key=task.key,
            ok=False,
            error_type=type(exc).__name__,
            error_message=str(exc) or repr(exc),
            traceback_digest=_digest(tb),
            duration_s=time.perf_counter() - started,
        )
    return ItemResult(
        key=task.key,
        ok=True,
        detail=str(detail),
        duration_s=time.perf_counter() - started,
    )


class _SerialExecutor:
    """An in-process stand-in for ``ProcessPoolExecutor`` used when ``workers == 1``.

    Spawning a subprocess to do one thing at a time buys nothing but pickling cost
    and makes debugging (and coverage measurement) far harder, so single-worker runs
    stay in the parent process.
    """

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> cf.Future[T]:
        """Run ``fn`` immediately and return an already-resolved future."""
        future: cf.Future[T] = cf.Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """No resources to release."""

    def __enter__(self) -> _SerialExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def default_workers() -> int:
    """Pick a sensible worker count.

    GDAL work is CPU- and I/O-bound in roughly equal measure, so we use the CPU
    count but cap it: past about 8 concurrent warps the bottleneck is usually disk,
    and each extra worker still costs a full window of resident memory.
    """
    return max(1, min(8, os.cpu_count() or 1))


def _pending_tasks(
    tasks: Iterable[Task],
    checkpoint: Checkpoint | None,
    report: BatchReport,
) -> Iterator[Task]:
    """Yield tasks that still need doing, counting resume-skips into ``report``."""
    done = checkpoint.completed_keys() if checkpoint else frozenset()
    for task in tasks:
        if task.key in done:
            report.skipped += 1
            continue
        yield task


def run_batch(
    tasks: Iterable[Task],
    worker: Callable[[Any], str],
    *,
    workers: int | None = None,
    on_error: OnError = OnError.SKIP,
    max_in_flight: int | None = None,
    checkpoint: Checkpoint | None = None,
    reporter: ProgressReporter | None = None,
    total: int | None = None,
) -> BatchReport:
    """Run ``worker`` over ``tasks`` in parallel with bounded memory.

    Args:
        tasks: Any iterable — a generator is preferred, and is consumed lazily.
        worker: A module-level callable taking ``task.payload``. It must be
            importable in a child process (no lambdas, no closures) and should
            return a short human-readable string describing what it produced.
        workers: Process count; ``None`` means :func:`default_workers`. A value of
            1 runs in-process.
        on_error: Failure policy, see :class:`OnError`.
        max_in_flight: Ceiling on simultaneously submitted tasks. Defaults to
            ``workers * 2``, which keeps every worker fed with one queued item
            without letting the pending set grow with job size.
        checkpoint: If given, completed keys are appended to it as they finish and
            already-recorded keys are skipped on the way in.
        reporter: Progress display; ``None`` means silent.
        total: Item count for ETA display when ``tasks`` is a sized collection.

    Returns:
        A :class:`BatchReport`. Note that results are *not* ordered: items complete
        in whatever order the pool finishes them. Nothing in this package depends
        on ordering; if you need it, sort by key at the call site.

    Raises:
        ValueError: If ``workers`` or ``max_in_flight`` is not positive.
    """
    if workers is None:
        workers = default_workers()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_in_flight is None:
        max_in_flight = workers * 2
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")

    reporter = reporter or NullReporter()
    report = BatchReport()
    started = time.perf_counter()
    reporter.on_start(total)

    source = _pending_tasks(tasks, checkpoint, report)
    executor: Any = (
        _SerialExecutor() if workers == 1 else cf.ProcessPoolExecutor(max_workers=workers)
    )

    try:
        with executor:
            in_flight: set[cf.Future[ItemResult]] = set()
            exhausted = False
            while True:
                while not exhausted and len(in_flight) < max_in_flight:
                    task = next(source, None)
                    if task is None:
                        exhausted = True
                        break
                    reporter.on_submit(task.key)
                    in_flight.add(executor.submit(execute_task, worker, task))
                if not in_flight:
                    break
                done, in_flight = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
                stop_now = False
                for future in done:
                    result = future.result()
                    report.total += 1
                    if result.ok:
                        report.succeeded += 1
                    else:
                        report.failed += 1
                        if on_error is not OnError.SKIP:
                            report.failures.append(result)
                        if on_error is OnError.STOP:
                            stop_now = True
                    if checkpoint is not None:
                        checkpoint.record(result)
                    reporter.on_result(result)
                if stop_now:
                    report.stopped_early = True
                    for future in in_flight:
                        future.cancel()
                    break
    finally:
        report.elapsed_s = time.perf_counter() - started
        reporter.on_finish(report)
    return report
