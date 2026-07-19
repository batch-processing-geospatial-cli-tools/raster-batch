"""Tests for the generic batch engine, using trivial pure-function workers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from _workers import always_fail, echo, fail_on_odd, report_pid, shout, slow_echo
from raster_batch.checkpoint import Checkpoint, read_completed_keys
from raster_batch.engine import (
    ItemResult,
    NullReporter,
    OnError,
    Task,
    default_workers,
    execute_task,
    run_batch,
)


class RecordingReporter:
    """A reporter that remembers the order of engine callbacks."""

    def __init__(self) -> None:
        self.started: list[int | None] = []
        self.submitted: list[str] = []
        self.results: list[ItemResult] = []
        self.finished = 0
        self.max_outstanding = 0

    def on_start(self, total: int | None) -> None:
        self.started.append(total)

    def on_submit(self, key: str) -> None:
        self.submitted.append(key)
        outstanding = len(self.submitted) - len(self.results)
        self.max_outstanding = max(self.max_outstanding, outstanding)

    def on_result(self, result: ItemResult) -> None:
        self.results.append(result)

    def on_finish(self, report: object) -> None:
        self.finished += 1


def tasks_for(values: list[object]) -> Iterator[Task]:
    return (Task(key=str(value), payload=value) for value in values)


def test_success_path_serial() -> None:
    report = run_batch(tasks_for(["a", "b", "c"]), shout, workers=1)
    assert report.total == 3
    assert report.succeeded == 3
    assert report.failed == 0
    assert report.ok


def test_success_path_parallel() -> None:
    report = run_batch(tasks_for([f"item{i}" for i in range(12)]), echo, workers=2)
    assert report.succeeded == 12
    assert report.ok


def test_parallel_actually_uses_child_processes() -> None:
    reporter = RecordingReporter()
    run_batch(tasks_for([f"i{n}" for n in range(8)]), report_pid, workers=2, reporter=reporter)
    pids = {result.detail for result in reporter.results}
    assert pids, "expected results"
    assert str(__import__("os").getpid()) not in pids


def test_single_worker_runs_in_process() -> None:
    reporter = RecordingReporter()
    run_batch(tasks_for(["x"]), report_pid, workers=1, reporter=reporter)
    assert reporter.results[0].detail == str(__import__("os").getpid())


def test_failure_isolation_keeps_going() -> None:
    report = run_batch(tasks_for(list(range(10))), fail_on_odd, workers=1, on_error=OnError.COLLECT)
    assert report.succeeded == 5
    assert report.failed == 5
    assert not report.ok
    assert {failure.error_type for failure in report.failures} == {"ValueError"}
    assert all(failure.traceback_digest for failure in report.failures)


def test_on_error_skip_does_not_retain_failures() -> None:
    report = run_batch(tasks_for(list(range(10))), fail_on_odd, workers=1, on_error=OnError.SKIP)
    assert report.failed == 5
    assert report.failures == []


def test_on_error_collect_retains_failures() -> None:
    report = run_batch(tasks_for(list(range(10))), fail_on_odd, workers=1, on_error=OnError.COLLECT)
    assert len(report.failures) == 5


def test_on_error_stop_aborts_early() -> None:
    report = run_batch(
        tasks_for([f"n{n}" for n in range(50)]),
        always_fail,
        workers=1,
        on_error=OnError.STOP,
    )
    assert report.stopped_early
    assert report.total < 50
    assert not report.ok


def test_bounded_in_flight_window() -> None:
    reporter = RecordingReporter()
    run_batch(
        tasks_for([f"n{n}" for n in range(40)]),
        slow_echo,
        workers=2,
        max_in_flight=3,
        reporter=reporter,
    )
    assert reporter.max_outstanding <= 3
    assert len(reporter.submitted) == 40
    assert len(reporter.results) == 40


def test_lazy_task_consumption() -> None:
    """The engine must not drain the task iterator before doing work."""
    pulled: list[int] = []

    def source() -> Iterator[Task]:
        for n in range(20):
            pulled.append(n)
            yield Task(key=str(n), payload=str(n))

    seen_during: list[int] = []

    class Watcher(NullReporter):
        def on_result(self, result: ItemResult) -> None:
            seen_during.append(len(pulled))

    run_batch(source(), echo, workers=1, max_in_flight=2, reporter=Watcher())
    assert seen_during[0] < 20, "iterator was fully drained before the first result"
    assert len(pulled) == 20


def test_results_are_not_order_guaranteed_but_complete() -> None:
    reporter = RecordingReporter()
    run_batch(tasks_for([f"k{n}" for n in range(20)]), echo, workers=2, reporter=reporter)
    assert {result.key for result in reporter.results} == {f"k{n}" for n in range(20)}


def test_reporter_lifecycle() -> None:
    reporter = RecordingReporter()
    run_batch(tasks_for(["a", "b"]), echo, workers=1, reporter=reporter, total=2)
    assert reporter.started == [2]
    assert reporter.finished == 1


def test_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="workers must be"):
        run_batch(tasks_for(["a"]), echo, workers=0)


def test_invalid_in_flight() -> None:
    with pytest.raises(ValueError, match="max_in_flight must be"):
        run_batch(tasks_for(["a"]), echo, workers=1, max_in_flight=0)


def test_empty_task_list() -> None:
    report = run_batch(iter([]), echo, workers=1)
    assert report.total == 0
    assert report.ok


def test_default_workers_is_sane() -> None:
    assert 1 <= default_workers() <= 8


def test_execute_task_never_raises() -> None:
    result = execute_task(always_fail, Task(key="k", payload="v"))
    assert not result.ok
    assert result.error_type == "RuntimeError"
    assert "nope: v" in result.error_message
    assert len(result.traceback_digest) == 12


def test_execute_task_records_duration() -> None:
    result = execute_task(slow_echo, Task(key="k", payload="v"))
    assert result.ok
    assert result.duration_s > 0


def test_checkpoint_round_trip_and_resume(tmp_path: Path) -> None:
    manifest = tmp_path / "state" / "run.jsonl"
    with Checkpoint(manifest) as checkpoint:
        first = run_batch(tasks_for(list(range(6))), fail_on_odd, workers=1, checkpoint=checkpoint)
    assert first.succeeded == 3
    assert first.failed == 3
    assert read_completed_keys(manifest) == {"0", "2", "4"}

    with Checkpoint(manifest, resume=True) as checkpoint:
        second = run_batch(tasks_for(list(range(6))), fail_on_odd, workers=1, checkpoint=checkpoint)
    assert second.skipped == 3
    assert second.total == 3
    assert second.failed == 3


def test_dead_letter_file_is_written(tmp_path: Path) -> None:
    manifest = tmp_path / "run.jsonl"
    with Checkpoint(manifest) as checkpoint:
        run_batch(tasks_for([1, 3]), fail_on_odd, workers=1, checkpoint=checkpoint)
    dead = manifest.with_suffix(".failed.jsonl")
    lines = [line for line in dead.read_text().splitlines() if line]
    assert len(lines) == 2
    assert "traceback_digest" in lines[0]
    assert "ValueError" in lines[0]


def test_resume_without_manifest_starts_fresh(tmp_path: Path) -> None:
    checkpoint = Checkpoint(tmp_path / "absent.jsonl", resume=True)
    assert checkpoint.completed_keys() == frozenset()
    checkpoint.close()


def test_truncated_manifest_line_is_ignored(tmp_path: Path) -> None:
    manifest = tmp_path / "run.jsonl"
    manifest.write_text(
        '{"key": "a", "status": "done"}\n\n{"key": "b", "status": "failed"}\n{"key": "c", "sta'
    )
    assert read_completed_keys(manifest) == {"a"}


def test_checkpoint_close_is_idempotent(tmp_path: Path) -> None:
    checkpoint = Checkpoint(tmp_path / "run.jsonl")
    checkpoint.record(ItemResult(key="a", ok=True, detail="d"))
    checkpoint.close()
    checkpoint.close()
    assert '"status": "done"' in (tmp_path / "run.jsonl").read_text()
