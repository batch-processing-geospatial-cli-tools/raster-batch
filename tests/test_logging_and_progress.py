"""Tests for structured logging and the progress reporters."""

from __future__ import annotations

import json
import logging

from rich.console import Console

from raster_batch.engine import BatchReport, ItemResult
from raster_batch.logging_setup import JsonFormatter, configure_logging
from raster_batch.progress import NullReporter, PlainReporter, RichReporter, make_reporter


def test_json_formatter_includes_extras() -> None:
    record = logging.LogRecord("raster_batch", logging.INFO, "f.py", 1, "hello", None, None)
    record.key = "data/a.tif"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello"
    assert payload["level"] == "info"
    assert payload["logger"] == "raster_batch"
    assert payload["key"] == "data/a.tif"
    assert "ts" in payload


def test_json_formatter_records_exception_type() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "raster_batch", logging.ERROR, "f.py", 1, "failed", None, __import__("sys").exc_info()
        )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["exc_type"] == "ValueError"


def test_configure_logging_replaces_handlers() -> None:
    first = configure_logging(json_output=False)
    second = configure_logging(json_output=True, verbose=True)
    assert first is second
    assert len(second.handlers) == 1
    assert isinstance(second.handlers[0].formatter, JsonFormatter)
    assert second.level == logging.DEBUG


def test_make_reporter_disabled() -> None:
    assert isinstance(make_reporter(Console(), enabled=False), NullReporter)


def test_make_reporter_non_tty() -> None:
    console = Console(force_terminal=False, file=__import__("io").StringIO())
    assert isinstance(make_reporter(console, enabled=True), PlainReporter)


def test_make_reporter_tty() -> None:
    console = Console(force_terminal=True, file=__import__("io").StringIO())
    assert isinstance(make_reporter(console, enabled=True), RichReporter)


def test_null_reporter_is_inert() -> None:
    reporter = NullReporter()
    reporter.on_start(3)
    reporter.on_submit("a")
    reporter.on_result(ItemResult(key="a", ok=True))
    reporter.on_finish(BatchReport())


def test_plain_reporter_output() -> None:
    stream = __import__("io").StringIO()
    console = Console(file=stream, force_terminal=False, width=200)
    reporter = PlainReporter(console)
    reporter.on_start(2)
    reporter.on_submit("a")
    reporter.on_result(ItemResult(key="a", ok=True, detail="wrote a"))
    reporter.on_result(
        ItemResult(key="b", ok=False, error_type="ItemError", error_message="bad file")
    )
    reporter.on_finish(BatchReport(succeeded=1, failed=1, elapsed_s=1.25))
    output = stream.getvalue()
    assert "Starting batch over 2 items." in output
    assert "[1/2] ok a wrote a" in output
    assert "[2/2] FAILED b ItemError: bad file" in output
    assert "1 ok, 1 failed" in output


def test_plain_reporter_without_total() -> None:
    stream = __import__("io").StringIO()
    reporter = PlainReporter(Console(file=stream, force_terminal=False, width=200))
    reporter.on_start(None)
    reporter.on_result(ItemResult(key="a", ok=True))
    assert "unknown number of items" in stream.getvalue()
    assert "[1] ok a" in stream.getvalue()


def test_rich_reporter_renders_and_stops() -> None:
    stream = __import__("io").StringIO()
    console = Console(file=stream, force_terminal=True, width=100)
    reporter = RichReporter(console)
    reporter.on_start(3)
    reporter.on_submit("data/a.tif")
    reporter.on_submit("data/b.tif")
    reporter.on_result(ItemResult(key="data/a.tif", ok=True, detail="ok"))
    reporter.on_result(ItemResult(key="data/b.tif", ok=False, error_type="ItemError"))
    reporter.on_finish(BatchReport(succeeded=1, failed=1, elapsed_s=0.5))
    output = stream.getvalue()
    assert "1 ok" in output
    assert reporter._live is None
