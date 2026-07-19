"""Logging configuration: human text by default, one JSON object per line on request.

Pipelines that ship logs to Loki, CloudWatch or a file-based collector need machine
parseable records; humans at a terminal do not. Rather than making callers choose a
format string, ``--log-json`` swaps the formatter and everything downstream is
unchanged.

Background: `Logging Spatial Transformations to Structured JSON
<https://www.batch-processing.com/spatial-batch-processing-async-workflows/error-handling-in-spatial-pipelines/logging-spatial-transformation-results-to-structured-json/>`_.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

LOGGER_NAME = "raster_batch"

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object.

    Any extra keyword passed through ``logger.info(..., extra={...})`` is merged into
    the object, which is how per-item fields (``key``, ``duration_s``) reach the log
    without a bespoke record class.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` to a compact JSON line."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_type"] = getattr(record.exc_info[0], "__name__", "unknown")
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(*, json_output: bool, verbose: bool = False) -> logging.Logger:
    """Install a single stderr handler on the package logger and return it.

    Handlers are replaced rather than appended so that repeated CLI invocations in
    the same process (as in tests) do not produce duplicated lines.

    Args:
        json_output: Emit structured JSON instead of plain text.
        verbose: Lower the threshold to DEBUG.

    Returns:
        The configured ``raster_batch`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(
        JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(message)s")
    )
    logger.addHandler(stream)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger
