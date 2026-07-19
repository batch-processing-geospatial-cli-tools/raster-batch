"""Append-only JSONL checkpointing and a dead-letter file for failed items.

A batch that has been running for six hours and dies on item 38,000 must not start
over. The manifest is written as JSON Lines, appended and flushed per item, because
that format survives a hard kill: a truncated final line is discarded on read and
everything before it is still valid.

Background: `Checkpointing for Interrupted Spatial Batch Jobs
<https://www.batch-processing.com/spatial-batch-processing-async-workflows/progress-tracking-in-batch-jobs/implementing-checkpointing-for-interrupted-spatial-batches/>`_
and `Building a Dead-Letter Queue for Failed Geometry Transforms
<https://www.batch-processing.com/spatial-batch-processing-async-workflows/error-handling-in-spatial-pipelines/building-a-dead-letter-queue-for-failed-geometry-transforms/>`_.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from raster_batch.engine import ItemResult


class Checkpoint:
    """Records item outcomes so an interrupted run can resume.

    Successes go to ``manifest_path``; failures go to ``dead_letter_path`` *and* the
    manifest (marked failed) so that a resume does not silently retry an item that
    deterministically fails — retrying poison items forever is the classic way a
    "resumable" pipeline turns into an infinite loop. Only ``status == "done"``
    entries are treated as complete.

    Args:
        manifest_path: JSONL file of every item outcome.
        dead_letter_path: JSONL file of failures only; defaults to the manifest path
            with a ``.failed.jsonl`` suffix.
        resume: When True, previously recorded successes are loaded and skipped.
    """

    def __init__(
        self,
        manifest_path: Path,
        dead_letter_path: Path | None = None,
        *,
        resume: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.dead_letter_path = dead_letter_path or manifest_path.with_suffix(".failed.jsonl")
        self.resume = resume
        self._done: frozenset[str] = read_completed_keys(manifest_path) if resume else frozenset()
        self._manifest: IO[str] | None = None
        self._dead: IO[str] | None = None

    def completed_keys(self) -> frozenset[str]:
        """Return keys already recorded as done in a previous run."""
        return self._done

    def record(self, result: ItemResult) -> None:
        """Append one outcome, flushing immediately so a kill loses at most one line."""
        entry: dict[str, object] = {
            "key": result.key,
            "status": "done" if result.ok else "failed",
            "detail": result.detail,
            "duration_s": round(result.duration_s, 6),
        }
        if not result.ok:
            entry |= {
                "error_type": result.error_type,
                "error_message": result.error_message,
                "traceback_digest": result.traceback_digest,
            }
        self._write(self._manifest_file(), entry)
        if not result.ok:
            self._write(self._dead_letter_file(), entry)

    @staticmethod
    def _write(handle: IO[str], entry: Mapping[str, object]) -> None:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()

    def _manifest_file(self) -> IO[str]:
        if self._manifest is None:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._manifest = self.manifest_path.open("a", encoding="utf-8")
        return self._manifest

    def _dead_letter_file(self) -> IO[str]:
        if self._dead is None:
            self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            self._dead = self.dead_letter_path.open("a", encoding="utf-8")
        return self._dead

    def close(self) -> None:
        """Close any open handles. Safe to call more than once."""
        for handle in (self._manifest, self._dead):
            if handle is not None:
                handle.close()
        self._manifest = None
        self._dead = None

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_completed_keys(manifest_path: Path) -> frozenset[str]:
    """Read the keys marked done in a manifest, tolerating a truncated last line.

    A partially written final line is the normal state of a file whose process was
    killed, so it is dropped rather than treated as corruption.
    """
    if not manifest_path.exists():
        return frozenset()
    keys: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and entry.get("status") == "done":
                key = entry.get("key")
                if isinstance(key, str):
                    keys.add(key)
    return frozenset(keys)
