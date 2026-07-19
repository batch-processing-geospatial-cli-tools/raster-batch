"""Progress displays for batch runs.

Three flavours, chosen automatically: a live Rich dashboard when stdout is a real
terminal, plain one-line-per-item text when it is a pipe or a CI log, and nothing at
all when the caller asks for silence. Emitting ANSI cursor movement into a log file
produces megabytes of unreadable escape sequences, so the TTY check is not a nicety.

Background: `Detecting Non-TTY Output and Disabling Rich Color
<https://www.batch-processing.com/cli-architecture-design-patterns/rich-console-output-progress-bars/detecting-non-tty-output-and-disabling-rich-color/>`_
and `Rendering a Live Rich Dashboard for Batch Raster Jobs
<https://www.batch-processing.com/cli-architecture-design-patterns/rich-console-output-progress-bars/rendering-a-live-rich-dashboard-for-batch-raster-jobs/>`_.
"""

from __future__ import annotations

from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from raster_batch.engine import BatchReport, ItemResult, NullReporter

__all__ = ["NullReporter", "PlainReporter", "RichReporter", "make_reporter"]

_ACTIVE_LINES = 6


class PlainReporter:
    """One line per completed item, no cursor control.

    Suitable for CI logs and for piping into another process. Lines are written to
    stderr so that command output on stdout stays machine-readable.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._total: int | None = None
        self._seen = 0

    def on_start(self, total: int | None) -> None:
        """Announce the size of the job when it is known up front."""
        self._total = total
        label = f"{total} items" if total is not None else "an unknown number of items"
        self.console.print(f"Starting batch over {label}.")

    def on_submit(self, key: str) -> None:
        """Submissions are not interesting without a live display."""

    def on_result(self, result: ItemResult) -> None:
        """Print the outcome of one item."""
        self._seen += 1
        position = f"[{self._seen}/{self._total}]" if self._total else f"[{self._seen}]"
        if result.ok:
            self.console.print(f"{position} ok {result.key} {result.detail}".rstrip())
        else:
            self.console.print(
                f"{position} FAILED {result.key} {result.error_type}: {result.error_message}"
            )

    def on_finish(self, report: BatchReport) -> None:
        """Print the closing summary."""
        self.console.print(
            f"Done in {report.elapsed_s:.1f}s: {report.succeeded} ok, "
            f"{report.failed} failed, {report.skipped} skipped."
        )


class RichReporter:
    """A live bar with ETA and throughput plus a tail of in-flight items.

    The active-item panel is capped at a handful of lines: with 8 workers you want
    to see what is currently chewing, but an unbounded list would redraw the whole
    terminal on every completion.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[rate]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        self._task_id = self._progress.add_task("processing", total=None, rate="")
        self._live: Live | None = None
        self._active: deque[str] = deque(maxlen=_ACTIVE_LINES)
        self._failed = 0

    def _render(self) -> Group:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim")
        for key in self._active:
            table.add_row(Text(f"working {key}", overflow="ellipsis"))
        if self._failed:
            table.add_row(Text(f"{self._failed} failed so far", style="red"))
        return Group(self._progress, table)

    def on_start(self, total: int | None) -> None:
        """Start the live display."""
        self._progress.update(self._task_id, total=total)
        self._live = Live(self._render(), console=self.console, refresh_per_second=8)
        self._live.start()

    def on_submit(self, key: str) -> None:
        """Show a newly submitted item in the active panel."""
        self._active.append(key)
        self._refresh()

    def on_result(self, result: ItemResult) -> None:
        """Advance the bar and drop the item from the active panel."""
        if result.key in self._active:
            self._active.remove(result.key)
        if not result.ok:
            self._failed += 1
        self._progress.advance(self._task_id, 1)
        task = self._progress.tasks[0]
        rate = f"{task.speed:.1f}/s" if task.speed else ""
        self._progress.update(self._task_id, rate=rate)
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def on_finish(self, report: BatchReport) -> None:
        """Tear down the live display and print a summary line."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        style = "green" if report.ok else "yellow"
        self.console.print(
            Text(
                f"Done in {report.elapsed_s:.1f}s: {report.succeeded} ok, "
                f"{report.failed} failed, {report.skipped} skipped.",
                style=style,
            )
        )


def make_reporter(
    console: Console, *, enabled: bool
) -> NullReporter | PlainReporter | RichReporter:
    """Pick the right reporter for the current output stream.

    Args:
        console: The Rich console the CLI is writing status to.
        enabled: False when the user passed ``--no-progress``.

    Returns:
        A live dashboard on a TTY, plain lines otherwise, or a no-op when disabled.
    """
    if not enabled:
        return NullReporter()
    if console.is_terminal:
        return RichReporter(console)
    return PlainReporter(console)
