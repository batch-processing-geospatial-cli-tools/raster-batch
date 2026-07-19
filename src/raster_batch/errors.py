"""Error types and process exit codes.

The CLI distinguishes *usage* errors (the job was never viable — bad EPSG code, a
source file that does not exist, an output directory that cannot be written) from
*item* errors (the job ran, but some rasters failed). They get different exit codes
so a shell pipeline can tell "you called me wrong" apart from "3 of 40000 tiles were
corrupt", which are very different operational situations.
"""

from __future__ import annotations

from typing import Final

EXIT_OK: Final = 0
"""Every item succeeded."""

EXIT_USAGE: Final = 1
"""The invocation itself was invalid; no work was attempted."""

EXIT_PARTIAL_FAILURE: Final = 2
"""The batch ran to completion but at least one item failed."""


class RasterBatchError(Exception):
    """Base class for every error this package raises deliberately."""


class UsageError(RasterBatchError):
    """The user's invocation cannot produce a job.

    Raised before any worker starts. The CLI turns this into exit code 1 with a
    single readable line rather than a traceback, because these are user-facing
    mistakes and not bugs.
    """


class ItemError(RasterBatchError):
    """A single unit of work failed.

    Carried across the process boundary as a plain record rather than a live
    exception object: exceptions from GDAL-backed libraries are not always
    picklable, so the worker converts them to text before returning.
    """
