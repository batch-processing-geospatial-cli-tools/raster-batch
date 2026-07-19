"""Module-level worker functions for the engine tests.

They live in their own module (not inside a test function) because
``ProcessPoolExecutor`` pickles the callable by qualified name; a closure or a local
function cannot cross the process boundary.
"""

from __future__ import annotations

import os
import time


def echo(payload: str) -> str:
    """Return the payload unchanged."""
    return payload


def shout(payload: str) -> str:
    """Uppercase the payload."""
    return payload.upper()


def fail_on_odd(payload: int) -> str:
    """Raise for odd numbers so failure isolation can be exercised."""
    if payload % 2:
        raise ValueError(f"odd payload {payload}")
    return f"even {payload}"


def always_fail(payload: str) -> str:
    """Raise a distinctive error for every item."""
    raise RuntimeError(f"nope: {payload}")


def report_pid(payload: str) -> str:
    """Return the process id that handled the item."""
    return str(os.getpid())


def slow_echo(payload: str) -> str:
    """Echo after a short sleep, so in-flight windows can be observed."""
    time.sleep(0.01)
    return payload
