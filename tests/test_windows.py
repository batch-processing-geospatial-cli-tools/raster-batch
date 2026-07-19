"""Tests for window sizing, the mechanism behind the memory guarantee."""

from __future__ import annotations

import pytest

from raster_batch.windows import (
    bytes_per_row,
    iter_row_windows,
    iter_tile_windows,
    rows_per_window,
)


def test_bytes_per_row() -> None:
    assert bytes_per_row(1000, 3, "uint8") == 3000
    assert bytes_per_row(1000, 1, "float32") == 4000


def test_rows_per_window_respects_budget() -> None:
    assert rows_per_window(1000, 1, "uint8", 10_000) == 10
    assert rows_per_window(1000, 4, "float32", 16_000) == 1


def test_rows_per_window_never_zero() -> None:
    """A single row larger than the budget still yields one row, not an empty window."""
    assert rows_per_window(1_000_000, 4, "float64", 1024) == 1


def test_row_windows_cover_raster_exactly() -> None:
    windows = list(iter_row_windows(100, 55, 1, "uint8", 1000))
    assert sum(int(window.height) for window in windows) == 55
    assert all(int(window.width) == 100 for window in windows)
    assert int(windows[0].row_off) == 0
    assert int(windows[-1].row_off) + int(windows[-1].height) == 55


def test_row_windows_do_not_overlap() -> None:
    covered: list[int] = []
    for window in iter_row_windows(64, 48, 1, "uint8", 512):
        covered.extend(range(int(window.row_off), int(window.row_off) + int(window.height)))
    assert covered == list(range(48))


def test_single_window_when_budget_is_large() -> None:
    windows = list(iter_row_windows(64, 48, 1, "uint8", 1 << 30))
    assert len(windows) == 1
    assert int(windows[0].height) == 48


def test_tile_windows_grid_shape() -> None:
    tiles = list(iter_tile_windows(1000, 1000, 512))
    assert len(tiles) == 4
    sizes = sorted((int(window.width), int(window.height)) for _, _, window in tiles)
    assert sizes == [(488, 488), (488, 512), (512, 488), (512, 512)]


def test_tile_windows_indices_and_offsets() -> None:
    tiles = list(iter_tile_windows(100, 60, 32))
    assert len(tiles) == 4 * 2
    by_index = {(col, row): window for col, row, window in tiles}
    assert int(by_index[(0, 0)].col_off) == 0
    assert int(by_index[(3, 1)].col_off) == 96
    assert int(by_index[(3, 1)].row_off) == 32
    assert int(by_index[(3, 1)].width) == 4
    assert int(by_index[(3, 1)].height) == 28


def test_tile_windows_cover_every_pixel() -> None:
    tiles = iter_tile_windows(97, 53, 16)
    area = sum(int(window.width) * int(window.height) for _, _, window in tiles)
    assert area == 97 * 53


def test_tile_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tile size"):
        list(iter_tile_windows(10, 10, 0))
