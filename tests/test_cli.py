"""End-to-end CLI tests, asserting exit codes, files on disk and geospatial results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rasterio
from rasterio.crs import CRS
from typer.testing import CliRunner

from conftest import RasterFactory
from raster_batch.cli import app
from raster_batch.errors import EXIT_OK, EXIT_PARTIAL_FAILURE, EXIT_USAGE

runner = CliRunner()


def sources(directory: Path) -> list[str]:
    return sorted(str(path) for path in directory.glob("*.tif"))


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == EXIT_OK
    for command in ("reproject", "clip", "tile", "convert", "info"):
        assert command in result.output


def test_reproject_batch(raster_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "reproject",
            *sources(raster_dir),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(out),
            "--workers",
            "2",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    written = sorted(out.glob("*.tif"))
    assert len(written) == 5
    for path in written:
        with rasterio.open(path) as dataset:
            assert dataset.crs == CRS.from_epsg(3857)


def test_reproject_rejects_bad_epsg(raster_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "reproject",
            *sources(raster_dir),
            "--dst-crs",
            "EPSG:banana",
            "--out",
            str(tmp_path / "o"),
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "Not a usable CRS" in result.output


def test_reproject_rejects_missing_source(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "reproject",
            str(tmp_path / "nope.tif"),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(tmp_path / "o"),
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "not found" in result.output


def test_reproject_rejects_unwritable_out_dir(wgs84_raster: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")
    result = runner.invoke(
        app,
        [
            "reproject",
            str(wgs84_raster),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(blocker),
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "not a directory" in result.output


def test_corrupt_input_yields_partial_failure(
    corrupt_raster: Path, wgs84_raster: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "reproject",
            str(wgs84_raster),
            str(corrupt_raster),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(tmp_path / "o"),
            "--workers",
            "1",
            "--on-error",
            "collect",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_PARTIAL_FAILURE
    assert "ItemError" in result.output
    assert (tmp_path / "o" / "wgs84.tif").exists()


def test_on_error_stop_exits_two(corrupt_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "reproject",
            str(corrupt_raster),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(tmp_path / "o"),
            "--workers",
            "1",
            "--on-error",
            "stop",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_PARTIAL_FAILURE


def test_clip_cli_bounds(wgs84_raster: Path, tmp_path: Path) -> None:
    with rasterio.open(wgs84_raster) as src:
        left, bottom, right, top = src.bounds
        res = src.res[0]
    box = (left + 8 * res, bottom + 8 * res, right - 8 * res, top - 8 * res)
    out = tmp_path / "clipped"
    result = runner.invoke(
        app,
        [
            "clip",
            str(wgs84_raster),
            "--bounds",
            ",".join(str(value) for value in box),
            "--out",
            str(out),
            "--workers",
            "1",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    with rasterio.open(out / "wgs84.tif") as dataset:
        assert dataset.width == 48
        assert dataset.height == 32
        assert dataset.bounds.left == pytest.approx(box[0], abs=res)
        assert dataset.bounds.top == pytest.approx(box[3], abs=res)


def test_clip_cli_rejects_malformed_bounds(wgs84_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "clip",
            str(wgs84_raster),
            "--bounds",
            "1,2,3",
            "--out",
            str(tmp_path / "o"),
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "four comma-separated" in result.output


def test_clip_cli_rejects_bad_bounds_crs(wgs84_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "clip",
            str(wgs84_raster),
            "--bounds",
            "0,0,1,1",
            "--bounds-crs",
            "nonsense:1",
            "--out",
            str(tmp_path / "o"),
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_USAGE


def test_tile_cli_grid_and_transforms(make_raster: RasterFactory, tmp_path: Path) -> None:
    src = make_raster("grid.tif", width=100, height=60)
    out = tmp_path / "tiles"
    result = runner.invoke(
        app,
        ["tile", str(src), "--size", "32", "--out", str(out), "--workers", "2", "--no-progress"],
    )
    assert result.exit_code == EXIT_OK, result.output
    tiles = sorted(out.glob("*.tif"))
    assert len(tiles) == 4 * 2  # ceil(100/32) x ceil(60/32)

    with rasterio.open(src) as parent:
        parent_transform = parent.transform
        res = parent.res[0]
    with rasterio.open(out / "grid_0000_0000.tif") as first:
        assert (first.width, first.height) == (32, 32)
        assert first.transform.c == pytest.approx(parent_transform.c)
    with rasterio.open(out / "grid_0001_0003.tif") as edge:
        assert (edge.width, edge.height) == (100 - 96, 60 - 32)
        assert edge.transform.c == pytest.approx(parent_transform.c + 96 * res)
        assert edge.transform.f == pytest.approx(parent_transform.f - 32 * res)


def test_tile_cli_rejects_bad_size(wgs84_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["tile", str(wgs84_raster), "--size", "0", "--out", str(tmp_path / "o")]
    )
    assert result.exit_code == EXIT_USAGE
    assert "--size" in result.output


def test_tile_cli_rejects_corrupt_source(corrupt_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["tile", str(corrupt_raster), "--out", str(tmp_path / "o")])
    assert result.exit_code == EXIT_USAGE
    assert "Cannot open" in result.output


def test_convert_cli_compression(raster_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "lzw"
    result = runner.invoke(
        app,
        [
            "convert",
            *sources(raster_dir),
            "--format",
            "GTiff",
            "--compress",
            "lzw",
            "--out",
            str(out),
            "--workers",
            "2",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    written = sorted(out.glob("*.tif"))
    assert len(written) == 5
    for path in written:
        with rasterio.open(path) as dataset:
            assert dataset.profile["compress"].lower() == "lzw"


def test_convert_cli_cog(make_raster: RasterFactory, tmp_path: Path) -> None:
    src = make_raster("wide.tif", width=600, height=600)
    out = tmp_path / "cog"
    result = runner.invoke(
        app,
        [
            "convert",
            str(src),
            "--format",
            "COG",
            "--blocksize",
            "256",
            "--out",
            str(out),
            "--workers",
            "1",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    with rasterio.open(out / "wide.tif") as dataset:
        assert dataset.block_shapes[0] == (256, 256)
        assert dataset.overviews(1)


def test_info_table(wgs84_raster: Path, utm_raster: Path) -> None:
    result = runner.invoke(app, ["info", str(wgs84_raster), str(utm_raster)])
    assert result.exit_code == EXIT_OK
    assert "wgs84.tif" in result.output
    assert "utm.tif" in result.output
    assert "float32" in result.output


def test_info_on_corrupt_only_is_usage_error(corrupt_raster: Path) -> None:
    result = runner.invoke(app, ["info", str(corrupt_raster)])
    assert result.exit_code == EXIT_USAGE


def test_info_partial_failure(wgs84_raster: Path, corrupt_raster: Path) -> None:
    result = runner.invoke(app, ["info", str(wgs84_raster), str(corrupt_raster)])
    assert result.exit_code == EXIT_PARTIAL_FAILURE
    assert "wgs84.tif" in result.output


def test_checkpoint_and_resume(raster_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "state" / "run.jsonl"
    out = tmp_path / "out"
    common = [
        "reproject",
        *sources(raster_dir),
        "--dst-crs",
        "EPSG:3857",
        "--out",
        str(out),
        "--workers",
        "1",
        "--no-progress",
        "--checkpoint",
        str(manifest),
    ]
    first = runner.invoke(app, common)
    assert first.exit_code == EXIT_OK, first.output
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    assert len(entries) == 5
    assert all(entry["status"] == "done" for entry in entries)

    # Without --resume the outputs already exist, so every item now fails.
    second = runner.invoke(app, common)
    assert second.exit_code == EXIT_PARTIAL_FAILURE

    third = runner.invoke(app, [*common, "--resume", "--log-json"])
    assert third.exit_code == EXIT_OK
    summary = json.loads([line for line in third.output.splitlines() if line.startswith("{")][-1])
    assert summary["skipped"] == 5
    assert summary["succeeded"] == 0
    # A resumed run must not append duplicate entries for work it skipped.
    assert len([line for line in manifest.read_text().splitlines() if line]) == 10


def test_resume_without_checkpoint_is_usage_error(wgs84_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "reproject",
            str(wgs84_raster),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(tmp_path / "o"),
            "--resume",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_USAGE
    assert "--resume needs --checkpoint" in result.output


def test_overwrite_flag(wgs84_raster: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    args = [
        "reproject",
        str(wgs84_raster),
        "--dst-crs",
        "EPSG:3857",
        "--out",
        str(out),
        "--workers",
        "1",
        "--no-progress",
    ]
    assert runner.invoke(app, args).exit_code == EXIT_OK
    assert runner.invoke(app, args).exit_code == EXIT_PARTIAL_FAILURE
    assert runner.invoke(app, [*args, "--overwrite"]).exit_code == EXIT_OK


def test_dead_letter_file(corrupt_raster: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "run.jsonl"
    result = runner.invoke(
        app,
        [
            "reproject",
            str(corrupt_raster),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(tmp_path / "o"),
            "--workers",
            "1",
            "--no-progress",
            "--checkpoint",
            str(manifest),
        ],
    )
    assert result.exit_code == EXIT_PARTIAL_FAILURE
    dead = json.loads(manifest.with_suffix(".failed.jsonl").read_text().splitlines()[0])
    assert dead["status"] == "failed"
    assert dead["error_type"] == "ItemError"
    assert len(dead["traceback_digest"]) == 12


def test_log_json_emits_parseable_lines(wgs84_raster: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "reproject",
            str(wgs84_raster),
            "--dst-crs",
            "EPSG:3857",
            "--out",
            str(tmp_path / "o"),
            "--workers",
            "1",
            "--no-progress",
            "--log-json",
        ],
    )
    assert result.exit_code == EXIT_OK
    lines = [line for line in result.output.splitlines() if line.startswith("{")]
    assert lines, result.output
    payload = json.loads(lines[-1])
    assert payload["message"] == "batch finished"
    assert payload["succeeded"] == 1
    assert payload["level"] == "info"


def test_progress_falls_back_to_plain_lines(raster_dir: Path, tmp_path: Path) -> None:
    """CliRunner's stream is not a TTY, so the plain reporter must be selected."""
    result = runner.invoke(
        app,
        ["convert", *sources(raster_dir), "--out", str(tmp_path / "o"), "--workers", "1"],
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "Starting batch over 5 items." in result.output
    assert "[5/5] ok" in result.output
    assert "\x1b[" not in result.output


def test_max_in_flight_option(raster_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "convert",
            *sources(raster_dir),
            "--out",
            str(tmp_path / "o"),
            "--workers",
            "2",
            "--max-in-flight",
            "2",
            "--no-progress",
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    assert len(list((tmp_path / "o").glob("*.tif"))) == 5
