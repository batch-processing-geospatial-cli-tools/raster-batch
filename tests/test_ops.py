"""Tests for the raster operations, asserting geospatial results rather than file size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from conftest import RasterFactory
from raster_batch.errors import ItemError, UsageError
from raster_batch.ops import (
    ClipPayload,
    ConvertPayload,
    OutputFormat,
    ReprojectPayload,
    ResamplingName,
    TilePayload,
    clip_one,
    convert_one,
    describe,
    parse_bounds,
    parse_crs,
    reproject_one,
    tile_one,
)


def test_parse_crs_accepts_epsg() -> None:
    assert parse_crs("EPSG:3857").to_epsg() == 3857


def test_parse_crs_rejects_nonsense() -> None:
    with pytest.raises(UsageError, match="Not a usable CRS"):
        parse_crs("EPSG:notacode")


def test_parse_crs_rejects_unknown_authority_code() -> None:
    with pytest.raises(UsageError):
        parse_crs("EPSG:999999")


def test_parse_bounds_ok() -> None:
    assert parse_bounds(" -1, -2 ,3,4 ") == (-1.0, -2.0, 3.0, 4.0)


@pytest.mark.parametrize(
    "value",
    ["1,2,3", "a,b,c,d", "5,5,1,1", "0,0,0,0", ""],
)
def test_parse_bounds_rejects_bad_input(value: str) -> None:
    with pytest.raises(UsageError):
        parse_bounds(value)


def test_reproject_sets_target_crs(wgs84_raster: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out" / "reprojected.tif"
    detail = reproject_one(
        ReprojectPayload(
            src=str(wgs84_raster),
            dst=str(dst),
            dst_crs="EPSG:3857",
            resampling=ResamplingName.BILINEAR,
        )
    )
    assert dst.exists()
    assert "EPSG:3857" in detail
    with rasterio.open(dst) as dataset:
        assert dataset.crs == CRS.from_epsg(3857)
        assert dataset.count == 1
        assert dataset.width > 0 and dataset.height > 0


def test_reproject_preserves_bands_and_dtype(utm_raster: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out" / "utm_to_wgs84.tif"
    reproject_one(
        ReprojectPayload(src=str(utm_raster), dst=str(dst), dst_crs="EPSG:4326", window_mb=1)
    )
    with rasterio.open(dst) as dataset:
        assert dataset.crs == CRS.from_epsg(4326)
        assert dataset.count == 3
        assert dataset.dtypes[0] == "float32"


def test_reproject_roundtrip_keeps_bounds_close(wgs84_raster: Path, tmp_path: Path) -> None:
    """Warping out and back must land within a pixel of the original footprint."""
    mid = tmp_path / "mid.tif"
    back = tmp_path / "back.tif"
    reproject_one(ReprojectPayload(src=str(wgs84_raster), dst=str(mid), dst_crs="EPSG:3857"))
    reproject_one(ReprojectPayload(src=str(mid), dst=str(back), dst_crs="EPSG:4326"))
    with rasterio.open(wgs84_raster) as original, rasterio.open(back) as result:
        tolerance = original.res[0] * 2
        assert result.bounds.left == pytest.approx(original.bounds.left, abs=tolerance)
        assert result.bounds.top == pytest.approx(original.bounds.top, abs=tolerance)


def test_reproject_refuses_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ItemError, match="does not exist"):
        reproject_one(
            ReprojectPayload(
                src=str(tmp_path / "gone.tif"), dst=str(tmp_path / "o.tif"), dst_crs="EPSG:3857"
            )
        )


def test_reproject_refuses_corrupt_source(corrupt_raster: Path, tmp_path: Path) -> None:
    with pytest.raises(ItemError, match="Cannot open"):
        reproject_one(
            ReprojectPayload(
                src=str(corrupt_raster), dst=str(tmp_path / "o.tif"), dst_crs="EPSG:3857"
            )
        )


def test_reproject_refuses_crs_less_source(make_raster: RasterFactory, tmp_path: Path) -> None:
    src = make_raster("nocrs.tif", crs=None)
    with pytest.raises(ItemError, match="no CRS"):
        reproject_one(
            ReprojectPayload(src=str(src), dst=str(tmp_path / "o.tif"), dst_crs="EPSG:3857")
        )


def test_existing_output_is_protected(wgs84_raster: Path, tmp_path: Path) -> None:
    dst = tmp_path / "exists.tif"
    dst.write_bytes(b"")
    with pytest.raises(ItemError, match="already exists"):
        reproject_one(ReprojectPayload(src=str(wgs84_raster), dst=str(dst), dst_crs="EPSG:3857"))


def test_overwrite_flag_replaces_output(wgs84_raster: Path, tmp_path: Path) -> None:
    dst = tmp_path / "exists.tif"
    dst.write_bytes(b"")
    reproject_one(
        ReprojectPayload(src=str(wgs84_raster), dst=str(dst), dst_crs="EPSG:3857", overwrite=True)
    )
    with rasterio.open(dst) as dataset:
        assert dataset.crs == CRS.from_epsg(3857)


def test_clip_bounds_within_tolerance(wgs84_raster: Path, tmp_path: Path) -> None:
    with rasterio.open(wgs84_raster) as src:
        left, bottom, right, top = src.bounds
        res = src.res[0]
    box = (left + 10 * res, bottom + 5 * res, right - 10 * res, top - 5 * res)
    dst = tmp_path / "clipped.tif"
    clip_one(ClipPayload(src=str(wgs84_raster), dst=str(dst), bounds=box))
    with rasterio.open(dst) as dataset:
        assert dataset.bounds.left == pytest.approx(box[0], abs=res)
        assert dataset.bounds.right == pytest.approx(box[2], abs=res)
        assert dataset.bounds.bottom == pytest.approx(box[1], abs=res)
        assert dataset.bounds.top == pytest.approx(box[3], abs=res)
        assert dataset.width == 44
        assert dataset.height == 38


def test_clip_preserves_pixel_values(wgs84_raster: Path, tmp_path: Path) -> None:
    """The clipped window must contain exactly the source pixels, not resampled ones."""
    dst = tmp_path / "clipped.tif"
    with rasterio.open(wgs84_raster) as src:
        left, top = src.bounds.left, src.bounds.top
        res = src.res[0]
        expected = src.read(1, window=rasterio.windows.Window(4, 4, 20, 20))
    box = (left + 4 * res, top - 24 * res, left + 24 * res, top - 4 * res)
    clip_one(ClipPayload(src=str(wgs84_raster), dst=str(dst), bounds=box))
    with rasterio.open(dst) as dataset:
        np.testing.assert_array_equal(dataset.read(1), expected)


def test_clip_with_foreign_bounds_crs(utm_raster: Path, tmp_path: Path) -> None:
    from rasterio.warp import transform_bounds

    with rasterio.open(utm_raster) as src:
        wgs = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
        src_crs = src.crs
    inset = (
        wgs[0] + (wgs[2] - wgs[0]) * 0.25,
        wgs[1] + (wgs[3] - wgs[1]) * 0.25,
        wgs[0] + (wgs[2] - wgs[0]) * 0.75,
        wgs[1] + (wgs[3] - wgs[1]) * 0.75,
    )
    dst = tmp_path / "clipped_utm.tif"
    clip_one(ClipPayload(src=str(utm_raster), dst=str(dst), bounds=inset, bounds_crs="EPSG:4326"))
    with rasterio.open(dst) as dataset:
        assert dataset.crs == src_crs
        assert 0 < dataset.width < 64
        assert 0 < dataset.height < 48


def test_clip_outside_raster_fails(wgs84_raster: Path, tmp_path: Path) -> None:
    with pytest.raises(ItemError, match="do not intersect"):
        clip_one(
            ClipPayload(
                src=str(wgs84_raster),
                dst=str(tmp_path / "empty.tif"),
                bounds=(100.0, 10.0, 101.0, 11.0),
            )
        )


def test_clip_streams_in_small_windows(utm_raster: Path, tmp_path: Path) -> None:
    """A 1 MiB budget forces several stripes; the result must still be complete."""
    dst = tmp_path / "streamed.tif"
    with rasterio.open(utm_raster) as src:
        bounds = src.bounds
        expected = src.read()
    clip_one(ClipPayload(src=str(utm_raster), dst=str(dst), bounds=tuple(bounds), window_mb=1))
    with rasterio.open(dst) as dataset:
        np.testing.assert_array_equal(dataset.read(), expected)


def test_tile_transform_origin_is_shifted(wgs84_raster: Path, tmp_path: Path) -> None:
    with rasterio.open(wgs84_raster) as src:
        parent_transform = src.transform
        res = src.res[0]
    dst = tmp_path / "tile.tif"
    tile_one(
        TilePayload(src=str(wgs84_raster), dst=str(dst), col_off=16, row_off=8, width=16, height=8)
    )
    with rasterio.open(dst) as dataset:
        assert dataset.width == 16
        assert dataset.height == 8
        assert dataset.transform.c == pytest.approx(parent_transform.c + 16 * res)
        assert dataset.transform.f == pytest.approx(parent_transform.f - 8 * res)
        assert dataset.transform.a == pytest.approx(parent_transform.a)


def test_tile_pixels_match_source_window(wgs84_raster: Path, tmp_path: Path) -> None:
    dst = tmp_path / "tile.tif"
    with rasterio.open(wgs84_raster) as src:
        expected = src.read(1, window=rasterio.windows.Window(32, 16, 16, 16))
    tile_one(
        TilePayload(
            src=str(wgs84_raster), dst=str(dst), col_off=32, row_off=16, width=16, height=16
        )
    )
    with rasterio.open(dst) as dataset:
        np.testing.assert_array_equal(dataset.read(1), expected)


def test_tile_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ItemError, match="does not exist"):
        tile_one(
            TilePayload(
                src=str(tmp_path / "gone.tif"),
                dst=str(tmp_path / "t.tif"),
                col_off=0,
                row_off=0,
                width=4,
                height=4,
            )
        )


def test_convert_gtiff_applies_compression(wgs84_raster: Path, tmp_path: Path) -> None:
    dst = tmp_path / "lzw.tif"
    convert_one(ConvertPayload(src=str(wgs84_raster), dst=str(dst), compress="lzw"))
    with rasterio.open(dst) as dataset:
        assert dataset.profile["compress"].lower() == "lzw"
        assert dataset.profile["tiled"] is True


def test_convert_preserves_data_and_nodata(wgs84_raster: Path, tmp_path: Path) -> None:
    dst = tmp_path / "converted.tif"
    with rasterio.open(wgs84_raster) as src:
        expected = src.read()
        nodata = src.nodata
    convert_one(ConvertPayload(src=str(wgs84_raster), dst=str(dst), window_mb=1))
    with rasterio.open(dst) as dataset:
        np.testing.assert_array_equal(dataset.read(), expected)
        assert dataset.nodata == nodata


def test_convert_to_cog(make_raster: RasterFactory, tmp_path: Path) -> None:
    src = make_raster("big.tif", width=600, height=600)
    dst = tmp_path / "out.tif"
    detail = convert_one(
        ConvertPayload(
            src=str(src), dst=str(dst), driver=OutputFormat.COG, compress="deflate", blocksize=256
        )
    )
    assert "COG" in detail
    with rasterio.open(dst) as dataset:
        assert dataset.profile["compress"].lower() == "deflate"
        assert dataset.block_shapes[0] == (256, 256)
        assert dataset.overviews(1), "a COG should carry overviews"


def test_convert_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ItemError, match="does not exist"):
        convert_one(ConvertPayload(src=str(tmp_path / "gone.tif"), dst=str(tmp_path / "o.tif")))


def test_describe_reports_header(utm_raster: Path) -> None:
    info = describe(str(utm_raster))
    assert info.crs.startswith("EPSG:32610")
    assert (info.width, info.height) == (64, 48)
    assert info.count == 3
    assert info.dtype == "float32"
    assert info.nodata is None
    assert info.resolution == (30.0, 30.0)
    assert info.driver == "GTiff"


def test_describe_reports_nodata(wgs84_raster: Path) -> None:
    assert describe(str(wgs84_raster)).nodata == 0


def test_describe_rejects_corrupt(corrupt_raster: Path) -> None:
    with pytest.raises(ItemError, match="Cannot open"):
        describe(str(corrupt_raster))


def test_unwritable_output_directory(wgs84_raster: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    with pytest.raises(ItemError, match="Cannot create output directory"):
        reproject_one(
            ReprojectPayload(src=str(wgs84_raster), dst=str(blocker / "x.tif"), dst_crs="EPSG:3857")
        )
