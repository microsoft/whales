from pathlib import Path

import fiona
import pytest
import rasterio

from generate_interesting_points import main, set_up_parser

DATA_DIR = Path(__file__).parent / "data"


def run_detector(tmp_path, *extra_args):
    input_path = DATA_DIR / "synthetic.tif"
    output_path = tmp_path / "interesting.geojson"

    args = set_up_parser().parse_args(
        [
            "--input_fn",
            str(input_path),
            "--output_fn",
            str(output_path),
            "--method",
            "big_window",
            "--big_window_size",
            "64",
            "--difference_threshold",
            "10",
            "--area_threshold",
            "1",
            "--max_area_threshold",
            "10",
            *extra_args,
        ]
    )

    main(args)

    with fiona.open(output_path) as collection:
        return list(collection)


def test_detector_finds_bright_feature_in_synthetic_raster(tmp_path):
    with rasterio.open(DATA_DIR / "synthetic.tif") as dataset:
        assert dataset.count == 1
        assert dataset.shape == (32, 32)
        assert dataset.dtypes == ("float32",)
        assert dataset.crs.to_string() == "EPSG:32618"
        assert dataset.nodata == 0
        assert dataset.profile["compress"] == "deflate"
        assert dataset.profile["tiled"] is True

    features = run_detector(tmp_path)
    assert len(features) == 1
    assert features[0]["geometry"]["type"] == "Point"
    assert features[0]["geometry"]["coordinates"] == pytest.approx((500_015, 985))
    assert features[0]["properties"]["area"] == pytest.approx(4)
    assert features[0]["properties"]["deviation"] > 10


def test_detector_reprojects_and_applies_study_area(tmp_path):
    features = run_detector(
        tmp_path,
        "--study_area_fn",
        str(DATA_DIR / "study_area.geojson"),
        "--difference_threshold",
        "4",
    )

    assert len(features) == 1
    assert features[0]["geometry"]["coordinates"] == pytest.approx((500_015, 985))


def test_detector_applies_land_mask(tmp_path):
    features = run_detector(
        tmp_path,
        "--land_mask_fn",
        str(DATA_DIR / "land_mask.geojson"),
    )

    assert features == []
