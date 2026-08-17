import fiona
import pytest
import rasterio

from generate_interesting_points import main, set_up_parser
from tests.data.generate import generate_synthetic_raster


def test_detector_finds_bright_feature_in_synthetic_raster(tmp_path):
    input_path = tmp_path / "synthetic.tif"
    output_path = tmp_path / "interesting.geojson"
    generate_synthetic_raster(input_path)

    with rasterio.open(input_path) as dataset:
        assert dataset.profile["compress"] == "deflate"
        assert dataset.profile["tiled"] is True

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
        ]
    )

    main(args)

    with fiona.open(output_path) as collection:
        features = list(collection)

    assert len(features) == 1
    assert features[0]["geometry"]["type"] == "Point"
    assert features[0]["properties"]["area"] == pytest.approx(4)
    assert features[0]["properties"]["deviation"] > 10
