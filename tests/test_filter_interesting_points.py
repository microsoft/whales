
import fiona
from shapely.geometry import box, mapping, Point
import rasterio
import numpy as np

from utilities.filter_interesting_points import process_features


def test_filter_converts_polygons_and_filters_by_percentile(tmp_path):
    input_path = tmp_path / "input.geojson"
    output_path = tmp_path / "output.geojson"
    pan_path = tmp_path / "pan.tif"
    schema = {
        "geometry": "Polygon",
        "properties": {"deviation_mean": "float"},
    }

    with fiona.open(
        input_path,
        "w",
        driver="GeoJSON",
        crs="EPSG:32618",
        schema=schema,
    ) as collection:
        for x, deviation in [(0, 1.0), (10, 10.0)]:
            collection.write(
                {
                    "type": "Feature",
                    "geometry": mapping(box(x, 0, x + 2, 2)),
                    "properties": {"deviation_mean": deviation},
                }
            )
    
    with rasterio.open(pan_path, 'w', driver='GTiff', height=10, width=20, count=1, dtype=np.uint8, crs='EPSG:32618', transform=rasterio.transform.from_origin(0, 10, 1, 1)) as dst:
        dst.write(np.ones((10, 20), dtype=np.uint8), 1)

    process_features(
        input_path,
        output_path,
        pan_image_path=pan_path,
        filter_by_percentile=50,
    )

    with fiona.open(output_path) as collection:
        features = list(collection)

    assert len(features) == 1
    assert features[0]["geometry"]["type"] == "Point"
    assert features[0]["geometry"]["coordinates"] == (11, 1)
    assert features[0]["properties"]["deviation"] == 10


def test_point_geometry_is_processed_correctly(tmp_path):
    input_path = tmp_path / "input.geojson"
    output_path = tmp_path / "output.geojson"
    pan_path = tmp_path / "pan.tif"
    mul_path = tmp_path / "mul.tif"
    schema = {
        "geometry": "Point",
        "properties": {},
    }

    with fiona.open(
        input_path,
        "w",
        driver="GeoJSON",
        crs="EPSG:32618",
        schema=schema,
    ) as collection:
        collection.write(
            {
                "type": "Feature",
                "geometry": mapping(Point(10, 5)),
                "properties": {},
            }
        )

    transform = rasterio.transform.from_origin(0, 10, 1, 1)
    with rasterio.open(pan_path, 'w', driver='GTiff', height=10, width=20, count=1, dtype=np.uint8, crs='EPSG:32618', transform=transform) as dst:
        data = np.ones((10, 20), dtype=np.uint8) * 123
        dst.write(data, 1)

    with rasterio.open(mul_path, 'w', driver='GTiff', height=10, width=20, count=8, dtype=np.uint8, crs='EPSG:32618', transform=transform) as dst:
        # Green band (3) = 100, NIR band (8) = 50
        data = np.zeros((8, 10, 20), dtype=np.uint8)
        data[2, :, :] = 100
        data[7, :, :] = 50
        for i in range(8):
            dst.write(data[i], i + 1)

    process_features(
        input_path,
        output_path,
        pan_image_path=pan_path,
        raster_path=mul_path,
    )

    with fiona.open(output_path) as collection:
        features = list(collection)

    assert len(features) == 1
    feature = features[0]
    assert feature["geometry"]["type"] == "Point"
    assert feature["geometry"]["coordinates"] == (10, 5)
    assert "pan_value" in feature["properties"]
    assert feature["properties"]["pan_value"] == 123
    assert "ndwi" in feature["properties"]
    assert np.isclose(feature["properties"]["ndwi"], (100 - 50) / (100 + 50))
    assert "water" in feature["properties"]
    assert feature["properties"]["water"] == "probably water"
