import fiona
from shapely.geometry import box, mapping

from utilities.ip_pg2pt_filter import calculate_ndwi_from_geojson


def test_filter_converts_polygons_and_filters_by_percentile(tmp_path):
    input_path = tmp_path / "input.geojson"
    output_path = tmp_path / "output.geojson"
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

    calculate_ndwi_from_geojson(
        input_path,
        output_path,
        filter_by_percentile=50,
    )

    with fiona.open(output_path) as collection:
        features = list(collection)

    assert len(features) == 1
    assert features[0]["geometry"]["type"] == "Point"
    assert features[0]["geometry"]["coordinates"] == (11, 1)
    assert features[0]["properties"]["deviation"] == 10
