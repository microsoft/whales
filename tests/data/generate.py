import argparse
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping


def generate_synthetic_raster(output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    data = rng.normal(100, 1, size=(1, 32, 32)).astype(np.float32)
    data[0, 14:16, 14:16] = 150

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:32618",
        transform=from_origin(500_000, 1_000, 1, 1),
        nodata=0,
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dataset:
        dataset.write(data)

    return output_path


def generate_boundary(output_path: Path, geometry, crs: str) -> Path:
    output_path = Path(output_path)
    schema = {"geometry": "Polygon", "properties": {}}
    with fiona.open(
        output_path,
        "w",
        driver="GeoJSON",
        crs=crs,
        schema=schema,
    ) as collection:
        collection.write(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {},
            }
        )
    return output_path


def generate_test_data(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_synthetic_raster(output_dir / "synthetic.tif")

    study_area = mapping(box(500_010, 980, 500_020, 990))
    generate_boundary(
        output_dir / "study_area.geojson",
        transform_geom("EPSG:32618", "EPSG:4326", study_area),
        "EPSG:4326",
    )

    generate_boundary(
        output_dir / "land_mask.geojson",
        mapping(box(500_014, 984, 500_016, 986)),
        "EPSG:32618",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate the raster and boundary data used by detector tests."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).parent,
    )
    args = parser.parse_args()
    generate_test_data(args.output_dir)


if __name__ == "__main__":
    main()
