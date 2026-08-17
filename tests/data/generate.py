import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


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
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dataset:
        dataset.write(data)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate the synthetic GeoTIFF used by detector tests."
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("synthetic.tif"),
    )
    args = parser.parse_args()
    generate_synthetic_raster(args.output_path)


if __name__ == "__main__":
    main()
