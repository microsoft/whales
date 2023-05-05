# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os

import rasterio
import rasterio.mask
import rasterio.windows
import fiona
import fiona.transform
import numpy as np
import shapely.geometry
import imageio
import json
from tqdm import tqdm


def set_up_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name", required=True, type=str, help="Name of the run"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--input_fn",
        type=str,
        help="Input filename containing interesting points",
    )
    parser.add_argument(
        "--image_url",
        required=True,
        type=str,
        help="Input filename containing interesting points",
    )
    parser.add_argument(
        "--buffer",
        default=50,
        type=int,
        help="Amount to buffer around each point (in meters)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to overwrite existing output",
    )
    parser.add_argument(
        "--dar",
        required=False,
        type=str,
        help="dar to log for this run (if this isn't set we try to interpret the image URL)",
    )
    parser.add_argument(
        "--catid",
        required=False,
        type=str,
        help="Catid to log for this run (if this isn't set we try to interpret the image URL)",
    )
    parser.add_argument(
        "--date",
        required=False,
        type=str,
        help="Date to log for this run (if this isn't set we try to interpret the image URL)",
    )

    return parser


def main(args):

    assert args.input_fn.endswith(".geojson")
    assert args.image_url.endswith(".tif")
    if os.path.exists(args.output_dir) and not args.overwrite:
        raise ValueError(
            "Output directory already exists. Set --overwrite to overwrite existing output."
        )

    os.makedirs(os.path.join(args.output_dir, "patches"), exist_ok=True)

    if args.dar is not None:
        dar = args.dar
    else:
        dar = args.image_url.split("/")[-2]

    if args.catid is not None:
        catid = args.catid
    else:
        catid = args.image_url.split("/")[-1].split("_")[0].split("-")[1]

    if args.date is not None:
        date = args.date
    else:
        date = args.image_url.split("/")[-1].split("_")[0].split("-")[0]

    # Load the points from file
    idxs = []
    latlons = []
    geoms = []
    fns = []
    with fiona.open(args.input_fn) as f:
        src_crs = f.crs["init"]
        for row in tqdm(f):
            geom = fiona.transform.transform_geom(src_crs, "epsg:4326", row["geometry"])
            lon, lat = geom["coordinates"]
            latlons.append((lat, lon))
            geoms.append(geom)
            idxs.append(row["id"])

    # Crop out image patches for each point
    with rasterio.open(args.image_url) as f:
        y_res, x_res = f.res
        dst_crs = f.crs.to_string()

        for i, geom in enumerate(tqdm(geoms)):
            idx = idxs[i]
            warped_geom = fiona.transform.transform_geom("epsg:4326", dst_crs, geom)
            warped_shape = shapely.geometry.shape(warped_geom)
            warped_envelope_shape = warped_shape.buffer(args.buffer).envelope
            warped_envelope_geom = shapely.geometry.mapping(warped_envelope_shape)

            out_image, _ = rasterio.mask.mask(
                f, [warped_envelope_geom], crop=True, all_touched=True
            )
            img = np.rollaxis(out_image, 0, 3)
            img = np.clip(img, 0, 255).astype(np.uint8).copy()

            output_fn = os.path.join(args.output_dir, "patches", f"{idx}.png")
            imageio.imwrite(output_fn, img)
            fns.append(output_fn)

    # Write out the inputs.csv file
    with open(os.path.join(args.output_dir, "inputs.csv"), "w") as f:
        f.write("idx,lat,lon,fn\n")
        for i, idx in enumerate(idxs):
            lat, lon = latlons[i]
            fn = fns[i]

            f.write(f"{idx},{lat},{lon},{fn}\n")

    # Write out the metadata.json file
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        metadata = {
            "run_name": args.run_name,
            "image_url": args.image_url,
            "input_fn": args.input_fn,
            "date": date,
            "dar": dar,
            "catid": catid,
            "buffer_amount": args.buffer,
            "x_res": x_res,
            "y_res": y_res,
            "dst_crs": dst_crs,
        }
        f.write(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
