# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os
import time

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.features
import rasterio.io
import rasterio.mask
import shapely.geometry
import torch
from tqdm import tqdm

import whales.methods


def set_up_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_url", required=True, type=str, help="URL of COG to process"
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory (will be created if it doesn't exist)",
    )
    output_group.add_argument(
        "--output_fn",
        type=str,
        help="Filename to write output to (parent directories will be created if they don't exist)",  # noqa: E501
    )
    parser.add_argument(
        "--land_mask_fn",
        required=False,
        type=str,
        help="Filename of a vector file that contains a land mask",
    )
    parser.add_argument(
        "--study_area_fn",
        required=False,
        type=str,
        help="Filename of a vector file that contains the study area",
    )
    parser.add_argument(
        "--method",
        choices=["big_window", "rolling_window", "gmm"],
        default="big_window",
        help="Method to use for standardization",
    )
    parser.add_argument(
        "--big_window_size",
        default=1024,
        type=int,
        help="Window size to use for the `big_window` method",
    )
    parser.add_argument(
        "--area_threshold",
        default=9 * 0.25,
        type=float,
        help="Minimum size feature to keep",
    )
    parser.add_argument(
        "--difference_threshold",
        default=30,
        type=float,
        help="Threshold (in stdevs) for determining an interesting pixel",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to overwrite existing output",
    )
    parser.add_argument(
        "--gpu",
        required=False,
        type=int,
        help="GPU to use (if available)",
    )

    return parser


def main(args):
    output_dir = None
    if args.output_dir is None:
        output_dir = os.path.dirname(args.output_fn)
    else:
        output_dir = args.output_dir

    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir, exist_ok=False)

    if args.output_fn is None:
        output_fn_part = os.path.basename(args.input_url)[:-4] + "-pt.geojson"
        output_fn = os.path.join(output_dir, output_fn_part)
    else:
        if not args.output_fn.endswith(".geojson"):
            print("Output filename must end with '.geojson'")
            return
        output_fn = args.output_fn

    if os.path.exists(output_fn):
        if not args.overwrite:
            print("Output file already exists use `--overwrite` to overwrite")
            return
        else:
            os.remove(output_fn)

    if args.land_mask_fn is not None and not os.path.exists(args.land_mask_fn):
        print(f"Land mask file '{args.land_mask_fn}' does not exist")
        return

    if args.study_area_fn is not None and not os.path.exists(args.study_area_fn):
        print(f"Land mask file '{args.land_mask_fn}' does not exist")
        return

    if args.method == "rolling_window" and args.gpu is None:
        print("GPU is required for rolling window method")
        return

    if args.gpu is not None and not torch.cuda.is_available():
        print("GPU requested but CUDA is not available")
        return

    print("Reading data")
    tic = time.time()
    if args.land_mask_fn is None and args.study_area_fn is None:
        with rasterio.open(args.input_url) as f:
            nodata = f.nodata
            data = f.read()[:3, :, :]
            profile = f.profile
    else:
        land_mask = None
        study_area = None
        if args.land_mask_fn is not None:
            with fiona.open(args.land_mask_fn) as f:
                assert len(f) == 1
                land_mask_crs = f.crs["init"]
                land_mask = next(iter(f))["geometry"]
        if args.study_area_fn is not None:
            with fiona.open(args.study_area_fn) as f:
                assert len(f) == 1
                study_area_crs = f.crs["init"]
                study_area = next(iter(f))["geometry"]

        with rasterio.open(args.input_url) as f:
            crs = f.crs.to_string().lower()
            nodata = f.nodata
            if study_area is None:
                study_area = shapely.geometry.mapping(shapely.geometry.box(*f.bounds))
            elif crs != study_area_crs:
                study_area = fiona.transform.transform_geom(
                    study_area_crs, crs, study_area
                )

            if land_mask is None:
                geom = study_area
            else:
                if crs != land_mask_crs:
                    land_mask = fiona.transform.transform_geom(
                        land_mask_crs, crs, land_mask
                    )
                geom = shapely.geometry.mapping(
                    shapely.geometry.shape(study_area).difference(
                        shapely.geometry.shape(land_mask)
                    )
                )

            data, transform = rasterio.mask.mask(f, [geom], crop=True)
            data = data[:3, :, :]
            profile = f.profile
            profile["transform"] = transform
    print(f"Finished loading data in {time.time() - tic} seconds\n")

    print("Calculating deviations")
    tic = time.time()
    if args.method == "big_window":
        deviations = whales.methods.apply_chunked_standardization(data, args.big_window_size, nodata=nodata)
    elif args.method == "rolling_window":
        device = torch.device(f"cuda:{args.gpu}")
        deviations = whales.methods.apply_rolling_standardization(
            data, device, 10000, 51
        )
    elif args.method == "gmm":
        pass
    deviations = np.absolute(deviations).sum(axis=0)
    deviations[np.isnan(deviations)] = 0
    print(f"Note, the 99.95th percentile is {np.percentile(deviations, 99.95)}")
    print(f"Finished calculating deviations in {time.time() - tic} seconds\n")

    print("Computing connected features")
    tic = time.time()
    thresholded_deviations = deviations > args.difference_threshold
    outputs = list(
        rasterio.features.shapes(
            thresholded_deviations.astype(np.uint8),
            mask=None,
            connectivity=8,
            transform=profile["transform"],
        )
    )

    # Calculate the mean deviation for each feature -- we do this with a memory file for speed
    all_vals = []
    with rasterio.io.MemoryFile() as memfile:
        t_profile = {
            "driver": "GTiff",
            "height": deviations.shape[0],
            "width": deviations.shape[1],
            "count": 1,
            "dtype": deviations.dtype,
            "crs": profile["crs"],
            "transform": profile["transform"],
        }
        with memfile.open(**t_profile) as dataset:
            dataset.write(deviations, 1)

        with memfile.open() as f:
            for geom, val in tqdm(outputs):
                if val == 1:
                    data, _ = rasterio.mask.mask(f, [geom], crop=True)
                    all_vals.append(float(data.mean()))
                else:
                    all_vals.append(float("inf"))
    print(f"Found {len(outputs)} features in {time.time() - tic} seconds\n")

    print("Writing output")
    tic = time.time()
    schema = {
        "geometry": "Point",
        "properties": {"id": "int", "area": "float", "deviation": "float"},
    }

    count = 0
    with fiona.open(
        output_fn,
        "w",
        driver="GeoJSON",
        crs=profile["crs"].to_string(),
        schema=schema,
    ) as f:
        for i, (geom, val) in enumerate(tqdm(outputs)):
            shape = shapely.geometry.shape(geom)
            area = shape.area
            if val == 1 and area > args.area_threshold:
                row = {
                    "type": "Feature",
                    "geometry": shapely.geometry.mapping(shape.centroid),
                    "properties": {
                        "id": i,
                        "area": area,
                        "deviation": all_vals[i],
                    },
                }
                f.write(row)
                count += 1

    print(
        f"Wrote {count} features to '{output_fn}' in" + f" {time.time() - tic} seconds"
    )


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
