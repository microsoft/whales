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
        "--input_fn", required=True, type=str, help="URL of COG to process"
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
        help="Path to a vector file containing a single polygon feature representing land areas "
             "to exclude from processing. The land mask will be subtracted from the study area. "
             "Can be in any CRS; will be reprojected to match the input raster if needed.",
    )
    parser.add_argument(
        "--study_area_fn",
        required=False,
        type=str,
        help="Path to a vector file containing a single polygon feature defining the region of "
             "interest to analyze. If not provided, the full extent of the input raster is used. "
             "Can be in any CRS; will be reprojected to match the input raster if needed.",
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
        help="Minimum size feature to keep (in map units, e.g., square meters if data is in a UTM projection)",
    )
    parser.add_argument(
        "--difference_threshold",
        default=30,
        type=float,
        help="Threshold (in stdevs) for determining an interesting pixel",
    )
    parser.add_argument(
        "--auto_difference_threshold",
        action="store_true",
        help="Set the difference_threshold automatically based on distribution of deviations"
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
    parser.add_argument(
        "--bands",
        required=False,
        type=str,
        default=None,
        help="Comma-separated list of band indices (1-based) to use, e.g., '1,2,3' for RGB. "
             "If not specified, all bands are used.",
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
        output_fn_part = os.path.basename(args.input_fn)[:-4] + "-pt.geojson"
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
        print(f"Study area file '{args.study_area_fn}' does not exist")
        return

    if args.method == "rolling_window" and args.gpu is None:
        print("GPU is required for rolling window method")
        return

    if args.gpu is not None and not torch.cuda.is_available():
        print("GPU requested but CUDA is not available")
        return

    if not os.path.exists(args.input_fn) and not args.input_fn.startswith(("http://", "https://", "s3://")):
        print(f"Input file '{args.input_fn}' does not exist")
        return

    print("Reading data")
    tic = time.time()
    
    # Parse band indices if specified
    band_indices = None
    if args.bands is not None:
        try:
            band_indices = [int(b.strip()) for b in args.bands.split(",")]
        except ValueError:
            print(f"Invalid bands specification '{args.bands}'. Expected comma-separated integers (1-based).")
            return
    
    if args.land_mask_fn is None and args.study_area_fn is None:
        with rasterio.open(args.input_fn) as f:
            nodata = f.nodata
            if band_indices is not None:
                data = f.read(band_indices)
            else:
                data = f.read()
            profile = f.profile
    else:
        land_mask = None
        study_area = None
        if args.land_mask_fn is not None:
            with fiona.open(args.land_mask_fn) as f:
                if len(f) != 1:
                    print(f"Land mask file must contain exactly 1 feature (found {len(f)}). "
                          "This file should contain a polygon representing land areas to exclude from processing.")
                    return
                land_mask_crs = f.crs.to_string().lower() if hasattr(f.crs, 'to_string') else f.crs.get("init", str(f.crs)).lower()
                land_mask = next(iter(f))["geometry"]
        if args.study_area_fn is not None:
            with fiona.open(args.study_area_fn) as f:
                if len(f) != 1:
                    print(f"Study area file must contain exactly 1 feature (found {len(f)}). "
                          "This file should contain a polygon defining the region of interest to analyze.")
                    return
                study_area_crs = f.crs.to_string().lower() if hasattr(f.crs, 'to_string') else f.crs.get("init", str(f.crs)).lower()
                study_area = next(iter(f))["geometry"]

        with rasterio.open(args.input_fn) as f:
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

            data, transform = rasterio.mask.mask(f, [geom], crop=True, indexes=band_indices)
            profile = f.profile
            profile["transform"] = transform
    
    print(f"Loaded {data.shape[0]} bands with shape {data.shape[1:]}")
    print(f"Finished loading data in {time.time() - tic:.2f} seconds\n")

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
        raise NotImplementedError("GMM method is not yet implemented")
    deviations = np.absolute(deviations).sum(axis=0)
    deviations[np.isnan(deviations)] = 0
    print(f"Note, the 99.95th percentile is {np.percentile(deviations, 99.95)}")
    print(f"Finished calculating deviations in {time.time() - tic} seconds\n")

    if args.auto_difference_threshold:
        difference_threshold = np.percentile(deviations, 99.95)
    else:
        difference_threshold = args.difference_threshold

    print("Computing connected features")
    tic = time.time()

    thresholded_deviations = (deviations > difference_threshold)
    outputs = list(
        rasterio.features.shapes(
            thresholded_deviations.astype(np.uint8),
            mask=None,
            connectivity=8,
            transform=profile["transform"],
        )
    )

    # Calculate the mean deviation for each feature and check for all-zero pixels
    # We use memory files for speed
    all_vals = []
    has_zero_pixels = []
    
    base_profile = {
        "driver": "GTiff",
        "height": deviations.shape[0],
        "width": deviations.shape[1],
        "count": 1,
        "crs": profile["crs"],
        "transform": profile["transform"],
    }
    
    with rasterio.io.MemoryFile() as dev_memfile, rasterio.open(args.input_fn) as src:
        # Write deviations to memory file
        dev_profile = {**base_profile, "dtype": deviations.dtype}
        with dev_memfile.open(**dev_profile) as dataset:
            dataset.write(deviations, 1)

        with dev_memfile.open() as dev_f:
            for geom, val in tqdm(outputs):
                if val == 1:
                    feature_devs, _ = rasterio.mask.mask(dev_f, [geom], crop=True, filled=False)
                    all_vals.append(float(feature_devs.mean()))

                    # Check original imagery for all-zero pixels (only within the geometry)
                    feature_data, _ = rasterio.mask.mask(src, [geom], crop=True, indexes=band_indices, filled=False)
                    valid_mask = ~feature_data.mask[0]  # Pixels inside the geometry
                    all_zeros = np.all(feature_data.data == 0, axis=0)
                    has_zero_pixels.append(np.any(all_zeros & valid_mask))
                else:
                    all_vals.append(float("inf"))
                    has_zero_pixels.append(True)
    print(f"Found {len(outputs)} features in {time.time() - tic} seconds\n")

    print("Writing output")
    tic = time.time()
    schema = {
        "geometry": "Point",
        "properties": {"id": "int", "area": "float", "deviation": "float"},
    }

    count = 0
    max_area = args.area_threshold * 5
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
            if val == 1 and area > args.area_threshold and area <= max_area and not has_zero_pixels[i]:
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
