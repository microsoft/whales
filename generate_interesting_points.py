# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.features
import rasterio.io
import rasterio.mask
import shapely.geometry
import torch
from rasterio.enums import Resampling
from tqdm import tqdm

import whales.methods
import whales.utils

torch.set_num_threads(os.cpu_count())

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
        "--rolling_window_size",
        default=51,
        type=int,
        help="Kernel size to use for the `rolling_window` method",
    )
    parser.add_argument(
        "--min_stdev",
        default=None,
        type=int,
        help="Minimum standard deviation to use as the denominator for the 'rolling_window' and 'big_window' methods "
             "(reduces anomalous high deviation scores in areas of low variance)",
    )
    parser.add_argument(
        "--area_threshold",
        default=9 * 0.25,
        type=float,
        help="Minimum size feature to keep (in map units, e.g., square meters if data is in a UTM projection)",
    )
    parser.add_argument(
        "--max_area_threshold",
        type=float,
        help="Maximum size feature to keep in map units (default: 5 * area_threshold)",
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
    parser.add_argument(
        "--return-full-shapes",
        action="store_true",
        help="Output full polygon shapes with deviation statistics (mean, max, std) instead of "
             "centroid points with mean deviation only",
    )
    #TODO delete deviation raster if false
    parser.add_argument(
        "--write-deviation-raster",
        action="store_true",
        help="Write out the deviation values in raster form (for debugging thresholds)",
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
            print(f'Nodata value: {nodata}')
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
        deviations = whales.methods.apply_chunked_standardization(data, args.big_window_size, min_stdev=args.min_stdev, nodata=nodata)
    elif args.method == "rolling_window":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            print(f"GPU found. Running on {device}.")
        else:
            device = torch.device("cpu")
            print("No GPU detected. Falling back to CPU.")
        deviations = whales.methods.apply_rolling_standardization(
            data, device, 10000, args.rolling_window_size, min_stdev=args.min_stdev, nodata=nodata
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

    base_profile = {
        "driver": "GTiff",
        "height": deviations.shape[0],
        "width": deviations.shape[1],
        "count": 1,
        "crs": profile["crs"],
        "transform": profile["transform"],
    }

    # Write deviations with overviews to disk for parallelized rasterio.mask.mask
    dev_profile = {**base_profile, "dtype": deviations.dtype}
    output_deviations_fn = output_fn.replace(".geojson", "_deviations.tif")
    with rasterio.open(
        output_deviations_fn,
        "w",
        **dev_profile,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="YES",
    ) as dst:
        dst.write(deviations, 1)
    print(f"Wrote deviations to {output_deviations_fn}")

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

    indexed_outputs = list(enumerate(outputs))
    chunk_size=500
    max_workers=os.cpu_count()
    chunks = [
        indexed_outputs[i : i + chunk_size]
        for i in range(0, len(indexed_outputs), chunk_size)
    ]

    task_args = [(chunk, output_deviations_fn, args.input_fn, band_indices) for chunk in chunks]

    # Calculate deviation statistics for each feature and check for all-zero pixels
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunk_result in tqdm(
            executor.map(whales.utils.process_geometry_chunk, task_args), total=len(chunks)
        ):
            all_results.extend(chunk_result)

    # Re-sort the results by the original index to fix any out-of-order returns
    all_results.sort(key=lambda x: x[0])
    all_means = [r[1] for r in all_results]
    all_maxes = [r[2] for r in all_results]
    all_stds = [r[3] for r in all_results]
    has_zero_pixels = [r[4] for r in all_results]
    print(f"Found {len(outputs)} features in {time.time() - tic} seconds\n")

    print("Writing output")
    tic = time.time()
    if args.return_full_shapes:
        schema = {
            "geometry": "Polygon",
            "properties": {
                "id": "int",
                "area": "float",
                "deviation_mean": "float",
                "deviation_max": "float",
                "deviation_std": "float",
            },
        }
    else:
        schema = {
            "geometry": "Point",
            "properties": {"id": "int", "area": "float", "deviation": "float"},
        }

    count = 0
    max_area = args.area_threshold * 5 if args.max_area_threshold is None else args.max_area_threshold
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
                if args.return_full_shapes:
                    row = {
                        "type": "Feature",
                        "geometry": shapely.geometry.mapping(shape),
                        "properties": {
                            "id": i,
                            "area": area,
                            "deviation_mean": all_means[i],
                            "deviation_max": all_maxes[i],
                            "deviation_std": all_stds[i],
                        },
                    }
                else:
                    row = {
                        "type": "Feature",
                        "geometry": shapely.geometry.mapping(shape.centroid),
                        "properties": {
                            "id": i,
                            "area": area,
                            "deviation": all_means[i],
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
