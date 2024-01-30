# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""This script is used to pre-process a set of 1B Maxar images.
"""
import argparse
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import fiona
import pandas as pd
from tqdm import tqdm

import whales.maxar_utils as maxar_utils
from whales.utils import pansharpen


def setup_parser():
    parser = argparse.ArgumentParser(description="Preprocess Maxar data")

    parser.add_argument(
        "--pan_dir",
        type=str,
        required=True,
        help="Input panchromatic directory",
    )
    parser.add_argument(
        "--ms_dir",
        type=str,
        required=True,
        help="Input multispectral directory",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        required=True,
        help="Input name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs",
        help="Output directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def main(args):
    # check if the input directory exists
    if not os.path.exists(args.pan_dir):
        raise ValueError("Input panchromatic directory does not exist")
    if not os.path.exists(args.ms_dir):
        raise ValueError("Input multispectral directory does not exist")

    index_geojson_fn = os.path.join(args.output_dir, f"{args.input_name}_index.geojson")
    target_dir = os.path.join(args.output_dir, args.input_name)

    # check to make sure none of the files already exist
    if any(
        [
            os.path.exists(fn)
            for fn in [
                index_geojson_fn,
                target_dir
            ]
        ]
    ):
        print("Output files already exist")
        if not args.overwrite:
            print("Exiting to avoid data loss, use `--overwrite` to overwrite files.")
            return
        else:
            print("Overwriting existing files")

    # Create output directories
    os.makedirs(target_dir, exist_ok=True)

    # Get the list of files
    pan_fns = maxar_utils.get_files_from_maxar_directory(args.pan_dir)
    ms_fns = maxar_utils.get_files_from_maxar_directory(args.ms_dir)

    # Reproject the files and save as COG
    if args.verbose:
        print("Converting input files to reprojected COGs")
    pan_cog_fn = maxar_utils.convert_maxar_til_to_reprojected_cog(pan_fns["til"], verbose=args.verbose)
    ms_cog_fn = maxar_utils.convert_maxar_til_to_reprojected_cog(ms_fns["til"], verbose=args.verbose)
    pan_cog_fn = Path(pan_cog_fn)
    ms_cog_fn = Path(ms_cog_fn)
    output_pan_cog_fn = os.path.join(target_dir, pan_cog_fn.name)
    output_ms_cog_fn = os.path.join(target_dir, ms_cog_fn.name)
    output_ms_resampled_cog_fn = os.path.join(target_dir, f"{ms_cog_fn.stem}_reprojected_resampled-to-pan.tif")
    output_pansharpened_cog_fn = os.path.join(target_dir, f"{ms_cog_fn.stem}_pansharpened.tif")
    output_geojson_fn = os.path.join(target_dir, ms_cog_fn.name.replace("_reprojected.tif", "_metadata.geojson"))

    shutil.copyfile(pan_cog_fn, output_pan_cog_fn)
    shutil.copyfile(ms_cog_fn, output_ms_cog_fn)

    # Resample the multispectral image to the size of the panchromatic image so we can
    # pansharpen easily. TODO: Should we do this in memory?
    if args.verbose:
        print("Resampling each multispectral image to the size of the panchromatic image")
    maxar_utils.merge_panchromatic_and_multispectral(
        output_pan_cog_fn, output_ms_cog_fn, output_ms_resampled_cog_fn
    )

    # Pansharpen
    if args.verbose:
        print("Pansharpening the multispectral image")
    pansharpen(
        panchromatic_fn=output_pan_cog_fn,
        multispectral_fn=output_ms_resampled_cog_fn,
        output_fn=output_pansharpened_cog_fn,
        idx_red=4,
        idx_green=2,
        idx_blue=1,
        idx_nir=6,
        method="simple_brovey",
        output_bands=[4, 2, 1],
        verbose=args.verbose,
    )

    # Grab metadata from the XML files and save to GeoJSON
    if args.verbose:
        print("Gathering metadata")
    # Example of how to get a dictionary of metadata from the XML file
    #metadata = maxar_utils.get_maxar_metadata(ms_fns["xml"])
    maxar_utils.build_index(
        output_pansharpened_cog_fn,
        pan_fns["xml"],
        output_geojson_fn
    )


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
