# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This script is used to pre-process a set of 1B Maxar images.

It takes as input:
- An input directory that is assumed to contain unzipped folders of 1B Maxar products.
    Here each folder will be a multispectral or panchromatic scene and contain
    unprojected TIF of NTF files.
- An input name that is used to name the _set_ of output files
- An output directory that is used to store the output files

"""
import argparse
import os
import shutil
import subprocess
from collections import defaultdict

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

    # ------------------------------------------------
    # Part 1 - convert to COG and reproject to UTM
    # ------------------------------------------------
    pan_fns = maxar_utils.get_files_from_maxar_directory(args.pan_dir)
    ms_fns = maxar_utils.get_files_from_maxar_directory(args.ms_dir)

    print("Converting input files to reprojected COGs")
    output_pan_fn = maxar_utils.convert_maxar_til_to_reprojected_cog(pan_fns["til"], verbose=args.verbose)
    output_ms_fn = maxar_utils.convert_maxar_til_to_reprojected_cog(ms_fns["til"], verbose=args.verbose)

    print("Created:")
    print(f" - {output_pan_fn}")
    print(f" - {output_ms_fn}")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
