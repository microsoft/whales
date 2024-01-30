# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp


def get_pareto_points(points, minimize_y=False):
    """Return the Pareto points of a set of points.

    If `maximize_y` is True, then points with the largest y value are returned, else
    points with the smallest y value are returned.
    In all cases, points with largest x value are returned.
    """
    points = sorted(points, key=lambda x: x[0], reverse=True)
    pareto_points = [points[0]]
    for point in points[1:]:
        # check the direction that we are optimizing for
        check = point[1] >= pareto_points[-1][1]
        if minimize_y:
            check = point[1] <= pareto_points[-1][1]

        if check:
            if point[0] < pareto_points[-1][0]:
                pareto_points.append(point)
            else:
                pareto_points[-1] = point
    return pareto_points


def pansharpen(
    panchromatic_fn: str,
    multispectral_fn: str,
    output_fn: str,
    geom: Optional[Dict[str, Any]] = None,
    idx_red: int = 1,
    idx_green: int = 2,
    idx_blue: int = 3,
    idx_nir: int = 4,
    method: str = "simple_brovey",
    W: float = 0.1,
    output_bands: List[int] = [2, 1, 0, 3],
    force_byte: bool = False,
    verbose: bool = True,
):
    """This function is used to pansharpen a given multispectral image using its
    corresponding panchromatic image with one of the following algorithms: Simple Brovey,
    Simple Mean, ESRI, Brovey.

    This is a heavily adapted version of the code from https://github.com/ThomasWangWeiHong/Simple-Pansharpening-Algorithms.

    Some notes:
    - The output file is saved in cloud optimized geotiff format.
    - If the height and width of the multispectral imagery is smaller than that of the
        panchromatic data we assume it needs to be resampled and do so with linear
        resampling. Else, the height and width of both must match (i.e. the multispectral
        imagery should be manually resampled to match).
    - We cast the inputs as float32s then cast back to the dtype of the panchromatic imagery
        when actually performing the pansharpening steps. TODO: find out how this might
        break things.
    - We do _not_ check to see if the aspect ratios of the panchromatic data and
        multispectral data approximately match. TODO: do this.

    Args:
        panchromatic_fn: The path to a GeoTIFF containing the panchromatic band
        multispectral_fn: The path to a GeoTIFF containing the multispectral bands
        output_fn: The path to write the pansharpened result (must not exist)
        geom: Either a GeoJSON description of the subset area of the input to pansharpen
            or None to pansharpen the entire input
        idx_red: Zero based index of the red band in the multispectral image
        idx_green: Zero based index of the green band in the multispectral image
        idx_blue: Zero based index of the blue band in the multispectral image
        idx_nir: Zero based index of the NIR band in the multispectral image
        method: The type of pansharpening method to use. Options are: simple_brovey,
            brovey, simple_mean, esri.
        W: Weight value to use for the brovey pansharpening methods
        output_bands: List of the multispectral bands to pansharpen and include in the
            output file
        force_byte: Flag the converts the output to byte format without scaling
        verbose: Flag to enable printing timing information and progress during execution

    Returns:
        pansharpened_image: The pansharpened multispectral bands in C x H x W format

    Raises:
        ValueError: if the `method` parameter is not valid or the dimensions of the
            inputs don't make sense
        IOError: if the `output_fn` parameter already exists as a file
    """

    if method not in {"simple_brovey", "simple_mean", "esri", "brovey"}:
        raise ValueError(f"Method '{method}' not recognized")
    if os.path.exists(output_fn):
        raise IOError("The output file already exists")

    eps = 0.00001

    # Read panchromatic data
    tic = time.time()
    with rasterio.open(panchromatic_fn) as f:
        pan_metadata = f.profile
        pan_crs = f.crs.to_string()
        if geom is not None:
            geom = rasterio.warp.transform_geom("epsg:4326", pan_crs, geom)
            pan_img, pan_transform = rasterio.mask.mask(f, [geom], crop=True)
        else:
            pan_img = f.read()
            pan_transform = f.transform
        pan_img = pan_img.squeeze().astype(np.float32)
        assert (
            len(pan_img.shape) == 2
        ), "The panchromatic input should contain a single band"
        pan_height = pan_img.shape[0]
        pan_width = pan_img.shape[1]
    if verbose:
        print(
            f"Finished reading the panchromatic data in {time.time() - tic:0.2f} seconds"
        )

    # Read multispectral data
    tic = time.time()
    with rasterio.open(multispectral_fn) as f:
        assert (
            f.crs.to_string() == pan_crs
        ), "The two inputs should be in the same projection"
        if geom is not None:
            multispectral_img, _ = rasterio.mask.mask(f, [geom], crop=True)
        else:
            multispectral_img = f.read()
        multispectral_img = np.rollaxis(multispectral_img, 0, 3).astype(np.float32)
        multispectral_height = multispectral_img.shape[0]
        multispectral_width = multispectral_img.shape[1]
    if verbose:
        print(
            f"Finished reading the multispectral data in {time.time() - tic:0.2f} seconds"
        )

    # Resample the multispectral data if needed
    tic = time.time()
    if multispectral_height < pan_height and multispectral_width < pan_width:
        multispectral_img = cv2.resize(
            multispectral_img, (pan_width, pan_height), interpolation=cv2.INTER_LINEAR
        )
    elif multispectral_height == pan_height and multispectral_width == pan_width:
        pass  # The inputs are the same size, we assume that the multispectral data has
        # been resampled to the dimensions of the panchromatic data
    else:
        raise ValueError(
            f"The dimensions of the multispectral imagery ({multispectral_height},"
            + f" {multispectral_width}) are unexpected given the dimensions of"
            + f" the panchromatic image ({pan_height}, {pan_width})"
        )
    if verbose:
        print(
            "Finished resizing the multispectral data in"
            + f" {time.time() - tic:0.2f} seconds"
        )

    # Do pansharpening
    tic = time.time()
    pansharpened_image = np.zeros(
        (len(output_bands), pan_height, pan_width), dtype=pan_metadata["dtype"]
    )
    if method == "simple_brovey":
        all_in = (
            multispectral_img[:, :, idx_red]
            + multispectral_img[:, :, idx_green]
            + multispectral_img[:, :, idx_blue]
            + multispectral_img[:, :, idx_nir]
        )
        for i, band in enumerate(output_bands):
            pansharpened_image[i, :, :] = np.multiply(
                multispectral_img[:, :, band], (pan_img / (all_in + eps))
            )

    elif method == "simple_mean":
        for i, band in enumerate(output_bands):
            pansharpened_image[i, :, :] = 0.5 * (
                multispectral_img[:, :, band] + pan_img
            )

    elif method == "esri":
        adj = pan_img - multispectral_img.mean(axis=2)
        for i, band in enumerate(output_bands):
            pansharpened_image[i, :, :] = multispectral_img[:, :, band] + adj

    elif method == "brovey":
        dnf = (pan_img - W * multispectral_img[:, :, idx_nir]) / (
            W * multispectral_img[:, :, idx_red]
            + W * multispectral_img[:, :, idx_green]
            + W * multispectral_img[:, :, idx_blue]
            + eps
        )
        for i, band in enumerate(output_bands):
            pansharpened_image[i, :, :] = multispectral_img[:, :, band] * dnf
    if verbose:
        print(f"Finished pansharpening in {time.time() - tic:0.2f} seconds")

    # Write output
    tic = time.time()
    new_metadata = {
        "driver": "COG",
        "dtype": pan_metadata["dtype"] if not force_byte else "uint8",
        "nodata": None,
        "width": pan_width,
        "height": pan_height,
        "count": len(output_bands),
        "crs": pan_crs,
        "transform": pan_transform,
        "compress": "lzw",
        "predictor": 2,
        "BIGTIFF": "YES",
    }
    with rasterio.open(output_fn, "w", **new_metadata) as f:
        if force_byte:
            f.write(pansharpened_image.astype(np.uint8))
        else:
            f.write(pansharpened_image)
    if verbose:
        print(f"Finished writing output in {time.time() - tic:0.2f} seconds")

    return pansharpened_image
