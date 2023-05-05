# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import time

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.mask
import shapely.geometry

DATASETS = {
    "fretwell": {
        "pan": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/fretwell_september_19_2012/Fretwell_12SEP19142103_PAN.tif",
        "mul": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/fretwell_september_19_2012/Fretwell_12SEP19142103_MUL_reprojected_resampled-to-pan.tif",
        "land_mask": "data/valdes_land_mask.gpkg",
        "study_area": "data/fretwell_et_al_2014_study_extent.geojson",
        "preprocessed": False,
    },
    "hannah": {
        "pan": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-5556-Hannah-WV3/14OCT16-10400100032A3700_panchromatic.vrt",
        "mul": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-5556-Hannah-WV3/14OCT16-10400100032A3700_multispectral.vrt",
        "land_mask": "data/valdes_land_mask.gpkg",
        "study_area": "data/valdes2014_study_extent.geojson",
        "preprocessed": False,
    },
    "hannah1": {
        "pan": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-5556-Hannah-WV3/14OCT16-10400100032A3700_panchromatic_study_area.tif",
        "mul": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-5556-Hannah-WV3/14OCT16-10400100032A3700_multispectral_study_area.tif",
        "land_mask": "data/valdes_land_mask.gpkg",
        "study_area": "data/valdes2014_study_extent.geojson",
        "preprocessed": True,
    },
    "hodul1": {
        "pan": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-4384-April-2021-right-whales-Cape-Cod-Bay/21APR24-1040010067D36B00_panchromatic.vrt",
        "mul": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-4384-April-2021-right-whales-Cape-Cod-Bay/21APR24-1040010067D36B00_multispectral.vrt",
        "land_mask": "data/cb_2021_us_state_500k.gpkg",
        "study_area": "data/hodul_et_al_2022_study_extent.geojson",
        "preprocessed": False,
    },
    "hodul2": {
        "pan": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-4384-April-2021-right-whales-Cape-Cod-Bay/21APR24-10400100674B2100_panchromatic.vrt",
        "mul": "https://gaiasatellite.blob.core.windows.net/gaia-images/Data/postprocessed/DAR-4384-April-2021-right-whales-Cape-Cod-Bay/21APR24-10400100674B2100_multispectral.vrt",
        "land_mask": "data/cb_2021_us_state_500k.gpkg",
        "study_area": "data/hodul_et_al_2022_study_extent.geojson",
        "preprocessed": False,
    },
}


def load_data_preprocessed(pan_uri, mul_uri, verbose=True):
    tic = time.time()
    with rasterio.open(pan_uri) as f:
        pan_crs = f.crs.to_string().lower()
        assert f.profile["count"] == 1
        pan_data = f.read()

    with rasterio.open(mul_uri) as f:
        assert f.crs.to_string().lower() == pan_crs
        profile = f.profile
        transform = profile["transform"]
        mul_data = f.read()
    if verbose:
        print(f"Finished reading data in {time.time() - tic:0.2f} seconds")

    assert (
        pan_data.shape[1] == mul_data.shape[1]
        and pan_data.shape[2] == mul_data.shape[2]
    )
    assert pan_data.dtype == mul_data.dtype
    data = np.concatenate([pan_data, mul_data], axis=0)

    return data, transform


def load_data(
    pan_uri, mul_uri, land_mask_uri, study_area_uri, preprocessed=False, verbose=True
):
    if preprocessed:
        return load_data_preprocessed(pan_uri, mul_uri, verbose)

    with fiona.open(study_area_uri) as f:
        study_area_crs = f.crs["init"]
        assert len(f) == 1
        study_area_shape = shapely.geometry.shape(next(iter(f))["geometry"])

    with fiona.open(land_mask_uri) as f:
        land_mask_crs = f.crs["init"]
        assert len(f) == 1

        row = next(iter(f))
        if land_mask_crs != study_area_crs:
            geom = fiona.transform.transform_geom(
                land_mask_crs, study_area_crs, row["geometry"]
            )
        else:
            geom = row["geometry"]
        land_mask_shape = shapely.geometry.shape(geom)
        data_mask_geom = shapely.geometry.mapping(study_area_shape - land_mask_shape)

    tic = time.time()
    with rasterio.open(pan_uri) as f:
        assert f.crs.to_string().lower() == study_area_crs
        assert f.profile["count"] == 1
        pan_data, transform = rasterio.mask.mask(f, [data_mask_geom], crop=True)

    with rasterio.open(mul_uri) as f:
        assert f.crs.to_string().lower() == study_area_crs
        mul_data, transform = rasterio.mask.mask(f, [data_mask_geom], crop=True)
    if verbose:
        print(f"Finished reading data in {time.time() - tic:0.2f} seconds")

    assert (
        pan_data.shape[1] == mul_data.shape[1]
        and pan_data.shape[2] == mul_data.shape[2]
    )
    assert pan_data.dtype == mul_data.dtype
    data = np.concatenate([pan_data, mul_data], axis=0)

    return data, transform


def load_data_from_set(name):
    assert name in DATASETS
    return load_data(
        DATASETS[name]["pan"],
        DATASETS[name]["mul"],
        DATASETS[name]["land_mask"],
        DATASETS[name]["study_area"],
        DATASETS[name]["preprocessed"],
    )
