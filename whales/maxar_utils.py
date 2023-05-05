# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import subprocess
from datetime import datetime
from pathlib import Path
from xml.dom import minidom

import fiona
import fiona.transform
import pytz
import rasterio
import rasterio.warp
import shapely.geometry
import utm
from timezonefinder import TimezoneFinder

WV3_BANDS = [
    "C",  # 0
    "B",  # 1
    "G",  # 2
    "Y",  # 3
    "R",  # 4
    "RE",  # 5
    "N",  # 6
    "N2",  # 7
]


def ensure_path_directory_representation(directory):
    if isinstance(directory, str):
        directory = Path(directory)

    if directory.is_dir():
        return directory
    else:
        raise IOError("Input is not a directory")


def get_files_from_maxar_directory(directory):
    directory = ensure_path_directory_representation(directory)

    xml_file = list(directory.glob("*.XML"))
    has_xml = len(xml_file) == 1

    til_file = list(directory.glob("*.TIL"))
    has_til = len(til_file) == 1

    tif_file = list(directory.glob("*.TIF"))
    has_tif = len(tif_file) == 1

    ntf_file = list(directory.glob("*.NTF"))
    has_ntf = len(ntf_file) == 1

    if has_til and has_xml:
        files = {"xml": xml_file[0], "til": til_file[0]}
        if has_ntf and has_tif:
            raise IOError("Has both NTF and TIF files")
        elif has_tif:
            files["img"] = tif_file[0]
        elif has_ntf:
            files["img"] = ntf_file[0]
        else:
            raise IOError("Missing NTF or TIF files")
        return files
    else:
        raise IOError("Not a maxar directory")


def get_reprojected_files_from_maxar_directory(directory):
    directory = ensure_path_directory_representation(directory)

    reprojected_file = list(directory.glob("*_reprojected.tif"))
    has_reprojected = len(reprojected_file) == 1

    if has_reprojected:
        return {"img": reprojected_file[0]}
    else:
        raise IOError("Not a maxar directory")


def convert_maxar_dir_to_cog(directory, target_extension="TIL", verbose=False):
    directory = ensure_path_directory_representation(directory)

    reprojected_file = list(directory.glob("*_reprojected.tif"))
    has_reprojected = len(reprojected_file) == 1

    if has_reprojected:
        if verbose:
            print("This directory has already been converted to a reprojected format")
    else:
        fns = list(directory.glob(f"*.{target_extension}"))
        assert len(fns) == 1
        input_fn = fns[0]
        output_fn = input_fn.parent / input_fn.name.replace(
            f".{target_extension}", "_cog.tif"
        )

        if output_fn.exists():
            if verbose:
                print("File already exists")
        else:
            if verbose:
                print("Converting...")
            command = [
                "gdalwarp",
                "-q",
                "-co",
                "BIGTIFF=YES",
                "-co",
                "NUM_THREADS=ALL_CPUS",
                "-co",
                "COMPRESS=LZW",
                "-co",
                "PREDICTOR=2",
                "-of",
                "COG",
                str(input_fn),
                str(output_fn),
            ]
            subprocess.call(command)
        return output_fn


def convert_maxar_cog_to_reprojected(directory, verbose=False):
    directory = ensure_path_directory_representation(directory)

    fns = list(directory.glob("*_cog.tif"))
    assert len(fns) == 1
    input_fn = fns[0]
    output_fn = input_fn.parent / input_fn.name.replace("_cog.tif", "_reprojected.tif")

    if output_fn.exists():
        if verbose:
            print("File already exists")
        return output_fn

    with rasterio.open(input_fn) as f:
        old_crs = f.crs
        bounds = shapely.geometry.box(*f.bounds)
        lat = bounds.centroid.y
        lon = bounds.centroid.x

    if old_crs is None:
        raise IOError("CRS doesn't exist!")

    if old_crs == "EPSG:4326":
        if verbose:
            print("Converting...")
        projection_dict = {"proj": "utm", "zone": utm.latlon_to_zone_number(lat, lon)}
        if lat < 0:
            projection_dict["south"] = True
        new_crs = rasterio.crs.CRS.from_dict(projection_dict)

        command = [
            "gdalwarp",
            "-q",
            "-t_srs",
            new_crs.to_string(),
            "-co",
            "BIGTIFF=YES",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "PREDICTOR=2",
            "-of",
            "COG",
            str(input_fn),
            str(output_fn),
        ]
        subprocess.call(command)
        os.remove(input_fn)

    else:  # assume the COG is already projected
        if verbose:
            print("No need to convert, file is already projected, just renaming")
        os.rename(input_fn, output_fn)

    return output_fn


def convert_maxar_reprojected_to_resampled(directory, verbose=False):
    directory = ensure_path_directory_representation(directory)

    fns = list(directory.glob("*_reprojected.tif"))
    assert len(fns) == 1
    input_fn = fns[0]
    output_fn = input_fn.parent / input_fn.name.replace(
        "_reprojected.tif", "_reprojected_resampled.tif"
    )

    if output_fn.exists():
        if verbose:
            print("File already exists")
    else:
        if verbose:
            print("Converting...")
        command = [
            "gdalwarp",
            "-q",
            "-of",
            "COG",
            "-tr",
            "0.3",
            "0.3",
            "-r",
            "bilinear",
            "-co",
            "BIGTIFF=YES",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "PREDICTOR=2",
            input_fn,
            output_fn,
        ]
        subprocess.call(command)
    return output_fn


def merge_panchromatic_and_multispectral(pan_fn, ms_fn, output_fn):
    if not os.path.exists(output_fn):
        with rasterio.open(pan_fn) as f:
            height = f.height
            width = f.width
            left, bottom, right, top = f.bounds
            crs = f.crs.to_string()

        command = [
            "gdalwarp",
            "-q",
            "-t_srs",
            crs,
            "-of",
            "COG",
            "-te",
            str(left),
            str(bottom),
            str(right),
            str(top),
            "-ts",
            str(width),
            str(height),
            "-r",
            "bilinear",
            "-co",
            "BIGTIFF=YES",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "PREDICTOR=2",
            ms_fn,
            output_fn,
        ]
        subprocess.call(command)

    return output_fn


def get_maxar_metadata(directory, return_flat=True, timezone_finder=None):
    directory = ensure_path_directory_representation(directory)

    fns = list(directory.glob("*.XML"))
    assert len(fns) == 1
    input_fn = str(fns[0])

    xmldoc = minidom.parse(input_fn)

    root = xmldoc.getElementsByTagName("IMD")[0]

    product_level = root.getElementsByTagName("PRODUCTLEVEL")[0].firstChild.data
    image_descriptor = root.getElementsByTagName("IMAGEDESCRIPTOR")[0].firstChild.data
    bandid = root.getElementsByTagName("BANDID")[0].firstChild.data

    catid = xmldoc.getElementsByTagName("CATID")[0].firstChild.data

    product_catalog_id = root.getElementsByTagName("PRODUCTCATALOGID")[
        0
    ].firstChild.data

    if len(root.getElementsByTagName("CHILDCATALOGID")) > 0:
        child_catalog_id = root.getElementsByTagName("CHILDCATALOGID")[
            0
        ].firstChild.data
    else:
        child_catalog_id = ""

    num_rows = int(root.getElementsByTagName("NUMROWS")[0].firstChild.data)
    num_cols = int(root.getElementsByTagName("NUMCOLUMNS")[0].firstChild.data)

    ullon, ullat, lrlon, lrlat = None, None, None, None

    bands = []
    for node in root.childNodes:
        if node.nodeName.startswith("BAND_"):
            bands.append(node.nodeName)

            if (ullon is None) or (ullat is None) or (lrlon is None) or (lrlat is None):
                ullon = float(node.getElementsByTagName("ULLON")[0].firstChild.data)
                ullat = float(node.getElementsByTagName("ULLAT")[0].firstChild.data)
                lrlon = float(node.getElementsByTagName("LRLON")[0].firstChild.data)
                lrlat = float(node.getElementsByTagName("LRLAT")[0].firstChild.data)

    center_lat = (ullat + lrlat) / 2
    center_lon = (ullon + lrlon) / 2

    image_element = root.getElementsByTagName("IMAGE")[0]

    satid = image_element.getElementsByTagName("SATID")[0].firstChild.data
    min_off_nadir_view_angle = float(
        image_element.getElementsByTagName("MINOFFNADIRVIEWANGLE")[0].firstChild.data
    )
    max_off_nadir_view_angle = float(
        image_element.getElementsByTagName("MAXOFFNADIRVIEWANGLE")[0].firstChild.data
    )
    mean_off_nadir_view_angle = float(
        image_element.getElementsByTagName("MEANOFFNADIRVIEWANGLE")[0].firstChild.data
    )

    mean_collected_col_gsd = float(
        image_element.getElementsByTagName("MEANCOLLECTEDCOLGSD")[0].firstChild.data
    )
    mean_collected_row_gsd = float(
        image_element.getElementsByTagName("MEANCOLLECTEDROWGSD")[0].firstChild.data
    )
    mean_collected_gsd = float(
        image_element.getElementsByTagName("MEANCOLLECTEDGSD")[0].firstChild.data
    )

    row_uncertainty = float(
        image_element.getElementsByTagName("ROWUNCERTAINTY")[0].firstChild.data
    )
    col_uncertainty = float(
        image_element.getElementsByTagName("COLUNCERTAINTY")[0].firstChild.data
    )

    cloud_cover = float(
        image_element.getElementsByTagName("CLOUDCOVER")[0].firstChild.data
    )

    str_date = image_element.getElementsByTagName("FIRSTLINETIME")[0].firstChild.data
    utc_date = (
        datetime.strptime(str_date, "%Y-%m-%dT%H:%M:%S.%fZ")
        .replace(microsecond=0)
        .replace(tzinfo=pytz.utc)
    )
    if timezone_finder is None:
        local_tz = TimezoneFinder().timezone_at(lng=center_lon, lat=center_lat)
    else:
        local_tz = timezone_finder.timezone_at(lng=center_lon, lat=center_lat)
    local_date = utc_date.astimezone(pytz.timezone(local_tz))

    metadata = {
        "satid": satid,
        "image_descriptor": image_descriptor,
        "product_catalog_id": product_catalog_id,
        "child_catalog_id": child_catalog_id,
        "catid": catid,
        "bandid": bandid,
        "date_string": str_date,
        "datetime_utc": utc_date.strftime("%Y-%m-%dT%H:%M:%S %Z"),
        "datetime_local": local_date.strftime("%Y-%m-%dT%H:%M:%S %Z"),
        "timezone_string": local_tz,
        "mean_off_nadir_view_angle": mean_off_nadir_view_angle,
        "gsd_row": mean_collected_row_gsd,
        "gsd_col": mean_collected_col_gsd,
        "gsd_mean": mean_collected_gsd,
        "pixel_uncertainty_row": row_uncertainty,
        "pixel_uncertainty_col": col_uncertainty,
        "num_bands": len(bands),
        "center_point_lat": center_lat,
        "center_point_lon": center_lon,
        "height": num_rows,
        "width": num_cols,
        "cloud_cover": cloud_cover,
    }
    return metadata


def build_index(root_directory, output_fn):
    if isinstance(root_directory, Path):
        root_directory = str(root_directory)
    if isinstance(output_fn, Path):
        output_fn = str(output_fn)

    assert os.path.isdir(root_directory)
    assert not os.path.exists(output_fn)

    fn_list = []
    for root, dirs, fns in os.walk(root_directory):
        if len(dirs) == 0:
            try:
                files = get_reprojected_files_from_maxar_directory(root)
                fn_list.append(
                    (
                        str(files["img"]),
                        str(root),
                    )
                )
            except IOError as e:
                print(f"ERROR at {root}, {str(e)}")

    schema = {
        "geometry": "Polygon",
        "properties": {
            "url": "str",
            "fn": "str",
            "directory": "str",
            "crs": "str",
            "x_res": "float",
            "y_res": "float",
            # The below is returned by the XML metadata
            "satid": "str",
            "image_descriptor": "str",
            "product_catalog_id": "str",
            "child_catalog_id": "str",
            "catid": "str",
            "bandid": "str",
            "date_string": "str",
            "datetime_utc": "str",
            "datetime_local": "str",
            "timezone_string": "str",
            "mean_off_nadir_view_angle": "float",
            "gsd_row": "float",
            "gsd_col": "float",
            "gsd_mean": "float",
            "pixel_uncertainty_row": "float",
            "pixel_uncertainty_col": "float",
            "num_bands": "int",
            "center_point_lat": "float",
            "center_point_lon": "float",
            "height": "int",
            "width": "int",
            "cloud_cover": "float",
        },
    }

    TIMEZONE_FINDER = TimezoneFinder(in_memory=True)

    with fiona.open(
        output_fn, "w", driver="GeoJSON", crs="EPSG:4326", schema=schema
    ) as f:
        for i, (fn, directory) in enumerate(fn_list):
            if i % 10 == 0:
                print(f"{i}/{len(fn_list)}")

            url = fn.replace(
                "/mnt/blobfuse/gaia-images/",
                "https://gaiasatellite.blob.core.windows.net/gaia-images/",
            ).replace(" ", "%20")

            with rasterio.open(url) as g:
                crs = g.crs.to_string() if g.crs is not None else "EPSG:4326"

                geom = shapely.geometry.mapping(shapely.geometry.box(*g.bounds))
                warped_geom = rasterio.warp.transform_geom(crs, "EPSG:4326", geom)

                properties = {
                    "url": url,
                    "fn": fn,
                    "directory": directory,
                    "crs": crs,
                    "x_res": g.profile["transform"].a,
                    "y_res": g.profile["transform"].e,
                }

                properties.update(
                    get_maxar_metadata(directory, timezone_finder=TIMEZONE_FINDER)
                )

                properties["height"] = g.height
                properties["width"] = g.width

                row = {
                    "type": "Feature",
                    "geometry": warped_geom,
                    "properties": properties,
                }
                f.write(row)
