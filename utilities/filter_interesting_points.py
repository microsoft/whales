
import fiona
import rasterio
import rasterio.mask
import numpy as np
import argparse
import os
import logging
import json
import xml.etree.ElementTree as ET

from shapely.geometry import shape, mapping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_up_parser():
    parser = argparse.ArgumentParser(description='Process GeoJSON features, optionally using raster data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('geojson_path', help='Path to the input GeoJSON file')
    parser.add_argument('output_geojson_path', help='Path to save the output GeoJSON file')
    parser.add_argument('pan_image_path', help='Path to the panchromatic image')
    parser.add_argument('--filter-by-ndwi',
                        help='Path to the input multiband raster file (optional, ndwi > 0.3)')
    parser.add_argument('--green-band-idx', type=int, default=3, help='1-based index of the green band')
    parser.add_argument('--nir-band-idx', type=int, default=8, help='1-based index of the NIR band')
    parser.add_argument('--filter-by-pan-threshold', type=float,
                        help='Filter by panchromatic image pixel value (optional)')
    parser.add_argument('--filter-by-percentile', type=float,
                        help='Filter by top N percentile of mean deviation scores (optional)')
    return parser


def get_water_class(ndwi) -> str:
    """
    Determines the water classification based on the NDWI value.
    Returns the water classification string.
    """
    if ndwi < 0.3:
        return 'not water'
    elif ndwi < 0.5:
        return 'probably water'
    else:
        return 'water'


def get_ndwi_value(green_band, nir_band) -> float:
    """
    Calculates the final NDWI value for a feature from green and NIR bands,
    handling both single values and arrays.
    """
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green_band - nir_band) / (green_band + nir_band)
    if isinstance(ndwi, np.ndarray):
        return np.nanmean(ndwi)
    return ndwi


def process_features(geojson_path, output_geojson_path, pan_image_path, raster_path=None,
                     green_band_idx=3, nir_band_idx=8,
                     filter_by_pan_threshold=None, filter_by_percentile=None) -> int:
    """
    Processes GeoJSON features, calculates NDWI and pan values, and filters them.
    """

    if raster_path and not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    if not os.path.exists(pan_image_path):
        raise FileNotFoundError(f"Panchromatic image file not found: {pan_image_path}")
    min_band_idx = min(green_band_idx, nir_band_idx)
    if min_band_idx < 1:
        raise ValueError("Band indices must be 1-based.")

    if filter_by_percentile is not None:
        if not 0 <= filter_by_percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100.")

    logging.info(f"Reading GeoJSON file: {geojson_path}")
    with fiona.open(geojson_path, 'r') as collection:
        features = list(collection)
        schema = collection.schema
        crs = collection.crs

    new_features = []

    logging.info(f"Processing rasters")
    pan_src = rasterio.open(pan_image_path)
    src = rasterio.open(raster_path) if raster_path else None

    skip_ndwi = False
    try:
        check_band_count(green_band_idx, nir_band_idx, src)
    except ValueError as e:
        logging.warning(e)
        logging.warning("Skipping NDWI calculation")
        skip_ndwi = True

    # Adjust to 0-based index for array access
    green_idx, nir_idx = green_band_idx - 1, nir_band_idx - 1

    try:
        logging.info("Processing features")
        for feature in features:
            properties = dict(feature['properties'])
            geom = shape(feature['geometry'])

            if geom.geom_type == 'Point':
                coords = [geom.x, geom.y]
                if src and not skip_ndwi:
                    try:
                        for val in src.sample([coords]):
                            green_band = val[green_idx].astype(float)
                            nir_band = val[nir_idx].astype(float)
                            ndwi_val = get_ndwi_value(green_band, nir_band)
                            properties['ndwi'] = ndwi_val
                            properties['water'] = get_water_class(ndwi_val)
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Skipping NDWI calculation for point feature due to error: {e}")
                        properties["ndwi"] = -1
                        properties["water"] = "not water"

                try:
                    for val in pan_src.sample([coords]):
                        properties['pan_value'] = float(val[0])
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping pan value calculation for point feature due to error: {e}")
                    properties["pan_value"] = float('inf')


                new_feature = {'type': 'Feature', 'geometry': mapping(geom), 'properties': properties}
                new_features.append(new_feature)

            elif geom.geom_type == 'Polygon':
                if src and not skip_ndwi:
                    try:
                        out_image, _ = rasterio.mask.mask(src, [geom], all_touched=True, crop=True, nodata=src.nodata)
                        green_band = out_image[green_idx, :, :].astype(float)
                        nir_band = out_image[nir_idx, :, :].astype(float)
                        ndwi_val = get_ndwi_value(green_band, nir_band)
                        properties['ndwi'] = ndwi_val
                        properties['water'] = get_water_class(ndwi_val)
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Skipping NDWI calculation for polygon feature due to error: {e}")
                        properties["ndwi"] = -1
                        properties["water"] = "not water"

                try:
                    pan_image, _ = rasterio.mask.mask(pan_src, [geom], crop=True, filled=False)
                    mean_pan_value = pan_image.mean()
                    if not isinstance(mean_pan_value, np.ma.core.MaskedConstant):
                        properties['pan_value'] = float(mean_pan_value)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping pan value calculation for polygon feature due to error: {e}")
                    properties["pan_value"] = float('inf')

                centroid = geom.centroid
                new_feature = {'type': 'Feature', 'geometry': mapping(centroid), 'properties': properties}
                new_features.append(new_feature)
    finally:
        if src:
            src.close()
        pan_src.close()

    # Consolidated filtering
    final_features = []
    
    # Pre-calculate percentile threshold if needed
    percentile_threshold = None
    if filter_by_percentile is not None:
        score_property = 'deviation_mean'
        scores = [f['properties'].get(score_property) for f in new_features if f['properties'].get(score_property) is not None]
        if scores:
            percentile_threshold = np.percentile(scores, filter_by_percentile)
            logging.info(f"Filtering by percentile: {filter_by_percentile} ({score_property} >= {percentile_threshold:.2f})")
        else:
            logging.warning(f"'{score_property}' not found in features. Skipping percentile filter.")

    for feature in new_features:
        # Pan threshold filter
        if filter_by_pan_threshold is not None and feature["properties"].get("pan_value", float('inf')) >= filter_by_pan_threshold:
            continue
        # NDWI filter
        if raster_path and not skip_ndwi and feature['properties'].get('ndwi', -1) < 0.3:
            continue
        # Percentile filter
        if percentile_threshold is not None and feature['properties'].get(score_property, -1) < percentile_threshold:
            continue
        
        final_features.append(feature)

    # Rename 'deviation_mean' to 'deviation'
    for feature in final_features:
        if 'deviation_mean' in feature['properties']:
            feature['properties']['deviation'] = feature['properties'].pop('deviation_mean')

    # Update schema
    if 'deviation_mean' in schema['properties']:
        schema['properties']['deviation'] = schema['properties'].pop('deviation_mean')
    if raster_path and not skip_ndwi:
        schema['properties'].update({'ndwi': 'float', 'water': 'str'})
    schema['properties']['pan_value'] = 'float'
    schema['geometry'] = 'Point'

    # Write the output even if the feature count is 0 as a record of the run
    logging.info(f"Saving {len(final_features)} features to: {output_geojson_path}")
    with fiona.open(output_geojson_path, 'w', driver='GeoJSON', crs=crs, schema=schema) as collection:
        collection.writerecords(final_features)

    return len(final_features)


def check_band_count(green_band_idx, nir_band_idx, src):
    if src:
        max_band_idx = max(green_band_idx, nir_band_idx)
        if src.count < max_band_idx:
            raise ValueError(
                f"Raster has {src.count} bands, but green band index is {green_band_idx} "
                f"and NIR band index is {nir_band_idx}."
            )


def cli():
    args = set_up_parser().parse_args()
    if os.path.exists(args.output_geojson_path):
        logging.warning(f"Output file '{args.output_geojson_path}' already exists. Skipping.")
        return

    num_filtered_points = process_features(
        args.geojson_path, args.output_geojson_path, args.pan_image_path,
        args.filter_by_ndwi, args.green_band_idx, args.nir_band_idx,
        args.filter_by_pan_threshold, args.filter_by_percentile
    )

    # Handle metadata
    source_metadata_path = os.path.splitext(args.geojson_path)[0] + "_meta.json"
    if os.path.exists(source_metadata_path):
        with open(source_metadata_path, 'r') as f:
            source_metadata = json.load(f)
    else:
        source_metadata = "Source metadata not found."

    xml_metadata_path = os.path.splitext(args.pan_image_path)[0] + ".xml"
    if os.path.exists(xml_metadata_path):
        image_id = os.path.basename(os.path.splitext(xml_metadata_path)[0])
        tree = ET.parse(xml_metadata_path)
        root = tree.getroot()

        image_metadata = {"image_id": image_id}
        catalog_id_tag = 'CATID'
        catalog_id_element = root.find(f".//{catalog_id_tag}")
        if catalog_id_element is not None:
            image_metadata["catalog_id"] = catalog_id_element.text
        else:
            msg = f"{catalog_id_tag} tag not found in XML."
            image_metadata["catalog_id"] = msg
            logging.warning(f"{image_id}: {msg}")

        pgc_imd_element = root.find("PGC_IMD")
        if pgc_imd_element is not None:
            image_metadata["image_processing_settings"] = {child.tag.lower(): child.text for child in pgc_imd_element}
        else:
            msg = "PGC_IMD tag not found in XML."
            image_metadata["image_processing_settings"] = msg
            logging.warning(f"{image_id}: {msg}")

    else:
        image_metadata = "Image metadata not found."

    density_analysis = {}
    if args.filter_by_ndwi:
        logging.info(f"Calculating water area from: {args.filter_by_ndwi}")
        with rasterio.open(args.filter_by_ndwi) as src:
            try:
                check_band_count(args.green_band_idx, args.nir_band_idx, src)
            except ValueError as e:
                logging.warning(e)
                water_area_sq_km = 0
            else:
                pixel_area_sq_km = (src.res[0] * src.res[1]) / 1_000_000
                green_band = src.read(args.green_band_idx).astype(float)
                nir_band = src.read(args.nir_band_idx).astype(float)
                np.seterr(divide='ignore', invalid='ignore')
                ndwi = (green_band - nir_band) / (green_band + nir_band)
                water_mask = ndwi > 0.3
                water_area_sq_km = np.sum(water_mask) * pixel_area_sq_km

            if water_area_sq_km > 0:
                ip_water_density = num_filtered_points / water_area_sq_km
                logging.info(f"Total water area (NDWI > 0.3): {water_area_sq_km:.2f} sq km")
                logging.info(f"Water-masked interesting point density: {ip_water_density:.2f} / sq km")
            else:
                ip_water_density = 0
                logging.warning("No water area found. Cannot calculate density.")

            density_analysis = {
                "num_interesting_points": num_filtered_points,
                "water_area_sq_km": round(water_area_sq_km, 2),
                "ip_density_per_sq_km_water": round(ip_water_density, 2),
            }

    else:
        logging.info(f"Calculating total area from: {args.pan_image_path}")
        with rasterio.open(args.pan_image_path) as src:
            pixel_area_sq_km = (src.res[0] * src.res[1]) / 1_000_000
            pan_band = src.read(1, masked=True)
            valid_area_sq_km = pan_band.count() * pixel_area_sq_km

            logging.info(f"Total valid area: {valid_area_sq_km:.2f} sq km")
            if valid_area_sq_km > 0:
                ip_density = num_filtered_points / valid_area_sq_km
                logging.info(f"Interesting point density: {ip_density:.2f} / sq km")
            else:
                logging.warning("No valid data found. Cannot calculate density.")

            density_analysis = {
                "num_interesting_points": num_filtered_points,
                "valid_area_sq_km": round(valid_area_sq_km, 2),
                "ip_density_per_sqkm_valid_data": round(ip_density, 2),
            }

    output_metadata = {
        "image_metadata": image_metadata,
        "source_metadata": source_metadata,
        "filtering_parameters": {
            "filtered_points_fn": os.path.basename(args.output_geojson_path),
            "pan_image_fn": os.path.basename(args.pan_image_path),
            "filter_by_ndwi": os.path.basename(args.filter_by_ndwi)
            if args.filter_by_ndwi
            else None,
            "green_band_idx": args.green_band_idx,
            "nir_band_idx": args.nir_band_idx,
            "filter_by_pan_threshold": args.filter_by_pan_threshold,
            "filter_by_percentile": args.filter_by_percentile,
        },
        "density_analysis": density_analysis,
    }

    output_metadata_path = os.path.splitext(args.output_geojson_path)[0] + "_meta.json"
    with open(output_metadata_path, 'w') as f:
        json.dump(output_metadata, f, indent=2)
    logging.info(f"Wrote metadata to {output_metadata_path}")


if __name__ == '__main__':
    cli()
