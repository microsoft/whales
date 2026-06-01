
import fiona
import rasterio
import rasterio.mask
import numpy as np
import argparse
import os
import logging
from shapely.geometry import shape, mapping

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_ndwi_from_geojson(geojson_path, output_geojson_path, raster_path=None, pan_image_path=None,
                                filter_by_percentile=None):
    """
    Calculates NDWI for features in a GeoJSON file based on a multiband raster.

    Args:
        geojson_path (str): Path to the input GeoJSON file.
        output_geojson_path (str): Path to save the output GeoJSON file.
        raster_path (str, optional): Path to the input multiband raster file.
        pan_image_path (str, optional): Path to the panchromatic image.
        filter_by_percentile (float, optional): Percentile threshold for filtering by mean deviation scores.
    """

    if filter_by_percentile is not None:
        if not 0 <= filter_by_percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100.")

    logging.info(f"Reading GeoJSON file: {geojson_path}")
    with fiona.open(geojson_path, 'r') as collection:
        features = list(collection)
        schema = collection.schema
        crs = collection.crs

    new_features = []

    src = rasterio.open(raster_path) if raster_path else None
    pan_src = rasterio.open(pan_image_path) if pan_image_path else None

    try:
        if src:
            logging.info(f"Reading raster file: {raster_path}")
        if pan_src:
            logging.info(f"Processing panchromatic image: {pan_image_path}")

        logging.info("Processing features...")
        for feature in features:
            properties = dict(feature['properties'])
            geom = shape(feature['geometry'])

            if src:
                try:
                    out_image, out_transform = rasterio.mask.mask(src, [geom], all_touched=True, crop=True,
                                                                  nodata=src.nodata)
                    green_band = out_image[2, :, :].astype(float)
                    nir_band = out_image[7, :, :].astype(float)
                    np.seterr(divide='ignore', invalid='ignore')
                    ndwi = (green_band - nir_band) / (green_band + nir_band)
                    mean_ndwi = np.nanmean(ndwi)
                    properties['ndwi'] = mean_ndwi
                    if mean_ndwi < 0.3:
                        properties['water'] = 'not water'
                    elif mean_ndwi < 0.5:
                        properties['water'] = 'probably water'
                    else:
                        properties['water'] = 'water'
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping NDWI calculation for feature due to error: {e}")

            if pan_src:
                try:
                    pan_image, _ = rasterio.mask.mask(pan_src, [geom], crop=True, nodata=pan_src.nodata)
                    masked_pan = np.ma.masked_equal(pan_image, pan_src.nodata)
                    mean_pan_value = masked_pan.mean()
                    if not isinstance(mean_pan_value, np.ma.core.MaskedConstant):
                        properties['pan_value'] = float(mean_pan_value)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Skipping pan value calculation for feature due to error: {e}")

            centroid = geom.centroid
            new_feature = {'type': 'Feature', 'geometry': mapping(centroid), 'properties': properties}
            new_features.append(new_feature)
    finally:
        if src:
            src.close()
        if pan_src:
            pan_src.close()

    # Filter features
    if pan_image_path:
        filtered_features = [f for f in new_features if f["properties"].get("pan_value", float('inf')) < 600]
    else:
        filtered_features = new_features

    if raster_path:
        filtered_features = [f for f in filtered_features if f['properties'].get('water') != 'not water']

    # Optional filtering by percentile
    if filter_by_percentile is not None:
        logging.info(f"Filtering by percentile: {filter_by_percentile}")
        score_property = 'deviation_mean'
        scores = [f['properties'][score_property] for f in filtered_features if score_property in f['properties']]
        if scores:
            percentile_threshold = np.percentile(scores, filter_by_percentile)
            filtered_features = [f for f in filtered_features if f['properties'].get(score_property,
                                                                                     0) >= percentile_threshold]
            logging.info(f"Filtered down to {len(filtered_features)} features.")

    # Rename 'deviation_mean' to 'deviation'
    for feature in filtered_features:
        if 'deviation_mean' in feature['properties']:
            feature['properties']['deviation'] = feature['properties'].pop('deviation_mean')

    # Update schema
    if 'deviation_mean' in schema['properties']:
        schema['properties']['deviation'] = schema['properties'].pop('deviation_mean')
    if raster_path:
        schema['properties'].update({'ndwi': 'float', 'water': 'str'})
    if pan_image_path:
        schema['properties']['pan_value'] = 'float'
    schema['geometry'] = 'Point'

    logging.info(f"Saving output to: {output_geojson_path}")
    with fiona.open(output_geojson_path, 'w', driver='GeoJSON', crs=crs, schema=schema) as collection:
        collection.writerecords(filtered_features)
    logging.info("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GeoJSON features, optionally using raster data.')
    parser.add_argument('geojson_path', help='Path to the input GeoJSON file')
    parser.add_argument('output_geojson_path', help='Path to save the output GeoJSON file')
    parser.add_argument('--filter-by-ndwi',
                        help='Path to the input multiband raster file (optional, ndwi > 0.3)')
    parser.add_argument('--filter-by-pan-image',
                        help='Path to the panchromatic image (optional, TOA < 600)')
    parser.add_argument('--filter-by-percentile', type=float,
                        help='Filter by top N percentile of mean deviation scores (optional)')
    args = parser.parse_args()

    if os.path.exists(args.output_geojson_path):
        logging.warning(f"Output file '{args.output_geojson_path}' already exists. Skipping.")
        exit()

    calculate_ndwi_from_geojson(args.geojson_path, args.output_geojson_path, args.filter_by_ndwi,
                                args.filter_by_pan_image, args.filter_by_percentile)
    logging.info("All operations finished.")
