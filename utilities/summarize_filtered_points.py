
import argparse
import os
import json
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_up_parser():
    parser = argparse.ArgumentParser(
        description="Summarize density analysis data from metadata JSON files into a single CSV."
    )
    parser.add_argument(
        'input_directory',
        help="Directory containing the '_meta.json' files to process."
    )
    parser.add_argument(
        'output_csv',
        help="Path to the output CSV file."
    )
    return parser


def cli():
    """
    Reads all metadata JSON files in a directory, extracts the 'density_analysis'
    section, and writes the data to a CSV file.

    Expects:
        input_directory (str): The path to the directory containing the metadata files.
        output_csv (str): The path to the output CSV file.
    """

    args = set_up_parser().parse_args()
    input_dir = args.input_directory
    output_csv = args.output_csv

    # Validate input directory
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found or is not a directory: {input_dir}")
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Validate output path
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    # Find all metadata files
    metadata_files = [f for f in os.listdir(input_dir) if f.endswith('_meta.json')]
    if not metadata_files:
        logging.warning(f"No '_meta.json' files found in {input_dir}")
        return

    logging.info(f"Found {len(metadata_files)} metadata files to process.")

    # Prepare to write CSV
    header = ['source_file', 'num_interesting_points', "valid_area_sq_km",
              'ip_density_per_sqkm_valid_data','water_area_sq_km', 'ip_density_per_sq_km_water']
    rows = []

    metadata_files.sort()
    for filename in metadata_files:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            density_analysis = metadata.get('density_analysis')
            if density_analysis and isinstance(density_analysis, dict):
                row_data = {
                    'source_file': os.path.basename(filename).replace('_meta.json', ''),
                    'num_interesting_points': density_analysis.get('num_interesting_points', 'NA'),
                    'water_area_sq_km': density_analysis.get('water_area_sq_km', 'NA'),
                    'ip_density_per_sq_km_water': density_analysis.get('ip_density_per_sq_km_water', 'NA'),
                    'valid_area_sq_km': density_analysis.get('valid_area_sq_km', 'NA'),
                    "ip_density_per_sqkm_valid_data": density_analysis.get('ip_density_per_sqkm_valid_data', 'NA'),
                }
                rows.append(row_data)
            else:
                logging.warning(f"'density_analysis' section not found or empty in {filename}")

        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {filename}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {filename}: {e}")

    # Write to CSV
    if not rows:
        logging.warning("No data to write to CSV.")
        return
        
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"Successfully wrote {len(rows)} rows to {output_csv}")
    except IOError:
        logging.error(f"Could not write to output file: {output_csv}")


if __name__ == '__main__':
    cli()


