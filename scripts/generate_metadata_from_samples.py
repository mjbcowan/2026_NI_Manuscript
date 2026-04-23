#!/usr/bin/env python3
"""
Generate metadata file from sample directory names

Automatically extracts:
- sample_id
- experiment_id (ex6, ex7, ex8)
- donor_id (DONOR1, DONOR2, etc.)
- timepoint (TP_A, TP_B, TP_C, TP_D)
- region (region_001, region_002, etc.)

Usage:
    python generate_metadata_from_samples.py \
        --results-dir ./results \
        --output metadata.csv
"""

import argparse
import re
from pathlib import Path
import pandas as pd


def parse_sample_id(sample_id: str) -> dict:
    """
    Parse sample ID into components - workflow specific
    """
    # Pattern: exp_donor_timepoint_region_num
    # pattern = r'^(ex\d+)_([A-Z0-9]+)_([A-Z0-9]+)_region_(\d+)$' # Pattern for NLM samples
    pattern = r'^(\d+w)_(\d+)_([A-Z])_(\d+)_region_(\d+)_subregion_(\d+)_uberon_stratified_(distal|proximal)$' # pattern for SDMJBC samples

    match = re.match(pattern, sample_id)

    if match:
        gest_age, donor, section_letter, section_num, region_num, subregion_num, location = match.groups()

        return {
            'sample_id': sample_id,
            'subregion': subregion_num,
            'location': location  # 'proximal' or 'distal'
        }
    else:
        return {
            'sample_id': sample_id,
            'subregion': 'unknown',
            'location': 'unknown'
        }

def find_samples(results_dir: Path, pattern: str = '*/phase_3/*_phenotyped.zarr') -> list:
    """Find all phenotyped zarr stores"""
    zarr_paths = sorted(results_dir.glob(pattern))
    sample_ids = [p.stem for p in zarr_paths]  # Gets filename without .zarr extension


    return sample_ids


def generate_metadata(results_dir: Path, output_path: Path, pattern: str):
    """Generate metadata CSV from samples"""
    print(f"Scanning for samples in: {results_dir}")

    # Find all samples
    sample_ids = find_samples(results_dir, pattern)
    print(f"Found {len(sample_ids)} samples")

    if len(sample_ids) == 0:
        print("No samples found! Check your results directory.")
        return

    # Parse each sample ID
    metadata_records = []
    for sample_id in sample_ids:
        record = parse_sample_id(sample_id)
        metadata_records.append(record)

    # Create DataFrame
    metadata = pd.DataFrame(metadata_records)


    # Sort by sample_id
    metadata = metadata.sort_values(['sample_id'])

    # Save
    metadata.to_csv(output_path, index=False)
    print(f"\nMetadata saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("METADATA SUMMARY")
    print("="*80)

    print(f"\nTotal samples: {len(metadata)}")

    print("\nBy location:")
    for loc, count in metadata['location'].value_counts().items():
        print(f"  {loc}: {count} samples")

    print("\nBy subregion:")
    for sub, count in metadata['subregion'].value_counts().sort_index().items():
        print(f"  {sub}: {count} samples")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate metadata CSV from sample directory structure'
    )

    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('./results'),
        help='Results directory containing sample zarr stores (default: ./results)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./metadata.csv'),
        help='Output metadata CSV file (default: ./metadata.csv)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*/phase_3/*_phenotyped.zarr',
        help='Pattern to match zarr files (default: */phase_3/*_phenotyped.zarr)'
    )

    args = parser.parse_args()

    generate_metadata(args.results_dir, args.output, args.pattern)


if __name__ == '__main__':
    main()
