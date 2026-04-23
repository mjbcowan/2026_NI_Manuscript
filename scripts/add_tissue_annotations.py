#!/usr/bin/env python3
"""
Add Tissue Annotations to Spatial Proteomics Data

This script demonstrates how to integrate tissue-level annotations into
the spatial proteomics workflow. It can be run after Phase 3 (phenotyping)
to add regional analysis capabilities.

Usage Examples
--------------

# Example 1: Add annotations from GeoJSON file
python add_tissue_annotations.py \
    --zarr-path ./output/phase_3/sample1_phenotyped.zarr \
    --annotations ./annotations/sample1_regions.geojson \
    --output-dir ./output/phase_4

# Example 2: Create simple rectangular regions programmatically
python add_tissue_annotations.py \
    --zarr-path ./output/phase_3/sample1_phenotyped.zarr \
    --create-demo-regions \
    --n-regions 4 \
    --output-dir ./output/phase_4

# Example 3: Batch processing
python add_tissue_annotations.py \
    --batch-dir ./output/phase_3 \
    --annotations-dir ./annotations \
    --output-dir ./output/phase_4

Features
--------
- Adds tissue annotations to SpatialData shapes layer
- Assigns cells to tissue regions via spatial join
- Computes region-level summary statistics
- Generates visualizations of tissue regions with cell phenotypes
- Exports region-level tables for downstream analysis
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
import seaborn as sns

import spatialdata as sd
from spatialdata import SpatialData

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase4_tissue_annotation import (
    TissueAnnotationPhase,
    TissueAnnotationConfig,
    add_tissue_annotations,
    assign_cells_to_regions,
    compute_region_summary,
    plot_tissue_regions
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_regions(
    sdata: SpatialData,
    n_regions: int = 4,
    grid_layout: bool = True
) -> gpd.GeoDataFrame:
    """
    Create demo tissue regions for testing.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object to get image dimensions
    n_regions : int
        Number of regions to create
    grid_layout : bool
        If True, create grid layout; else random polygons

    Returns
    -------
    gpd.GeoDataFrame
        Demo tissue annotations
    """
    logger.info(f"Creating {n_regions} demo regions")

    # Get image dimensions
    if len(sdata.images) > 0:
        img_key = list(sdata.images.keys())[0]
        img = sdata.images[img_key]
        height, width = img.shape[-2:]
    elif 'table_intensities' in sdata.tables:
        table = sdata.tables['table_intensities']
        if 'x' in table.obs.columns and 'y' in table.obs.columns:
            width = table.obs['x'].max()
            height = table.obs['y'].max()
        else:
            width, height = 1000, 1000
    else:
        width, height = 1000, 1000

    regions = []

    if grid_layout:
        # Create grid layout
        if n_regions == 4:
            grid_rows, grid_cols = 2, 2
        elif n_regions == 9:
            grid_rows, grid_cols = 3, 3
        else:
            grid_rows = int(np.sqrt(n_regions))
            grid_cols = int(np.ceil(n_regions / grid_rows))

        region_width = width / grid_cols
        region_height = height / grid_rows

        region_types = ['tumor', 'stroma', 'immune', 'necrotic']

        for i in range(grid_rows):
            for j in range(grid_cols):
                if len(regions) >= n_regions:
                    break

                x0 = j * region_width
                y0 = i * region_height
                x1 = x0 + region_width
                y1 = y0 + region_height

                # Create polygon with slight margins
                margin = 10
                poly = box(x0 + margin, y0 + margin, x1 - margin, y1 - margin)

                region_id = f"region_{len(regions) + 1}"
                region_type = region_types[len(regions) % len(region_types)]

                regions.append({
                    'region_id': region_id,
                    'region_type': region_type,
                    'geometry': poly
                })

    else:
        # Create random rectangular regions
        region_types = ['tumor', 'stroma', 'immune', 'necrotic']

        for i in range(n_regions):
            # Random position and size
            x0 = np.random.uniform(0, width * 0.5)
            y0 = np.random.uniform(0, height * 0.5)
            w = np.random.uniform(width * 0.2, width * 0.4)
            h = np.random.uniform(height * 0.2, height * 0.4)

            poly = box(x0, y0, x0 + w, y0 + h)

            regions.append({
                'region_id': f"region_{i + 1}",
                'region_type': region_types[i % len(region_types)],
                'geometry': poly
            })

    gdf = gpd.GeoDataFrame(regions, crs=None)
    logger.info(f"Created {len(gdf)} demo regions")

    return gdf


def process_single_sample(
    zarr_path: Path,
    annotations: Optional[gpd.GeoDataFrame],
    output_dir: Path,
    config: TissueAnnotationConfig
) -> None:
    """
    Process a single sample with tissue annotations.

    Parameters
    ----------
    zarr_path : Path
        Path to input zarr file
    annotations : gpd.GeoDataFrame or None
        Tissue annotations (None to create demo)
    output_dir : Path
        Output directory
    config : TissueAnnotationConfig
        Configuration
    """
    logger.info(f"Processing {zarr_path.name}")

    # Load SpatialData
    sdata = sd.read_zarr(zarr_path)

    # Run tissue annotation phase
    phase = TissueAnnotationPhase(config)

    if annotations is None:
        # Create demo regions
        annotations = create_demo_regions(sdata, n_regions=4)

    result = phase.execute(sdata, tissue_annotations=annotations)

    # Save output
    sample_name = zarr_path.stem.replace('_phenotyped', '').replace('_filtered', '')
    output_zarr = output_dir / f"{sample_name}_tissue_annotated.zarr"
    result.sdata.write(output_zarr, overwrite=True)

    logger.info(f"Saved annotated data to {output_zarr}")

    # Generate visualisations
    logger.info("Generating visualisations")

    # Plot 1: Tissue regions with cell phenotypes
    if 'cell_type' in result.sdata.tables[config.table_key].obs.columns:
        fig1 = plot_tissue_regions(
            result.sdata,
            color_by='cell_type',
            output_path=output_dir / f"{sample_name}_regions_by_phenotype.pdf"
        )
        plt.close(fig1)

    # Plot 2: Tissue regions with region assignments
    fig2 = plot_tissue_regions(
        result.sdata,
        color_by=config.cell_column_name,
        output_path=output_dir / f"{sample_name}_regions_overlay.pdf"
    )
    plt.close(fig2)

    # Export region summary
    if config.region_table_name in result.sdata.tables:
        region_table = result.sdata.tables[config.region_table_name]
        summary_path = output_dir / f"{sample_name}_region_summary.csv"
        region_table.obs.to_csv(summary_path)
        logger.info(f"Exported region summary to {summary_path}")

        # Create summary visualizations
        create_region_summary_plots(region_table, sample_name, output_dir)

    logger.info(f"Completed processing {sample_name}")


def create_region_summary_plots(
    region_table,
    sample_name: str,
    output_dir: Path
) -> None:
    """Create summary plots for region statistics."""
    obs = region_table.obs

    # Plot 1: Cell counts per region
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Cell counts
    if 'n_cells' in obs.columns:
        obs['n_cells'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Cell Counts per Region')
        axes[0, 0].set_ylabel('Number of Cells')
        axes[0, 0].set_xlabel('Region')

    # Cell density
    if 'cell_density' in obs.columns:
        obs['cell_density'].plot(kind='bar', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title('Cell Density per Region')
        axes[0, 1].set_ylabel('Cells per Unit Area')
        axes[0, 1].set_xlabel('Region')

    # Cell type proportions (stacked bar)
    pct_cols = [c for c in obs.columns if c.startswith('pct_') and not c.endswith('_count')]
    if pct_cols:
        pct_data = obs[pct_cols]
        pct_data.columns = [c.replace('pct_', '') for c in pct_data.columns]
        pct_data.T.plot(kind='bar', stacked=True, ax=axes[1, 0], legend=True)
        axes[1, 0].set_title('Cell Type Proportions per Region')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].set_xlabel('Cell Type')
        axes[1, 0].legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Mean marker expression (heatmap)
    mean_cols = [c for c in obs.columns if c.startswith('mean_') and
                 not any(x in c for x in ['cell_size', 'cell_area', 'circularity', 'aspect_ratio'])]
    if mean_cols and len(mean_cols) > 1:
        mean_data = obs[mean_cols]
        mean_data.columns = [c.replace('mean_', '') for c in mean_data.columns]

        # Normalize for heatmap
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        mean_data_scaled = pd.DataFrame(
            scaler.fit_transform(mean_data.T).T,
            index=mean_data.index,
            columns=mean_data.columns
        )

        sns.heatmap(
            mean_data_scaled.T,
            cmap='coolwarm',
            center=0,
            ax=axes[1, 1],
            cbar_kws={'label': 'Z-score'}
        )
        axes[1, 1].set_title('Mean Marker Expression per Region (Z-scored)')
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('Marker')

    plt.tight_layout()
    output_path = output_dir / f"{sample_name}_region_summary_plots.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved region summary plots to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Add tissue annotations to spatial proteomics data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--zarr-path',
        type=Path,
        help='Path to single zarr file'
    )
    input_group.add_argument(
        '--batch-dir',
        type=Path,
        help='Directory containing multiple zarr files'
    )

    # Annotation options
    annot_group = parser.add_mutually_exclusive_group(required=False)
    annot_group.add_argument(
        '--annotations',
        type=Path,
        help='Path to GeoJSON/shapefile with tissue annotations'
    )
    annot_group.add_argument(
        '--annotations-dir',
        type=Path,
        help='Directory with annotation files (for batch mode)'
    )
    annot_group.add_argument(
        '--create-demo-regions',
        action='store_true',
        help='Create demo regions programmatically'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )

    # Configuration
    parser.add_argument(
        '--tissue-layer-name',
        type=str,
        default='tissue_regions',
        help='Name for tissue annotation layer'
    )
    parser.add_argument(
        '--table-key',
        type=str,
        default='table_intensities',
        help='Cell table name in SpatialData'
    )
    parser.add_argument(
        '--region-column',
        type=str,
        default='region_id',
        help='Column name for region assignment'
    )
    parser.add_argument(
        '--spatial-relationship',
        type=str,
        choices=['within', 'intersects'],
        default='within',
        help='Spatial relationship for cell-region assignment'
    )
    parser.add_argument(
        '--n-regions',
        type=int,
        default=4,
        help='Number of demo regions to create'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing outputs'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create configuration
    config = TissueAnnotationConfig(
        output_dir=args.output_dir,
        tissue_layer_name=args.tissue_layer_name,
        table_key=args.table_key,
        cell_column_name=args.region_column,
        spatial_relationship=args.spatial_relationship,
        overwrite=args.overwrite
    )

    # Process samples
    if args.zarr_path:
        # Single sample mode
        if not args.zarr_path.exists():
            logger.error(f"Zarr file not found: {args.zarr_path}")
            sys.exit(1)

        # Load annotations
        annotations = None
        if args.annotations:
            logger.info(f"Loading annotations from {args.annotations}")
            annotations = gpd.read_file(args.annotations)
        elif args.create_demo_regions:
            # Will create demo regions in process_single_sample
            annotations = None
        else:
            logger.warning("No annotations provided, creating demo regions")
            annotations = None

        process_single_sample(args.zarr_path, annotations, args.output_dir, config)

    else:
        # Batch mode
        zarr_files = sorted(args.batch_dir.glob("**/*_phenotyped.zarr"))

        if not zarr_files:
            logger.error(f"No zarr files found in {args.batch_dir}")
            sys.exit(1)

        logger.info(f"Found {len(zarr_files)} zarr files")

        for zarr_path in zarr_files:
            sample_name = zarr_path.stem.replace('_phenotyped', '')

            # Find matching annotation file
            annotations = None
            if args.annotations_dir:
                # Try multiple formats
                for ext in ['.geojson', '.shp', '.gpkg']:
                    annot_path = args.annotations_dir / f"{sample_name}{ext}"
                    if annot_path.exists():
                        logger.info(f"Found annotations for {sample_name}: {annot_path}")
                        annotations = gpd.read_file(annot_path)
                        break

            if annotations is None and not args.create_demo_regions:
                logger.warning(f"No annotations found for {sample_name}, creating demo regions")

            try:
                process_single_sample(zarr_path, annotations, args.output_dir, config)
            except Exception as e:
                logger.error(f"Failed to process {sample_name}: {e}")
                continue
            
    logger.info("Tissue annotation workflow completed")



if __name__ == "__main__":
    main()
