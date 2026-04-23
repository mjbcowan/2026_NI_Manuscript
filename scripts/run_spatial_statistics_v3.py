#!/usr/bin/env python3
"""
1. Spatial QC and diagnostics
2. Neighborhood enrichment analysis
3. Spatial co-localszation analysis (Ripley's functions)
4. Niche/microenvironment identification
5. Spatial gene/protein expression patterns
6. Cell-cell interaction analysis

Usage:
    python run_spatial_statistics.py --zarr-dir ./results --output-dir ./spatial_analysis

    # Single sample
    python run_spatial_statistics.py --zarr-path ./results/sample1/phase_3/sample1_phenotyped.zarr

    # With specific phenotype column
    python run_spatial_statistics.py --zarr-dir ./results \
        --cell-type-col cell_type \
        --pixel-size 0.325
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Spatial analysis
import spatialdata as sd
import squidpy as sq
from scipy.spatial import distance
from scipy.stats import mannwhitneyu

# Single-cell
import scanpy as sc
import anndata as ad

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress squidpy palette errors by redirecting stderr temporarily
import contextlib

# Set environment variables to suppress additional warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Context manager to suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = stderr


# =============================================================================
# Configuration and Setup
# =============================================================================

class SpatialAnalysisConfig:
    """Configuration for spatial analysis following FAIR principles"""

    def __init__(
        self,
        table_name: str = 'table_intensities',
        cell_type_col: str = 'cell_type',
        spatial_coords: Tuple[str, str] = ('x', 'y'),
        n_neighbors: int = 6,
        radius: List[int] = None,
        coord_type: str = 'generic',
        n_perms: int = 1000,
        random_state: int = 42,
        pixel_size_um: Optional[float] = None,
        interaction_radius: float = 50.0,
        cooccurrence_distances: List[int] = None
    ):
        """
        Initialize spatial analysis configuration

        Parameters
        ----------
        table_name : str
            Name of the table in SpatialData object
        cell_type_col : str
            Column name containing cell type annotations
        spatial_coords : tuple
            Names of x and y coordinate columns
        n_neighbors : int
            Number of neighbors for graph construction
        radius : list
            Radii for spatial analysis (in micrometers)
        coord_type : str
            Type of coordinate system ('generic' or 'grid')
        n_perms : int
            Number of permutations for statistical testing
        random_state : int
            Random seed for reproducibility
        pixel_size_um : float, optional
            Pixel size in micrometers. If None, will try to detect from metadata
        interaction_radius : float
            Radius for cell-cell interaction analysis in micrometers (default: 50.0)
        cooccurrence_distances : list
            Distances for co-occurrence analysis in micrometers (default: [50, 100, 150])
        """
        self.table_name = table_name
        self.cell_type_col = cell_type_col
        self.spatial_coords = spatial_coords
        self.n_neighbors = n_neighbors
        self.radius = radius if radius else [20, 50, 100, 200]
        self.coord_type = coord_type
        self.n_perms = n_perms
        self.random_state = random_state
        self.pixel_size_um = pixel_size_um
        self.interaction_radius = interaction_radius
        self.cooccurrence_distances = cooccurrence_distances if cooccurrence_distances else [50, 100, 150]


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_spatialdata(zarr_path: Path, config: SpatialAnalysisConfig) -> ad.AnnData:
    """
    Load SpatialData and prepare AnnData for spatial analysis

    Parameters
    ----------
    zarr_path : Path
        Path to zarr store
    config : SpatialAnalysisConfig
        Analysis configuration

    Returns
    -------
    adata : AnnData
        Prepared AnnData object with spatial coordinates
    """
    logger.info(f"Loading SpatialData from: {zarr_path}")
    sdata = sd.read_zarr(zarr_path)

    # Extract table
    if config.table_name not in sdata.tables:
        raise ValueError(f"Table '{config.table_name}' not found. Available: {list(sdata.tables.keys())}")

    adata = sdata.tables[config.table_name].copy()

    # Verify spatial coordinates
    x_col, y_col = config.spatial_coords
    if x_col not in adata.obs.columns or y_col not in adata.obs.columns:
        raise ValueError(f"Spatial coordinates '{x_col}', '{y_col}' not found in obs")

    # Add spatial coordinates to obsm for squidpy compatibility
    adata.obsm['spatial'] = adata.obs[[x_col, y_col]].values

    # Try to find pixel size in metadata and convert to micrometers
    # Priority: config.pixel_size_um > metadata > default
    pixel_size_um = config.pixel_size_um

    if pixel_size_um is not None:
        logger.info(f"Using pixel size from config: {pixel_size_um} μm/pixel")
    else:
        # Check various possible locations for pixel size metadata
        # 1. Check image metadata if images exist
        if hasattr(sdata, 'images') and len(sdata.images) > 0:
            image_name = list(sdata.images.keys())[0]
            image = sdata.images[image_name]

            # Check for scale transform in image metadata
            if hasattr(image, 'attrs') and 'transform' in image.attrs:
                transform = image.attrs['transform']
                if 'scale' in transform:
                    pixel_size_um = transform['scale'][0]  
                    logger.info(f"Found pixel size in image transform: {pixel_size_um} μm/pixel")

        # 2. Check adata.uns for pixel size
        if pixel_size_um is None and 'pixel_size_um' in adata.uns:
            pixel_size_um = adata.uns['pixel_size_um']
            logger.info(f"Found pixel size in adata.uns: {pixel_size_um} μm/pixel")

        # 3. Check sdata metadata
        if pixel_size_um is None and hasattr(sdata, 'attrs') and 'pixel_size_um' in sdata.attrs:
            pixel_size_um = sdata.attrs['pixel_size_um']
            logger.info(f"Found pixel size in sdata.attrs: {pixel_size_um} μm/pixel")

        # 4. Default fallback
        if pixel_size_um is None:
            pixel_size_um = 0.325  # Default for CellDIVE imaging
            logger.warning(f"Pixel size not found in metadata. Using default: {pixel_size_um} μm/pixel")
            logger.warning(f"To override, use --pixel-size option or add pixel_size_um to your data")

    # Convert coordinates from pixels to micrometers
    coord_max = np.max(adata.obsm['spatial'])
    if coord_max > 100:  # Likely in pixels if values are large
        logger.info(f"Converting coordinates from pixels to micrometers (pixel size: {pixel_size_um} μm/pixel)")
        adata.obsm['spatial'] = adata.obsm['spatial'] * pixel_size_um
        adata.uns['spatial_units'] = 'micrometers'
        adata.uns['pixel_size_um'] = pixel_size_um

        # Also update the obs columns for consistency
        adata.obs[x_col] = adata.obs[x_col] * pixel_size_um
        adata.obs[y_col] = adata.obs[y_col] * pixel_size_um
    else:
        logger.info(f"Coordinates appear to already be in micrometers")
        adata.uns['spatial_units'] = 'micrometers'

    logger.info(f"Loaded {adata.n_obs:,} cells with {adata.n_vars} markers")

    # Verify cell type annotation
    if config.cell_type_col in adata.obs.columns:
        n_types = adata.obs[config.cell_type_col].nunique()
        logger.info(f"Found {n_types} unique cell types in '{config.cell_type_col}'")
        if n_types > 0:
            sample_types = list(adata.obs[config.cell_type_col].unique()[:10])
            if n_types <= 10:
                logger.info(f"  Cell types: {sample_types}")
            else:
                logger.info(f"  Sample cell types (first 10): {sample_types}...")
    else:
        logger.error(f"Cell type column '{config.cell_type_col}' not found!")

        # Show available cell type-related columns
        available_cols = list(adata.obs.columns)
        type_related = [col for col in available_cols if any(x in col.lower() for x in ['type', 'lineage', 'phenotype', 'cluster', 'label'])]

        if type_related:
            logger.error(f"Available cell type-related columns: {type_related}")
        logger.error(f"All available columns: {available_cols}")
        logger.error(f"Please specify a valid column using --cell-type-col")
        raise ValueError(f"Column '{config.cell_type_col}' not found in data")

    return adata


def prepare_adata_for_spatial(adata: ad.AnnData, config: SpatialAnalysisConfig) -> ad.AnnData:
    """
    Prepare AnnData object for spatial analysis

    Following scVerse best practices:
    - Ensure spatial coordinates are in obsm['spatial']
    - Add metadata for reproducibility
    - Validate cell type annotations
    """
    # Make a copy to avoid modifying original
    adata = adata.copy()
    # Convert tuple to list for h5ad compatibility
    adata.uns['spatial_analysis_config'] = {
        'table_name': config.table_name,
        'cell_type_col': config.cell_type_col,
        'spatial_coords': list(config.spatial_coords),  # Convert tuple to list
        'n_neighbors': config.n_neighbors,
        'coord_type': config.coord_type,
        'random_state': config.random_state
    }

    return adata


# =============================================================================
# Spatial QC and Diagnostics
# =============================================================================

def run_spatial_qc(
    adata: ad.AnnData,
    config: SpatialAnalysisConfig,
    output_dir: Path
) -> Dict:
    """
    Run spatial quality control checks

    QC Metrics:
    - Cell density across tissue
    - Spatial distribution of cell types
    - Edge effects
    - Clustering patterns

    Returns
    -------
    qc_stats : dict
        Dictionary with QC statistics
    """
    logger.info("Running spatial QC diagnostics...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qc_stats = {}

    # 1. Overall cell density
    coords = adata.obsm['spatial']
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    area_um2 = x_range * y_range

    qc_stats['total_cells'] = adata.n_obs
    qc_stats['area_um2'] = area_um2
    qc_stats['cell_density_per_mm2'] = (adata.n_obs / area_um2) * 1e6

    # 2. Cell type proportions
    if config.cell_type_col in adata.obs.columns:
        cell_type_counts = adata.obs[config.cell_type_col].value_counts()
        qc_stats['cell_type_proportions'] = (cell_type_counts / adata.n_obs).to_dict()

    # 3. Spatial distribution uniformity (using quadrat analysis)
    n_quadrats = 10
    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_quadrats + 1)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_quadrats + 1)

    quadrat_counts = []
    for i in range(n_quadrats):
        for j in range(n_quadrats):
            in_quadrat = (
                (coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i+1]) &
                (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j+1])
            )
            quadrat_counts.append(in_quadrat.sum())

    qc_stats['quadrat_variance'] = np.var(quadrat_counts)
    qc_stats['quadrat_mean'] = np.mean(quadrat_counts)
    qc_stats['variance_to_mean_ratio'] = qc_stats['quadrat_variance'] / qc_stats['quadrat_mean']

    # 4. Visualise spatial QC
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Cell density heatmap
    ax = axes[0, 0]
    ax.hexbin(coords[:, 0], coords[:, 1], gridsize=30, cmap='viridis')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(f'Cell Density\n({adata.n_obs:,} cells, {qc_stats["cell_density_per_mm2"]:.0f} cells/mm²)')

    # Cell type spatial distribution
    if config.cell_type_col in adata.obs.columns:
        ax = axes[0, 1]
        cell_types = adata.obs[config.cell_type_col].astype('category')
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                           c=cell_types.cat.codes,
                           s=1, alpha=0.5, cmap='tab20')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Cell Type Spatial Distribution')

        # Legend with cell type names
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=plt.cm.tab20(i/len(cell_types.cat.categories)),
                             markersize=8, label=cat)
                  for i, cat in enumerate(cell_types.cat.categories[:10])]  # Limit to 10
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                 fontsize=8, frameon=False)

    # Quadrat analysis
    ax = axes[1, 0]
    ax.hist(quadrat_counts, bins=20, edgecolor='black')
    ax.axvline(qc_stats['quadrat_mean'], color='red', linestyle='--',
              label=f'Mean: {qc_stats["quadrat_mean"]:.1f}')
    ax.set_xlabel('Cells per Quadrat')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Spatial Distribution\nVMR: {qc_stats["variance_to_mean_ratio"]:.2f}')
    ax.legend()

    # Cell type proportions
    if config.cell_type_col in adata.obs.columns:
        ax = axes[1, 1]
        cell_type_counts.head(15).plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of Cells')
        ax.set_title('Cell Type Abundance (Top 15)')

    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_qc_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Spatial QC complete. VMR = {qc_stats['variance_to_mean_ratio']:.2f}")

    return qc_stats


# =============================================================================
# Neighborhood Enrichment Analysis
# =============================================================================

def compute_neighborhood_enrichment(
    adata: ad.AnnData,
    config: SpatialAnalysisConfig,
    output_dir: Path
) -> pd.DataFrame:
    """
    Compute neighborhood enrichment using squidpy

    Tests which cell types are enriched in the neighborhood of other cell types
    using permutation testing.

    Following squidpy best practices for spatial interaction analysis.
    """
    logger.info("Computing neighborhood enrichment:")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.cell_type_col not in adata.obs.columns:
        logger.warning(f"Cell type column '{config.cell_type_col}' not found. Skipping.")
        return None

    # Ensure cell type column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[config.cell_type_col]):
        logger.info(f"Converting '{config.cell_type_col}' to categorical")
        adata.obs[config.cell_type_col] = adata.obs[config.cell_type_col].astype('category')

    # Check for sufficient cell types
    n_cell_types = adata.obs[config.cell_type_col].nunique()
    if n_cell_types < 2:
        logger.warning(f"Only {n_cell_types} cell type(s) found. Skipping neighborhood enrichment.")
        return None

    # Build spatial graph
    sq.gr.spatial_neighbors(
        adata,
        coord_type=config.coord_type,
        n_neighs=config.n_neighbors,
        spatial_key='spatial'
    )

    # Compute neighborhood enrichment
    try:
        sq.gr.nhood_enrichment(
            adata,
            cluster_key=config.cell_type_col,
            n_perms=config.n_perms,
            seed=config.random_state
        )
    except Exception as e:
        logger.error(f"Neighborhood enrichment failed: {e}")
        return None

    # Extract results and get cell type categories
    cell_type_categories = adata.obs[config.cell_type_col].cat.categories.tolist()
    enrichment_matrix = adata.uns[f'{config.cell_type_col}_nhood_enrichment']['zscore']

    enrichment_df = pd.DataFrame(
        enrichment_matrix,
        index=cell_type_categories,
        columns=cell_type_categories
    )

    # Visualise
    fig, ax = plt.subplots(figsize=(12, 10))

    try:
        sq.pl.nhood_enrichment(
            adata,
            cluster_key=config.cell_type_col,
            method='average',
            cmap='RdBu_r',
            vmin=-3,
            vmax=3,
            ax=ax,
            title='Neighborhood Enrichment (Z-score)'
        )
    except Exception as e:
        logger.warning(f"Squidpy plotting failed, using seaborn: {e}")

        sns.heatmap(
            enrichment_df,
            cmap='RdBu_r',
            center=0,
            vmin=-3,
            vmax=3,
            ax=ax,
            cbar_kws={'label': 'Z-score'},
            square=True,
            linewidths=0.5
        )
        ax.set_title('Neighborhood Enrichment (Z-score)')
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Cell Type')

    plt.tight_layout()
    plt.savefig(output_dir / 'neighborhood_enrichment.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Save enrichment matrix with properly aligned indices
    enrichment_df.to_csv(output_dir / 'neighborhood_enrichment_zscore.csv')

    logger.info("Neighborhood enrichment analysis complete")

    return enrichment_df


# =============================================================================
# Co-localisation Analysis (Ripley's Functions)
# =============================================================================

def compute_spatial_colocalization(
    adata: ad.AnnData,
    config: SpatialAnalysisConfig,
    output_dir: Path
) -> Dict:
    """
    Compute pairwise spatial co-localisation between cell types

    Uses squidpy's co-occurrence score to identify which cell types
    cluster together vs avoid each other.
    """
    logger.info("Computing pairwise spatial co-localisation:")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.cell_type_col not in adata.obs.columns:
        logger.warning(f"Cell type column '{config.cell_type_col}' not found. Skipping.")
        return None

    if not pd.api.types.is_categorical_dtype(adata.obs[config.cell_type_col]):
        adata.obs[config.cell_type_col] = adata.obs[config.cell_type_col].astype('category')

    adata.obs[config.cell_type_col] = adata.obs[config.cell_type_col].cat.remove_unused_categories()

    cell_types = adata.obs[config.cell_type_col].cat.categories
    n_cell_types = len(cell_types)

    logger.info(f"Analyzing {n_cell_types} cell types present in sample: {list(cell_types)}")

    if n_cell_types < 2:
        logger.warning(f"Only {n_cell_types} cell type(s) found. Skipping.")
        return None

    logger.info("Rebuilding spatial graph with current cell types...")
    sq.gr.spatial_neighbors(
        adata,
        coord_type=config.coord_type,
        n_neighs=config.n_neighbors,
        spatial_key='spatial'
    )

    try:
        # Check the coordinate scale
        coords = adata.obsm['spatial']
        coord_range = np.ptp(coords, axis=0)  # Peak-to-peak (max - min) for each dimension
        max_distance = np.max(coord_range)
        logger.info(f"Coordinate ranges: X={coord_range[0]:.1f}, Y={coord_range[1]:.1f}")
        logger.info(f"Max coordinate range: {max_distance:.1f}")

        # Check spatial units
        spatial_units = adata.uns.get('spatial_units', 'unknown')
        logger.info(f"Spatial units: {spatial_units}")

        # Use distance thresholds from config for interpretability
        distance_thresholds = config.cooccurrence_distances
        logger.info(f"Computing co-occurrence at distances: {distance_thresholds} μm")

        # Create bins: we need bin edges, so add 0 at start and a max value
        # Squidpy will compute co-occurrence in each interval
        distance_bins = np.array([0] + distance_thresholds + [max_distance])

        sq.gr.co_occurrence(
            adata,
            cluster_key=config.cell_type_col,
            spatial_key='spatial',
            interval=distance_bins
        )

        # Extract results
        co_occur_key = f'{config.cell_type_col}_co_occurrence'
        if co_occur_key not in adata.uns:
            logger.error(f"Co-occurrence results not found")
            return None

        co_occur = adata.uns[co_occur_key]

        occurrence = co_occur['occ']
        intervals = co_occur['interval']

        logger.info(f"Co-occurrence matrix shape: {occurrence.shape}")
        logger.info(f"Number of distance intervals: {len(intervals)}")
       
        if occurrence.shape[0] == n_cell_types and occurrence.shape[1] == n_cell_types:
            # Shape is (n_cell_types, n_cell_types, n_intervals), transpose it
            occurrence = np.transpose(occurrence, (2, 0, 1))
            logger.info(f"Transposed co-occurrence matrix to shape: {occurrence.shape}")

        n_intervals = occurrence.shape[0]

        # Verify shape matches expectations (n_intervals, n_cell_types, n_cell_types)
        expected_shape = (n_intervals, n_cell_types, n_cell_types)
        if occurrence.shape != expected_shape:
            logger.error(f"Shape mismatch: expected {expected_shape}, got {occurrence.shape}")
            logger.error(f"Cell types in sample: {list(cell_types)}")
            logger.error(f"All categories in column: {list(adata.obs[config.cell_type_col].cat.categories)}")
            return None

        target_distances = config.cooccurrence_distances
        plot_intervals = []
        for target in target_distances:
            # Find the interval index closest to our target distance
            idx = np.argmin(np.abs(intervals - target))
            plot_intervals.append(idx)

        # Determine global min/max for consistent color scaling across all distances
        all_values = []
        for interval_idx in plot_intervals:
            all_values.extend(occurrence[interval_idx].flatten())
        vmin_global = np.percentile(all_values, 5)  # Use 5th/95th percentile to avoid outliers
        vmax_global = np.percentile(all_values, 95)

        # Make symmetric around 0 for diverging colormap
        vmax_abs = max(abs(vmin_global), abs(vmax_global))
        vmin_global = -vmax_abs
        vmax_global = vmax_abs

        logger.info(f"Co-occurrence value range: [{vmin_global:.2f}, {vmax_global:.2f}]")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.flatten()

        for idx, interval_idx in enumerate(plot_intervals):
            ax = axes[idx]

            # Get co-occurrence matrix for this distance
            co_occur_matrix = occurrence[interval_idx]

            # Create DataFrame for better labels
            co_occur_df = pd.DataFrame(
                co_occur_matrix,
                index=cell_types,
                columns=cell_types
            )

            # Plot heatmap with data-driven color scale
            sns.heatmap(
                co_occur_df,
                ax=ax,
                cmap='RdBu_r',
                center=0,
                vmin=vmin_global,
                vmax=vmax_global,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Co-occurrence score'},
                annot=True if len(cell_types) <= 8 else False,
                fmt='.2f'
            )

            distance = intervals[interval_idx]
            ax.set_title(f'Distance: {distance:.0f} μm', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('')

            # Rotate labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        # Create dynamic title based on distances
        distances_str = ', '.join([f'{d}' for d in target_distances])
        plt.suptitle(f'Pairwise Cell Type Co-occurrence at {distances_str} μm', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'colocalization_heatmaps.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved colocalization heatmaps to {output_dir / 'colocalization_heatmaps.pdf'}")

        # Create main pairwise plot at the first distance (default 50 μm)
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use first target distance
        first_distance = target_distances[0]
        idx_first = plot_intervals[0]
        co_occur_first = occurrence[idx_first]
        co_occur_first_df = pd.DataFrame(
            co_occur_first,
            index=cell_types,
            columns=cell_types
        )

        # Mask diagonal (self-interactions)
        mask = np.eye(len(cell_types), dtype=bool)

        # Use the same color scale as the multi-panel plot
        sns.heatmap(
            co_occur_first_df,
            ax=ax,
            mask=mask,
            cmap='RdBu_r',
            center=0,
            vmin=vmin_global,
            vmax=vmax_global,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Co-occurrence score'},
            annot=True,
            fmt='.2f'
        )

        ax.set_title(f'Pairwise Co-occurrence at {first_distance} μm\n'
                    f'Red = Attraction | Blue = Avoidance', fontsize=14)
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Cell Type', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(output_dir / f'colocalization_pairwise_{first_distance}um.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved pairwise colocalization ({first_distance} μm) to {output_dir / f'colocalization_pairwise_{first_distance}um.pdf'}")

        # Save numerical results for the specific distances (50, 100, 150 μm)
        for idx, interval_idx in enumerate(plot_intervals):
            distance = intervals[interval_idx]
            co_occur_df = pd.DataFrame(
                occurrence[interval_idx],
                index=cell_types,
                columns=cell_types
            )
            co_occur_df.to_csv(output_dir / f'colocalization_{distance:.0f}um.csv')
            logger.info(f"Saved co-occurrence matrix at {distance:.0f} μm")

        logger.info(f"Pairwise co-localization analysis complete. Saved to {output_dir}")

        return co_occur

    except Exception as e:
        logger.error(f"Co-occurrence computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def identify_spatial_niches(
    adata: ad.AnnData,
    config: SpatialAnalysisConfig,
    output_dir: Path,
    n_clusters: int = 10
) -> ad.AnnData:
    """
    Identify spatial niches/microenvironments using spatial clustering

    Approach:
    1. Build spatial neighborhood graph
    2. For each cell, compute neighborhood composition
    3. Cluster cells based on neighborhood composition
    4. Visualise spatial niches
    """
    logger.info("Identifying spatial niches...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.cell_type_col not in adata.obs.columns:
        logger.warning(f"Cell type column '{config.cell_type_col}' not found. Skipping.")
        return adata

    # Ensure categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[config.cell_type_col]):
        adata.obs[config.cell_type_col] = adata.obs[config.cell_type_col].astype('category')

    # Check for sufficient cell types
    n_cell_types = adata.obs[config.cell_type_col].nunique()
    if n_cell_types < 2:
        logger.warning(f"Only {n_cell_types} cell type(s) found. Skipping niche identification.")
        return adata

    try:
        # Build spatial graph if not exists
        if 'spatial_connectivities' not in adata.obsp:
            sq.gr.spatial_neighbors(
                adata,
                coord_type=config.coord_type,
                n_neighs=config.n_neighbors,
                spatial_key='spatial'
            )

        # Compute neighborhood composition manually instead of using centrality_scores
        logger.info("Computing neighborhood composition from spatial graph...")

        # Get the spatial connectivity matrix
        connectivity = adata.obsp['spatial_connectivities']
        cell_types = adata.obs[config.cell_type_col].cat.categories

        # Create one-hot encoding of cell types
        cell_type_matrix = pd.get_dummies(adata.obs[config.cell_type_col])

        # Compute neighborhood composition for each cell
        # For each cell, count the cell types in its neighborhood
        nhood_composition = connectivity @ cell_type_matrix.values

        # Normalize by number of neighbors to get proportions
        n_neighbors = np.array(connectivity.sum(axis=1)).flatten()
        # Avoid division by zero for isolated cells
        n_neighbors[n_neighbors == 0] = 1
        nhood_composition = nhood_composition / n_neighbors[:, np.newaxis]

        # Add to adata.obs with proper column names
        composition_cols = []
        for i, cell_type in enumerate(cell_types):
            col_name = f'{config.cell_type_col}_nhood_{cell_type}'
            adata.obs[col_name] = nhood_composition[:, i]
            composition_cols.append(col_name)

        logger.info(f"Computed neighborhood composition for {len(composition_cols)} cell types")

        # Create feature matrix for clustering
        X_nhood = adata.obs[composition_cols].values

        # Adjust n_clusters if needed
        n_cells = adata.n_obs
        n_clusters_adj = min(n_clusters, max(2, n_cells // 100))  # At least 100 cells per cluster

        if n_clusters_adj < n_clusters:
            logger.info(f"Adjusting n_clusters from {n_clusters} to {n_clusters_adj} due to small sample size")

        # Perform clustering on neighborhood composition
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters_adj, random_state=config.random_state, n_init=10)
        adata.obs['spatial_niche'] = kmeans.fit_predict(X_nhood).astype(str)
        adata.obs['spatial_niche'] = adata.obs['spatial_niche'].astype('category')

        # Visualise spatial niches
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Spatial distribution
        coords = adata.obsm['spatial']
        ax = axes[0]
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=adata.obs['spatial_niche'].cat.codes,
            s=2, alpha=0.6, cmap='tab20'
        )
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Spatial Niches (n={n_clusters_adj})')
        plt.colorbar(scatter, ax=ax, label='Niche ID')

        # Niche composition
        ax = axes[1]
        niche_composition = pd.crosstab(
            adata.obs['spatial_niche'],
            adata.obs[config.cell_type_col],
            normalize='index'
        ) * 100

        niche_composition.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        ax.set_xlabel('Spatial Niche')
        ax.set_ylabel('Cell Type Composition (%)')
        ax.set_title('Niche Cellular Composition')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'spatial_niches.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Save niche composition table
        niche_composition.to_csv(output_dir / 'spatial_niche_composition.csv')

        logger.info(f"Identified {n_clusters_adj} spatial niches")

    except Exception as e:
        logger.error(f"Niche identification failed: {e}")
        import traceback
        traceback.print_exc()

    return adata


# =============================================================================
# Spatial Expression Patterns
# =============================================================================

def analyze_spatial_expression(
    adata: ad.AnnData,
    config: SpatialAnalysisConfig,
    output_dir: Path,
    n_top_markers: int = 10
) -> Dict:
    """
    Analyse spatial patterns of marker expression

    Computes:
    - Moran's I (spatial autocorrelation)
    - Geary's C (spatial autocorrelation)
    """
    logger.info("Analyzing spatial expression patterns...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Build spatial graph if not exists
        if 'spatial_connectivities' not in adata.obsp:
            sq.gr.spatial_neighbors(
                adata,
                coord_type=config.coord_type,
                n_neighs=config.n_neighbors,
                spatial_key='spatial'
            )

        # Compute spatial autocorrelation (Moran's I)
        sq.gr.spatial_autocorr(
            adata,
            mode='moran',
            n_perms=config.n_perms,
            n_jobs=-1,
            seed=config.random_state
        )

        morans_i = adata.uns['moranI'].sort_values('I', ascending=False)

        # Save results
        morans_i.to_csv(output_dir / 'morans_i_statistics.csv')

        # Visualize top spatially variable markers
        top_markers = morans_i.head(n_top_markers).index.tolist()

        if len(top_markers) > 0:
            n_cols = min(4, len(top_markers))
            n_rows = int(np.ceil(len(top_markers) / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if len(top_markers) == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            coords = adata.obsm['spatial']

            for idx, marker in enumerate(top_markers):
                ax = axes[idx]
                marker_idx = list(adata.var_names).index(marker)
                values = adata.X[:, marker_idx]
                if hasattr(values, 'toarray'):
                    values = values.toarray().flatten()

                scatter = ax.scatter(coords[:, 0], coords[:, 1],
                                   c=values, s=2, cmap='viridis', alpha=0.6)
                ax.set_xlabel('X (μm)')
                ax.set_ylabel('Y (μm)')
                moran_value = morans_i.loc[marker, 'I']
                pval = morans_i.loc[marker, 'pval_norm']
                ax.set_title(f'{marker}\nMoran\'s I = {moran_value:.3f} (p={pval:.3e})')
                plt.colorbar(scatter, ax=ax)

            # Hide extra subplots
            for idx in range(len(top_markers), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(output_dir / 'spatial_expression_patterns.pdf', dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Spatial autocorrelation analysis complete. Top marker: {morans_i.index[0]}")

        return {'morans_i': morans_i}

    except Exception as e:
        logger.error(f"Spatial autocorrelation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# Cell-Cell Interaction Analysis
# =============================================================================

def analyze_cell_interactions(
    adata: ad.AnnData,
    config: SpatialAnalysisConfig,
    output_dir: Path
) -> pd.DataFrame:
    """
    Analyse cell-cell physical interactions

    Computes pairwise distances and identifies significant interactions
    at the radius specified in config.interaction_radius (in micrometers)
    """
    logger.info("Analyzing cell-cell interactions:")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.cell_type_col not in adata.obs.columns:
        logger.warning(f"Cell type column '{config.cell_type_col}' not found. Skipping.")
        return None

    # Ensure categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[config.cell_type_col]):
        adata.obs[config.cell_type_col] = adata.obs[config.cell_type_col].astype('category')

    # Check for sufficient cell types
    n_cell_types = adata.obs[config.cell_type_col].nunique()
    if n_cell_types < 2:
        logger.warning(f"Only {n_cell_types} cell type(s) found. Skipping interaction analysis.")
        return None

    try:
        # Get interaction radius from config (already in micrometers)
        interaction_radius = config.interaction_radius
        logger.info(f"Using interaction radius: {interaction_radius:.1f} μm")

        # Build interaction graph at specified radius
        # Coordinates are already in micrometers from load_spatialdata()
        sq.gr.spatial_neighbors(
            adata,
            coord_type=config.coord_type,
            radius=interaction_radius,
            spatial_key='spatial'
        )

        # Count interactions between cell types
        interactions = []
        cell_types = adata.obs[config.cell_type_col].cat.categories

        adjacency = adata.obsp['spatial_connectivities']

        for i, ct1 in enumerate(cell_types):
            mask1 = adata.obs[config.cell_type_col] == ct1
            cells1 = np.where(mask1)[0]

            for j, ct2 in enumerate(cell_types):
                if j < i:  # Only compute upper triangle
                    continue

                mask2 = adata.obs[config.cell_type_col] == ct2
                cells2 = np.where(mask2)[0]

                # Count interactions
                n_interactions = adjacency[cells1, :][:, cells2].sum()

                interactions.append({
                    'cell_type_1': ct1,
                    'cell_type_2': ct2,
                    'n_interactions': n_interactions,
                    'n_cells_1': len(cells1),
                    'n_cells_2': len(cells2)
                })

        interactions_df = pd.DataFrame(interactions)

        # Normalize by cell counts
        interactions_df['interactions_per_cell'] = (
            interactions_df['n_interactions'] /
            (interactions_df['n_cells_1'] * interactions_df['n_cells_2'])
        )

        # Save results
        interactions_df.to_csv(output_dir / 'cell_cell_interactions.csv', index=False)

        # Visualise interaction network
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create interaction matrix
        interaction_matrix = pd.pivot_table(
            interactions_df,
            values='interactions_per_cell',
            index='cell_type_1',
            columns='cell_type_2',
            fill_value=0
        )

        # Make symmetric
        for ct1 in interaction_matrix.index:
            for ct2 in interaction_matrix.columns:
                if ct1 != ct2:
                    val = max(
                        interaction_matrix.loc[ct1, ct2],
                        interaction_matrix.loc[ct2, ct1] if ct2 in interaction_matrix.index and ct1 in interaction_matrix.columns else 0
                    )
                    interaction_matrix.loc[ct1, ct2] = val
                    if ct2 in interaction_matrix.index and ct1 in interaction_matrix.columns:
                        interaction_matrix.loc[ct2, ct1] = val

        sns.heatmap(
            interaction_matrix,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Interactions per cell pair'},
            square=True
        )
        ax.set_title(f'Cell-Cell Interactions (r={interaction_radius}μm)')

        plt.tight_layout()
        plt.savefig(output_dir / 'cell_cell_interaction_matrix.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Cell-cell interaction analysis complete")

        return interactions_df

    except Exception as e:
        logger.error(f"Cell-cell interaction analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Comprehensive Report Generation
# =============================================================================

def generate_analysis_report(
    adata: ad.AnnData,
    config: SpatialAnalysisConfig,
    qc_stats: Dict,
    output_dir: Path,
    sample_id: str
):
    """
    Generate comprehensive analysis report
    """
    logger.info("Generating analysis report:")
    output_dir = Path(output_dir)

    report_path = output_dir / f'{sample_id}_spatial_analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPATIAL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Table: {config.table_name}\n")
        f.write(f"Cell Type Column: {config.cell_type_col}\n")
        f.write(f"Spatial Coordinates: {config.spatial_coords}\n")
        f.write(f"Spatial Units: micrometers (μm)\n")
        f.write(f"Pixel Size: {config.pixel_size_um} μm/pixel\n")
        f.write(f"N Neighbors: {config.n_neighbors}\n")
        f.write(f"Analysis Radii: {config.radius} μm\n")
        f.write(f"Interaction Radius: {config.interaction_radius} μm\n")
        f.write(f"N Permutations: {config.n_perms}\n")
        f.write(f"Random State: {config.random_state}\n\n")

        f.write("QUALITY CONTROL\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Cells: {qc_stats.get('total_cells', 'N/A'):,}\n")
        f.write(f"Tissue Area: {qc_stats.get('area_um2', 0):.2f} μm²\n")
        f.write(f"Cell Density: {qc_stats.get('cell_density_per_mm2', 0):.0f} cells/mm²\n")
        f.write(f"Spatial Distribution (VMR): {qc_stats.get('variance_to_mean_ratio', 0):.2f}\n")
        f.write(f"  (VMR ~1: random, >1: clustered, <1: regular)\n\n")

        if 'cell_type_proportions' in qc_stats:
            f.write("CELL TYPE COMPOSITION\n")
            f.write("-"*80 + "\n")
            for ct, prop in sorted(qc_stats['cell_type_proportions'].items(),
                                  key=lambda x: x[1], reverse=True):
                f.write(f"  {ct}: {prop*100:.2f}%\n")
            f.write("\n")

        f.write("ANALYSES PERFORMED\n")
        f.write("-"*80 + "\n")
        f.write("✓ Spatial QC and diagnostics\n")
        f.write("✓ Neighborhood enrichment analysis\n")
        f.write("✓ Spatial co-localization (Ripley's functions)\n")
        f.write("✓ Spatial niche identification\n")
        f.write("✓ Spatial expression patterns (Moran's I)\n")
        f.write("✓ Cell-cell interaction analysis\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-"*80 + "\n")
        f.write("  spatial_qc_summary.pdf\n")
        f.write("  neighborhood_enrichment.pdf\n")
        f.write("  neighborhood_enrichment_zscore.csv\n")
        f.write("  ripley_L_function.pdf\n")
        f.write("  spatial_niches.pdf\n")
        f.write("  spatial_niche_composition.csv\n")
        f.write("  spatial_expression_patterns.pdf\n")
        f.write("  morans_i_statistics.csv\n")
        f.write("  cell_cell_interactions.csv\n")
        f.write("  cell_cell_interaction_matrix.pdf\n\n")

        f.write("="*80 + "\n")
        f.write("Analysis complete!\n")
        f.write("="*80 + "\n")

    logger.info(f"Report saved to: {report_path}")


# =============================================================================
# Main Workflow
# =============================================================================

def run_complete_spatial_analysis(
    zarr_path: Path,
    output_dir: Path,
    config: SpatialAnalysisConfig,
    sample_id: Optional[str] = None
):
    """
    Run complete spatial analysis workflow
    """
    # Extract sample ID from path if not provided
    if sample_id is None:
        # Use the zarr filename (without .zarr extension) as sample ID
        sample_id = zarr_path.stem

    logger.info("="*80)
    logger.info(f"SPATIAL ANALYSIS: {sample_id}")
    logger.info("="*80)

    # Create output directory
    output_dir = Path(output_dir) / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    adata = load_spatialdata(zarr_path, config)
    adata = prepare_adata_for_spatial(adata, config)

    # Run analyses
    qc_stats = run_spatial_qc(adata, config, output_dir)

    enrichment_df = compute_neighborhood_enrichment(adata, config, output_dir)

    colocalization_results = compute_spatial_colocalization(adata, config, output_dir)

    adata = identify_spatial_niches(adata, config, output_dir, n_clusters=10)

    expression_results = analyze_spatial_expression(adata, config, output_dir, n_top_markers=10)

    interactions_df = analyze_cell_interactions(adata, config, output_dir)

    # Generate report
    generate_analysis_report(adata, config, qc_stats, output_dir, sample_id)

    # Save annotated adata
    adata.write_h5ad(output_dir / f'{sample_id}_spatial_annotated.h5ad')

    logger.info("="*80)
    logger.info(f"Spatial analysis complete for {sample_id}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

    return adata


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Comprehensive spatial statistics analysis following best practices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse all samples in results directory
  python run_spatial_statistics.py --zarr-dir ./results --output-dir ./spatial_analysis

  # Analyse single sample
  python run_spatial_statistics.py --zarr-path ./results/sample1/phase_3/sample1_phenotyped.zarr

  # Custom cell type column
  python run_spatial_statistics.py --zarr-dir ./results --cell-type-col lineage_detailed
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--zarr-path',
        type=Path,
        help='Path to single zarr store'
    )
    input_group.add_argument(
        '--zarr-dir',
        type=Path,
        help='Directory containing multiple zarr stores (will search for */phase_3/*_phenotyped.zarr)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./spatial_analysis'),
        help='Output directory for results (default: ./spatial_analysis)'
    )

    # Configuration
    parser.add_argument(
        '--table-name',
        type=str,
        default='table_intensities',
        help='Name of table in SpatialData (default: table_intensities)'
    )
    parser.add_argument(
        '--cell-type-col',
        type=str,
        default='cell_type',
        help='Column name for cell type annotations (default: cell_type). Use this to specify which column contains your cell type labels (e.g., clustered_cell_type, lineage, phenotype, etc.)'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=6,
        help='Number of neighbors for spatial graph (default: 6)'
    )
    parser.add_argument(
        '--radius',
        nargs='+',
        type=int,
        default=[20, 50, 100, 200],
        help='Radii for spatial analysis in μm (default: 20 50 100 200)'
    )
    parser.add_argument(
        '--n-perms',
        type=int,
        default=1000,
        help='Number of permutations for statistical testing (default: 1000)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--zarr-pattern',
        type=str,
        default='*/phase_3/*_phenotyped.zarr',
        help='Pattern to match zarr files (default: */phase_3/*_phenotyped.zarr)'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=None,
        help='Pixel size in micrometers (default: auto-detect or 0.325 for CODEX)'
    )
    parser.add_argument(
        '--interaction-radius',
        type=float,
        default=50.0,
        help='Radius for cell-cell interaction analysis in micrometers (default: 50.0)'
    )
    parser.add_argument(
        '--cooccurrence-distances',
        nargs='+',
        type=int,
        default=[50, 100, 150],
        help='Distances for co-occurrence analysis in μm (default: 50 100 150)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Create configuration
    config = SpatialAnalysisConfig(
        table_name=args.table_name,
        cell_type_col=args.cell_type_col,
        n_neighbors=args.n_neighbors,
        radius=args.radius,
        n_perms=args.n_perms,
        random_state=args.random_state,
        pixel_size_um=args.pixel_size,
        interaction_radius=args.interaction_radius,
        cooccurrence_distances=args.cooccurrence_distances
    )

    # Find zarr files
    if args.zarr_path:
        zarr_paths = [args.zarr_path]
    else:
        zarr_paths = sorted(args.zarr_dir.glob(args.zarr_pattern))
        logger.info(f"Found {len(zarr_paths)} zarr stores")

    if len(zarr_paths) == 0:
        logger.error("No zarr stores found!")
        sys.exit(1)

    # Process each zarr
    for zarr_path in tqdm(zarr_paths, desc="Processing samples"):
        try:
            run_complete_spatial_analysis(
                zarr_path=zarr_path,
                output_dir=args.output_dir,
                config=config
            )
        except Exception as e:
            logger.error(f"Failed to process {zarr_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("="*80)
    logger.info("✓ All spatial analyses complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()