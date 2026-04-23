"""
phase0_segmentation.py - Cellpose segmentation for WSI when masks unavailable
============================================================================
Uses Cellpose to segment nuclei and cells, then creates SpatialData-compliant
masks and feature tables.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import tifffile
from cellpose import models, core
from skimage import measure
import anndata as ad
import spatialdata as sd
from spatialdata.models import (
    Image2DModel, 
    Labels2DModel,
    PointsModel, 
    ShapesModel, 
    TableModel
)
from spatialdata.transformations import Identity
from shapely.geometry import Polygon
import geopandas as gpd

import time
from contextlib import contextmanager

@contextmanager
def timer(description):
    """Context manager for timing code blocks with formatted output."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    
    # Format time nicely
    if elapsed < 1:
        time_str = f"{elapsed*1000:.1f}ms"
    elif elapsed < 60:
        time_str = f"{elapsed:.2f}s"
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        time_str = f"{minutes}m {seconds:.1f}s"
    
    print(f"{description}: {time_str}")

@dataclass
class SegmentationConfig:
    """Configuration for Cellpose segmentation."""
    nucleus_channels: List[int]
    cytoplasm_channels: List[int]
    nucleus_diameter: float = 30.0
    cytoplasm_diameter: float = 60.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    use_gpu: bool = True
    channel_aggregation: str = "max"  # 'max', 'mean', or 'sum'


def aggregate_channels(
    img: np.ndarray, 
    channels: List[int], 
    method: str = "max"
) -> np.ndarray:
    """
    Aggregate multiple channels for Cellpose input.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (C, Y, X)
    channels : List[int]
        Channel indices to aggregate
    method : str
        Aggregation method ('max', 'mean', 'sum')
    
    Returns
    -------
    np.ndarray
        Aggregated 2D image (Y, X)
    """
    if len(channels) == 1:
        return img[channels[0]]
    
    channel_data = img[channels]
    
    if method == "max":
        aggregated = np.max(channel_data, axis=0)
    elif method == "mean":
        aggregated = np.mean(channel_data, axis=0)
    elif method == "sum":
        aggregated = np.sum(channel_data, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    # Normalise to uint8 for Cellpose
    if aggregated.dtype != np.uint8:
        aggregated = (
            (aggregated - aggregated.min()) / 
            (aggregated.max() - aggregated.min() + 1e-8) * 255
        ).astype(np.uint8)
    
    return aggregated


def segment_nuclei_and_cells(
    img: np.ndarray,
    config: SegmentationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment nuclei and cells using Cellpose.
    
    Uses 'nuclei' model for nuclear segmentation and 'cyto2' for cell segmentation.
    Works with both old and new Cellpose versions.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (C, Y, X)
    config : SegmentationConfig
        Segmentation configuration
    
    Returns
    -------
    nucleus_masks : np.ndarray
        Nuclear segmentation masks (Y, X)
    cell_masks : np.ndarray
        Cell segmentation masks (Y, X)
    """
    # Check GPU availability
    gpu_available = core.use_gpu()
    print(f"GPU available: {gpu_available}")
    
    # Check image size
    _, height, width = img.shape
    print(f"Image size: {height}x{width} pixels")
    
    print("Initializing Cellpose models...")
    
    # Initialize models - use model_type to get correct pretrained models
    nucleus_model = models.CellposeModel(
        gpu=config.use_gpu and gpu_available,
        model_type='nuclei'
    )
    
    cyto_model = models.CellposeModel(
        gpu=config.use_gpu and gpu_available,
        model_type='cyto2'
    )
    
    # Segment nuclei
    print("Segmenting nuclei with 'nuclei' model...")
    nucleus_img = aggregate_channels(
        img, 
        config.nucleus_channels, 
        config.channel_aggregation
    )
    
    # Basic eval call without tile parameter (version-agnostic)
    nucleus_masks, flows, styles = nucleus_model.eval(
        nucleus_img,
        diameter=config.nucleus_diameter,
        flow_threshold=config.flow_threshold,
        cellprob_threshold=config.cellprob_threshold,
        batch_size=8  # Process in smaller batches for memory efficiency
    )
    
    print(f"Detected {len(np.unique(nucleus_masks)) - 1} nuclei")
    
    # Segment cells
    print("Segmenting cells with 'cyto2' model...")
    cyto_img = aggregate_channels(
        img, 
        config.cytoplasm_channels, 
        config.channel_aggregation
    )
    
    # cyto2 expects: [cytoplasm, nucleus] as 2-channel input
    input_img = np.stack([cyto_img, nucleus_img], axis=-1)
    
    cell_masks, flows, styles = cyto_model.eval(
        input_img,
        diameter=config.cytoplasm_diameter,
        flow_threshold=config.flow_threshold,
        cellprob_threshold=config.cellprob_threshold,
        channels=[1, 2],  # [cytoplasm=1, nucleus=2] for 2-channel input
        batch_size=8
    )
    
    print(f" ✓ Detected {len(np.unique(cell_masks)) - 1} cells")
    
    return nucleus_masks, cell_masks


def assign_nuclei_and_compute_cytoplasm(cell_masks, nucleus_masks):
    """
    Nucleus-to-cell assignment with timing and edge case handling.
    """
    from scipy.sparse import csr_matrix
    
    print("  Assigning nuclei to cells...")
    
    with timer("Building overlap matrix"):
        # Build overlap matrix (vectorized)
        cell_ids = cell_masks.flatten()
        nucleus_ids = nucleus_masks.flatten()
        valid = (cell_ids > 0) & (nucleus_ids > 0)
        
        n_nuclei = nucleus_masks.max()
        n_cells = cell_masks.max()
        
        if not valid.any():
            print("No nucleus-cell overlaps found!")
            return np.zeros_like(cell_masks), np.zeros_like(nucleus_masks)
        
        # Count overlaps between each nucleus and cell
        overlap_matrix = csr_matrix(
            (np.ones(valid.sum()), (nucleus_ids[valid] - 1, cell_ids[valid] - 1)),
            shape=(n_nuclei, n_cells)
        )
    
    with timer("Computing nucleus assignments"):
        # Find maximum overlap for each nucleus
        max_overlaps = overlap_matrix.max(axis=1).toarray().flatten()
        nucleus_to_cell = overlap_matrix.argmax(axis=1).A1 + 1
        
        # Identify orphaned nuclei (no overlap with any cell)
        orphaned_mask = (max_overlaps == 0)
        nucleus_to_cell[orphaned_mask] = 0
    
    with timer("Remapping nucleus masks"):
        lookup = np.zeros(n_nuclei + 1, dtype=cell_masks.dtype)
        lookup[1:] = nucleus_to_cell
        aligned_nucleus_masks = lookup[nucleus_masks]
    
    with timer("Computing cytoplasm masks"):
        # Compute cytoplasm masks: cell - aligned_nucleus
        cytoplasm_masks = np.where(
            (cell_masks > 0) & (aligned_nucleus_masks == 0),
            cell_masks,
            0
        )
    
    with timer("Generating diagnostics"):
        # Diagnostic information
        orphaned_nuclei = np.where(orphaned_mask)[0] + 1
        cells_with_nuclei = np.unique(aligned_nucleus_masks[aligned_nucleus_masks > 0])
        all_cells = np.arange(1, n_cells + 1)
        anucleate_cells = np.setdiff1d(all_cells, cells_with_nuclei)
        cell_nucleus_counts = np.bincount(nucleus_to_cell[nucleus_to_cell > 0])
        multinucleated_cells = np.where(cell_nucleus_counts > 1)[0]
    
    # Print summary
    print(f"Assigned {n_nuclei - len(orphaned_nuclei)}/{n_nuclei} nuclei to cells")
    if len(orphaned_nuclei) > 0:
        print(f"{len(orphaned_nuclei)} orphaned nuclei (outside any cell)")
    if len(anucleate_cells) > 0:
        print(f"{len(anucleate_cells)} anucleate cells (no nucleus)")
    if len(multinucleated_cells) > 0:
        print(f"{len(multinucleated_cells)} multinucleated cells")
    
    return cytoplasm_masks, aligned_nucleus_masks

def extract_regionprops_features(
    masks: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    prefix: str = "",
    channel_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract morphology and intensity features using scikit-image regionprops.
    
    Parameters
    ----------
    masks : np.ndarray
        Segmentation masks
    intensity_image : Optional[np.ndarray]
        Optional multichannel image for intensity measurements
    prefix : str
        Prefix for column names
    channel_names : Optional[List[str]]
        Channel names for intensity columns (e.g., ['DAPI', 'CD45', 'CD3'])
        If None, uses ch0, ch1, ch2, etc.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with features
    """
    print(f"  Extracting {prefix} features...")
    
    # Morphology features - this is already fast
    morph_props = measure.regionprops_table(
        masks,
        properties=[
            'label', 'area', 'perimeter', 'eccentricity',
            'solidity', 'extent', 'major_axis_length',
            'minor_axis_length', 'centroid'
        ]
    )
    
    df = pd.DataFrame(morph_props)
    print(f"Morphology: {len(df)} objects")
    
    if intensity_image is not None:
        print(f"    Computing intensity features:")
        
        n_channels = intensity_image.shape[0] if intensity_image.ndim == 3 else 1
        
        if intensity_image.ndim == 3:
            # Use regionprops with intensity_image for each channel (c.f. looping method)
            for ch_idx in range(n_channels):
                # Use regionprops_table with intensity_image
                ch_props = measure.regionprops_table(
                    masks,
                    intensity_image=intensity_image[ch_idx],
                    properties=['intensity_mean']
                )
                
                # Use actual channel name if provided, otherwise use index
                if channel_names and ch_idx < len(channel_names):
                    channel_name = channel_names[ch_idx]
                else:
                    channel_name = f'ch{ch_idx}'
                
                df[f'{prefix}_{channel_name}_mean'] = ch_props['intensity_mean']
            
            print(f"Intensity: {n_channels} channels")
        else:
            # Single channel case
            ch_props = measure.regionprops_table(
                masks,
                intensity_image=intensity_image,
                properties=['intensity_mean']
            )
            df[f'{prefix}_intensity_mean'] = ch_props['intensity_mean']
    
    # Add prefix to columns (EXCEPT the intensity columns we just added)
    if prefix:
        # Get columns that don't already have the prefix
        cols_to_rename = [col for col in df.columns 
                         if col != 'label' and not col.startswith(f'{prefix}_')]
        df = df.rename(columns={
            col: f"{prefix}_{col}" 
            for col in cols_to_rename
        })
    
    return df

def create_shapes_from_masks(masks: np.ndarray) -> gpd.GeoDataFrame:
    """Create polygon shapes from segmentation masks."""
    cell_ids = np.unique(masks)[1:]
    geometries = []
    
    for cell_id in cell_ids:
        contours = measure.find_contours(masks == cell_id, 0.5)
        if len(contours) > 0:
            contour = max(contours, key=len)
            if len(contour) >= 3:
                polygon = Polygon(contour[:, [1, 0]])
                geometries.append({
                    'geometry': polygon,
                    'cell_id': int(cell_id)
                })
    
    return gpd.GeoDataFrame(geometries)


def segment_and_create_tables(
    img: np.ndarray,
    config: SegmentationConfig,
    channel_names: List[str],
    image_name: str = "fluorescence",
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, np.ndarray], ad.AnnData, ad.AnnData, gpd.GeoDataFrame]:
    """
    Complete segmentation workflow: segment WSI and create tables.

    Returns SpatialData-compliant masks and AnnData tables.

    Parameters
    ----------
    img : np.ndarray
        Input image (C, Y, X)
    config : SegmentationConfig
        Segmentation configuration
    channel_names : List[str]
        Channel names for intensity measurements
    image_name : str
        Image identifier
    metadata : Optional[Dict[str, Any]]
        Sample metadata to propagate to AnnData .obs (e.g., sample_id, donor_id,
        experiment_id, timepoint, disease_status, batch, etc.)

    Returns
    -------
    masks_dict : Dict[str, np.ndarray]
        Dictionary with keys 'cell', 'nucleus', 'cytoplasm' containing mask arrays
        Compatible with SpatialData Labels2DModel
    morphology_table : ad.AnnData
        Morphology AnnData table
    intensity_table : ad.AnnData
        Intensity AnnData table
    centroids_points : gpd.GeoDataFrame
        Cell centroids as PointsModel-compatible GeoDataFrame
    """
    print("="*80)
    print("PHASE 0: Cellpose Segmentation (No mask provided)")
    print("="*80)
    
    total_start = time.perf_counter()
    
    # Segment
    with timer("Cellpose segmentation (nuclei + cells)"):
        nucleus_masks, cell_masks = segment_nuclei_and_cells(img, config)
    
    with timer("Nucleus-to-cell assignment + cytoplasm computation"):
        cytoplasm_masks, aligned_nucleus_masks = assign_nuclei_and_compute_cytoplasm(
            cell_masks,
            nucleus_masks
        )
    
    # Extract features for each compartment
    print("Extracting features")
    with timer("Cell feature extraction"):
        cell_features = extract_regionprops_features(
            cell_masks, 
            intensity_image=img, 
            prefix='cell',
            channel_names=channel_names
        )
    
    with timer("Nucleus feature extraction"):
        nucleus_features = extract_regionprops_features(
            aligned_nucleus_masks, 
            intensity_image=img, 
            prefix='nucleus',
            channel_names=channel_names
        )
    
    with timer("Cytoplasm feature extraction"):
        cytoplasm_features = extract_regionprops_features(
            cytoplasm_masks, 
            intensity_image=img, 
            prefix='cytoplasm',
            channel_names=channel_names
        )
    
    with timer("Merging feature tables"):
        # Merge all features
        features_df = cell_features.merge(
            nucleus_features, on='label', how='left'
        ).merge(
            cytoplasm_features, on='label', how='left'
        )
    
    with timer("Creating AnnData tables"):
        # Create obs data
        obs_data = {
            'cell_id': features_df['label'].astype(str).values,
            'x': features_df['cell_centroid-1'].values,
            'y': features_df['cell_centroid-0'].values,
            'region': 'cell_segmentation',
            'instance_id': features_df['label'].values,
        }

        # Add nucleus-to-cell ratio
        if 'nucleus_area' in features_df.columns and 'cell_area' in features_df.columns:
            nucleus_area = np.array(features_df['nucleus_area'])
            cell_area = np.array(features_df['cell_area'])
            with np.errstate(divide='ignore', invalid='ignore'):
                obs_data['nucleus_to_cell_ratio'] = np.where(
                    cell_area > 0,
                    nucleus_area / cell_area,
                    np.nan
                )

        # PROPAGATE METADATA to obs for batch correction and downstream analysis
        if metadata:
            # Define standard metadata fields that should be propagated
            metadata_fields = [
                'sample_id', 'donor_id', 'experiment_id', 'timepoint',
                'disease_status', 'fov_id', 'batch', 'microscope',
                'acquisition_date', 'tissue_type', 'condition'
            ]

            n_cells = len(features_df)
            for key in metadata_fields:
                if key in metadata:
                    # Replicate scalar metadata across all cells
                    obs_data[key] = metadata[key]

            # Also allow custom metadata fields
            for key, value in metadata.items():
                if key not in obs_data and key not in metadata_fields:
                    obs_data[key] = value

            print(f"    ✓ Propagated {sum(k in metadata for k in metadata_fields)} metadata fields to .obs")

        obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in features_df['label']])
        
        # Create morphology table
        morphology_features = [
            'cell_area', 'cell_perimeter', 'cell_eccentricity',
            'cell_solidity', 'cell_extent', 
            'cell_major_axis_length', 'cell_minor_axis_length',
            'nucleus_area', 'nucleus_perimeter', 'nucleus_eccentricity',
            'cytoplasm_area'
        ]
        
        features_df_renamed = features_df.rename(columns={'cell_area': 'cell_size'})
        available_morphology = [f for f in morphology_features if f in features_df.columns]
        if 'cell_size' in features_df_renamed.columns:
            available_morphology = ['cell_size'] + [f for f in available_morphology if f != 'cell_area']
        
        X_morphology = features_df_renamed[available_morphology].values.astype(np.float32)
        var_morphology = pd.DataFrame({
            'feature_type': ['morphology'] * len(available_morphology)
        }, index=available_morphology)
        
        morphology_adata = ad.AnnData(
            X=X_morphology,
            obs=obs_df.copy(),
            var=var_morphology,
            dtype=np.float32
        )
        
        morphology_adata = TableModel.parse(
            morphology_adata,
            region='cell_segmentation',
            region_key='region',
            instance_key='instance_id'
        )
        
        # Create intensity table
        intensity_cols = [f'cell_{ch}_mean' for ch in channel_names]
        available_intensity_cols = [col for col in intensity_cols if col in features_df.columns]
        X_intensity = features_df[available_intensity_cols].values.astype(np.float32)
        
        var_intensity = pd.DataFrame({
            'feature_type': ['intensity'] * len(available_intensity_cols),
            'channel_name': channel_names[:len(available_intensity_cols)]
        }, index=channel_names[:len(available_intensity_cols)])
        
        intensity_adata = ad.AnnData(
            X=X_intensity,
            obs=obs_df.copy(),
            var=var_intensity,
            dtype=np.float32
        )
        
        intensity_adata = TableModel.parse(
            intensity_adata,
            region='cell_segmentation',
            region_key='region',
            instance_key='instance_id'
        )
        
        # Create centroids
        centroids_df = pd.DataFrame({
            'x': features_df['cell_centroid-1'].values,
            'y': features_df['cell_centroid-0'].values,
            'cell_id': features_df['label'].values,
        })
        
        centroids_points = PointsModel.parse(
            centroids_df,
            coordinates={'x': 'x', 'y': 'y'},
            transformations={'global': Identity()}
        )
    
    # Create masks dictionary
    masks_dict = {
        'cell': cell_masks,
        'nucleus': aligned_nucleus_masks,
        'cytoplasm': cytoplasm_masks
    }
    
    total_elapsed = time.perf_counter() - total_start
    
    # Create masks dictionary for SpatialData compliance
    # All masks use the same cell IDs for consistent region linking
    masks_dict = {
        'cell': cell_masks,
        'nucleus': aligned_nucleus_masks,
        'cytoplasm': cytoplasm_masks
    }

    print(f"Created morphology table: {morphology_adata.shape}")
    print(f"Created intensity table: {intensity_adata.shape}")
    print(f"Created centroids: {len(centroids_points)} cells")
    print(f"Generated 3 mask layers: cell, nucleus, cytoplasm")
    print(f"Total segmentation time: {total_elapsed:.2f}s")
    print("="*80)
    
    return masks_dict, morphology_adata, intensity_adata, centroids_points
