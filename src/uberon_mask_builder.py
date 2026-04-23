"""
UBERON-Stratified Mask Creation Module

This module implements a refactored approach to creating multi-instance masks
from GeoJSON annotations, organized by UBERON anatomical ontology terms.

Key improvements over sequential rasterization:
1. Separate mask creation per UBERON term
2. Explicit hierarchical layering configuration
3. Semantic label IDs derived from UBERON terms
4. Enhanced validation and quality control
5. Full SpatialData and scVerse compatibility
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from shapely.geometry import Point

import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from shapely.geometry import Point, Polygon, MultiPolygon
from rasterio import features
from rasterio.transform import Affine
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel
from spatialdata.transformations import Identity
import anndata as ad

from shapely import affinity

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Core Configuration Functions
# ============================================================================

def load_hierarchy_config(config_path: Union[str, Path]) -> Dict:
    """
    Load UBERON hierarchy configuration from JSON file.

    Parameters
    ----------
    config_path : str or Path
        Path to hierarchy configuration JSON file

    Returns
    -------
    dict
        Hierarchy configuration dictionary
    """
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return json.load(f)


def extract_layer_order(hierarchy_dict: Dict, root_key: str = 'tissue') -> List[str]:
    """
    Extract flattened UBERON term order from hierarchical config.

    Uses depth-first traversal: parent before children, ensuring that
    broader anatomical structures are layered before specialized ones.

    Parameters
    ----------
    hierarchy_dict : dict
        Nested hierarchy configuration
    root_key : str, default='tissue'
        Root key in hierarchy dictionary

    Returns
    -------
    list of str
        Ordered list of UBERON term IDs
    """
    layer_order = []

    def _traverse(node_dict):
        """Depth-first traversal to build layer order."""
        for label, data in node_dict.items():
            # Extract UBERON term ID
            aliases = data.get('aliases', [])
            if aliases:
                term_id = aliases[0]
                # Normalize format
                if 'UBERON_' in term_id:
                    term_id = term_id.replace('UBERON_', 'UBERON:')
                layer_order.append(term_id)

            # Recurse into children
            children = data.get('children', {})
            if children:
                _traverse(children)

    # Start traversal from root
    if root_key in hierarchy_dict:
        root_data = hierarchy_dict[root_key]
        # Add root term if it has aliases
        root_aliases = root_data.get('aliases', [])
        if root_aliases:
            root_term = root_aliases[0]
            if 'UBERON_' in root_term:
                root_term = root_term.replace('UBERON_', 'UBERON:')
            layer_order.append(root_term)

        # Traverse children
        children = root_data.get('children', {})
        if children:
            _traverse(children)
    else:
        # If no root key, traverse entire dict
        _traverse(hierarchy_dict)

    return layer_order


def uberon_to_label_id(uberon_term: str) -> int:
    """
    Convert UBERON term to semantic label ID.

    Extracts numeric suffix from UBERON term for use as mask label.
    Example: UBERON:0001981 → 1981

    Parameters
    ----------
    uberon_term : str
        UBERON term ID (e.g., 'UBERON:0001981')

    Returns
    -------
    int
        Numeric label ID
    """
    if ':' in uberon_term:
        return int(uberon_term.split(':')[1])
    else:
        # Fallback for malformed terms
        return hash(uberon_term) % 65535


def create_uberon_to_label_mapping(uberon_terms: List[str]) -> Dict[str, int]:
    """
    Create mapping from UBERON terms to semantic label IDs.

    Parameters
    ----------
    uberon_terms : list of str
        List of UBERON term IDs

    Returns
    -------
    dict
        Mapping of UBERON term → label ID
    """
    return {term: uberon_to_label_id(term) for term in uberon_terms}


# ============================================================================
# GeoJSON Processing Functions
# ============================================================================

def group_annotations_by_uberon(
    annotations_gdf: gpd.GeoDataFrame,
    uberon_column: str = 'uberon_term_id'
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Split annotations GeoDataFrame by UBERON term.

    Parameters
    ----------
    annotations_gdf : GeoDataFrame
        GeoDataFrame containing all annotations with UBERON terms
    uberon_column : str, default='uberon_term_id'
        Column name containing UBERON term IDs

    Returns
    -------
    dict
        Mapping of UBERON term → GeoDataFrame subset
    """
    uberon_groups = {}

    for term in annotations_gdf[uberon_column].dropna().unique():
        uberon_groups[term] = annotations_gdf[
            annotations_gdf[uberon_column] == term
        ].copy()

    return uberon_groups


def validate_uberon_coverage(
    uberon_groups: Dict[str, gpd.GeoDataFrame],
    expected_terms: List[str]
) -> pd.DataFrame:
    """
    Validate that expected UBERON terms are present in annotations.

    Parameters
    ----------
    uberon_groups : dict
        Mapping of UBERON term → GeoDataFrame
    expected_terms : list of str
        Expected UBERON terms from hierarchy config

    Returns
    -------
    DataFrame
        Coverage report with columns: uberon_term, status, n_annotations
    """
    report = []

    for term in expected_terms:
        if term in uberon_groups:
            n_annotations = len(uberon_groups[term])
            status = 'present' if n_annotations > 0 else 'empty'
        else:
            n_annotations = 0
            status = 'missing'

        report.append({
            'uberon_term': term,
            'status': status,
            'n_annotations': n_annotations
        })

    return pd.DataFrame(report)


# ============================================================================
# Mask Creation Functions
# ============================================================================

def create_single_uberon_mask(
    uberon_gdf: gpd.GeoDataFrame,
    mask_shape: Tuple[int, int],
    label_id: int,
    transform: Optional[Affine] = None
) -> np.ndarray:
    """
    Create mask for a single UBERON term.

    All polygons for this term are rasterized with the same label ID.

    Parameters
    ----------
    uberon_gdf : GeoDataFrame
        GeoDataFrame containing annotations for single UBERON term
    mask_shape : tuple of int
        Output mask shape (height, width)
    label_id : int
        Label value to assign to all pixels in this mask
    transform : Affine, optional
        Coordinate transformation. If None, uses identity transform

    Returns
    -------
    ndarray
        2D mask array with shape mask_shape
    """
    if transform is None:
        transform = Affine.identity()

    # Create shapes for rasterization
    shapes = [(geom, label_id) for geom in uberon_gdf.geometry if geom is not None]

    if not shapes:
        # Return empty mask if no valid geometries
        return np.zeros(mask_shape, dtype=np.uint16)

    # Rasterize all polygons at once with same label
    mask = features.rasterize(
        shapes,
        out_shape=mask_shape,
        transform=transform,
        dtype=np.uint16,
        all_touched=False  # Use strict containment
    )

    return mask


def create_uberon_stratified_masks(
    annotations_gdf: gpd.GeoDataFrame,
    mask_shape: Tuple[int, int],
    uberon_to_label_id: Dict[str, int],
    uberon_column: str = 'uberon_term_id',
    transform: Optional[Affine] = None
) -> Dict[str, np.ndarray]:
    """
    Create separate masks for each UBERON term in annotations.

    Parameters
    ----------
    annotations_gdf : GeoDataFrame
        GeoDataFrame containing all annotations with UBERON terms
    mask_shape : tuple of int
        Output mask shape (height, width)
    uberon_to_label_id : dict
        Mapping of UBERON term to label ID
    uberon_column : str, default='uberon_term_id'
        Column name containing UBERON term IDs
    transform : Affine, optional
        Coordinate transformation

    Returns
    -------
    dict
        Mapping of UBERON term to 2D mask array
    """
    # Group annotations by UBERON term
    uberon_groups = group_annotations_by_uberon(annotations_gdf, uberon_column)

    # Create mask for each UBERON term
    uberon_masks = {}

    for uberon_term, term_gdf in uberon_groups.items():
        if uberon_term not in uberon_to_label_id:
            warnings.warn(
                f"UBERON term '{uberon_term}' not in label mapping. Skipping."
            )
            continue

        label_id = uberon_to_label_id[uberon_term]

        mask = create_single_uberon_mask(
            term_gdf,
            mask_shape,
            label_id,
            transform
        )

        uberon_masks[uberon_term] = mask

    return uberon_masks


def create_composite_mask(
    uberon_masks: Dict[str, np.ndarray],
    layer_order: List[str],
    mask_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Assemble composite mask by layering individual UBERON masks.

    Later terms in layer_order overwrite earlier terms in overlapping regions.
    This implements the hierarchical priority system.

    Parameters
    ----------
    uberon_masks : dict
        Mapping of UBERON term → mask array
    layer_order : list of str
        Ordered list of UBERON terms (low to high priority)
    mask_shape : tuple of int
        Output mask shape (height, width)

    Returns
    -------
    ndarray
        Composite 2D mask array
    """
    composite = np.zeros(mask_shape, dtype=np.uint16)

    # Track which terms were actually layered
    layered_terms = []

    # Layer masks in order (low priority first, high priority last)
    for uberon_term in layer_order:
        if uberon_term in uberon_masks:
            term_mask = uberon_masks[uberon_term]
            # Overwrite where this mask has values
            composite[term_mask > 0] = term_mask[term_mask > 0]
            layered_terms.append(uberon_term)

    # Handle terms not in layer order (add at beginning as lowest priority)
    unlayered_terms = set(uberon_masks.keys()) - set(layer_order)

    if unlayered_terms:
        warnings.warn(
            f"UBERON terms not in hierarchy layer order: {unlayered_terms}. "
            "These will be added at lowest priority."
        )

        # Add unlayered terms first (lowest priority)
        for term in unlayered_terms:
            term_mask = uberon_masks[term]
            # Only fill where composite is still zero
            composite[(composite == 0) & (term_mask > 0)] = term_mask[
                (composite == 0) & (term_mask > 0)
            ]

    return composite


# ============================================================================
# Validation Functions
# ============================================================================

def validate_mask_coverage(
    uberon_masks: Dict[str, np.ndarray],
    composite_mask: np.ndarray,
    min_pixels_per_term: int = 10
) -> pd.DataFrame:
    """
    Validate mask coverage and detect potential issues.

    Parameters
    ----------
    uberon_masks : dict
        Individual UBERON term masks
    composite_mask : ndarray
        Final composite mask
    min_pixels_per_term : int, default=10
        Minimum pixels required per term

    Returns
    -------
    DataFrame
        Coverage report with columns: uberon_term, n_pixels_individual,
        n_pixels_composite, status
    """
    report = []

    for term, term_mask in uberon_masks.items():
        n_pixels_individual = (term_mask > 0).sum()

        # Count pixels in composite (may be less due to overwriting)
        label_id = term_mask[term_mask > 0][0] if n_pixels_individual > 0 else 0
        n_pixels_composite = (composite_mask == label_id).sum()

        # Determine status
        if n_pixels_individual == 0:
            status = 'empty'
        elif n_pixels_composite == 0:
            status = 'completely_overwritten'
        elif n_pixels_composite < min_pixels_per_term:
            status = 'low_coverage'
        elif n_pixels_composite < n_pixels_individual * 0.1:
            status = 'mostly_overwritten'
        else:
            status = 'ok'

        report.append({
            'uberon_term': term,
            'n_pixels_individual': int(n_pixels_individual),
            'n_pixels_composite': int(n_pixels_composite),
            'coverage_ratio': float(n_pixels_composite / n_pixels_individual) if n_pixels_individual > 0 else 0.0,
            'status': status
        })

    return pd.DataFrame(report)


def validate_critical_structures(
    coverage_report: pd.DataFrame,
    critical_terms: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that critical anatomical structures are preserved.

    Parameters
    ----------
    coverage_report : DataFrame
        Output from validate_mask_coverage
    critical_terms : list of str
        UBERON terms that must be preserved (e.g., vascular structures)

    Returns
    -------
    bool
        True if all critical structures preserved
    list of str
        List of missing/compromised critical terms
    """
    issues = []

    for term in critical_terms:
        term_row = coverage_report[coverage_report['uberon_term'] == term]

        if term_row.empty:
            issues.append(f"{term}: not found in coverage report")
        else:
            status = term_row.iloc[0]['status']
            n_pixels = term_row.iloc[0]['n_pixels_composite']

            if status == 'completely_overwritten' or n_pixels == 0:
                issues.append(f"{term}: completely overwritten in composite")
            elif status == 'low_coverage':
                issues.append(f"{term}: low coverage ({n_pixels} pixels)")

    return len(issues) == 0, issues


# ============================================================================
# SpatialData Integration Functions
# ============================================================================

def create_lookup_table(
    uberon_masks: Dict[str, np.ndarray],
    uberon_to_label_id: Dict[str, int],
    annotations_gdf: gpd.GeoDataFrame,
    uberon_column: str = 'uberon_term_id'
) -> pd.DataFrame:
    """
    Create lookup table mapping label IDs to UBERON terms and metadata.

    Parameters
    ----------
    uberon_masks : dict
        Individual UBERON term masks
    uberon_to_label_id : dict
        Mapping of UBERON term to label ID
    annotations_gdf : GeoDataFrame
        Original annotations with metadata
    uberon_column : str, default='uberon_term_id'
        Column containing UBERON terms

    Returns
    -------
    DataFrame
        Lookup table indexed by label_id
    """
    lookup_data = []

    for uberon_term, term_mask in uberon_masks.items():
        label_id = uberon_to_label_id[uberon_term]
        n_pixels = (term_mask > 0).sum()

        # Get representative metadata from first annotation
        term_annotations = annotations_gdf[
            annotations_gdf[uberon_column] == uberon_term
        ]

        if len(term_annotations) > 0:
            first_annotation = term_annotations.iloc[0]
            uberon_label = first_annotation.get('uberon_label', uberon_term)
            annotation_type = first_annotation.get('annotation_type', 'unknown')
        else:
            uberon_label = uberon_term
            annotation_type = 'unknown'

        lookup_data.append({
            'label_id': label_id,
            'uberon_term_id': uberon_term,
            'uberon_label': uberon_label,
            'annotation_type': annotation_type,
            'n_annotations': len(term_annotations),
            'n_pixels': int(n_pixels)
        })

    lookup_df = pd.DataFrame(lookup_data)
    lookup_df = lookup_df.set_index('label_id').sort_index()

    return lookup_df


def add_mask_to_spatialdata(
    sdata: SpatialData,
    composite_mask: np.ndarray,
    lookup_table: pd.DataFrame,
    layer_order: List[str],
    mask_name: str = 'tissue_regions',
    reference_image_key: Optional[str] = None
) -> SpatialData:
    """
    Add composite mask and metadata to SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object to modify
    composite_mask : ndarray
        Composite mask array
    lookup_table : DataFrame
        Lookup table mapping label IDs to metadata
    layer_order : list of str
        UBERON term layer order used
    mask_name : str, default='tissue_regions'
        Name for mask in sdata.labels
    reference_image_key : str, optional
        Key of reference image for coordinate alignment

    Returns
    -------
    SpatialData
        Updated SpatialData object
    """
    # Parse mask with Labels2DModel
    mask_element = Labels2DModel.parse(
        composite_mask,
        transformations={reference_image_key or "global": Identity()}
    )

    # Add metadata attributes
    mask_element.attrs['uberon_layer_order'] = layer_order
    mask_element.attrs['creation_date'] = datetime.now().isoformat()
    mask_element.attrs['method'] = 'uberon_stratified_layering_v2'
    mask_element.attrs['n_labels'] = int(len(lookup_table))

    # Add mask to labels
    sdata.labels[mask_name] = mask_element

    # Add lookup table to tables (convert to AnnData-compatible format)
    # Store as attributes for now (could be converted to AnnData if needed)
    sdata.labels[mask_name].attrs['lookup_table'] = lookup_table.to_dict('index')

    return sdata


# ============================================================================
# Cell Assignment Functions
# ============================================================================

def assign_cells_to_regions(
    cell_coords: np.ndarray,
    composite_mask: np.ndarray,
    lookup_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign cells to tissue regions based on mask.

    Uses fast vectorized lookup of cell coordinates in mask.

    Parameters
    ----------
    cell_coords : ndarray
        Cell coordinates as N×2 array [x, y]
    composite_mask : ndarray
        Composite mask array
    lookup_table : DataFrame
        Lookup table indexed by label_id

    Returns
    -------
    DataFrame
        Cell assignments with columns: cell_idx, x, y, label_id,
        uberon_term_id, uberon_label
    """
    # Ensure integer coordinates
    x_coords = cell_coords[:, 0].astype(int)
    y_coords = cell_coords[:, 1].astype(int)

    # Clip to mask bounds
    h, w = composite_mask.shape
    x_coords = np.clip(x_coords, 0, w - 1)
    y_coords = np.clip(y_coords, 0, h - 1)

    # Vectorized lookup
    label_ids = composite_mask[y_coords, x_coords]

    # Create assignment dataframe
    assignments = pd.DataFrame({
        'cell_idx': np.arange(len(cell_coords)),
        'x': cell_coords[:, 0],
        'y': cell_coords[:, 1],
        'label_id': label_ids
    })

    # Merge with lookup table to get UBERON terms
    assignments = assignments.merge(
        lookup_table[['uberon_term_id', 'uberon_label']],
        left_on='label_id',
        right_index=True,
        how='left'
    )

    # Mark unassigned cells (label_id = 0)
    assignments.loc[assignments['label_id'] == 0, 'uberon_term_id'] = 'unassigned'
    assignments.loc[assignments['label_id'] == 0, 'uberon_label'] = 'unassigned'

    return assignments


# ============================================================================
# High-Level Workflow Functions
# ============================================================================

def create_uberon_stratified_mask_workflow(
    annotations_gdf: gpd.GeoDataFrame,
    mask_shape: Tuple[int, int],
    hierarchy_config_path: Union[str, Path],
    uberon_column: str = 'uberon_term_id',
    transform: Optional[Affine] = None,
    critical_structures: Optional[List[str]] = None,
    min_pixels_per_term: int = 10
) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """
    Complete workflow for creating UBERON-stratified masks.

    This is the main entry point function that orchestrates the entire
    mask creation process.

    Parameters
    ----------
    annotations_gdf : GeoDataFrame
        GeoDataFrame containing annotations with UBERON terms
    mask_shape : tuple of int
        Output mask shape (height, width)
    hierarchy_config_path : str or Path
        Path to UBERON hierarchy JSON config
    uberon_column : str, default='uberon_term_id'
        Column containing UBERON terms
    transform : Affine, optional
        Coordinate transformation
    critical_structures : list of str, optional
        UBERON terms that must be preserved
    min_pixels_per_term : int, default=10
        Minimum pixels per term for validation

    Returns
    -------
    composite_mask : ndarray
        Final composite mask
    uberon_masks : dict
        Individual UBERON term masks
    lookup_table : DataFrame
        Label ID to UBERON term mapping
    coverage_report : DataFrame
        Validation coverage report
    """
    # Load hierarchy configuration
    hierarchy_config = load_hierarchy_config(hierarchy_config_path)
    layer_order = extract_layer_order(hierarchy_config)

    print(f"Loaded hierarchy with {len(layer_order)} UBERON terms")

    # Create label ID mapping
    uberon_to_label_id = create_uberon_to_label_mapping(layer_order)

    # Create individual masks per UBERON term
    print("Creating individual UBERON masks")
    uberon_masks = create_uberon_stratified_masks(
        annotations_gdf,
        mask_shape,
        uberon_to_label_id,
        uberon_column,
        transform
    )

    print(f"Created {len(uberon_masks)} individual masks")

    # Create composite mask
    print("Assembling composite mask")
    composite_mask = create_composite_mask(
        uberon_masks,
        layer_order,
        mask_shape
    )

    # Create lookup table
    lookup_table = create_lookup_table(
        uberon_masks,
        uberon_to_label_id,
        annotations_gdf,
        uberon_column
    )

    # Validate coverage
    print("Validating mask coverage")
    coverage_report = validate_mask_coverage(
        uberon_masks,
        composite_mask,
        min_pixels_per_term
    )

    # Validate critical structures if specified
    if critical_structures:
        is_valid, issues = validate_critical_structures(
            coverage_report,
            critical_structures
        )

        if not is_valid:
            warnings.warn(
                f"Critical structure validation failed:\n" +
                "\n".join(issues)
            )

    print("Mask creation complete!")
    print(f"Composite mask: {composite_mask.shape}, "
          f"{len(np.unique(composite_mask)) - 1} unique labels")

    return composite_mask, uberon_masks, lookup_table, coverage_report

def validate_coordinate_overlap(
    cell_coords: np.ndarray,  # Shape (n_cells, 2) with [x, y]
    annotation_gdf: gpd.GeoDataFrame,
    min_overlap_threshold: float = 0.01,
    sample_size: int = 1000
) -> Dict:
    """
    Validate spatial overlap between cell coordinates and annotation regions.
    
    Returns diagnostic information about coordinate alignment.
    """
    # Sample cells for faster validation
    n_cells = len(cell_coords)
    if n_cells > sample_size:
        sample_indices = np.random.choice(n_cells, sample_size, replace=False)
        sampled_coords = cell_coords[sample_indices]
    else:
        sampled_coords = cell_coords
    
    # Create point geometries
    cell_points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in sampled_coords],
        crs=annotation_gdf.crs
    )
    
    # Compute bounds
    cell_bounds = cell_points.total_bounds  # [minx, miny, maxx, maxy]
    annot_bounds = annotation_gdf.total_bounds
    
    # Check spatial overlap
    overlap_x = not (cell_bounds[2] < annot_bounds[0] or cell_bounds[0] > annot_bounds[2])
    overlap_y = not (cell_bounds[3] < annot_bounds[1] or cell_bounds[1] > annot_bounds[3])
    has_bbox_overlap = overlap_x and overlap_y
    
    # Spatial join to count actual overlaps
    joined = gpd.sjoin(cell_points, annotation_gdf, how='left', predicate='within')
    n_cells_in_regions = joined['index_right'].notna().sum()
    overlap_fraction = n_cells_in_regions / len(cell_points)
    
    # Generate results
    results = {
        'has_bbox_overlap': has_bbox_overlap,
        'n_cells_sampled': len(cell_points),
        'n_cells_in_regions': int(n_cells_in_regions),
        'overlap_fraction': float(overlap_fraction),
        'cell_bounds': cell_bounds.tolist(),
        'annotation_bounds': annot_bounds.tolist(),
        'alignment_status': 'unknown'
    }
    
    # Determine alignment status
    if not has_bbox_overlap:
        results['alignment_status'] = 'CRITICAL_MISMATCH'
        logger.error(
            f"CRITICAL: No spatial overlap between cells and annotations!\n"
            f"   Cell bounds: {cell_bounds}\n"
            f"   Annotation bounds: {annot_bounds}"
        )
    elif overlap_fraction < min_overlap_threshold:
        results['alignment_status'] = 'LOW_OVERLAP'
        logger.warning(
            f" WARNING: Only {overlap_fraction*100:.2f}% of cells overlap with annotations.\n"
            f"   Expected at least {min_overlap_threshold*100:.1f}%."
        )
    else:
        results['alignment_status'] = 'GOOD'
        logger.info(
            f"Coordinate alignment validated: {overlap_fraction*100:.1f}% overlap "
            f"({n_cells_in_regions}/{len(cell_points)} cells)"
        )
    
    return results

from shapely import affinity

def apply_coordinate_transform(
    gdf: gpd.GeoDataFrame,
    offset: Optional[Tuple[float, float]] = None,
    scale: Optional[Tuple[float, float]] = None,
    rotation_degrees: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Apply coordinate transformation to GeoDataFrame geometries.

    FIXED: Uses explicit shapely.affinity transforms for reliability.
    """
    gdf_transformed = gdf.copy()

    if offset is not None:
        # CRITICAL FIX: Use shapely.affinity.translate explicitly
        gdf_transformed['geometry'] = gdf_transformed['geometry'].apply(
            lambda geom: affinity.translate(geom, xoff=offset[0], yoff=offset[1])
        )
        logger.info(f"Applied offset: X{offset[0]:+.1f}, Y{offset[1]:+.1f}")

    if scale is not None:
        gdf_transformed['geometry'] = gdf_transformed['geometry'].apply(
            lambda geom: affinity.scale(geom, xfact=scale[0], yfact=scale[1], origin=(0, 0))
        )
        logger.info(f"Applied scale: {scale}")

    if rotation_degrees is not None:
        gdf_transformed['geometry'] = gdf_transformed['geometry'].apply(
            lambda geom: affinity.rotate(geom, rotation_degrees, origin=(0, 0))
        )
        logger.info(f"Applied rotation: {rotation_degrees}°")

    return gdf_transformed


def validate_coordinate_overlap(
    cell_coords: np.ndarray,
    annotation_gdf: gpd.GeoDataFrame,
    min_overlap_threshold: float = 0.01,
    sample_size: int = 1000,
    use_full_validation: bool = False
) -> Dict:
    """
    Validate spatial overlap between cell coordinates and annotation regions.

    FIXED:
    - Better CRS handling
    - Option for full validation (no sampling)
    - More robust spatial join
    """
    # Sample cells for faster validation (unless full validation requested)
    n_cells = len(cell_coords)
    if use_full_validation or n_cells <= sample_size:
        sampled_coords = cell_coords
        sample_indices = np.arange(n_cells)
    else:
        sample_indices = np.random.choice(n_cells, sample_size, replace=False)
        sampled_coords = cell_coords[sample_indices]

    # Create point geometries with explicit CRS handling
    cell_points = gpd.GeoDataFrame(
        {'cell_index': sample_indices},
        geometry=[Point(x, y) for x, y in sampled_coords],
        crs=None  # Start with no CRS
    )

    # Align CRS
    if annotation_gdf.crs is not None:
        cell_points = cell_points.set_crs(annotation_gdf.crs)

    # Compute bounds
    cell_bounds = cell_points.total_bounds  # [minx, miny, maxx, maxy]
    annot_bounds = annotation_gdf.total_bounds

    # Check bounding box overlap
    overlap_x = not (cell_bounds[2] < annot_bounds[0] or cell_bounds[0] > annot_bounds[2])
    overlap_y = not (cell_bounds[3] < annot_bounds[1] or cell_bounds[1] > annot_bounds[3])
    has_bbox_overlap = overlap_x and overlap_y

    # Spatial join to count actual overlaps
    try:
        joined = gpd.sjoin(cell_points, annotation_gdf, how='left', predicate='within')
        n_cells_in_regions = joined['index_right'].notna().sum()
        overlap_fraction = n_cells_in_regions / len(cell_points)
    except Exception as e:
        logger.error(f"Spatial join failed: {e}")
        n_cells_in_regions = 0
        overlap_fraction = 0.0

    # Generate results
    results = {
        'has_bbox_overlap': has_bbox_overlap,
        'n_cells_sampled': len(cell_points),
        'n_cells_in_regions': int(n_cells_in_regions),
        'overlap_fraction': float(overlap_fraction),
        'cell_bounds': cell_bounds.tolist(),
        'annotation_bounds': annot_bounds.tolist(),
        'alignment_status': 'unknown'
    }

    # Determine alignment status
    if not has_bbox_overlap:
        results['alignment_status'] = 'CRITICAL_MISMATCH'
    elif overlap_fraction < min_overlap_threshold:
        results['alignment_status'] = 'LOW_OVERLAP'
    else:
        results['alignment_status'] = 'GOOD'

    return results


def auto_detect_coordinate_offset(
    cell_coords: np.ndarray,
    annotations: gpd.GeoDataFrame,
    min_overlap_threshold: float = 0.95,
    test_axes_separately: bool = True
) -> Tuple[Optional[Tuple[float, float]], Dict]:
    """
    Automatically detect coordinate offset needed to align annotations with cells

    Returns:
        (best_offset, best_result) tuple
    """
    # Get bounds
    cell_gdf_temp = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in cell_coords[:1000]])
    cell_bounds = cell_gdf_temp.total_bounds
    annot_bounds = annotations.total_bounds

    print("Coordinate Analysis:")
    print(f"  Cells:       X=[{cell_bounds[0]:.0f}, {cell_bounds[2]:.0f}], Y=[{cell_bounds[1]:.0f}, {cell_bounds[3]:.0f}]")
    print(f"  Annotations: X=[{annot_bounds[0]:.0f}, {annot_bounds[2]:.0f}], Y=[{annot_bounds[1]:.0f}, {annot_bounds[3]:.0f}]")

    # Calculate candidate offsets
    offset_min = (
        cell_bounds[0] - annot_bounds[0],
        cell_bounds[1] - annot_bounds[1]
    )
    offset_max = (
        cell_bounds[2] - annot_bounds[2],
        cell_bounds[3] - annot_bounds[3]
    )

    cell_center = ((cell_bounds[0] + cell_bounds[2]) / 2, (cell_bounds[1] + cell_bounds[3]) / 2)
    annot_center = ((annot_bounds[0] + annot_bounds[2]) / 2, (annot_bounds[1] + annot_bounds[3]) / 2)
    offset_center = (cell_center[0] - annot_center[0], cell_center[1] - annot_center[1])

    print(f"\nCalculated Offsets:")
    print(f"  To align MIN corners:   X{offset_min[0]:+.0f}, Y{offset_min[1]:+.0f}")
    print(f"  To align MAX corners:   X{offset_max[0]:+.0f}, Y{offset_max[1]:+.0f}")
    print(f"  To align centers:       X{offset_center[0]:+.0f}, Y{offset_center[1]:+.0f}")

    # Build test cases
    test_offsets = [
        ((0, 0), "No offset (baseline)"),
        (offset_min, "Align MIN corners"),
        (offset_max, "Align MAX corners"),
        (offset_center, "Align centers"),
    ]

    # Add axis-specific tests
    if test_axes_separately:
        test_offsets.extend([
            ((offset_min[0], 0), "X-only (MIN)"),
            ((0, offset_min[1]), "Y-only (MIN)"),
            ((offset_center[0], 0), "X-only (center)"),
            ((0, offset_center[1]), "Y-only (center)"),
        ])

    print(f"\n🔍 Testing {len(test_offsets)} offset hypotheses:")

    best_overlap = 0
    best_offset = None
    best_result = None
    best_description = None

    for offset, description in test_offsets:
        # Apply offset to annotations (NOT to cells!)
        annotations_test = apply_coordinate_transform(annotations, offset=offset)

        # Validate overlap
        result = validate_coordinate_overlap(
            cell_coords,
            annotations_test,
            min_overlap_threshold=min_overlap_threshold,
            sample_size=min(2000, len(cell_coords)),  # Larger sample for accuracy
            use_full_validation=False
        )

        overlap_pct = result['overlap_fraction'] * 100

        print(f"{description:30s} X{offset[0]:+7.0f}, Y{offset[1]:+7.0f}: {overlap_pct:5.1f}% overlap")

        if result['overlap_fraction'] > best_overlap:
            best_overlap = result['overlap_fraction']
            best_offset = offset
            best_result = result
            best_description = description

    print(f"\n{'='*60}")
    if best_overlap >= min_overlap_threshold:
        print(f"Found excellent alignment!")
        print(f"   Strategy: {best_description}")
        print(f"   Offset: X{best_offset[0]:+.0f}, Y{best_offset[1]:+.0f}")
        print(f"   Overlap: {best_overlap*100:.1f}%")
    else:
        print(f"Best alignment still below threshold")
        print(f"   Strategy: {best_description}")
        print(f"   Offset: X{best_offset[0]:+.0f}, Y{best_offset[1]:+.0f}")
        print(f"   Overlap: {best_overlap*100:.1f}% (need {min_overlap_threshold*100:.0f}%)")
    print(f"{'='*60}")

    return best_offset, best_result


print("Fixed alignment functions loaded")

def visualize_mask_layers(
    uberon_masks,
    composite_mask,
    lookup_table,
    terms_to_show=None,
    figsize=(20, 5),
    downsample_factor=4,
    figure_output_path=None,
    file_format="png"
):
    """
    Visualize individual UBERON mask layers alongside composite.

    Parameters
    ----------
    uberon_masks : dict
        Individual UBERON masks
    composite_mask : ndarray
        Final composite mask
    lookup_table : DataFrame
        Lookup table with UBERON term metadata
    terms_to_show : list of str, optional
        Specific terms to visualize. If None, shows top 3 by pixel count
    figsize : tuple, default=(20, 5)
        Figure size
    downsample_factor : int, default=4
        Factor to downsample masks for faster rendering
    figure_output_path : str, optional
        Path to save the figure. Defaults to 'default_path.<format>' in current working directory
    file_format : str, default="png"
        Output file format. Supported formats: 'png', 'pdf', 'svg', 'jpg'
    """
    if figure_output_path is None:
        figure_output_path = os.path.join(os.getcwd(), f"default_path.{file_format}")

    # Select terms to show
    if terms_to_show is None:
        # Show top 3 terms by pixel count
        terms_to_show = lookup_table.nlargest(3, 'n_pixels')['uberon_term_id'].tolist()
    
    n_terms = len(terms_to_show)
    fig, axes = plt.subplots(1, n_terms + 1, figsize=figsize)
    
    if n_terms == 0:
        axes = [axes]
    
    # Downsample for visualization
    composite_viz = composite_mask[::downsample_factor, ::downsample_factor]
    
    # Show individual masks
    for idx, term in enumerate(terms_to_show):
        if term in uberon_masks:
            mask_viz = uberon_masks[term][::downsample_factor, ::downsample_factor]
            label = lookup_table[lookup_table['uberon_term_id'] == term]['uberon_label'].iloc[0]
            n_pixels = (uberon_masks[term] > 0).sum()
            
            axes[idx].imshow(mask_viz, cmap='viridis', interpolation='nearest')
            axes[idx].set_title(f"{label}\n{term}\n{n_pixels:,} pixels", fontsize=10)
            axes[idx].axis('off')
    
    # Show composite
    axes[-1].imshow(composite_viz, cmap='tab20', interpolation='nearest')
    axes[-1].set_title(f"Composite Mask\n{len(np.unique(composite_mask))-1} labels", fontsize=10)
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(figure_output_path, format=file_format, bbox_inches='tight', dpi=300)
    print(f"Saved to: {figure_output_path}")
    return fig

import numpy.ma as ma

def visualize_mask_with_annotations(
    composite_mask,
    annotations_gdf,
    figsize=(12, 12),
    downsample_factor=4,
    show_boundaries=True,
    figure_output_path=None,
    file_format="png"
):
    """
    Visualise composite mask with annotation boundaries overlaid.
    
    Parameters
    ----------
    composite_mask : ndarray
        Composite mask
    annotations_gdf : GeoDataFrame
        Original annotations
    figsize : tuple, default=(12, 12)
        Figure size
    downsample_factor : int, default=4
        Downsampling factor for faster rendering
    show_boundaries : bool, default=True
        Whether to show annotation boundaries
    """
    if figure_output_path is None:
        figure_output_path = os.path.join(os.getcwd(), f"default_path.{file_format}")


    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Downsample mask
    mask_viz = composite_mask[::downsample_factor, ::downsample_factor]
    
    # Mask background (0 values) for proper color mapping
    mask_viz_masked = ma.masked_equal(mask_viz, 0)
    
    # Show mask with masked background
    ax.imshow(mask_viz_masked, cmap='Set2', interpolation='nearest', alpha=0.7)
    
    # Overlay annotation boundaries
    if show_boundaries:
        for idx, row in annotations_gdf.iterrows():
            if row.geometry.geom_type == 'Polygon':
                x, y = row.geometry.exterior.xy
                ax.plot(
                    np.array(x) / downsample_factor, 
                    np.array(y) / downsample_factor,
                    color='white', linewidth=0.5, alpha=0.3
                )
    
    ax.set_title('Composite Mask with Annotation Boundaries', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    plt.savefig(figure_output_path, format=file_format, bbox_inches='tight', dpi=300)
    print(f"Saved to: {figure_output_path}")

    return fig