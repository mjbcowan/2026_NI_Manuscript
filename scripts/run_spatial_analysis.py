"""
run_spatial_analysis.py - Parameterised spatial transcriptomics/proteomics pipeline
Usage: python run_spatial_analysis.py --data-dir ./data --signatures signatures.yaml
      or
      python run_spatial_analysis.py --tiff-path ./data/sample1.ome.tiff --signatures signatures.yaml
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import yaml
import json
import tifffile
import spatialdata as sd
from spatialdata.models import Labels2DModel
from spatialdata.transformations import Identity
from spatialdata import SpatialData, to_polygons
import logging

from typing import Optional, Tuple, List

import sys

# Add src to path - use the src directory in THIS project
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# External imports (ensure they're in your PYTHONPATH)
from phase1_tiff import ingest_tiff_to_spatialdata, parse_csv_to_mudata_tables
from phase1b_normalise import normalize_intensity_by_area, choose_arcsinh_cofactor
from phase2_v4 import flag_cells_at_qc_risk, run_complete_qc_workflow, filter_sdata_by_qc_efficient
from phase3_intensity_phenotype_v6 import (
    run_complete_phenotyping_and_labeling, 
    run_complete_umap_cell_type_workflow, 
    run_gmm_phenotyping_workflow, 
    plot_gmm_phenotypes_spatial
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_signatures(signatures_path):
    """Load cell type signatures from YAML or JSON file."""
    signatures_path = Path(signatures_path)

    if not signatures_path.exists():
        raise FileNotFoundError(f"Signatures file not found: {signatures_path}")

    with open(signatures_path, 'r') as f:
        if signatures_path.suffix in ['.yaml', '.yml']:
            signatures = yaml.safe_load(f)
        elif signatures_path.suffix == '.json':
            signatures = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {signatures_path.suffix}. Use .yaml, .yml, or .json")

    logger.info(f"Loaded {len(signatures)} cell type signatures from {signatures_path}")
    return signatures


def extract_sample_id_from_tiff(tiff_path):
    """
    Extract sample ID from TIFF filename.
    Expects format: {sample_id}.ome.tiff
    Returns the sample_id portion.
    """
    tiff_path = Path(tiff_path)
    # Remove .ome.tiff or .tiff extension
    sample_id = tiff_path.name.replace('.ome.tiff', '').replace('.ome.tif', '')
    logger.info(f"Detected sample_id: {sample_id}")
    return sample_id


def find_sample_files(data_dir, sample_id=None):
    """
    Find all required files for a sample in the data directory.
    Mask and table are now OPTIONAL - will trigger segmentation if missing.
    
    Parameters
    ----------
    data_dir : Path
        Directory to search for files
    sample_id : str, optional
        If provided, look for specific sample. Otherwise, find first available sample.
    
    Returns
    -------
    dict : Dictionary with 'sample_id', 'tiff_path', 'mask_path', 'table_path'
           mask_path and table_path can be None (will trigger segmentation)
    """
    data_dir = Path(data_dir)
    
    if sample_id:
        # Look for specific sample
        tiff_path = data_dir / f"{sample_id}.ome.tiff"
        if not tiff_path.exists():
            tiff_path = data_dir / f"{sample_id}.ome.tif"
        
        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found for sample: {sample_id}")
        
        # Optional mask
        mask_path = data_dir / f"{sample_id}_whole_cell.tiff"
        if not mask_path.exists():
            mask_path = data_dir / f"{sample_id}_whole_cell.tif"
        if not mask_path.exists():
            logger.warning(f"Mask not found for {sample_id} - will use segmentation")
            mask_path = None
        
        # Optional table
        table_path = data_dir / f"cell_table_arcsinh_transformed_{sample_id}.csv"
        if not table_path.exists():
            logger.warning(f"Table not found for {sample_id} - will use segmentation")
            table_path = None
        
        return {
            'sample_id': sample_id,
            'tiff_path': tiff_path,
            'mask_path': mask_path,
            'table_path': table_path
        }
    
    else:
        # Auto-detect first sample - PRIORITIZE complete samples, but accept incomplete
        tiff_files = list(data_dir.glob("*.ome.tiff")) + list(data_dir.glob("*.ome.tif"))
        
        if not tiff_files:
            raise FileNotFoundError(f"No .ome.tiff files found in {data_dir}")
        
        # First pass: try to find a complete sample (with mask and table)
        for tiff_path in sorted(tiff_files):
            detected_sample_id = extract_sample_id_from_tiff(tiff_path)
            
            mask_path = data_dir / f"{detected_sample_id}_whole_cell.tiff"
            if not mask_path.exists():
                mask_path = data_dir / f"{detected_sample_id}_whole_cell.tif"
            
            table_path = data_dir / f"cell_table_arcsinh_transformed_{detected_sample_id}.csv"
            
            if mask_path.exists() and table_path.exists():
                logger.info(f"Auto-detected COMPLETE sample: {detected_sample_id}")
                return {
                    'sample_id': detected_sample_id,
                    'tiff_path': tiff_path,
                    'mask_path': mask_path,
                    'table_path': table_path
                }
        
        # Second pass: if no complete sample, use first TIFF and trigger segmentation
        logger.warning("No complete samples found - will use segmentation for first TIFF")
        tiff_path = sorted(tiff_files)[0]
        detected_sample_id = extract_sample_id_from_tiff(tiff_path)
        
        # Check for optional files
        mask_path = data_dir / f"{detected_sample_id}_whole_cell.tiff"
        if not mask_path.exists():
            mask_path = data_dir / f"{detected_sample_id}_whole_cell.tif"
        if not mask_path.exists():
            mask_path = None
        
        table_path = data_dir / f"cell_table_arcsinh_transformed_{detected_sample_id}.csv"
        if not table_path.exists():
            table_path = None
        
        logger.info(f"Auto-detected INCOMPLETE sample: {detected_sample_id}")
        logger.info(f"  Has mask: {mask_path is not None}")
        logger.info(f"  Has table: {table_path is not None}")
        
        return {
            'sample_id': detected_sample_id,
            'tiff_path': tiff_path,
            'mask_path': mask_path,
            'table_path': table_path
        }


def extract_channel_names(sdata, nuclear_stains=['DAPI_INIT', 'DAPI_FINAL']):
    """
    Extract channel names from the fluorescence image element.
    Removes nuclear stains and returns list of marker names.
    """
    img = sdata.images['fluorescence']

    # Handle both DataTree (multiscale) and DataArray (single scale)
    if hasattr(img, 'items'):  # DataTree with multiple scales
        # Get the highest resolution level (usually 'scale0')
        level_data = list(img.values())[0]
        channel_names = list(level_data.coords['c'].values)
    else:  # Single scale DataArray
        channel_names = list(img.coords['c'].values)

    # Remove nuclear stains
    marker_names = [ch for ch in channel_names if ch not in nuclear_stains]

    logger.info(f"Detected {len(channel_names)} total channels")
    logger.info(f"Using {len(marker_names)} markers (excluded: {nuclear_stains})")
    logger.info(f"Markers: {marker_names}")

    return marker_names, channel_names

def check_and_prepare_segmentation(
    tiff_path: Path,
    data_dir: Path,
    sample_id: str,
    nuclear_stains: List[str],
    channel_names: List[str]
) -> Tuple[Optional[Path], Optional[Path], bool]:
    """
    Check if mask and table exist. If not, prepare for segmentation.
    
    Returns
    -------
    mask_path : Path or None
        Path to mask if exists
    table_path : Path or None
        Path to table if exists
    needs_segmentation : bool
        True if segmentation is needed
    """
    mask_path = data_dir / f"{sample_id}_whole_cell.tiff"
    if not mask_path.exists():
        mask_path = data_dir / f"{sample_id}_whole_cell.tif"
    
    table_path = data_dir / f"cell_table_arcsinh_transformed_{sample_id}.csv"
    
    has_mask = mask_path.exists()
    has_table = table_path.exists()
    
    if not has_mask or not has_table:
        logger.warning(f"Missing {'mask' if not has_mask else ''}"
                      f"{' and ' if not has_mask and not has_table else ''}"
                      f"{'table' if not has_table else ''} for {sample_id}")
        logger.info("Will perform Cellpose segmentation...")
        return None, None, True
    
    return mask_path, table_path, False


def run_analysis(
    sample_id,
    tiff_path,
    mask_path,
    table_path,
    output_base_dir,
    signatures_path,
    nuclear_stains=['DAPI_INIT', 'DAPI_FINAL'],
    pixel_size_um=0.325,
    microscope="CellDive",
    objective="20x",
    created_by="MJBC",
    pipeline_version="0.1.0",
    n_mads=3.0,
    gmm_probability_threshold=0.7,
    gmm_min_combined_prob=0.3,
    visualize="False",
    # ADD SEGMENTATION PARAMS
    nucleus_channels=None,  # e.g., [0] for DAPI
    cytoplasm_channels=None,  # e.g., [1,2,3] for membrane markers
    nucleus_diameter=30.0,
    cytoplasm_diameter=60.0,
):
    """
    Run complete spatial analysis pipeline for a single sample.

    Parameters
    ----------
    sample_id : str
        Sample identifier (extracted from filename)
    tiff_path : Path
        Path to OME-TIFF file
    mask_path : Path
        Path to segmentation mask file
    table_path : Path
        Path to cell table CSV
    output_base_dir : str or Path
        Base output directory (phase subdirs will be created)
    signatures_path : str or Path
        Path to YAML/JSON file containing cell type signatures
    nuclear_stains : list
        List of nuclear stain channel names to exclude from marker analysis
    pixel_size_um : float
        Pixel size in micrometers
    microscope : str
        Microscope type
    objective : str
        Objective magnification
    created_by : str
        Pipeline user/operator ID
    pipeline_version : str
        Version of analysis pipeline
    n_mads : float
        Number of MADs for QC outlier detection
    gmm_probability_threshold : float
        GMM probability threshold for phenotype assignment
    gmm_min_combined_prob : float
        Minimum combined probability for GMM phenotyping
    """
    # Parse sample ID
    parts = sample_id.split("_")
    if len(parts) >= 4:
        experiment_id, donor_id, tp_code = parts[0], parts[1], parts[2]
        fov_num = parts[-1]

        TIMEPOINT_MAP = {"BL": "baseline", "D21": "D21", "D128": "D128"}
        fov_id = f"FOV_{fov_num}"
        timepoint = TIMEPOINT_MAP.get(tp_code, tp_code)
    else:
        # Fallback if sample_id doesn't match expected format
        logger.warning(f"Sample ID {sample_id} doesn't match expected format. Using defaults.")
        experiment_id = "unknown"
        donor_id = sample_id
        timepoint = "unknown"
        fov_id = "FOV_000"

    # Setup directories
    output_base_dir = Path(output_base_dir)
    output_phase1_dir = output_base_dir / "phase_1"
    output_phase2_dir = output_base_dir / "phase_2"
    output_phase3_dir = output_base_dir / "phase_3"

    for dir_path in [output_phase1_dir, output_phase2_dir, output_phase3_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Convert paths to Path objects
    tiff_path = Path(tiff_path)
    mask_path = Path(mask_path) if mask_path is not None else None
    table_path = Path(table_path) if table_path is not None else None

    # Output paths
    output_zarr_phase1 = output_phase1_dir / f"{sample_id}.zarr"
    output_zarr_phase2_flagged = output_phase2_dir / f"{sample_id}_flagged.zarr"
    output_zarr_phase2_filtered = output_phase2_dir / f"{sample_id}_filtered.zarr"
    output_zarr_phase3 = output_phase3_dir / f"{sample_id}_phenotyped.zarr"

    logger.info("="*80)
    logger.info(f"PHASE 1: Ingestion - {sample_id}")
    logger.info("="*80)
    logger.info(f" TIFF: {tiff_path}")
    logger.info(f" Mask: {mask_path if mask_path else 'None (will segment)'}")
    logger.info(f" Table: {table_path if table_path else 'None (will segment)'}")

    # Ingest TIFF to SpatialData
    image_element = ingest_tiff_to_spatialdata(
        tiff_path=tiff_path,
        sample_id=sample_id,
        donor_id=donor_id,
        fov_id=fov_id,
        experiment_id=experiment_id,
        disease_status="NA",
        timepoint=timepoint,
        acquisition_date="NA",
        microscope=microscope,
        objective=objective,
        created_by=created_by,
        pipeline_version=pipeline_version,
        nuclear_stains=nuclear_stains,
        pixel_size_um=pixel_size_um,
    )

    # CHECK IF SEGMENTATION IS NEEDED
    if mask_path is None or table_path is None:
        logger.info("No mask or table provided - performing segmentation")
        
        # Import segmentation module
        from phase0_segmentation import (
            SegmentationConfig, 
            segment_and_create_tables
        )
        
        # Load image data
        img_data = tifffile.imread(tiff_path)
        if img_data.ndim == 2:
            img_data = img_data[np.newaxis, :, :]
        elif img_data.ndim == 3 and img_data.shape[0] > img_data.shape[2]:
            img_data = np.transpose(img_data, (2, 0, 1))
        
        # Get channel names from image_element
        if hasattr(image_element, 'coords'):
            all_channel_names = list(image_element.coords['c'].values)
        else:
            all_channel_names = [f"Channel_{i}" for i in range(img_data.shape[0])]
        
        # Configure segmentation
        if nucleus_channels is None:
            # Auto-detect DAPI channels
            nucleus_channels = [i for i, name in enumerate(all_channel_names) 
                               if any(nuc in name.upper() for nuc in ['DAPI', 'HOECHST'])]
            if not nucleus_channels:
                nucleus_channels = [0]  # Default to first channel
        
        if cytoplasm_channels is None:
            # Use all non-nuclear channels
            cytoplasm_channels = [i for i in range(len(all_channel_names)) 
                                 if i not in nucleus_channels]
        
        config = SegmentationConfig(
            nucleus_channels=nucleus_channels,
            cytoplasm_channels=cytoplasm_channels,
            nucleus_diameter=nucleus_diameter,
            cytoplasm_diameter=cytoplasm_diameter,
        )

        # Prepare metadata for propagation to AnnData tables
        segmentation_metadata = {
            'sample_id': sample_id,
            'donor_id': donor_id,
            'fov_id': fov_id,
            'experiment_id': experiment_id,
            'timepoint': timepoint,
            'microscope': microscope,
            'objective': objective,
            'created_by': created_by,
            'pipeline_version': pipeline_version,
        }

        # Run segmentation
        masks_dict, morph_table, intensity_table, cell_centroids = \
            segment_and_create_tables(
                img=img_data,
                config=config,
                channel_names=all_channel_names,
                image_name=sample_id,
                metadata=segmentation_metadata
            )

        # Parse all three mask types to Labels2DModel for SpatialData compliance
        cell_mask = Labels2DModel.parse(
            masks_dict['cell'],
            dims=("y", "x"),
            transformations={"global": Identity()}
        )

        nucleus_mask = Labels2DModel.parse(
            masks_dict['nucleus'],
            dims=("y", "x"),
            transformations={"global": Identity()}
        )

        cytoplasm_mask = Labels2DModel.parse(
            masks_dict['cytoplasm'],
            dims=("y", "x"),
            transformations={"global": Identity()}
        )

        # Set primary mask for backwards compatibility
        mask = cell_mask
        
    else:
        # Use existing mask and table
        mask = Labels2DModel.parse(
            tifffile.imread(mask_path),
            dims=("y", "x"),
            transformations={"global": Identity()}
        )

        # Prepare metadata for propagation to AnnData tables
        csv_metadata = {
            'sample_id': sample_id,
            'donor_id': donor_id,
            'fov_id': fov_id,
            'experiment_id': experiment_id,
            'timepoint': timepoint,
            'microscope': microscope,
            'objective': objective,
            'created_by': created_by,
            'pipeline_version': pipeline_version,
        }

        morph_table, intensity_table, cell_centroids = parse_csv_to_mudata_tables(
            table_path,
            mask_path=mask_path,
            metadata=csv_metadata
        )
    
    # Create SpatialData object with all mask types
    # For segmented samples, include nucleus and cytoplasm masks
    # For pre-segmented samples, only include cell mask
    labels_dict = {"cell_segmentation": mask}

    # Add nucleus and cytoplasm masks if available (from segmentation)
    if mask_path is None or table_path is None:
        labels_dict["nucleus_segmentation"] = nucleus_mask
        labels_dict["cytoplasm_segmentation"] = cytoplasm_mask
        logger.info("Added nucleus and cytoplasm segmentation masks to SpatialData")

    sdata = SpatialData(
        images={"fluorescence": image_element},
        labels=labels_dict,
        tables={
            "table_morphology": morph_table,
            "table_intensities": intensity_table
        }
    )

    # Add shapes from cell segmentation
    sdata["cell_shapes"] = to_polygons(sdata["cell_segmentation"])

    sdata = normalize_intensity_by_area(
    sdata,
    intensity_table="table_intensities",
    morphology_table="table_morphology",
    area_column="cell_size",  
    area_var="cell_size", 
    cofactor=5.0,
    use_arcsinh=True,
    store_raw=True
)
    
    # Extract channel names and continue with rest of pipeline
    marker_names, all_channel_names = extract_channel_names(sdata, nuclear_stains)

    # Save phase1 output
    logger.info(f"Saving Phase 1 output to {output_zarr_phase1}")
    sdata.write(output_zarr_phase1, overwrite=True)

    logger.info("="*80)
    logger.info(f"PHASE 2: QC and Filtering - {sample_id}")
    logger.info("="*80)

    # Load and run QC
    sdata = sd.read_zarr(output_zarr_phase1)

    sdata = flag_cells_at_qc_risk(
        sdata, 
        intensity_table="table_intensities", 
        morphology_table="table_morphology", 
        n_mads=n_mads
    )

    sdata = run_complete_qc_workflow(
        sdata,
        intensity_table="table_intensities",
        morphology_table="table_morphology",
        shapes_key="cell_shapes",
        output_dir=output_phase2_dir,
        exclude_markers=nuclear_stains,
        plot_umap=True,
        plot_spatial=True,
        plot_marker_expression=True,
        visualize=False
    )

    sdata.write(output_zarr_phase2_flagged, overwrite=True)
    sdata = sd.read_zarr(output_zarr_phase2_flagged)

    sdata_filtered = filter_sdata_by_qc_efficient(
        sdata,
        intensity_table="table_intensities",
        morphology_table="table_morphology",
        shapes_key="cell_shapes",
        labels_keys=[
        "cell_segmentation",
        "cytoplasm_segmentation",
        "nucleus_segmentation"
        ],
        qc_column="is_qc_pass",
        filter_labels=True # Bad practice but for efficiency improvment
    )

    logger.info(f"Saving Phase 2 output to {output_zarr_phase2_filtered}")
    sdata_filtered.write(output_zarr_phase2_filtered, overwrite=True)
    sdata_filtered = sd.read_zarr(output_zarr_phase2_filtered)

    logger.info("="*80)
    logger.info(f"PHASE 3: Phenotyping - {sample_id}")
    logger.info("="*80)

    # Load signatures from file
    gmm_signatures = load_signatures(signatures_path)

    # Run phenotyping with automatically detected markers
    cell_types = run_complete_phenotyping_and_labeling(
        sdata_filtered,
        signatures=gmm_signatures,
        Sample=sample_id,
        marker_names=all_channel_names,  # Use ALL channel names for phenotyping
        table_key='table_intensities',
        consensus_method='consensus',
        save_dir=output_phase3_dir
    )

    # Clean up intermediate columns
    cols_to_remove = [
        col for col in sdata_filtered.tables['table_intensities'].obs.columns 
        if any(x in col for x in ['_pos_otsu', '_pos_triangle', '_intensity', 
                                   '_level', '_consensus', '_pos_multiotsu'])
    ]
    if cols_to_remove:
        logger.info(f"Removing {len(cols_to_remove)} intermediate phenotyping columns")
        sdata_filtered.tables['table_intensities'].obs.drop(columns=cols_to_remove, inplace=True)

    # UMAP workflow
    sdata_filtered = run_complete_umap_cell_type_workflow(
        sdata_filtered,
        table_key='table_intensities',
        exclude_patterns=nuclear_stains,
        cell_type_col='cell_type',
        output_dir=output_phase3_dir
    )

    # GMM phenotyping workflow
    gmm_results, phenotype_probs = run_gmm_phenotyping_workflow(
        sdata_filtered,
        signatures=gmm_signatures,
        table_key='table_intensities',
        marker_names=marker_names,  # Use marker names (excluding nuclear stains)
        probability_threshold=gmm_probability_threshold,
        min_combined_prob=gmm_min_combined_prob,
        save_dir=output_phase3_dir
    )

    # Spatial visualization
    plot_gmm_phenotypes_spatial(
        sdata_filtered,
        table_key='table_intensities',
        shapes_key='cell_shapes',
        phenotype_col='phenotype_gmm',
        save_path=f"{output_phase3_dir}/{sample_id}_gmm_phenotypes_spatial.pdf",
        figsize=(12, 12),
        alpha=0.8,
        outline_width=0.3,
        exclude_unlabelled=False,
        show_legend=True,
        visualize = False
    )

    logger.info("="*80)
    logger.info(f"PHASE 3: Clustering & Phenotyping - {sample_id}")
    logger.info("="*80)

    from phase3_unsupervised_clustering import (
        run_unsupervised_clustering,
        visualize_clustering_results,
        compute_cluster_marker_profiles
    )

    logger.info("Step 1: Unsupervised clustering (Leiden algorithm)")
    sdata_filtered = run_unsupervised_clustering(
        sdata_filtered,
        table_key="table_intensities",
        exclude_markers=nuclear_stains,
        exclude_patterns=None,
        use_layer=None,  # Uses .X (arcsinh-transformed data)
        n_neighbors=15,
        n_pcs=20,
        resolutions=[0.3, 0.5, 0.8, 1.0, 1.5],  # Test multiple resolutions
        random_state=42
        )  
    
    # Step 2: Visualize clustering results
    logger.info("Step 2: Visualizing clustering results")
    visualize_clustering_results(
        sdata_filtered,
        table_key="table_intensities",
        resolutions=[0.3, 0.5, 0.8, 1.0, 1.5],
        save_dir=output_phase3_dir,
        visualize=False
    )

    logger.info("Step 3: Computing cluster marker profiles")
    cluster_profiles = compute_cluster_marker_profiles(
        sdata_filtered,
        table_key="table_intensities",
        clustering_key="leiden_res_0.8",
        exclude_markers= ["DAPI_INIT", "DAPI_FINAL"],
        use_layer=None,
        save_dir=output_phase3_dir,
        visualize=False
    )

    logger.info(f"Saving Phase 3 output to {output_zarr_phase3}")
    sdata_filtered.write(output_zarr_phase3, overwrite=True)

    logger.info("="*80)
    logger.info(f"✓ Pipeline complete for {sample_id}")
    logger.info("="*80)

    return sdata_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Run spatial transcriptomics/proteomics analysis pipeline",
        epilog="""
Examples:
  # Auto-detect sample from directory
  python run_spatial_analysis.py --data-dir ./batch --signatures signatures.yaml

  # Specify a specific TIFF file
  python run_spatial_analysis.py --tiff-path ./batch/sample1.ome.tiff --signatures signatures.yaml

  # Process sample in current directory with custom output
  python run_spatial_analysis.py --data-dir . --output-dir ./results --signatures signatures.yaml
        """
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing sample files (will auto-detect sample)"
    )
    input_group.add_argument(
        "--tiff-path",
        type=Path,
        help="Path to specific .ome.tiff file (will infer sample ID from filename)"
    )

    # Required arguments
    parser.add_argument(
        "--signatures",
        type=Path,
        required=True,
        help="Path to YAML or JSON file with cell type signatures"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd().parent / "output",
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--nuclear-stains",
        nargs="+",
        default=["DAPI_INIT", "DAPI_FINAL"],
        help="Nuclear stain channel names to exclude (default: DAPI_INIT DAPI_FINAL)"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.325,
        help="Pixel size in micrometers (default: 0.325)"
    )
    parser.add_argument(
        "--microscope",
        default="CellDive",
        help="Microscope type (default: CellDive)"
    )
    parser.add_argument(
        "--objective",
        default="20x",
        help="Objective magnification (default: 20x)"
    )
    parser.add_argument(
        "--created-by",
        default="MJBC",
        help="Pipeline operator ID (default: MJBC)"
    )
    parser.add_argument(
        "--pipeline-version",
        default="0.1.0",
        help="Pipeline version (default: 0.1.0)"
    )
    parser.add_argument(
        "--n-mads",
        type=float,
        default=3.0,
        help="Number of MADs for QC outlier detection (default: 3.0)"
    )
    parser.add_argument(
        "--gmm-prob-threshold",
        type=float,
        default=0.5,
        help="GMM probability threshold (default: 0.5)"
    )
    parser.add_argument(
        "--gmm-min-prob",
        type=float,
        default=0.1,
        help="GMM minimum combined probability (default: 0.1)"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Display plots interactively (default: save only)"
    )
    # Add to argument parser
    parser.add_argument(
        "--nucleus-channels",
        nargs="+",
        type=int,
        help="Channel indices for nuclear segmentation (e.g., 0). Auto-detected if not provided."
    )
    parser.add_argument(
        "--cytoplasm-channels",
        nargs="+",
        type=int,
        help="Channel indices for cytoplasm segmentation. Auto-detected if not provided."
    )
    parser.add_argument(
        "--nucleus-diameter",
        type=float,
        default=30.0,
        help="Expected nucleus diameter in pixels (default: 30.0)"
    )
    parser.add_argument(
        "--cytoplasm-diameter",
        type=float,
        default=60.0,
        help="Expected cell diameter in pixels (default: 60.0)"
    )

    args = parser.parse_args()

    # Determine sample files
    if args.tiff_path:
        # Extract sample ID from TIFF filename
        tiff_path = Path(args.tiff_path)
        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

        sample_id = extract_sample_id_from_tiff(tiff_path)
        data_dir = tiff_path.parent

        # Find associated files
        sample_files = find_sample_files(data_dir, sample_id)
    else:
        # Auto-detect sample from directory
        sample_files = find_sample_files(args.data_dir)

    # Run analysis
    run_analysis(
        sample_id=sample_files['sample_id'],
        tiff_path=sample_files['tiff_path'],
        mask_path=sample_files['mask_path'],
        table_path=sample_files['table_path'],
        output_base_dir=args.output_dir,
        signatures_path=args.signatures,
        nuclear_stains=args.nuclear_stains,
        pixel_size_um=args.pixel_size,
        microscope=args.microscope,
        objective=args.objective,
        created_by=args.created_by,
        pipeline_version=args.pipeline_version,
        n_mads=args.n_mads,
        gmm_probability_threshold=args.gmm_prob_threshold,
        gmm_min_combined_prob=args.gmm_min_prob,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
