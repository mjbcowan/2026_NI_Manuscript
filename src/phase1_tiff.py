"""
TIFF Image Ingestion Pipeline

extracts channel names from OME-TIFF with multiple fallback strategies.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import warnings
import xml.etree.ElementTree as ET
import io

# Core dependencies
import xarray as xr
import tifffile
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

# Optional for enhanced reading
try:
    from aicsimageio import AICSImage
    from aicsimageio.readers import TiffReader
    HAS_AICSIMAGEIO = True
except ImportError:
    HAS_AICSIMAGEIO = False
    warnings.warn("aicsimageio not available. Install with: pip install aicsimageio")


def extract_channel_names_from_ome_xml(ome_xml_string: str) -> Optional[List[str]]:
    """
    Parse OME-XML string to extract channel names.
    
    Handles multiple OME-XML schema versions [web:12].
    
    Parameters
    ----------
    ome_xml_string : str
        Raw OME-XML metadata from TIFF ImageDescription
        
    Returns
    -------
    list of str or None
        Channel names if found, None otherwise
    """
    if not ome_xml_string:
        return None
    
    try:
        root = ET.parse(io.StringIO(ome_xml_string))
        
        # Try different OME schema versions
        namespaces_to_try = [
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'},
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'},
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2013-06'},
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2012-06'},
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2011-06'},
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2010-06'},
        ]
        
        for namespaces in namespaces_to_try:
            # Try to find channels in first Image
            channels = root.findall('ome:Image[1]/ome:Pixels/ome:Channel', namespaces)
            
            if channels:
                # Extract Name attribute [web:12]
                channel_names = []
                for channel in channels:
                    name = channel.attrib.get('Name')
                    if name:
                        channel_names.append(name)
                    else:
                        # Fallback to ID if Name not present
                        channel_id = channel.attrib.get('ID', f'Channel_{len(channel_names)}')
                        channel_names.append(channel_id)
                
                if channel_names:
                    return channel_names
        
        # If namespaced search failed, try without namespace (malformed OME-XML)
        channels = root.findall('.//Channel')
        if channels:
            channel_names = []
            for channel in channels:
                name = channel.attrib.get('Name') or channel.attrib.get('name')
                if name:
                    channel_names.append(name)
            if channel_names:
                return channel_names
                
    except Exception as e:
        warnings.warn(f"Failed to parse OME-XML: {e}")
        return None
    
    return None


def extract_ome_metadata_robust(tiff_path: Path) -> Dict[str, Any]:
    """
    Enhanced OME metadata extraction with multiple fallback strategies.
    
    Attempts extraction:
    1. aicsimageio
    2. tifffile OME-XML parsin
    3. Basic TIFF tag reading
    
    Parameters
    ----------
    tiff_path : Path
        Path to OME-TIFF file
        
    Returns
    -------
    dict
        Metadata dictionary with channel_names, pixel_size_um, etc.
    """
    metadata = {
        'channel_names': None,
        'n_channels': None,
        'pixel_size_um': None,
        'shape': None,
        'dtype': None,
    }
    
    # Strategy 1: aicsimageio first
    if HAS_AICSIMAGEIO:
        try:
            img = AICSImage(tiff_path, reader=TiffReader)
            
            # Get channel names using aicsimageio's method
            channel_names = img.get_channel_names()
            
            # Validate that we got real names, not defaults
            if channel_names and not all(name.startswith('Channel:') for name in channel_names):
                metadata['channel_names'] = channel_names
                print(f"  ✓ Channel names from aicsimageio: {channel_names}")
            
            # Get other metadata
            metadata['n_channels'] = img.dims.C if 'C' in img.dims else 1
            
            if img.physical_pixel_sizes.X is not None:
                metadata['pixel_size_um'] = float(img.physical_pixel_sizes.X)
            
            metadata['shape'] = img.shape
            metadata['dims'] = img.dims.order
            
        except Exception as e:
            warnings.warn(f"aicsimageio extraction failed: {e}")
    
    # Strategy 2: Direct OME-XML parsing with tifffile
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            # Get OME-XML from first page description
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                ome_xml = tif.ome_metadata
            elif tif.pages[0].description:
                # Check if description looks like OME-XML
                desc = tif.pages[0].description
                if '<?xml' in desc and 'OME' in desc:
                    ome_xml = desc
                else:
                    ome_xml = None
            else:
                ome_xml = None
            
            # Parse OME-XML for channel names
            if ome_xml and metadata['channel_names'] is None:
                channel_names = extract_channel_names_from_ome_xml(ome_xml)
                if channel_names:
                    metadata['channel_names'] = channel_names
                    print(f"  ✓ Channel names from OME-XML: {channel_names}")
            
            # Get basic image info if not already set
            page = tif.pages[0]
            if metadata['shape'] is None:
                metadata['shape'] = page.shape
            if metadata['dtype'] is None:
                metadata['dtype'] = page.dtype
            
            # Determine channel count if not set
            if metadata['n_channels'] is None:
                if len(tif.pages) > 1:
                    metadata['n_channels'] = len(tif.pages)
                elif len(page.shape) == 3:
                    # Assume CYX format
                    metadata['n_channels'] = page.shape[0]
                else:
                    metadata['n_channels'] = 1
            
            # Extract pixel size from TIFF tags if not set
            if metadata['pixel_size_um'] is None:
                # Check for resolution tags
                if 'XResolution' in page.tags:
                    x_res_tag = page.tags['XResolution'].value
                    if isinstance(x_res_tag, tuple) and len(x_res_tag) == 2:
                        # Resolution is in DPI by default, convert to micrometers
                        dpi = x_res_tag[0] / x_res_tag[1]
                        # Check resolution unit
                        unit = page.tags.get('ResolutionUnit')
                        if unit and unit.value == 3:  # Centimeter
                            metadata['pixel_size_um'] = 10000.0 / dpi
                        else:  # Inch or unknown
                            metadata['pixel_size_um'] = 25400.0 / dpi
                
                # Also check for ImageJ-style metadata
                if 'ImageDescription' in page.tags:
                    desc = page.tags['ImageDescription'].value
                    if isinstance(desc, str) and 'spacing=' in desc:
                        # Try to extract spacing from ImageJ format
                        try:
                            spacing_line = [l for l in desc.split('\n') if 'spacing=' in l][0]
                            spacing = float(spacing_line.split('=')[1])
                            metadata['pixel_size_um'] = spacing
                        except:
                            pass
                            
    except Exception as e:
        warnings.warn(f"tifffile metadata extraction failed: {e}")
    
    return metadata


def ingest_tiff_to_spatialdata(
    tiff_path: Path,
    sample_id: str,
    donor_id: str,
    fov_id: str,
    experiment_id: str,
    disease_status: str,
    timepoint: str,
    acquisition_date: str,
    created_by: str,
    pipeline_version: str,
    microscope: str,
    objective: str,
    nuclear_stains: List[str],
    pixel_size_um: Optional[float] = None,
    channel_names: Optional[List[str]] = None,
    coordinate_system: str = "global",
    image_layer_name: Optional[str] = None,
    **additional_metadata
) -> SpatialData:
    """
    FIXED: Main ingestion function
    
    Parameters
    ----------
    tiff_path : Path
        Path to input OME-TIFF file
    sample_id : str
        Sample identifier (required)
    donor_id : str
        Donor identifier (required)
    fov_id : str
        Field of view identifier (required)
    experiment_id : str
        Experiment batch identifier (required)
    disease_status : str
        Disease status (required)
    timepoint : str
        Timepoint identifier (required)
    acquisition_date : str
        ISO 8601 date (required)
    created_by : str
        Pipeline creator (required)
    pipeline_version : str
        Version for reproducibility (required)
    microscope : str
        Microscope identifier (required)
    objective : str
        Objective specification (required)
    nuclear_stains : list of str
        List of nuclear marker channel names (required)
    pixel_size_um : float, optional
        Physical pixel size. If None, extracted from OME metadata
    channel_names : list of str, optional
        EXPLICIT channel names to use. If provided, overrides OME metadata.
        If None, attempts extraction from OME-XML. If extraction fails,
        raises an error requiring explicit specification.
    coordinate_system : str
        Coordinate system name (default: 'global')
    image_layer_name : str, optional
        Name for image layer. If None, uses sample_id
    **additional_metadata
        Additional metadata fields
        
    Returns
    -------
    SpatialData
        SpatialData object with image and metadata
        
    Examples
    --------
    >>> # With explicit channel names (recommended if OME-XML is incomplete)
    >>> sdata = ingest_tiff_to_spatialdata(
    ...     tiff_path=Path("sample_001.tif"),
    ...     sample_id="sample_001",
    ...     donor_id="donor_A",
    ...     fov_id="FOV_01",
    ...     experiment_id="exp_2024_01",
    ...     disease_status="tumor",
    ...     timepoint="baseline",
    ...     acquisition_date="2024-01-15T10:30:00",
    ...     created_by="imaging_pipeline_v1",
    ...     pipeline_version="1.0.0",
    ...     microscope="Zeiss_LSM900",
    ...     objective="20x/0.8NA",
    ...     nuclear_stains=["DAPI"],
    ...     channel_names=["DAPI", "CD45", "PanCK", "CD3", "CD8", "CD20", "CD68"],
    ...     pixel_size_um=0.325,
    ... )
    """
    tiff_path = Path(tiff_path)
    
    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
    
    print(f"{'='*80}")
    print(f"TIFF INGESTION TO SPATIALDATA")
    print(f"{'='*80}")
    print(f"File: {tiff_path.name}")
    print(f"Sample ID: {sample_id}")
    print(f"FOV ID: {fov_id}")
    print()
    
    # Step 1: Extract OME metadata
    print("[1/4] Extracting OME metadata with robust parser")
    ome_meta = extract_ome_metadata_robust(tiff_path)
    
    # Handle pixel size
    if pixel_size_um is None:
        pixel_size_um = ome_meta.get('pixel_size_um')
        if pixel_size_um is None:
            raise ValueError(
                "pixel_size_um not found in OME metadata and not provided.\n"
                "Please specify pixel_size_um explicitly as a function parameter."
            )
    print(f"Pixel size: {pixel_size_um} µm")
    
    # Handle channel names
    if channel_names is not None:
        # User-provided names take precedence
        print(f"Using provided channel names: {channel_names}")
    else:
        # Try to extract from metadata
        channel_names = ome_meta.get('channel_names')
        
        if channel_names is None:
            # Generate defaults but warn user
            n_channels = ome_meta.get('n_channels', 1)
            channel_names = [f"Channel_{i}" for i in range(n_channels)]
            warnings.warn(
                f"\n{'='*80}\n"
                f"WARNING: Channel names not found in OME-XML metadata!\n"
                f"Using default names: {channel_names}\n\n"
                f"To fix this, provide explicit channel_names parameter:\n"
                f"  channel_names=['DAPI', 'CD45', 'PanCK', ...]\n"
                f"{'='*80}\n"
            )
        else:
            print(f"Extracted channel names from OME-XML: {channel_names}")
    
    # Validate nuclear stains are in channel names
    missing_stains = [s for s in nuclear_stains if s not in channel_names]
    if missing_stains:
        raise ValueError(
            f"Nuclear stains {missing_stains} not found in channel_names {channel_names}.\n"
            f"Check your spelling or update nuclear_stains parameter."
        )
    
    # Step 2: Load image data
    print("\n[2/4] Loading image data")
    if HAS_AICSIMAGEIO:
        img = AICSImage(tiff_path, reader=TiffReader)
        image_data = img.get_image_data("CYX")
    else:
        with tifffile.TiffFile(tiff_path) as tif:
            image_data = tif.asarray()
            
            # Standardize to CYX
            if image_data.ndim == 2:
                image_data = image_data[np.newaxis, :, :]
            elif image_data.ndim == 3:
                if image_data.shape[0] > image_data.shape[2]:
                    # Likely YXC, transpose to CYX
                    image_data = np.transpose(image_data, (2, 0, 1))
    
    print(f"Shape: {image_data.shape} (C, Y, X)")
    print(f"Dtype: {image_data.dtype}")
    
    # Validate channel count matches
    if image_data.shape[0] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: image has {image_data.shape[0]} channels "
            f"but {len(channel_names)} channel names provided: {channel_names}"
        )
    
    # Step 3: Create metadata
    print("\n[3/4] Creating FAIR metadata...")
    metadata = {
        # Core identifiers
        'sample_id': sample_id,
        'donor_id': donor_id,
        'fov_id': fov_id,
        'experiment_id': experiment_id,
        
        # Biological context
        'disease_status': disease_status,
        'timepoint': timepoint,
        
        # Provenance
        'acquisition_date': acquisition_date,
        'created_by': created_by,
        'pipeline_version': pipeline_version,
        'ingestion_timestamp': datetime.now().isoformat(),
        
        # Technical metadata
        'microscope': microscope,
        'objective': objective,
        'pixel_size_um': float(pixel_size_um),
        'data_type': 'fluorescence_microscopy',
        'coordinate_system': coordinate_system,
        
        # Channel information
        'channel_names': channel_names,
        'n_channels': len(channel_names),
        'nuclear_stains': nuclear_stains,
    }
    metadata.update(additional_metadata)
    print(f"Metadata fields: {len(metadata)}")
    
    # Step 4: Create SpatialData Image element
    print("\n[4/4] Creating SpatialData object")
    
    # Create xarray DataArray with proper coordinates
    dims = ['c', 'y', 'x']
    coords = {
        'c': channel_names,
        'y': np.arange(image_data.shape[1]),
        'x': np.arange(image_data.shape[2])
    }
    
    data_array = xr.DataArray(
        image_data,
        dims=dims,
        coords=coords,
        attrs=metadata
    )
    
    # Parse with SpatialData Image2DModel
    transformations = {coordinate_system: Identity()}
    
    image_element = Image2DModel.parse(
        data_array,
        dims=dims,
        transformations=transformations,
        c_coords=channel_names
    )
    
    return image_element

import pandas as pd
import numpy as np
import anndata as ad
from spatialdata.models import TableModel, PointsModel
from spatialdata.transformations import Identity
from typing import Optional, Tuple


def parse_csv_to_mudata_tables(
    csv_path: str,
    mask_path: Optional[str] = None,
    fov: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    """
    Parse CSV

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    mask_path : Optional[str]
        Path to mask file for validation
    fov : Optional[str]
        Field of view identifier to filter data
    metadata : Optional[Dict[str, Any]]
        Sample metadata to propagate to AnnData .obs (e.g., sample_id, donor_id,
        experiment_id, timepoint, disease_status, batch, etc.)

    Returns
    -------
    morphology_adata : ad.AnnData
        Morphology table
    intensity_adata : ad.AnnData
        Intensity table
    centroids_points : GeoDataFrame
        Cell centroids
    """
    import tifffile
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter by FOV if specified
    if fov is not None and 'fov' in df.columns:
        df = df[df['fov'] == fov].copy()
    
    # Validate against mask
    if mask_path is not None:
        mask = tifffile.imread(mask_path)
        mask_labels = set(np.unique(mask)) - {0}
        csv_labels = set(df['label'].values)
        
        if csv_labels != mask_labels:
            missing_in_mask = csv_labels - mask_labels
            if missing_in_mask:
                print(f"{len(missing_in_mask)} labels in CSV but not in mask")
                print(f"Removing these from table")
                df = df[df['label'].isin(mask_labels)]
    
    df['label'] = df['label'].astype(np.uint32)
    
    # Define column groups
    intensity_channels = []
    morphology_features = ['cell_size']
    
    # Create shared obs
    obs_data = pd.DataFrame({
        'cell_id': df['label'].astype(str).values,
        'x': df['centroid-1'].values,
        'y': df['centroid-0'].values,
        'region': 'cell_segmentation',
        'instance_id': df['label'].values,
    })

    if 'fov' in df.columns:
        obs_data['fov'] = df['fov'].values
    if 'mask_type' in df.columns:
        obs_data['mask_type'] = df['mask_type'].values

    # PROPAGATE METADATA to obs for batch correction and downstream analysis
    if metadata:
        # Define standard metadata fields that should be propagated
        metadata_fields = [
            'sample_id', 'donor_id', 'experiment_id', 'timepoint',
            'disease_status', 'fov_id', 'batch', 'microscope',
            'acquisition_date', 'tissue_type', 'condition'
        ]

        for key in metadata_fields:
            if key in metadata:
                # Replicate scalar metadata across all cells
                obs_data[key] = metadata[key]

        # Also allow custom metadata fields
        for key, value in metadata.items():
            if key not in obs_data.columns and key not in metadata_fields:
                obs_data[key] = value

        print(f"✓ Propagated {sum(k in metadata for k in metadata_fields)} metadata fields to .obs")

    obs_data.index = obs_data['cell_id']
    
    # === MORPHOLOGY TABLE ===
    available_morph = [f for f in morphology_features if f in df.columns]
    morph_X = df[available_morph].values
    morph_var = pd.DataFrame(index=available_morph, data={'feature_type': 'morphology'})
    morphology_adata = ad.AnnData(X=morph_X, obs=obs_data.copy(), var=morph_var, dtype=np.float32)
    morphology_adata = TableModel.parse(
        morphology_adata,
        region='cell_segmentation',
        region_key='region',
        instance_key='instance_id'
    )
    
    # === INTENSITY TABLE ===
    available_channels = [ch for ch in intensity_channels if ch in df.columns]
    intensity_X = df[available_channels].values
    intensity_var = pd.DataFrame(
        index=available_channels,
        data={'feature_type': 'intensity', 'channel_name': available_channels}
    )
    intensity_adata = ad.AnnData(X=intensity_X, obs=obs_data.copy(), var=intensity_var, dtype=np.float32)
    intensity_adata = TableModel.parse(
        intensity_adata,
        region='cell_segmentation',
        region_key='region',
        instance_key='instance_id'
    )
    
    # === CENTROID POINTS ===
    centroids_df = pd.DataFrame({
        'x': df['centroid-1'].values,
        'y': df['centroid-0'].values,
        'cell_id': df['label'].values,
    })
    
    if 'fov' in df.columns:
        centroids_df['fov'] = df['fov'].values
    if 'mask_type' in df.columns:
        centroids_df['mask_type'] = df['mask_type'].values
    
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Identity
    
    centroids_points = PointsModel.parse(
        centroids_df,
        coordinates={'x': 'x', 'y': 'y'},
        transformations={'global': Identity()}  # This is actually fine - it's how the attrs are stored
    )
    
    print(f"Created morphology table: {morphology_adata.shape}")
    print(f"Created intensity table: {intensity_adata.shape}")
    print(f"Created centroids points: {len(centroids_points)} cells")
    
    return morphology_adata, intensity_adata, centroids_points