# Add to phase1_tiff.py or create phase1b_normalize.py

import numpy as np
import anndata as ad
from spatialdata import SpatialData
from typing import Optional
import warnings
import pandas as pd

def normalize_intensity_by_area(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    area_column: str = "cell_size",
    area_var: str = "cellarea",    
    cofactor: float = 5.0, # Default - between 1 and 5 typical
    use_arcsinh: bool = True,
    store_raw: bool = True,
) -> SpatialData:
    """
    Normalise intensity measurements by cell area and apply arcsinh transformation.
    
    Parameters
    ----------
    area_column : str
        Column name in morphology_table.obs for area (e.g., 'cell_size')
    area_var : str
        Variable name in morphology_table.var_names if area not in .obs (e.g., 'cellarea')
        This is used when morphology features are stored in .X
    """
    
    if intensity_table not in sdata.tables:
        raise ValueError(f"Intensity table '{intensity_table}' not found")
    if morphology_table not in sdata.tables:
        raise ValueError(f"Morphology table '{morphology_table}' not found")
        
    adata_int = sdata.tables[intensity_table]
    adata_morph = sdata.tables[morphology_table]
    
    # Verify alignment
    if not np.array_equal(adata_int.obs_names, adata_morph.obs_names):
        raise ValueError("Intensity and morphology tables must be aligned")
    
    # Get area - try multiple locations
    area = None
    
    # 1. Try .obs first (e.g., pre-computed data)
    if area_column in adata_morph.obs.columns:
        area = adata_morph.obs[area_column].values
        print(f"Using area from .obs['{area_column}']")
    
    # 2. Try .var_names (e.g., from segmentation)
    elif area_var in adata_morph.var_names:
        area_idx = list(adata_morph.var_names).index(area_var)
        area = adata_morph.X[:, area_idx].flatten()
        print(f"Using area from .X[:, '{area_var}'] (index {area_idx})")
    
    # 3. Fallback
    elif adata_morph.n_vars == 1:
        area = adata_morph.X[:, 0].flatten()
        warnings.warn(f"Using first morphology feature as area (only 1 variable)")
    
    else:
        # Provide helpful error message
        available_obs = list(adata_morph.obs.columns)
        available_vars = list(adata_morph.var_names)
        raise ValueError(
            f"Cannot find area!\n"
            f"  - Tried .obs['{area_column}']: not found\n"
            f"  - Tried .var_names['{area_var}']: not found\n"
            f"  - Morphology table has {adata_morph.n_vars} variables (not 1)\n"
            f"  Available in .obs: {available_obs}\n"
            f"  Available in .var_names: {available_vars}\n"
            f"  Hint: Set area_var='cellarea' or area_column to match your data"
        )
    
    # Validate area
    if np.any(area <= 0):
        n_invalid = np.sum(area <= 0)
        warnings.warn(f"Found {n_invalid} cells with area <= 0. Setting to minimum valid area.")
        min_valid = np.min(area[area > 0])
        area = np.maximum(area, min_valid)
    
    print("=" * 70)
    print(f"INTENSITY NORMALIZATION - Area & Arcsinh")
    print("=" * 70)
    print(f"Cells: {adata_int.n_obs}")
    print(f"Markers: {adata_int.n_vars}")
    print(f"Area range: {area.min():.2f} - {area.max():.2f} pixels")
    
    # Store raw intensities
    if store_raw and 'raw' not in adata_int.layers:
        adata_int.layers['raw'] = adata_int.X.copy()
        print(f"\n✓ Stored raw intensities in .layers['raw']")
    
    # Step 1: Area normalization
    area_normalized = adata_int.X / area[:, np.newaxis]
    adata_int.layers['area_normalized'] = area_normalized
    
    print(f"\nStep 1: Area normalization")
    print(f"  Formula: intensity / cell_area")
    print(f"  Raw range: {adata_int.X.min():.2f} - {adata_int.X.max():.2f}")
    print(f"  Normalized range: {area_normalized.min():.4f} - {area_normalized.max():.4f}")
    
    # Step 2: Arcsinh transformation
    if use_arcsinh:
        arcsinh_data = np.arcsinh(area_normalized / cofactor)
        
        adata_int.layers['asinh'] = arcsinh_data
        
        # Set .X to transformed data (scverse convention)
        adata_int.X = arcsinh_data
        
        print(f"\nStep 2: Arcsinh transformation")
        print(f"  Formula: arcsinh(area_normalized / {cofactor})")
        print(f"  Range: {arcsinh_data.min():.4f} - {arcsinh_data.max():.4f}")
        print(f"\n✓ Set .X = .layers['asinh'] (default for downstream)")
        print(f"✓ Available layers: {list(adata_int.layers.keys())}")
    else:
        adata_int.X = area_normalized
        print(f"\nSkipped arcsinh (use_arcsinh=False)")
        print(f"✓ Set .X = .layers['area_normalized']")
    
    # Also add area to intensity table .obs for convenience
    adata_int.obs['cell_area'] = area
    
    # Metadata
    adata_int.uns['normalization'] = {
        'method': 'area_arcsinh' if use_arcsinh else 'area_only',
        'cofactor': float(cofactor) if use_arcsinh else None,
        'area_source': area_var if area_var in adata_morph.var_names else area_column,
        'timestamp': pd.Timestamp.now().isoformat(),
        'layers': {
            'raw': 'Raw unnormalized intensities',
            'area_normalized': 'Intensity / cell_area',
            'asinh': f'arcsinh(area_normalized / {cofactor})' if use_arcsinh else None
        },
        'default_layer': 'asinh' if use_arcsinh else 'area_normalized',
        'references': {
            'area_norm': 'SPACEc (Nature Comm 2025)',
            'arcsinh': 'Squidpy IMC tutorial, CATALYST Bioconductor'
        }
    }
    
    print(f"\n✓ Normalization metadata stored in .uns['normalization']")
    print("=" * 70)
    
    return sdata


def choose_arcsinh_cofactor(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    area_column: str = "cell_size",
    candidate_cofactors: list = [1, 5],
    plot: bool = True
) -> float:
    """
    Helper function to choose optimal arcsinh cofactor by examining intensity distribution.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    intensity_table : str
        Name of intensity table
    morphology_table : str
        Name of morphology table
    area_column : str
        Column for cell area
    candidate_cofactors : list
        List of cofactors to test
    plot : bool
        Whether to plot distributions
        
    Returns
    -------
    float
        Recommended cofactor
    """
    import matplotlib.pyplot as plt
    
    adata_int = sdata.tables[intensity_table]
    adata_morph = sdata.tables[morphology_table]
    area = adata_morph.obs[area_column].values if area_column in adata_morph.obs else adata_morph.X[:, 0]
    
    area_normalized = adata_int.X / area[:, np.newaxis]
    
    if plot:
        fig, axes = plt.subplots(1, len(candidate_cofactors), figsize=(4*len(candidate_cofactors), 4))
        if len(candidate_cofactors) == 1:
            axes = [axes]
    
    print(f"\nTesting arcsinh cofactors on area-normalized intensities:")
    print(f"{'Cofactor':<10} {'Mean':<10} {'Std':<10} {'Range':<20}")
    print("-" * 50)
    
    for i, cf in enumerate(candidate_cofactors):
        transformed = np.arcsinh(area_normalized / cf)
        mean_val = transformed.mean()
        std_val = transformed.std()
        range_val = f"{transformed.min():.2f} to {transformed.max():.2f}"
        
        print(f"{cf:<10} {mean_val:<10.3f} {std_val:<10.3f} {range_val:<20}")
        
        if plot:
            axes[i].hist(transformed.flatten(), bins=100, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Cofactor = {cf}')
            axes[i].set_xlabel('Arcsinh transformed intensity')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean={mean_val:.2f}')
            axes[i].legend()
    
    if plot:
        plt.tight_layout()
        plt.show()
    
    return 5  # Default for imaging data
