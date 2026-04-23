import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.filters import threshold_otsu, threshold_triangle, threshold_multiotsu
from scipy import stats
from sklearn.metrics import cohen_kappa_score

import scanpy as sc
import squidpy as sq

from typing import Dict, List, Optional, Tuple
from datetime import datetime

import os
import spatialdata_plot


# =================================================================
# PLOTTING UTILITIES - PDF export with optional visualization
# =================================================================

def save_plot_pdf(
    fig,
    output_path,
    visualize: bool = False,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save matplotlib figure as PDF with optional visualisation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : str or Path
        Output PDF file path
    visualize : bool, default False
        Whether to display the plot interactively after saving
    dpi : int, default 300
        Resolution in dots per inch for PDF output
    bbox_inches : str, default 'tight'
        Bounding box setting to minimize whitespace
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PDF
    fig.savefig(output_path, format='pdf', dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved plot: {output_path}")

    # Optionally display
    if visualize:
        plt.show()
    else:
        plt.close(fig)


def compute_thresholds_all_methods(intensity_values: np.ndarray,
                                   marker_name: str,
                                   remove_zeros: bool = True) -> Dict[str, float]:
    """
    Calculate thresholds using Otsu, Triangle, and Multi-Otsu methods.
    
    Parameters
    ----------
    intensity_values : np.ndarray
        Array of intensity values for a single marker
    marker_name : str
        Name of the marker for logging
    remove_zeros : bool, default=True
        Whether to exclude zero values before thresholding
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing threshold values for each method
    """
    if remove_zeros:
        values = intensity_values[intensity_values > 0]
    else:
        values = intensity_values
    
    if len(values) < 10:
        print(f"Warning: {marker_name} has <10 non-zero values")
        return {
            'otsu': 0, 'triangle': 0, 'multi_otsu_low': 0,
            'multi_otsu_mid': 0, 'multi_otsu_high': 0
        }
    
    thresholds = {}
    
    try:
        thresholds['otsu'] = threshold_otsu(values)
    except Exception as e:
        print(f"Otsu failed for {marker_name}: {e}")
        thresholds['otsu'] = 0
    
    try:
        thresholds['triangle'] = threshold_triangle(values)
    except Exception as e:
        print(f"Triangle failed for {marker_name}: {e}")
        thresholds['triangle'] = 0
    
    try:
        multi_thresholds = threshold_multiotsu(values, classes=4)
        thresholds['multi_otsu_low'] = multi_thresholds[0]
        thresholds['multi_otsu_mid'] = multi_thresholds[1]
        thresholds['multi_otsu_high'] = multi_thresholds[2]
    except Exception as e:
        print(f"Multi-Otsu failed for {marker_name}: {e}")
        thresholds['multi_otsu_low'] = 0
        thresholds['multi_otsu_mid'] = 0
        thresholds['multi_otsu_high'] = 0
    
    return thresholds


def get_marker_names_from_table(sdata, 
                                table_key: str = 'cell_intensity',
                                prefix: str = 'cyto_',
                                suffix: str = '_mean') -> Dict[str, str]:
    """
    Extract marker names and create mapping from short name to full variable name.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing the table
    table_key : str, default='cell_intensity'
        Key for the table in sdata.tables
    prefix : str, default='cyto_'
        Prefix to remove from variable names
    suffix : str, default='_mean'
        Suffix to remove from variable names
        
    Returns
    -------
    Dict[str, str]
        Mapping from short marker names to full variable names
    """
    table = sdata.tables[table_key]
    
    mapping = {}
    for var_name in table.var_names:
        short_name = var_name
        if prefix and short_name.startswith(prefix):
            short_name = short_name[len(prefix):]
        if suffix and short_name.endswith(suffix):
            short_name = short_name[:-len(suffix)]
        
        mapping[short_name] = var_name
    
    print(f"Available markers:")
    for short, full in mapping.items():
        print(f"  '{short}' -> '{full}'")
    
    return mapping


def phenotype_markers_otsu_triangle(sdata,
                                    table_key: str = 'cell_intensity',
                                    marker_names: Optional[List[str]] = None,
                                    methods: List[str] = ['otsu', 'triangle', 'multi_otsu'],
                                    use_layer: str = 'asinh',
                                    remove_zeros: bool = True,
                                    auto_detect_names: bool = True) -> Tuple[Dict, Dict]:
    """
    Perform cell phenotyping using Otsu and Triangle thresholding methods.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing cell intensity data
    table_key : str, default='cell_intensity'
        Key for the table in sdata.tables
    marker_names : Optional[List[str]], default=None
        List of marker names to process. If None, uses all markers in table
    methods : List[str], default=['otsu', 'triangle', 'multi_otsu']
        Thresholding methods to apply
    use_layer : str, default='asinh'
        Layer in the table to use for intensity values
    remove_zeros : bool, default=True
        Whether to exclude zero values before thresholding
    auto_detect_names : bool, default=True
        Whether to automatically detect and map marker names
        
    Returns
    -------
    Tuple[Dict, Dict]
        all_thresholds : Dictionary of threshold values per marker
        phenotype_results : Dictionary of statistics per marker
    """
    table = sdata.tables[table_key]
    
    # Get intensity data
    if use_layer in table.layers:
        intensity_matrix = table.layers[use_layer]
    else:
        print(f"Warning: Layer '{use_layer}' not found, using table.X")
        intensity_matrix = table.X
    
    # Convert to dense if sparse
    if hasattr(intensity_matrix, 'toarray'):
        intensity_matrix = intensity_matrix.toarray()
    
    # Auto-detect marker names
    if auto_detect_names:
        name_mapping = get_marker_names_from_table(sdata, table_key)
        
        if marker_names is not None:
            full_marker_names = []
            for name in marker_names:
                if name in table.var_names:
                    full_marker_names.append(name)
                elif name in name_mapping:
                    full_marker_names.append(name_mapping[name])
                else:
                    print(f"Warning: Marker '{name}' not found")
            marker_names = full_marker_names
        else:
            marker_names = table.var_names.tolist()
    else:
        if marker_names is None:
            marker_names = table.var_names.tolist()
    
    # Create clean name mapping
    short_names = {}
    for full_name in marker_names:
        clean_name = full_name
        if clean_name.startswith('cyto_'):
            clean_name = clean_name[5:]
        if clean_name.startswith('nuc_'):
            clean_name = clean_name[4:]
        if clean_name.endswith('_mean'):
            clean_name = clean_name[:-5]
        if clean_name.endswith('_intensity'):
            clean_name = clean_name[:-10]
        short_names[full_name] = clean_name
    
    print(f"\nProcessing {len(marker_names)} markers with methods: {methods}")
    
    # Store results
    all_thresholds = {}
    phenotype_results = {}
    
    marker_indices = {name: list(table.var_names).index(name) 
                     for name in marker_names if name in table.var_names}
    
    for full_marker_name, marker_idx in marker_indices.items():
        clean_name = short_names[full_marker_name]
        
        print(f"\nProcessing {clean_name} ({full_marker_name})...")
        
        values = intensity_matrix[:, marker_idx]
        
        thresholds = compute_thresholds_all_methods(values, clean_name, remove_zeros)
        all_thresholds[clean_name] = thresholds
        
        for method, thresh in thresholds.items():
            if not method.startswith('multi_otsu_'):
                print(f"  {method}: {thresh:.3f}")
        
        # Apply thresholds
        for method in methods:
            if method == 'otsu':
                table.obs[f'{clean_name}_pos_otsu'] = values > thresholds['otsu']
                table.obs[f'{clean_name}_intensity'] = values
                
            elif method == 'triangle':
                table.obs[f'{clean_name}_pos_triangle'] = values > thresholds['triangle']
                
            elif method == 'multi_otsu':
                categories = np.digitize(values, [
                    thresholds['multi_otsu_low'],
                    thresholds['multi_otsu_mid'],
                    thresholds['multi_otsu_high']
                ])
                table.obs[f'{clean_name}_level'] = pd.Categorical(
                    categories, categories=[0, 1, 2, 3], ordered=True
                )
                table.obs[f'{clean_name}_pos_multiotsu'] = values > thresholds['multi_otsu_low']
        
        # Statistics
        n_cells = len(values)
        phenotype_results[clean_name] = {
            'full_name': full_marker_name,
            'n_cells': n_cells,
            'n_positive_otsu': int((values > thresholds['otsu']).sum()),
            'n_positive_triangle': int((values > thresholds['triangle']).sum()),
            'pct_positive_otsu': float((values > thresholds['otsu']).sum() / n_cells * 100),
            'pct_positive_triangle': float((values > thresholds['triangle']).sum() / n_cells * 100),
            'mean_intensity': float(values.mean())
        }
    
    # Store metadata
    table.uns['phenotyping_thresholds'] = {
        'timestamp': datetime.now().isoformat(),
        'methods': methods,
        'thresholds': all_thresholds,
        'markers': list(all_thresholds.keys()),
        'references': {
            'otsu': 'Otsu (1979) IEEE Trans. Syst. Man Cybern.',
            'triangle': 'Zack et al. (1977) J. Histochem. Cytochem.',
        }
    }
    
    table.uns['phenotyping_stats'] = phenotype_results
    
    print(f"\nPhenotyping complete for {len(marker_indices)} markers")
    
    return all_thresholds, phenotype_results


# ============================================================================
# SECTION 2: CONSENSUS AND CELL TYPE ASSIGNMENT
# ============================================================================


def create_consensus_calls(sdata,
                          table_key: str = 'cell_intensity',
                          marker_names: List[str] = None,
                          agreement_threshold: float = 1.0,
                          use_clean_names: bool = True) -> Dict[str, str]:
    """
    Create consensus positivity calls from Otsu and Triangle methods.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing phenotyped cells
    table_key : str, default='cell_intensity'
        Key for the table in sdata.tables
    marker_names : List[str], default=None
        List of marker names to create consensus for
    agreement_threshold : float, default=1.0
        Threshold for consensus (1.0 = both methods agree, <1.0 = either method)
    use_clean_names : bool, default=True
        Whether marker names are already cleaned
        
    Returns
    -------
    Dict[str, str]
        Mapping of marker names used
    """
    table = sdata.tables[table_key]
    
    if marker_names is None:
        marker_names = list(set([col.replace('_pos_otsu', '').replace('_pos_triangle', '') 
                                for col in table.obs.columns 
                                if '_pos_otsu' in col or '_pos_triangle' in col]))
    
    print(f"Creating consensus calls for {len(marker_names)} markers...")
    
    name_mapping = {}
    
    for marker in marker_names:
        otsu_col = f'{marker}_pos_otsu'
        triangle_col = f'{marker}_pos_triangle'
        
        if otsu_col in table.obs.columns and triangle_col in table.obs.columns:
            otsu_pos = table.obs[otsu_col].values
            triangle_pos = table.obs[triangle_col].values
            
            if agreement_threshold == 1.0:
                consensus = otsu_pos & triangle_pos
            else:
                consensus = otsu_pos | triangle_pos
            
            table.obs[f'{marker}_consensus'] = consensus
            
            agreement = (otsu_pos == triangle_pos).sum() / len(otsu_pos)
            print(f"  {marker}: {agreement*100:.1f}% agreement, {consensus.sum()} consensus positive")
            
            name_mapping[marker] = marker
    
    print(f"\nConsensus calls created in table.obs['<marker>_consensus']")
    
    return name_mapping


def validate_cell_type_signatures(signatures: Dict[str, Dict],
                                  available_markers: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that cell type signatures use available markers.
    
    Parameters
    ----------
    signatures : Dict[str, Dict]
        Cell type signatures to validate
    available_markers : List[str]
        List of available marker names in the data
        
    Returns
    -------
    Tuple[bool, List[str]]
        is_valid : Whether all signatures are valid
        missing_markers : List of markers referenced but not available
    """
    missing_markers = set()
    
    for cell_type, signature in signatures.items():
        for marker in signature.get('positive', []):
            if marker not in available_markers:
                missing_markers.add(marker)
        for marker in signature.get('negative', []):
            if marker not in available_markers:
                missing_markers.add(marker)
    
    if missing_markers:
        print(f"\nWarning: The following markers are referenced in signatures but not available:")
        for marker in sorted(missing_markers):
            print(f"    - {marker}")
        print(f"\nAvailable markers: {sorted(available_markers)}")
        return False, list(missing_markers)
    
    return True, []


def assign_cell_types_combinatorial(sdata,
                                    signatures: Dict[str, Dict],
                                    table_key: str = 'cell_intensity',
                                    method: str = 'consensus',
                                    unlabelled_label: str = 'unlabelled',
                                    validate_signatures: bool = True) -> pd.Series:
    """
    Assign cell types based on combinatorial marker expression using provided signatures.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing phenotyped cells
    signatures : Dict[str, Dict]
        Cell type signatures. Each signature should have:
        - 'positive': List of markers that must be positive
        - 'negative': List of markers that must be negative
        - 'description': Optional description of the cell type
    table_key : str, default='cell_intensity'
        Key for the table in sdata.tables
    method : str, default='consensus'
        Method to use for positivity calls ('consensus', 'otsu', or 'triangle')
    unlabelled_label : str, default='unlabelled'
        Label for cells that don't match any signature
    validate_signatures : bool, default=True
        Whether to validate signatures against available markers
        
    Returns
    -------
    pd.Series
        Series of cell type assignments for each cell
        
    Examples
    --------
    >>> signatures = {
    ...     'T_cell': {
    ...         'positive': ['CD3', 'CD45'],
    ...         'negative': ['CD19'],
    ...         'description': 'T lymphocytes'
    ...     },
    ...     'B_cell': {
    ...         'positive': ['CD19', 'CD45'],
    ...         'negative': ['CD3'],
    ...         'description': 'B lymphocytes'
    ...     }
    ... }
    >>> cell_types = assign_cell_types_combinatorial(sdata, signatures)
    """
    table = sdata.tables[table_key]
    
    # Determine column suffix based on method
    if method == 'consensus':
        suffix = '_consensus'
    elif method == 'otsu':
        suffix = '_pos_otsu'
    elif method == 'triangle':
        suffix = '_pos_triangle'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'consensus', 'otsu', or 'triangle'")
    
    # Get available markers
    available_markers = list(set([col.replace(suffix, '') 
                                 for col in table.obs.columns 
                                 if col.endswith(suffix)]))
    
    # Validate signatures if requested
    if validate_signatures:
        is_valid, missing = validate_cell_type_signatures(signatures, available_markers)
        if not is_valid:
            raise ValueError(
                f"Signature validation failed. Missing markers: {missing}. "
                f"Set validate_signatures=False to skip validation."
            )
    
    # Initialize
    cell_types = pd.Series(unlabelled_label, index=table.obs.index)
    cell_type_matches = pd.DataFrame(0, index=table.obs.index, 
                                     columns=list(signatures.keys()))
    
    print(f"\nEvaluating {len(signatures)} cell type signatures using '{method}' method...")
    
    # Evaluate each signature
    for cell_type, signature in signatures.items():
        match_mask = pd.Series(True, index=table.obs.index)
        signature_valid = True
        
        # Positive markers
        for marker in signature.get('positive', []):
            col = f'{marker}{suffix}'
            if col in table.obs:
                match_mask &= table.obs[col]
            else:
                print(f"Warning: Column '{col}' not found for signature '{cell_type}'")
                signature_valid = False
                break
        
        # Negative markers
        if signature_valid:
            for marker in signature.get('negative', []):
                col = f'{marker}{suffix}'
                if col in table.obs:
                    match_mask &= ~table.obs[col]
                else:
                    print(f"Warning: Column '{col}' not found for signature '{cell_type}'")
                    signature_valid = False
                    break
        
        if signature_valid:
            cell_type_matches[cell_type] = match_mask.astype(int)
            n_matches = match_mask.sum()
            print(f"  {cell_type}: {n_matches} cells match")
    
    # Assign based on matches
    n_matches = cell_type_matches.sum(axis=1)
    
    # Unique matches
    unique_matches = n_matches == 1
    for cell_type in signatures.keys():
        mask = unique_matches & (cell_type_matches[cell_type] == 1)
        cell_types[mask] = cell_type
    
    # Multiple matches - use priority order (order in signature dict)
    priority_order = list(signatures.keys())
    multiple_matches = n_matches > 1
    
    if multiple_matches.sum() > 0:
        print(f"\n  Resolving {multiple_matches.sum()} cells with multiple matches using priority order...")
        
    for idx in cell_types[multiple_matches].index:
        matched_types = cell_type_matches.columns[cell_type_matches.loc[idx] == 1]
        for cell_type in priority_order:
            if cell_type in matched_types:
                cell_types[idx] = cell_type
                break
    
    # Store results
    table.obs['cell_type'] = cell_types
    table.obs['n_signature_matches'] = n_matches
    
    # Store signatures in uns for reproducibility
    table.uns['cell_type_signatures'] = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'signatures': signatures,
        'unlabelled_label': unlabelled_label
    }
    
    # Summary
    print("\n=== Cell Type Assignment Summary ===")
    type_counts = cell_types.value_counts()
    for cell_type, count in type_counts.items():
        pct = count / len(cell_types) * 100
        desc = signatures.get(cell_type, {}).get('description', '')
        desc_str = f" - {desc}" if desc else ""
        print(f"{cell_type}: {count} ({pct:.1f}%){desc_str}")
    
    return cell_types


# ============================================================================
# SECTION 3: VISUALISATION
# ============================================================================


def plot_cell_types_spatial(sdata,
                            table_key: str = 'cell_intensity',
                            Sample: str = 'Sample_1',
                            shapes_key: str = 'cell_shapes',
                            save_path: Optional[str] = None,
                            visualize: bool = False,
                            figsize: tuple = (12, 10),
                            alpha: float = 0.8,
                            outline_width: float = 0.5,
                            show_legend: bool = True):
    """
    Plot cell types using matplotlib polygon rendering (reliable, no display issues).

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with cell type annotations
    table_key : str, default='cell_intensity'
        Key for the table in sdata.tables
    Sample : str, default='Sample_1'
        Sample identifier for title
    shapes_key : str, default='cell_shapes'
        Key for the shapes in sdata.shapes
    save_path : Optional[str], default=None
        Path to save the figure
    visualize : bool, default=False
        Whether to display the plot
    figsize : tuple, default=(12, 10)
        Figure size
    alpha : float, default=0.8
        Transparency of cell fills
    outline_width : float, default=0.5
        Width of cell outlines
    show_legend : bool, default=True
        Whether to show legend
    """
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt
    from pathlib import Path

    table = sdata.tables[table_key]
    shapes = sdata.shapes[shapes_key]

    print("="*70)
    print("PLOTTING CELL TYPES (SPATIAL) - MATPLOTLIB")
    print("="*70)
    print(f"Table: {len(table)} cells")
    print(f"Shapes: {len(shapes)} polygons")

    # Setup region annotation
    table.obs['region'] = shapes_key
    sdata.set_table_annotates_spatialelement(table_key, region=shapes_key)

    # Get cell types - align by index
    if 'cell_type' not in table.obs.columns:
        raise ValueError("Column 'cell_type' not found in table.obs")

    # Verify alignment
    if len(table) != len(shapes):
        print(f"WARNING: Mismatch between table ({len(table)}) and shapes ({len(shapes)})")
        print("Attempting to align by index...")

    # Try to align by shared index
    common_idx = table.obs.index.intersection(shapes.index)
    if len(common_idx) == 0:
        print("No common indices - using positional alignment")
        cell_types = table.obs['cell_type'].values
    else:
        print(f"Found {len(common_idx)} common indices")
        cell_types = table.obs.loc[shapes.index, 'cell_type'].values

    # Get cell type counts
    type_counts = pd.Series(cell_types).value_counts()
    print(f"\nCell type distribution:")
    for ct, count in type_counts.items():
        print(f"  {ct:35s}: {count:6d} ({count/len(cell_types)*100:5.2f}%)")

    # Create color mapping
    unique_types = list(type_counts.index)
    n_types = len(unique_types)

    if n_types <= 10:
        colors = sns.color_palette('tab10', n_types)
    elif n_types <= 20:
        colors = sns.color_palette('tab20', n_types)
    else:
        colors = sns.color_palette('husl', n_types)

    type_to_color = {ct: colors[i] for i, ct in enumerate(unique_types)}

    # Add gray for unlabelled if present
    if 'unlabelled' in type_counts.index:
        type_to_color['unlabelled'] = (0.85, 0.85, 0.85)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    print("\nRendering polygons...")

    # Plot each cell as a polygon
    patches = []
    patch_colors = []

    n_plotted = 0
    for idx, (shape_idx, row) in enumerate(shapes.iterrows()):
        if idx >= len(cell_types):
            break

        ct = cell_types[idx]
        geom = row.geometry

        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            poly = Polygon(coords, closed=True)
            patches.append(poly)
            patch_colors.append(type_to_color.get(ct, (0.5, 0.5, 0.5)))
            n_plotted += 1
        elif geom.geom_type == 'MultiPolygon':
            for sub_poly in geom.geoms:
                coords = np.array(sub_poly.exterior.coords)
                poly = Polygon(coords, closed=True)
                patches.append(poly)
                patch_colors.append(type_to_color.get(ct, (0.5, 0.5, 0.5)))
                n_plotted += 1

    print(f"Plotted {n_plotted} polygons")

    # Add patches to axis
    collection = PatchCollection(patches, 
                                facecolors=patch_colors,
                                edgecolors='black',
                                linewidths=outline_width,
                                alpha=alpha)
    ax.add_collection(collection)

    # Set axis properties
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)

    # Title
    title = f'{Sample}, Cell Types\n{len(table)} cells, {n_types} types'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add legend
    if show_legend:
        from matplotlib.patches import Patch

        # Sort by count
        sorted_types = sorted(unique_types, 
                             key=lambda x: type_counts.get(x, 0), 
                             reverse=True)

        legend_elements = []
        for ct in sorted_types:
            count = type_counts.get(ct, 0)
            pct = count / len(cell_types) * 100
            label = f"{ct} (n={count}, {pct:.1f}%)"
            legend_elements.append(
                Patch(facecolor=type_to_color[ct], 
                     edgecolor='black', 
                     label=label)
            )

        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(1.02, 1),
                 loc='upper left',
                 frameon=True,
                 fancybox=True,
                 shadow=True,
                 fontsize=9,
                 title='Cell Type',
                 title_fontsize=11)

    # Grid for easier reading
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save or display
    if save_path:
        save_plot_pdf(fig, Path(save_path), visualize=visualize)
    else:
        if visualize:
            plt.show()
        else:
            plt.close(fig)

    print("\nPlotting complete!")

    return fig, ax




def plot_cell_type_composition(sdata,
                               table_key: str = 'cell_intensity',
                               Sample: str = 'Sample_1',
                               save_path: Optional[str] = None,
    visualize: bool = False):
    """
    Plot cell type composition.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with cell type annotations
    table_key : str, default='cell_intensity'
        Key for the table in sdata.tables
    save_path : Optional[str], default=None
        Path to save the figure
    """
    table = sdata.tables[table_key]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    type_counts = table.obs['cell_type'].value_counts()
    colors = sns.color_palette('tab20', len(type_counts))
    
    bars = axes[0].bar(range(len(type_counts)), type_counts.values, color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(type_counts)))
    axes[0].set_xticklabels(type_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Number of Cells')
    axes[0].set_title(f'{Sample}, Cell Type Abundance')
    
    for bar, count in zip(bars, type_counts.values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(table)*100:.1f}%)',
                    ha='center', va='bottom')
    
    # Pie chart
    wedges, texts, autotexts = axes[1].pie(type_counts.values, labels=type_counts.index,
                                           colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'{Sample}, Cell Type Proportions')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.tight_layout()

    if save_path:
        from pathlib import Path
        fig = plt.gcf()  # Get current figure
        save_plot_pdf(fig, Path(save_path), visualize=visualize)
    else:
        if visualize:
            plt.show()
        else:
            plt.close()


# ============================================================================
# SECTION 4: COMPLETE PIPELINE
# ============================================================================


def run_complete_phenotyping_and_labeling(sdata,
                                          signatures: Dict[str, Dict],
                                          Sample: str = 'Sample_1',
                                          table_key: str = 'cell_intensity',
                                          marker_names: Optional[List[str]] = None,
                                          consensus_method: str = 'consensus',
                                          validate_signatures: bool = True,
                                          save_dir: Optional[str] = None) -> pd.Series:
    """
    Complete end-to-end pipeline for cell phenotyping and labeling.
    
    This function performs the following steps:
    1. Marker thresholding using Otsu, Triangle, and Multi-Otsu methods
    2. Consensus call generation from multiple thresholding methods
    3. Cell type assignment based on provided signatures
    4. Visualisation of results
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing cell intensity data
    signatures : Dict[str, Dict]
        Cell type signatures defining marker combinations for each cell type.
        Each signature should contain:
        - 'positive': List[str] - Markers that must be positive
        - 'negative': List[str] - Markers that must be negative (optional)
        - 'description': str - Description of cell type (optional)
    table_key : str, default='cell_intensity'
        Key for the table in sdata.tables
    marker_names : Optional[List[str]], default=None
        List of marker names to process. If None, uses all markers in table
    consensus_method : str, default='consensus'
        Method to use for cell typing ('consensus', 'otsu', or 'triangle')
    validate_signatures : bool, default=True
        Whether to validate signatures against available markers
    save_dir : Optional[str], default=None
        Directory to save output figures
        
    Returns
    -------
    pd.Series
        Series of cell type assignments for each cell
        
    Examples
    --------
    >>> # Define your cell type signatures
    >>> my_signatures = {
    ...     'Stromal_Actin+': {
    ...         'positive': ['Actin'],
    ...         'negative': ['CD45', 'PD-L1'],
    ...         'description': 'Stromal fibroblasts or myofibroblasts'
    ...     },
    ...     'Immune_CD45+': {
    ...         'positive': ['CD45'],
    ...         'negative': [],
    ...         'description': 'Pan-immune cells'
    ...     }
    ... }
    >>> 
    >>> # Run the pipeline
    >>> cell_types = run_complete_phenotyping_and_labeling(
    ...     sdata,
    ...     signatures=my_signatures,
    ...     marker_names=["Actin", "Tubulin", "CD45", "PD-L1"],
    ...     save_dir='./phenotyping_output'
    ... )
    """
    print("="*70)
    print("COMPLETE PHENOTYPING AND LABELING PIPELINE")
    print("="*70)
    print(f"\nCell type signatures provided: {len(signatures)}")
    for cell_type, sig in signatures.items():
        pos = sig.get('positive', [])
        neg = sig.get('negative', [])
        print(f"  {cell_type}: +{pos}, -{neg}")
    
    # Create output directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    table = sdata.tables[table_key]
    
    # Step 1: Phenotyping
    if not any('_pos_otsu' in col for col in table.obs.columns):
        print("\nSTEP 1: MARKER THRESHOLDING")
        print("="*70)
        thresholds, stats = phenotype_markers_otsu_triangle(sdata, table_key, marker_names)
        clean_names = list(stats.keys())
    else:
        print("\nPhenotyping already complete, skipping Step 1")
        clean_names = list(set([col.replace('_pos_otsu', '') 
                               for col in table.obs.columns if '_pos_otsu' in col]))
    
    # Step 2: Consensus
    print("\nSTEP 2: CONSENSUS CALLS")
    print("="*70)
    create_consensus_calls(sdata, table_key, clean_names, agreement_threshold=1.0)
    
    # Step 3: Cell typing with provided signatures
    print("\nSTEP 3: CELL TYPE ASSIGNMENT")
    print("="*70)
    
    cell_types = assign_cell_types_combinatorial(
        sdata, 
        signatures=signatures,
        table_key=table_key, 
        method=consensus_method,
        validate_signatures=validate_signatures
    )
    
    # Step 4: Visualise
    if (cell_types != 'unlabelled').sum() > 0:
        print("\nSTEP 4: VISUALISATION")
        print("="*70)
        
        save_path = os.path.join(save_dir, f'{Sample}_cell_types_spatial.pdf') if save_dir else None
        plot_cell_types_spatial(sdata, 
                                table_key, 
                                Sample, 
                                save_path=save_path,
                                visualize=False)
        
        save_path = os.path.join(save_dir, f'{Sample}_cell_type_composition.pdf') if save_dir else None
        plot_cell_type_composition(sdata, table_key, Sample, save_path)
    else:
        print("\nWARNING:  All cells unlabelled - review signatures")
    
    return cell_types


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def create_example_signatures() -> Dict[str, Dict]:
    """
    Create example cell type signatures for common tissue types.
    
    This is a template function - modify it for your specific experiment.
    
    Returns
    -------
    Dict[str, Dict]
        Example cell type signatures
    """
    # Example 1: Tumor microenvironment
    tumor_signatures = {
        'Stromal_Actin+': {
            'positive': ['Actin'],
            'negative': ['CD45', 'PD-L1'],
            'description': 'Stromal fibroblasts or myofibroblasts'
        },
        'Tumor_PD-L1+': {
            'positive': ['PD-L1'],
            'negative': ['CD45'],
            'description': 'PD-L1+ tumor cells'
        }
    }
    
    return tumor_signatures


# Example usage:
"""
# 1. Define your cell type signatures
my_signatures = {
    'T_cell': {
        'positive': ['CD3', 'CD45'],
        'negative': ['CD19'],
        'description': 'T lymphocytes'
    },
    'B_cell': {
        'positive': ['CD19', 'CD45'],
        'negative': ['CD3'],
        'description': 'B lymphocytes'
    },
    'Macrophage': {
        'positive': ['CD68', 'CD45'],
        'negative': ['CD3', 'CD19'],
        'description': 'Macrophages'
    }
}

# 2. Run the complete pipeline with your signatures
cell_types = run_complete_phenotyping_and_labeling(
    sdata,
    signatures=my_signatures,
    marker_names=["CD3", "CD19", "CD68", "CD45"],
    consensus_method='consensus',
    save_dir='./phenotyping_output'
)

# 3. Or use the example signatures
example_sigs = create_example_signatures()
cell_types = run_complete_phenotyping_and_labeling(
    sdata,
    signatures=example_sigs,
    marker_names=["Actin", "Tubulin", "CD45", "PD-L1"],
    save_dir='./phenotyping_output'
)
"""

"""
UMAP Visualisation for Cell Type Analysis
Following scverse standards and SpatialData conventions
"""

"""
UMAP Visualisation for Cell Type Analysis
Following scverse standards and SpatialData conventions
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from spatialdata import SpatialData
from typing import Optional, List, Literal
from pathlib import Path
import argparse
import os


def exclude_markers_from_analysis(
    sdata: SpatialData,
    table_key: str = 'cell_intensity',
    exclude_markers: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> SpatialData:
    """
    Flag markers to exclude from UMAP analysis.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    table_key : str
        Name of the table containing intensity data
    exclude_markers : List[str], optional
        Specific marker names to exclude
    exclude_patterns : List[str], optional
        Patterns to match for exclusion (e.g., ['DAPI', 'EMPTY'])

    Returns
    -------
    SpatialData
        Modified SpatialData with 'use_for_analysis' flag in table.var
    """
    adata = sdata.tables[table_key]
    all_markers = adata.var_names.tolist()
    n_total = len(all_markers)

    # Initialize all markers as True
    use_for_analysis = np.ones(n_total, dtype=bool)

    print("=" * 70)
    print("MARKER EXCLUSION FOR UMAP")
    print("=" * 70)
    print(f"Total markers: {n_total}")

    # Exclude specific markers
    if exclude_markers is not None:
        print(f"Excluding {len(exclude_markers)} specified markers:")
        for marker in exclude_markers:
            if marker in all_markers:
                idx = all_markers.index(marker)
                use_for_analysis[idx] = False
                print(f"  - {marker}")
            else:
                print(f"Warning: {marker} not found")

    # Exclude by pattern
    if exclude_patterns is not None:
        print(f"Excluding by patterns: {exclude_patterns}")
        for pattern in exclude_patterns:
            pattern_lower = pattern.lower()
            for i, marker in enumerate(all_markers):
                if pattern_lower in marker.lower():
                    if use_for_analysis[i]:  # Only print if not already excluded
                        print(f"  - {marker}")
                    use_for_analysis[i] = False

    n_included = use_for_analysis.sum()
    n_excluded = n_total - n_included

    print()
    print(f"Analysis markers: {n_included}")
    print(f"Excluded markers: {n_excluded}")
    print("=" * 70)

    # Store in var
    adata.var['use_for_analysis'] = use_for_analysis

    return sdata


def compute_umap_with_marker_selection(
    sdata: SpatialData,
    table_key: str = 'cell_intensity',
    use_layer: Optional[str] = None,
    n_neighbors: int = 15,
    n_pcs: int = 20,
    random_state: int = 42,
    use_only_analysis_markers: bool = True,
    **kwargs
) -> SpatialData:
    """
    Compute UMAP embedding with optional marker exclusion.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    table_key : str
        Name of intensity table
    use_layer : str, optional
        Layer to use for computation. If None, uses adata.X (default)
    n_neighbors : int
        Number of neighbors for UMAP
    n_pcs : int
        Number of PCs to compute
    random_state : int
        Random seed for reproducibility
    use_only_analysis_markers : bool
        If True, uses only markers where var['use_for_analysis'] == True

    Returns
    -------
    SpatialData
        Modified SpatialData with UMAP in table.obsm['X_umap']
    """
    adata = sdata.tables[table_key]
    n_vars = adata.n_vars

    # Check if we should filter markers
    if use_only_analysis_markers and 'use_for_analysis' in adata.var:
        analysis_mask = adata.var['use_for_analysis'].values
        n_analysis_markers = analysis_mask.sum()

        if n_analysis_markers < n_vars:
            print(f"Using {n_analysis_markers} analysis markers (excluding {n_vars - n_analysis_markers})")
            adata_subset = adata[:, analysis_mask].copy()
        else:
            adata_subset = adata
    else:
        adata_subset = adata

    n_vars_used = adata_subset.n_vars

    print("=" * 70)
    print(f"Computing UMAP")
    print("=" * 70)
    print(f"Cells: {adata_subset.n_obs}, Markers: {n_vars_used}")

    # Adjust n_pcs if needed
    n_pcs_adjusted = min(n_pcs, n_vars_used - 1, adata_subset.n_obs - 1)

    print(f"Computing PCA (n_pcs={n_pcs_adjusted})...")
    sc.tl.pca(
        adata_subset,
        n_comps=n_pcs_adjusted,
        layer=use_layer,
        random_state=random_state,
        svd_solver='arpack'
    )

    var_explained = adata_subset.uns['pca']['variance_ratio']
    cumsum_var = np.cumsum(var_explained)
    print(f"  PC1-{n_pcs_adjusted} explain {cumsum_var[-1]*100:.1f}% of variance")

    # Compute neighbors
    n_pcs_for_neighbors = min(n_pcs_adjusted, 15)
    sc.pp.neighbors(
        adata_subset,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_for_neighbors,
        random_state=random_state
    )

    # Compute UMAP
    print("Computing UMAP")
    sc.tl.umap(adata_subset, random_state=random_state, **kwargs)

    # Copy back to original adata
    adata.obsm['X_umap'] = adata_subset.obsm['X_umap']
    if 'X_pca' in adata_subset.obsm:
        adata.obsm['X_pca'] = adata_subset.obsm['X_pca']
    adata.uns['pca'] = adata_subset.uns['pca']
    adata.uns['neighbors'] = adata_subset.uns.get('neighbors', {})
    adata.uns['umap'] = adata_subset.uns.get('umap', {})

    print("UMAP computed")
    print("=" * 70)

    return sdata


def plot_marker_expression_on_umap(
    sdata: SpatialData,
    table_key: str = 'cell_intensity',
    embedding: str = 'umap',
    markers: Optional[List[str]] = None,
    n_markers: Optional[int] = None,
    include_excluded_markers: bool = False,
    use_layer: Optional[str] = None,
    save_path: Optional[str] = None,
    ncols: int = 4,
    figsize_per_panel: tuple = (4, 4)
):
    """
    Plot marker expression patterns on UMAP.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    table_key : str
        Name of intensity table
    embedding : str
        Embedding to use ('umap', 'tsne', 'pca')
    markers : List[str], optional
        Specific markers to plot. If None, plots all analysis markers
    n_markers : int, optional
        Limit number of markers to plot
    include_excluded_markers : bool
        Whether to include markers flagged for exclusion
    use_layer : str, optional
        Layer to use for intensity values. If None, uses adata.X (default)
    save_path : str, optional
        Path to save figure
    ncols : int
        Number of columns in grid
    figsize_per_panel : tuple
        Size of each subplot
    """
    adata = sdata.tables[table_key]
    embedding_key = f'X_{embedding}'

    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found. Run compute_umap_with_marker_selection() first.")

    # Determine which markers to plot
    if markers is None:
        if 'use_for_analysis' in adata.var and not include_excluded_markers:
            analysis_mask = adata.var['use_for_analysis'].values
            markers = adata.var_names[analysis_mask].tolist()
            excluded_markers = adata.var_names[~analysis_mask].tolist()
            if excluded_markers:
                print(f"Excluding markers: {', '.join(excluded_markers)}")
        else:
            markers = adata.var_names.tolist()

    # Apply marker limit
    if n_markers is not None:
        markers = markers[:n_markers]

    n_markers_actual = len(markers)
    if n_markers_actual == 0:
        print("No markers to plot")
        return

    print(f"Plotting {n_markers_actual} markers on {embedding.upper()}...")

    # Create grid
    nrows = int(np.ceil(n_markers_actual / ncols))
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    # Get data
    if use_layer and use_layer in adata.layers:
        X_data = adata.layers[use_layer]
    else:
        X_data = adata.X

    if hasattr(X_data, 'toarray'):
        X_data = X_data.toarray()

    # Plot each marker
    for i, marker in enumerate(markers):
        ax = axes[i]
        try:
            marker_idx = list(adata.var_names).index(marker)
            marker_data = X_data[:, marker_idx]

            # Use scanpy's embedding plot
            sc.pl.embedding(
                adata,
                basis=embedding,
                color=marker,
                ax=ax,
                show=False,
                frameon=False,
                title=marker,
                cmap='viridis',
                layer=use_layer,
                vmin=np.percentile(marker_data, 1),
                vmax=np.percentile(marker_data, 99)
            )
        except Exception as e:
            print(f"Warning: Could not plot marker '{marker}': {e}")
            ax.text(0.5, 0.5, f"Error plotting\n{marker}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    # Hide unused subplots
    for i in range(n_markers_actual, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        from pathlib import Path
        save_plot_pdf(fig, Path(save_path), visualize=False)
    else:
        plt.close(fig)


def plot_cell_types_on_umap(
    sdata: SpatialData,
    table_key: str = 'cell_intensity',
    embedding: str = 'umap',
    cell_type_col: str = 'cell_type',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    palette: Optional[str] = None
):
    """
    Plot cell type annotations on UMAP.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    table_key : str
        Name of intensity table
    embedding : str
        Embedding to use ('umap', 'tsne', 'pca')
    cell_type_col : str
        Column name in obs containing cell type labels
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    palette : str, optional
        Color palette (e.g., 'tab20', 'Set3')
    """
    adata = sdata.tables[table_key]
    embedding_key = f'X_{embedding}'

    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found. Run compute_umap_with_marker_selection() first.")

    if cell_type_col not in adata.obs:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in table.obs")

    print(f"Plotting cell types on {embedding.upper()}...")

    # Get cell type counts
    cell_type_counts = adata.obs[cell_type_col].value_counts()
    n_types = len(cell_type_counts)

    print(f"Found {n_types} cell types:")
    for cell_type, count in cell_type_counts.items():
        pct = count / len(adata) * 100
        print(f"  {cell_type}: {count} ({pct:.1f}%)")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Set color palette
    if palette is None:
        if n_types <= 10:
            palette = 'tab10'
        elif n_types <= 20:
            palette = 'tab20'
        else:
            palette = 'husl'

    # Plot using scanpy
    sc.pl.embedding(
        adata,
        basis=embedding,
        color=cell_type_col,
        ax=ax,
        show=False,
        frameon=True,
        title=f'Cell Types ({n_types} types)',
        palette=palette,
        legend_loc='right margin',
        legend_fontsize=10
    )

    plt.tight_layout()

    if save_path:
        from pathlib import Path
        save_plot_pdf(fig, Path(save_path), visualize=False)
    else:
        plt.close(fig)


def run_complete_umap_cell_type_workflow(
    sdata: SpatialData,
    table_key: str = 'cell_intensity',
    exclude_markers: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    cell_type_col: str = 'cell_type',
    n_neighbors: int = 15,
    n_pcs: int = 20,
    random_state: int = 42,
    use_layer: Optional[str] = None,
    plot_markers: bool = True,
    n_markers_to_plot: Optional[int] = None,
    output_dir: str = './umap_plots',
    ncols: int = 4
) -> SpatialData:
    """
    Complete workflow: UMAP computation and visualisation of markers and cell types.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with cell_type annotations
    table_key : str
        Name of intensity table
    exclude_markers : List[str], optional
        Specific markers to exclude from UMAP (e.g., ['DAPI_INIT', 'DAPI_FINAL'])
    exclude_patterns : List[str], optional
        Patterns to exclude (e.g., ['DAPI', 'EMPTY'])
    cell_type_col : str
        Column name with cell type labels
    n_neighbors : int
        UMAP parameter
    n_pcs : int
        Number of PCs for UMAP
    random_state : int
        Random seed
    use_layer : str, optional
        Layer to use. If None, uses adata.X (default)
    plot_markers : bool
        Whether to plot individual marker expression
    n_markers_to_plot : int, optional
        Limit number of markers to plot
    output_dir : str
        Directory for output figures
    ncols : int
        Number of columns for marker grid

    Returns
    -------
    SpatialData
        Modified SpatialData with UMAP embedding
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPLETE UMAP + CELL TYPE VISUALISATION WORKFLOW")
    print("=" * 70)

    # Step 1: Exclude markers if requested
    if exclude_markers or exclude_patterns:
        print("\nStep 1: Excluding markers")
        sdata = exclude_markers_from_analysis(
            sdata,
            table_key=table_key,
            exclude_markers=exclude_markers,
            exclude_patterns=exclude_patterns
        )
    else:
        print("\nStep 1: No markers excluded")

    # Step 2: Compute UMAP
    print("\nStep 2: Computing UMAP")
    sdata = compute_umap_with_marker_selection(
        sdata,
        table_key=table_key,
        use_layer=use_layer,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=random_state,
        use_only_analysis_markers=True
    )

    # Step 3: Plot marker expression
    if plot_markers:
        print("\nStep 3: Plotting marker expression on UMAP")
        try:
            plot_marker_expression_on_umap(
                sdata,
                table_key=table_key,
                embedding='umap',
                n_markers=n_markers_to_plot,
                include_excluded_markers=False,
                use_layer=use_layer,
                save_path=str(output_dir / 'marker_expression_umap.pdf'),
                ncols=ncols
            )
        except Exception as e:
            print(f"Marker expression plotting failed: {e}")

    # Step 4: Plot cell types
    print("\nStep 4: Plotting cell types on UMAP...")
    try:
        plot_cell_types_on_umap(
            sdata,
            table_key=table_key,
            embedding='umap',
            cell_type_col=cell_type_col,
            save_path=str(output_dir / 'cell_types_umap.pdf')
        )
    except Exception as e:
        print(f"Cell type plotting failed: {e}")

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print("Files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")
    print("=" * 70)

    return sdata


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Basic usage - uses adata.X by default
sdata = run_complete_umap_cell_type_workflow(
    sdata,
    table_key='cell_intensity',
    exclude_patterns=['DAPI', 'EMPTY'],  # Exclude DAPI and empty channels
    cell_type_col='cell_type',
    output_dir='./umap_analysis'
)

# Example 2: Exclude specific markers
sdata = run_complete_umap_cell_type_workflow(
    sdata,
    table_key='cell_intensity',
    exclude_markers=['DAPI_INIT', 'DAPI_FINAL', 'FITC_EMPTY'],
    n_markers_to_plot=16,  # Only plot first 16 markers
    output_dir='./umap_analysis'
)

# Example 3: Use a specific layer if you have transformed data
sdata = run_complete_umap_cell_type_workflow(
    sdata,
    table_key='cell_intensity',
    use_layer='normalized',  # Use a specific layer
    exclude_patterns=['DAPI', 'EMPTY'],
    output_dir='./umap_analysis'
)

# Example 4: Step-by-step control
# Exclude markers
sdata = exclude_markers_from_analysis(
    sdata,
    table_key='cell_intensity',
    exclude_patterns=['DAPI', 'EMPTY']
)

# Compute UMAP (uses adata.X by default)
sdata = compute_umap_with_marker_selection(
    sdata,
    table_key='cell_intensity',
    n_neighbors=30,
    n_pcs=30
)

# Plot markers
plot_marker_expression_on_umap(
    sdata,
    table_key='cell_intensity',
    save_path='./marker_umap.pdf'
)

# Plot cell types
plot_cell_types_on_umap(
    sdata,
    table_key='cell_intensity',
    save_path='./cell_types_umap.pdf'
)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

# ============================================================================
# GMM-BASED PROBABILISTIC PHENOTYPING
# ============================================================================

def fit_gmm_per_marker(
    sdata,
    table_key: str = 'table',
    marker_names: Optional[List[str]] = None,
    use_layer: str = 'asinh',
    n_components: int = 2,
    use_bayesian: bool = True,
    min_cells: int = 100,
    min_separation: float = 1.5,  # NEW PARAMETER
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Fit Gaussian Mixture Models for each marker to get probabilistic positivity calls
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with cell intensity data
    table_key : str, default='table'
        Key for table in sdata.tables
    marker_names : Optional[List[str]], default=None
        Markers to process. If None, uses all markers.
    use_layer : str, default='asinh'
        Layer to use for intensities
    n_components : int, default=2
        Number of GMM components (2 = negative/positive)
    use_bayesian : bool, default=True
        Use BayesianGaussianMixture (better for uncertainty)
    min_cells : int, default=100
        Minimum cells needed for reliable GMM fitting
    min_separation : float, default=1.5
        Minimum Cohen's d separation between components.
        Below this, GMM is unreliable - use fallback thresholding.
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    Dict[str, Dict]
        GMM models and metadata per marker
    """
    
    table = sdata.tables[table_key]
    
    # Get intensity matrix
    if use_layer in table.layers:
        X = table.layers[use_layer]
    else:
        X = table.X
    
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Auto-detect marker names
    marker_mapping = {}
    for var_name in table.var_names:
        short_name = var_name
        for prefix in ['cyto_', 'nuc_', 'membrane_']:
            if short_name.startswith(prefix):
                short_name = short_name[len(prefix):]
        for suffix in ['_mean', '_intensity', '_sum']:
            if short_name.endswith(suffix):
                short_name = short_name[:-len(suffix)]
        marker_mapping[short_name] = var_name
    
    if marker_names is None:
        marker_names = list(marker_mapping.keys())
    
    print("="*70)
    print("PROBABILISTIC MARKER MODELING (GMM)")
    print("="*70)
    print(f"Fitting {n_components}-component GMM for {len(marker_names)} markers")
    print(f"Method: {'Bayesian' if use_bayesian else 'Standard'} GMM")
    print(f"Min separation threshold: {min_separation}")
    print(f"Total cells: {len(table)}\n")
    
    gmm_results = {}
    
    for marker in marker_names:
        if marker not in marker_mapping:
            warnings.warn(f"Marker '{marker}' not found, skipping")
            continue
        
        full_name = marker_mapping[marker]
        marker_idx = list(table.var_names).index(full_name)
        intensities = X[:, marker_idx].reshape(-1, 1)
        
        # Remove zeros and extreme outliers for better fitting
        valid_mask = (intensities.flatten() > 0) & (intensities.flatten() < np.percentile(intensities, 99.5))
        clean_intensities = intensities[valid_mask]
        
        if len(clean_intensities) < min_cells:
            warnings.warn(f"{marker}: Only {len(clean_intensities)} valid cells, skipping GMM")
            continue
        
        print(f"Processing {marker} ({full_name})...")
        print(f"  Valid cells: {len(clean_intensities)} / {len(intensities)}")
        
        # Fit GMM
        if use_bayesian:
            gmm = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type='full',
                weight_concentration_prior=0.1,
                max_iter=200,
                n_init=5,
                random_state=random_state
            )
        else:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                max_iter=200,
                n_init=10,
                random_state=random_state
            )
        
        gmm.fit(clean_intensities)
        
        # Identify negative vs positive components by mean
        means = gmm.means_.flatten()
        component_order = np.argsort(means)
        negative_idx = component_order[0]
        positive_idx = component_order[-1]
        
        negative_mean = means[negative_idx]
        positive_mean = means[positive_idx]
        
        # Calculate separation quality FIRST (before threshold calculation)
        pooled_std = np.sqrt(np.mean([gmm.covariances_[i, 0, 0] for i in range(n_components)]))
        separation = (positive_mean - negative_mean) / pooled_std
        
        # ========================================================================
        # CORRECTED THRESHOLD CALCULATION
        # ========================================================================
        
        if separation < min_separation:
            # Poor separation - use conservative fallback
            print(f" Separation ({separation:.2f}) below minimum ({min_separation})")
            print(f"Using robust fallback threshold")
            
            # Fallback: Use 95th percentile of negative component (2 std above mean)
            neg_std = np.sqrt(gmm.covariances_[negative_idx, 0, 0])
            threshold = negative_mean + 2 * neg_std
            
            # Safety check: use Otsu if available and higher
            try:
                from skimage.filters import threshold_otsu
                otsu_thresh = threshold_otsu(clean_intensities.flatten())
                threshold = max(threshold, otsu_thresh)
                print(f"      Fallback threshold: {threshold:.2f} (Otsu: {otsu_thresh:.2f})")
            except:
                print(f"      Fallback threshold: {threshold:.2f}")
            
            is_reliable = False
            
        else:
            # Good separation - use Bayes optimal threshold
            # This is where P(positive|x) = P(negative|x), i.e., posterior odds = 1
            
            neg_var = gmm.covariances_[negative_idx, 0, 0]
            pos_var = gmm.covariances_[positive_idx, 0, 0]
            neg_weight = gmm.weights_[negative_idx]
            pos_weight = gmm.weights_[positive_idx]
            
            # Solve quadratic equation for intersection of two Gaussians
            # See: https://en.wikipedia.org/wiki/Mixture_model#Optimal_classification
            if abs(neg_var - pos_var) > 1e-6:
                # Unequal variances - use quadratic solution
                a = 1/(2*neg_var) - 1/(2*pos_var)
                b = positive_mean/pos_var - negative_mean/neg_var
                c = (negative_mean**2)/(2*neg_var) - (positive_mean**2)/(2*pos_var) + \
                    np.log((pos_weight * np.sqrt(neg_var)) / (neg_weight * np.sqrt(pos_var)))
                
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    # Two solutions - pick the one between the means
                    x1 = (-b + np.sqrt(discriminant)) / (2*a)
                    x2 = (-b - np.sqrt(discriminant)) / (2*a)
                    
                    candidates = [x for x in [x1, x2] if negative_mean <= x <= positive_mean]
                    threshold = candidates[0] if candidates else (negative_mean + positive_mean) / 2
                else:
                    threshold = (negative_mean + positive_mean) / 2
            else:
                # Equal variances - simple midpoint weighted by priors
                threshold = (negative_mean + positive_mean) / 2 + \
                           neg_var * np.log(pos_weight / neg_weight) / (positive_mean - negative_mean)
            
            print(f"  Bayes optimal threshold: {threshold:.2f}")
            is_reliable = True
        
        # ========================================================================
        # QUALITY CHECKS
        # ========================================================================
        
        # Ensure threshold is reasonable
        if threshold < negative_mean:
            print(f"Threshold below negative mean! Adjusting to neg_mean + 1 std")
            threshold = negative_mean + np.sqrt(gmm.covariances_[negative_idx, 0, 0])
            is_reliable = False
        
        if threshold > positive_mean:
            print(f" Threshold above positive mean! Using midpoint")
            threshold = (negative_mean + positive_mean) / 2
            is_reliable = False
        
        # Predict probabilities for ALL cells (including zeros)
        proba_all = gmm.predict_proba(intensities)
        prob_positive = proba_all[:, positive_idx]
        prob_negative = proba_all[:, negative_idx]
        
        # Store results
        table.obs[f'{marker}_intensity'] = intensities.flatten()
        table.obs[f'{marker}_prob_positive'] = prob_positive
        table.obs[f'{marker}_prob_negative'] = prob_negative
        
        # Also store binary calls at 0.5 threshold for compatibility
        table.obs[f'{marker}_gmm_positive'] = prob_positive >= 0.5
        
        # Calculate BIC
        if hasattr(gmm, 'bic'):
            bic_score = gmm.bic(clean_intensities)
        else:
            # For Bayesian GMM, calculate approximate BIC
            bic_score = -2 * gmm.score(clean_intensities) * len(clean_intensities) + \
                        np.log(len(clean_intensities)) * (n_components * 3)
        
        gmm_results[marker] = {
            'model': gmm,
            'negative_idx': negative_idx,
            'positive_idx': positive_idx,
            'negative_mean': negative_mean,
            'positive_mean': positive_mean,
            'threshold': threshold,
            'separation': separation,
            'is_reliable': is_reliable,  # NEW: Flag for reliability
            'converged': gmm.converged_ if hasattr(gmm, 'converged_') else True,
            'n_iter': gmm.n_iter_ if hasattr(gmm, 'n_iter_') else None,
            'bic': bic_score,
            'fraction_positive': (prob_positive >= 0.5).mean(),
            'model_type': 'Bayesian' if use_bayesian else 'Standard'
        }
        
        print(f"  Negative mean: {negative_mean:.2f}")
        print(f"  Positive mean: {positive_mean:.2f}")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Separation: {separation:.2f} {'RELIABLE' if is_reliable else 'UNRELIABLE'}")
        print(f"  Fraction positive (p>0.5): {gmm_results[marker]['fraction_positive']:.1%}")
        print(f"  Converged: {gmm_results[marker]['converged']}")
        print()
    
    # Store metadata
    table.uns['gmm_phenotyping'] = {
        'timestamp': datetime.now().isoformat(),
        'n_components': n_components,
        'use_bayesian': use_bayesian,
        'use_layer': use_layer,
        'min_separation': min_separation,  # NEW
        'markers': list(gmm_results.keys()),
        'model_info': {k: {k2: v2 for k2, v2 in v.items() if k2 != 'model'} 
                       for k, v in gmm_results.items()}
    }
    
    print("GMM fitting complete")
    print(f"Results stored in table.obs['<marker>_prob_positive']")
    
    return gmm_results


def assign_phenotypes_probabilistic(
    sdata,
    signatures: Dict[str, Dict],
    table_key: str = 'table',
    probability_threshold: float = 0.5,
    min_combined_prob: float = 0.1,
    use_product: bool = True,
    priority_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Assign cell phenotypes using probabilistic marker calls from GMM.
    
    Instead of hard AND/OR logic, this uses probability multiplication
    to get graded phenotype assignments [web:68].
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with GMM probabilities
    signatures : Dict[str, Dict]
        Cell type signatures with 'positive' and 'negative' markers
    table_key : str, default='table'
        Table key
    probability_threshold : float, default=0.5
        Threshold for individual marker positivity
    min_combined_prob : float, default=0.1
        Minimum combined probability to assign phenotype
    use_product : bool, default=True
        Use probability product (stricter) vs mean (more lenient)
    priority_order : Optional[List[str]], default=None
        Priority order for resolving conflicts. If None, uses signature order.
        
    Returns
    -------
    pd.DataFrame
        Phenotype assignments with probabilities
    """
    
    table = sdata.tables[table_key]
    
    # Verify GMM probabilities exist
    prob_cols = [col for col in table.obs.columns if col.endswith('_prob_positive')]
    if len(prob_cols) == 0:
        raise ValueError("No GMM probabilities found. Run fit_gmm_per_marker first!")
    
    available_markers = [col.replace('_prob_positive', '') for col in prob_cols]
    
    print("="*70)
    print("PROBABILISTIC PHENOTYPE ASSIGNMENT")
    print("="*70)
    print(f"Method: {'Probability product' if use_product else 'Probability mean'}")
    print(f"Probability threshold: {probability_threshold}")
    print(f"Min combined probability: {min_combined_prob}")
    print(f"Available markers: {len(available_markers)}\n")
    
    n_cells = len(table)
    
    # Initialize tracking
    phenotype_probs = pd.DataFrame(0.0, index=table.obs.index, columns=signatures.keys())
    phenotype_details = {}
    
    # Calculate probability for each phenotype
    for pheno_name, signature in signatures.items():
        pos_markers = signature.get('positive', [])
        neg_markers = signature.get('negative', [])
        
        print(f"Computing {pheno_name}...")
        print(f"  Positive: {pos_markers}")
        print(f"  Negative: {neg_markers}")
        
        # Collect probabilities
        probs_list = []
        missing_markers = []
        
        # Positive markers
        for marker in pos_markers:
            prob_col = f'{marker}_prob_positive'
            if prob_col in table.obs.columns:
                probs_list.append(table.obs[prob_col].values)
            else:
                missing_markers.append(marker)
        
        # Negative markers (use probability of being negative)
        for marker in neg_markers:
            prob_col = f'{marker}_prob_negative'
            if prob_col in table.obs.columns:
                probs_list.append(table.obs[prob_col].values)
            elif f'{marker}_prob_positive' in table.obs.columns:
                # Use 1 - P(positive) as proxy
                probs_list.append(1 - table.obs[f'{marker}_prob_positive'].values)
            else:
                missing_markers.append(marker)
        
        if missing_markers:
            warnings.warn(f"{pheno_name}: Missing markers {missing_markers}")
        
        if len(probs_list) == 0:
            warnings.warn(f"{pheno_name}: No valid markers, skipping")
            continue
        
        # Combine probabilities
        probs_array = np.column_stack(probs_list)
        
        if use_product:
            # Product of probabilities (strict)
            combined_prob = np.prod(probs_array, axis=1)
        else:
            # Geometric mean (less strict)
            combined_prob = np.exp(np.mean(np.log(probs_array + 1e-10), axis=1))
        
        phenotype_probs[pheno_name] = combined_prob
        
        n_assigned = (combined_prob >= min_combined_prob).sum()
        mean_prob = combined_prob[combined_prob >= min_combined_prob].mean() if n_assigned > 0 else 0
        
        print(f"  Assigned: {n_assigned} cells (mean prob: {mean_prob:.3f})")
        
        phenotype_details[pheno_name] = {
            'n_assigned': n_assigned,
            'mean_prob': mean_prob,
            'missing_markers': missing_markers
        }
    
    print()
    
    # Assign cells to best-matching phenotype
    best_phenotype = phenotype_probs.idxmax(axis=1)
    best_probability = phenotype_probs.max(axis=1)
    
    # Apply minimum probability filter
    below_threshold = best_probability < min_combined_prob
    best_phenotype[below_threshold] = 'unlabelled'
    
    # Handle ties (multiple phenotypes with similar probability)
    prob_margin = phenotype_probs.apply(
        lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1] if len(row.nlargest(2)) > 1 else 1.0,
        axis=1
    )
    ambiguous = prob_margin < 0.1  # Flag cells with <0.1 difference
    
    # Store results
    table.obs['phenotype_gmm'] = best_phenotype.values
    table.obs['phenotype_probability'] = best_probability.values
    table.obs['phenotype_ambiguous'] = ambiguous.values
    table.obs['phenotype_margin'] = prob_margin.values
    
    # Store all probabilities
    for pheno in phenotype_probs.columns:
        table.obs[f'prob_{pheno}'] = phenotype_probs[pheno].values
    
    # Store metadata
    table.uns['phenotype_gmm'] = {
        'timestamp': datetime.now().isoformat(),
        'signatures': signatures,
        'probability_threshold': probability_threshold,
        'min_combined_prob': min_combined_prob,
        'use_product': use_product,
        'phenotype_details': phenotype_details
    }
    
    # Summary
    print("="*70)
    print("PHENOTYPE ASSIGNMENT SUMMARY")
    print("="*70)
    
    for pheno in signatures.keys():
        count = (best_phenotype == pheno).sum()
        pct = count / n_cells * 100
        if count > 0:
            mean_prob = best_probability[best_phenotype == pheno].mean()
            n_ambig = ambiguous[best_phenotype == pheno].sum()
            print(f"{pheno:40s}: {count:6d} ({pct:5.2f}%)  "
                  f"[prob: {mean_prob:.3f}, ambiguous: {n_ambig}]")
    
    unlabelled = (best_phenotype == 'unlabelled').sum()
    print(f"{'unlabelled':40s}: {unlabelled:6d} ({unlabelled/n_cells*100:5.2f}%)")
    
    print(f"\nTotal ambiguous assignments: {ambiguous.sum()} ({ambiguous.sum()/n_cells*100:.1f}%)")
    
    return phenotype_probs


def visualize_gmm_phenotyping(
    sdata,
    table_key: str = 'table',
    markers_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    visualize: bool = False
):
    """
    Visualize GMM-based phenotyping results.
    
    Creates diagnostic plots showing GMM fits and phenotype quality.
    """
    
    table = sdata.tables[table_key]
    
    if 'gmm_phenotyping' not in table.uns:
        raise ValueError("No GMM results found. Run fit_gmm_per_marker first!")
    
    gmm_info = table.uns['gmm_phenotyping']['model_info']
    
    if markers_to_plot is None:
        markers_to_plot = list(gmm_info.keys())[:6]  # Limit to 6 for visibility
    
    n_markers = len(markers_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_markers / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_markers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, marker in enumerate(markers_to_plot):
        ax = axes[idx]
        
        if marker not in gmm_info:
            ax.text(0.5, 0.5, f'{marker}\nNo GMM data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        info = gmm_info[marker]
        
        if f'{marker}_intensity' not in table.obs.columns:
            ax.text(0.5, 0.5, f'{marker}\nIntensity data not found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
            
        intensities = table.obs[f'{marker}_intensity'].values
        probs = table.obs[f'{marker}_prob_positive'].values
        
        # Plot histogram
        valid_intensities = intensities[intensities > 0]
        if len(valid_intensities) > 0:
            ax.hist(valid_intensities, bins=50, alpha=0.5, 
                   color='gray', density=True, label='Data')
            
            # Plot threshold
            ax.axvline(info['threshold'], color='red', linestyle='--', 
                      linewidth=2, label=f"Threshold: {info['threshold']:.2f}")
            
            # Plot component means
            ax.axvline(info['negative_mean'], color='blue', linestyle=':', 
                      alpha=0.7, label=f"Neg: {info['negative_mean']:.2f}")
            ax.axvline(info['positive_mean'], color='green', linestyle=':', 
                      alpha=0.7, label=f"Pos: {info['positive_mean']:.2f}")
            
            # Color by probability
            ax2 = ax.twinx()
            scatter = ax2.scatter(intensities, probs, c=probs, cmap='RdYlGn',
                                s=0.5, alpha=0.3, vmin=0, vmax=1)
            ax2.set_ylabel('P(positive)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim(0, 1)
            
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Density')
            ax.set_title(f"{marker}\nSep: {info['separation']:.2f}, "
                        f"{info['fraction_positive']:.1%} pos")
            ax.legend(loc='upper left', fontsize=8)
            ax.set_xlim(0, np.percentile(valid_intensities, 99))
    
    # Hide empty subplots
    for idx in range(n_markers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        save_path_obj = Path(save_path)
        save_path_fits = save_path_obj.parent / f"{save_path_obj.stem}_gmm_fits.pdf"
        save_plot_pdf(plt.gcf(), save_path_fits, visualize=False)
    else:
        plt.close()
    
    # Second figure: Phenotype probabilities
    
    # Second figure: Phenotype probabilities
    if 'phenotype_gmm' in table.obs.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Phenotype counts with probability
        ax = axes[0, 0]
        pheno_counts = table.obs['phenotype_gmm'].value_counts()
        colors = plt.cm.tab20(np.arange(len(pheno_counts)))
        bars = ax.barh(range(len(pheno_counts)), pheno_counts.values, color=colors)
        ax.set_yticks(range(len(pheno_counts)))
        ax.set_yticklabels(pheno_counts.index, fontsize=9)
        ax.set_xlabel('Number of Cells')
        ax.set_title('Phenotype Assignments')
        
        # Add probability annotations
        for i, (pheno, count) in enumerate(pheno_counts.items()):
            if pheno != 'unlabelled' and count > 0:
                mean_prob = table.obs[table.obs['phenotype_gmm'] == pheno]['phenotype_probability'].mean()
                ax.text(count, i, f' {count} (p={mean_prob:.2f})', 
                       va='center', fontsize=8)
        
        # Panel 2: Probability distribution
        ax = axes[0, 1]
        for pheno in pheno_counts.index[:10]:  # Top 10
            if pheno != 'unlabelled':
                probs = table.obs[table.obs['phenotype_gmm'] == pheno]['phenotype_probability']
                if len(probs) > 0:
                    ax.hist(probs, bins=20, alpha=0.5, label=pheno)
        ax.set_xlabel('Assignment Probability')
        ax.set_ylabel('Number of Cells')
        ax.set_title('Probability Distributions by Phenotype')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
        
        # Panel 3: Ambiguous assignments
        ax = axes[1, 0]
        ambig_counts = table.obs.groupby('phenotype_gmm')['phenotype_ambiguous'].sum().sort_values(ascending=False)
        if len(ambig_counts) > 0:
            ax.barh(range(len(ambig_counts)), ambig_counts.values)
            ax.set_yticks(range(len(ambig_counts)))
            ax.set_yticklabels(ambig_counts.index, fontsize=9)
            ax.set_xlabel('Number of Ambiguous Cells')
            ax.set_title('Ambiguous Assignments (margin < 0.1)')
        
        # Panel 4: UMAP if available
        ax = axes[1, 1]
        if 'X_umap' in table.obsm:
            umap = table.obsm['X_umap']
            phenotypes = table.obs['phenotype_gmm']
            
            for idx, pheno in enumerate(phenotypes.unique()[:10]):
                mask = phenotypes == pheno
                ax.scatter(umap[mask, 0], umap[mask, 1],
                          label=pheno, s=1, alpha=0.6, c=[colors[idx % len(colors)]])
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('GMM Phenotypes (UMAP)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5, fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Run UMAP first\n(compute_unsupervised_clustering)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            from pathlib import Path
            save_path_obj = Path(save_path)
            save_path_pheno = save_path_obj.parent / f"{save_path_obj.stem}_phenotypes.pdf"
            save_plot_pdf(plt.gcf(), save_path_pheno, visualize=False)
        else:
            plt.close()
    



def run_gmm_phenotyping_workflow(
    sdata,
    signatures: Dict[str, Dict],
    table_key: str = 'table',
    marker_names: List[str] = None,
    use_layer: str = 'asinh',
    probability_threshold: float = 0.5,
    min_combined_prob: float = 0.1,
    save_dir: Optional[str] = None
) -> Tuple[Dict, pd.DataFrame]:
    """
    Complete GMM-based phenotyping workflow.
    """
    
    print("="*70)
    print("COMPLETE GMM PHENOTYPING WORKFLOW")
    print("="*70)
    
    # Step 1: Fit GMMs
    print("\n### STEP 1: FIT GMM PER MARKER ###\n")
    gmm_results = fit_gmm_per_marker(
        sdata,
        table_key=table_key,
        marker_names=marker_names,
        use_layer=use_layer,
        use_bayesian=True
    )
    
    # Step 2: Assign phenotypes
    print("\n### STEP 2: PROBABILISTIC PHENOTYPE ASSIGNMENT ###\n")
    phenotype_probs = assign_phenotypes_probabilistic(
        sdata,
        signatures=signatures,
        table_key=table_key,
        probability_threshold=probability_threshold,
        min_combined_prob=min_combined_prob,
        use_product=True
    )
    
    # Step 3: Visualize
    print("\n### STEP 3: VISUALIZATION ###\n")
    save_path = f"{save_dir}/gmm_phenotyping.pdf" if save_dir else None
    visualize_gmm_phenotyping(
        sdata,
        table_key=table_key,
        markers_to_plot=marker_names[:6] if marker_names else None,
        save_path=save_path
    )
    
    return gmm_results, phenotype_probs

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection

def plot_gmm_phenotypes_spatial(
    sdata,
    table_key: str = 'table_intensities',
    shapes_key: str = 'cell_shapes',
    phenotype_col: str = 'phenotype_gmm',
    save_path: str = None,
    figsize: tuple = (12, 12),
    alpha: float = 0.8,
    outline_width: float = 0.5,
    exclude_unlabelled: bool = False,
    show_legend: bool = True,
    visualize: bool = False
):
    """
    Plot GMM phenotypes using manual polygon rendering (more reliable).
    """
    
    table = sdata.tables[table_key]
    shapes = sdata.shapes[shapes_key]
    
    print("="*70)
    print("PLOTTING GMM PHENOTYPES (SPATIAL) - FIXED")
    print("="*70)
    print(f"Table: {len(table)} cells")
    print(f"Shapes: {len(shapes)} polygons")
    
    # Verify alignment
    if len(table) != len(shapes):
        print(f"WARNING: Mismatch between table ({len(table)}) and shapes ({len(shapes)})")
        print("Attempting to align by index...")
    
    # Get phenotypes - align by index
    if phenotype_col not in table.obs.columns:
        raise ValueError(f"Column '{phenotype_col}' not found in table.obs")
    
    # Try to align by shared index
    common_idx = table.obs.index.intersection(shapes.index)
    if len(common_idx) == 0:
        print("No common indices - using positional alignment")
        phenotypes = table.obs[phenotype_col].values
    else:
        print(f"Found {len(common_idx)} common indices")
        phenotypes = table.obs.loc[shapes.index, phenotype_col].values
    
    # Get phenotype counts
    pheno_counts = pd.Series(phenotypes).value_counts()
    print(f"\nPhenotype distribution:")
    for pheno, count in pheno_counts.items():
        print(f"  {pheno:35s}: {count:6d} ({count/len(phenotypes)*100:5.2f}%)")
    
    # Create color mapping
    unique_phenotypes = [p for p in pheno_counts.index if p != 'unlabelled']
    if exclude_unlabelled:
        unique_phenotypes_to_plot = unique_phenotypes
    else:
        unique_phenotypes_to_plot = list(pheno_counts.index)
    
    n_phenotypes = len(unique_phenotypes_to_plot)
    
    if n_phenotypes <= 10:
        colors = sns.color_palette('tab10', n_phenotypes)
    elif n_phenotypes <= 20:
        colors = sns.color_palette('tab20', n_phenotypes)
    else:
        colors = sns.color_palette('husl', n_phenotypes)
    
    phenotype_to_color = {pheno: colors[i] for i, pheno in enumerate(unique_phenotypes_to_plot)}
    
    # Add gray for unlabelled if needed
    if not exclude_unlabelled and 'unlabelled' in pheno_counts.index:
        phenotype_to_color['unlabelled'] = (0.85, 0.85, 0.85)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    print("\nRendering polygons...")
    
    # Plot each cell as a polygon
    patches = []
    patch_colors = []
    
    n_plotted = 0
    for idx, (shape_idx, row) in enumerate(shapes.iterrows()):
        if idx >= len(phenotypes):
            break
            
        pheno = phenotypes[idx]
        
        # Skip unlabelled if requested
        if exclude_unlabelled and pheno == 'unlabelled':
            continue
        
        geom = row.geometry
        
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            poly = Polygon(coords, closed=True)
            patches.append(poly)
            patch_colors.append(phenotype_to_color.get(pheno, (0.5, 0.5, 0.5)))
            n_plotted += 1
        elif geom.geom_type == 'MultiPolygon':
            for sub_poly in geom.geoms:
                coords = np.array(sub_poly.exterior.coords)
                poly = Polygon(coords, closed=True)
                patches.append(poly)
                patch_colors.append(phenotype_to_color.get(pheno, (0.5, 0.5, 0.5)))
                n_plotted += 1
    
    print(f"Plotted {n_plotted} polygons")
    
    # Add patches to axis
    collection = PatchCollection(patches, 
                                facecolors=patch_colors,
                                edgecolors='black',
                                linewidths=outline_width,
                                alpha=alpha)
    ax.add_collection(collection)
    
    # Set axis properties
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel('X (μm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (μm)', fontsize=14, fontweight='bold')
    
    # Title
    fov = table.obs['fov'].iloc[0] if 'fov' in table.obs.columns else 'Unknown FOV'
    region = table.obs['region'].iloc[0] if 'region' in table.obs.columns else 'Unknown region'
    title = f'GMM Phenotypes - {fov} {region}\n{len(table)} cells, {n_phenotypes} phenotypes'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    if show_legend:
        from matplotlib.patches import Patch
        
        # Sort by count
        sorted_phenotypes = sorted(unique_phenotypes_to_plot, 
                                  key=lambda x: pheno_counts.get(x, 0), 
                                  reverse=True)
        
        legend_elements = []
        for pheno in sorted_phenotypes:
            count = pheno_counts.get(pheno, 0)
            pct = count / len(phenotypes) * 100
            label = f"{pheno} (n={count}, {pct:.1f}%)"
            legend_elements.append(
                Patch(facecolor=phenotype_to_color[pheno], 
                     edgecolor='black', 
                     label=label)
            )
        
        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(1.02, 1),
                 loc='upper left',
                 frameon=True,
                 fancybox=True,
                 shadow=True,
                 fontsize=9,
                 title='Phenotype',
                 title_fontsize=11)
    
    # Grid for easier reading
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()

    # Save or display
    if save_path:
        from pathlib import Path
        save_plot_pdf(fig, Path(save_path), visualize=visualize)
    else:
        if visualize:
            plt.show()
        else:
            plt.close(fig)

    print("\nPlotting complete!")

    return fig, ax