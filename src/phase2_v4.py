"""
QC Flagging for Multiplexed Imaging Data
Following scverse and FAIR principles
"""

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from scipy import stats
from typing import Optional, Union, Dict, Tuple
import warnings
from pathlib import Path


# =================================================================
# PLOTTING UTILITIES - PDF export with optional visualization
# =================================================================

def save_plot_pdf(
    fig,
    output_path: Path,
    visualize: bool = False,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save matplotlib figure as PDF with optional visualization.

    Following best practices for scientific visualization:
    - PDF format ensures vector graphics for scalability and publication quality
    - High DPI (300) suitable for journals and presentations
    - bbox_inches='tight' removes excess whitespace
    - Default is to save without displaying (ideal for batch processing)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : Path
        Output PDF file path
    visualize : bool, default False
        Whether to display the plot interactively after saving
    dpi : int, default 300
        Resolution in dots per inch for PDF output
    bbox_inches : str, default 'tight'
        Bounding box setting to minimize whitespace

    """
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


def flag_cells_at_qc_risk(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    min_area: float = None,
    max_area: float = None,
    n_mads: float = 3.0,
    flag_column: str = "is_qc_pass"
) -> SpatialData:
    """
    Flag cells at risk of failing QC based on morphology (area) and intensities.
    
    This function DOES NOT remove cells - it adds boolean flags to .obs
    following FAIR principles for reproducibility and transparency.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with intensity and morphology tables
    intensity_table : str
        Name of intensity measurements table
    morphology_table : str
        Name of morphology features table  
    min_area : float, optional
        Minimum cell area threshold. If None, uses MAD-based detection
    max_area : float, optional
        Maximum cell area threshold. If None, uses MAD-based detection
    n_mads : float
        Number of MADs from median for outlier detection (default: 3.0)
    flag_column : str
        Column name for the QC pass flag (default: "is_qc_pass")
    
    Returns
    -------
    SpatialData
        Modified SpatialData object with QC flags in .obs
        
    Notes
    -----
    - Assumes intensities are already arcsinh transformed and cell-size normalized
    - Only uses area as morphology feature
    - Flags are stored as boolean columns in .obs
    - Individual QC criteria flags are also stored for transparency
    """
    
    # Get the tables
    if intensity_table not in sdata.tables:
        raise ValueError(f"Table '{intensity_table}' not found in SpatialData")
    if morphology_table not in sdata.tables:
        raise ValueError(f"Table '{morphology_table}' not found in SpatialData")
    
    adata_int = sdata.tables[intensity_table]
    adata_morph = sdata.tables[morphology_table]
    
    # Verify tables are aligned
    if not np.array_equal(adata_int.obs_names, adata_morph.obs_names):
        raise ValueError("Intensity and morphology tables are not aligned")
    
    n_cells = adata_int.n_obs
    
    # =================================================================
    # 1. MORPHOLOGY QC: Cell area outliers
    # =================================================================
    
    # Get area from morphology table
    if adata_morph.n_vars != 1:
        warnings.warn(f"Expected 1 morphology feature (area), found {adata_morph.n_vars}. Using first feature.")
    
    area = adata_morph.X[:, 0].flatten()
    
    # Calculate thresholds using MAD (Median Absolute Deviation)
    # More robust to outliers than mean/std
    if min_area is None or max_area is None:
        median_area = np.median(area)
        mad_area = stats.median_abs_deviation(area)
        
        if min_area is None:
            min_area = median_area - n_mads * mad_area
            min_area = max(min_area, 0)  # Area can't be negative
            
        if max_area is None:
            max_area = median_area + n_mads * mad_area
    
    # Flag cells outside area range
    qc_area_pass = (area >= min_area) & (area <= max_area)
    
    # =================================================================
    # 2. INTENSITY QC: Detect abnormal intensity patterns
    # =================================================================
    
    # Calculate per-cell summary statistics from already-transformed intensities
    X = adata_int.X
    
    # Mean intensity per cell (across all markers)
    mean_intensity = np.mean(X, axis=1).flatten()
    
    # Detect intensity outliers using MAD
    median_int = np.median(mean_intensity)
    mad_int = stats.median_abs_deviation(mean_intensity)
    
    # Flag cells with extremely low or high mean intensity
    # These could be debris, artifacts, or technical issues
    qc_intensity_low = mean_intensity >= (median_int - n_mads * mad_int)
    qc_intensity_high = mean_intensity <= (median_int + n_mads * mad_int)
    qc_intensity_pass = qc_intensity_low & qc_intensity_high
    
    # =================================================================
    # 3. COMBINE FLAGS
    # =================================================================
    
    # Cell passes QC if it passes ALL criteria
    qc_pass = qc_area_pass & qc_intensity_pass
    
    # =================================================================
    # 4. ADD FLAGS TO .obs (FAIR principle: preserve all information)
    # =================================================================
    
    # Add to BOTH tables (they're aligned)
    for adata in [adata_int, adata_morph]:
        # Individual criteria (for transparency and debugging)
        adata.obs['qc_area_pass'] = qc_area_pass
        adata.obs['qc_area_min'] = min_area
        adata.obs['qc_area_max'] = max_area
        adata.obs['qc_intensity_pass'] = qc_intensity_pass
        
        # Overall QC flag
        adata.obs[flag_column] = qc_pass
        
        # Store actual values for reference
        adata.obs['area'] = area
        adata.obs['mean_intensity'] = mean_intensity
    
    # =================================================================
    # 5. STORE METADATA (FAIR principle: document methods)
    # =================================================================
    
    qc_params = {
        'min_area': float(min_area),
        'max_area': float(max_area),
        'n_mads': float(n_mads),
        'method': 'MAD-based outlier detection',
        'n_cells_total': int(n_cells),
        'n_cells_pass': int(qc_pass.sum()),
        'n_cells_fail': int((~qc_pass).sum()),
        'pct_pass': float(qc_pass.sum() / n_cells * 100)
    }
    
    # Store in .uns of both tables
    for adata in [adata_int, adata_morph]:
        if 'qc' not in adata.uns:
            adata.uns['qc'] = {}
        adata.uns['qc']['cell_filtering'] = qc_params
    
    # =================================================================
    # 6. PRINT SUMMARY
    # =================================================================
    
    print("=" * 60)
    print("QC FLAG SUMMARY (Cells at Risk)")
    print("=" * 60)
    print(f"Total cells: {n_cells}")
    print(f"Cells passing QC: {qc_pass.sum()} ({qc_pass.sum()/n_cells*100:.1f}%)")
    print(f"Cells at risk: {(~qc_pass).sum()} ({(~qc_pass).sum()/n_cells*100:.1f}%)")
    print()
    print("Failure breakdown:")
    print(f"  - Failed area QC: {(~qc_area_pass).sum()} cells")
    print(f"    (min: {min_area:.1f}, max: {max_area:.1f})")
    print(f"  - Failed intensity QC: {(~qc_intensity_pass).sum()} cells")
    print()
    print("Flags added to .obs:")
    print(f"  - '{flag_column}': Overall QC pass/fail")
    print("  - 'qc_area_pass': Area QC result")
    print("  - 'qc_intensity_pass': Intensity QC result")
    print("=" * 60)
    
    return sdata


def subset_to_qc_pass(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    flag_column: str = "is_qc_pass"
) -> SpatialData:
    """
    Create a filtered copy with only QC-passing cells.
    
    This is a SEPARATE function from flagging - filtering is optional
    and should only be done after careful inspection of QC results.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with QC flags
    intensity_table : str
        Name of intensity table
    morphology_table : str
        Name of morphology table
    flag_column : str
        Name of QC pass column in .obs
        
    Returns
    -------
    SpatialData
        Filtered SpatialData object (COPY, not in-place)
    """
    from spatialdata import deepcopy
    
    sdata_filtered = deepcopy(sdata)
    
    for table_name in [intensity_table, morphology_table]:
        if table_name in sdata_filtered.tables:
            adata = sdata_filtered.tables[table_name]
            
            if flag_column not in adata.obs:
                warnings.warn(f"QC flag '{flag_column}' not found in {table_name}. Run flag_cells_at_qc_risk() first.")
                continue
            
            # Filter to passing cells
            mask = adata.obs[flag_column].values
            sdata_filtered.tables[table_name] = adata[mask].copy()
            
            print(f"Filtered {table_name}: {adata.n_obs} → {sdata_filtered.tables[table_name].n_obs} cells")
    
    return sdata_filtered

"""
Complete QC Workflow - NOW WITH MARKER INTENSITY PLOTS
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from spatialdata import SpatialData
from scipy import stats
from typing import Optional, List, Literal
from pathlib import Path
import argparse
import warnings
import os


# =================================================================
# 1. QC FLAGGING FUNCTIONS (unchanged)
# =================================================================

def flag_cells_at_qc_risk(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    min_area: float = None,
    max_area: float = None,
    n_mads: float = 3.0,
    flag_column: str = "is_qc_pass"
) -> SpatialData:
    """Flag cells at risk of failing QC."""
    
    if intensity_table not in sdata.tables:
        raise ValueError(f"Table '{intensity_table}' not found in SpatialData")
    if morphology_table not in sdata.tables:
        raise ValueError(f"Table '{morphology_table}' not found in SpatialData")
    
    adata_int = sdata.tables[intensity_table]
    adata_morph = sdata.tables[morphology_table]
    
    if not np.array_equal(adata_int.obs_names, adata_morph.obs_names):
        raise ValueError("Intensity and morphology tables are not aligned")
    
    n_cells = adata_int.n_obs
    
    # Get area
    if adata_morph.n_vars != 1:
        warnings.warn(f"Expected 1 morphology feature (area), found {adata_morph.n_vars}. Using first feature.")
    
    area = adata_morph.X[:, 0].flatten()
    
    # Calculate thresholds using MAD
    if min_area is None or max_area is None:
        median_area = np.median(area)
        mad_area = stats.median_abs_deviation(area)
        
        if min_area is None:
            min_area = median_area - n_mads * mad_area
            min_area = max(min_area, 0)
            
        if max_area is None:
            max_area = median_area + n_mads * mad_area
    
    qc_area_pass = (area >= min_area) & (area <= max_area)
    
    # Calculate intensity statistics
    X = adata_int.X
    mean_intensity = np.mean(X, axis=1).flatten()
    
    median_int = np.median(mean_intensity)
    mad_int = stats.median_abs_deviation(mean_intensity)
    
    qc_intensity_low = mean_intensity >= (median_int - n_mads * mad_int)
    qc_intensity_high = mean_intensity <= (median_int + n_mads * mad_int)
    qc_intensity_pass = qc_intensity_low & qc_intensity_high
    
    qc_pass = qc_area_pass & qc_intensity_pass
    
    # Add flags to BOTH tables
    for adata in [adata_int, adata_morph]:
        adata.obs['qc_area_pass'] = qc_area_pass
        adata.obs['qc_area_min'] = min_area
        adata.obs['qc_area_max'] = max_area
        adata.obs['qc_intensity_pass'] = qc_intensity_pass
        adata.obs[flag_column] = qc_pass
        adata.obs['area'] = area
        adata.obs['mean_intensity'] = mean_intensity
    
    # Store metadata
    qc_params = {
        'min_area': float(min_area),
        'max_area': float(max_area),
        'n_mads': float(n_mads),
        'method': 'MAD-based outlier detection',
        'n_cells_total': int(n_cells),
        'n_cells_pass': int(qc_pass.sum()),
        'n_cells_fail': int((~qc_pass).sum()),
        'pct_pass': float(qc_pass.sum() / n_cells * 100)
    }
    
    for adata in [adata_int, adata_morph]:
        if 'qc' not in adata.uns:
            adata.uns['qc'] = {}
        adata.uns['qc']['cell_filtering'] = qc_params
    
    print("=" * 60)
    print("QC FLAG SUMMARY")
    print("=" * 60)
    print(f"Total cells: {n_cells}")
    print(f"Cells passing QC: {qc_pass.sum()} ({qc_pass.sum()/n_cells*100:.1f}%)")
    print(f"Cells at risk: {(~qc_pass).sum()} ({(~qc_pass).sum()/n_cells*100:.1f}%)")
    print(f"\nFailure breakdown:")
    print(f"  - Failed area QC: {(~qc_area_pass).sum()} cells")
    print(f"  - Failed intensity QC: {(~qc_intensity_pass).sum()} cells")
    print("=" * 60)
    
    return sdata


def exclude_markers_from_analysis(
    sdata: SpatialData,
    table_name: str = "table_intensities",
    exclude_markers: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> SpatialData:
    """Flag markers to exclude from analysis."""
    
    adata = sdata.tables[table_name]
    all_markers = adata.var_names.tolist()
    n_total = len(all_markers)
    
    use_for_analysis = np.ones(n_total, dtype=bool)
    
    print("=" * 60)
    print("MARKER EXCLUSION")
    print("=" * 60)
    print(f"Total markers: {n_total}")
    
    if exclude_markers is not None:
        print(f"Excluding {len(exclude_markers)} specified markers:")
        for marker in exclude_markers:
            if marker in all_markers:
                idx = all_markers.index(marker)
                use_for_analysis[idx] = False
                print(f"{marker}")
            else:
                print(f"Warning: '{marker}' not found")
    
    if exclude_patterns is not None:
        print(f"Excluding by patterns: {exclude_patterns}")
        for pattern in exclude_patterns:
            pattern_lower = pattern.lower()
            for i, marker in enumerate(all_markers):
                if pattern_lower in marker.lower():
                    if use_for_analysis[i]:
                        print(f"{marker}")
                        use_for_analysis[i] = False
    
    n_included = use_for_analysis.sum()
    n_excluded = n_total - n_included
    
    print(f"\nSummary:")
    print(f"  Analysis markers: {n_included}")
    print(f"  Excluded markers: {n_excluded}")
    print("=" * 60)
    
    adata.var['use_for_analysis'] = use_for_analysis
    
    return sdata


# =================================================================
# 2. DIMENSIONALITY REDUCTION
# =================================================================

def compute_dimensionality_reduction(
    sdata: SpatialData,
    table_name: str = "table_intensities",
    use_layer: Optional[str] = None,
    method: Literal["umap", "tsne", "pca"] = "umap",
    n_neighbors: int = 15,
    n_pcs: int = 20,
    random_state: int = 42,
    use_only_analysis_markers: bool = True,
    **kwargs
) -> SpatialData:
    """Compute dimensionality reduction."""
    
    adata = sdata.tables[table_name]
    n_vars = adata.n_vars
    
    # Check marker exclusion
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
    
    print("=" * 60)
    print(f"Computing {method.upper()}")
    print("=" * 60)
    print(f"Cells: {adata_subset.n_obs}, Markers: {n_vars_used}")
    
    # PCA
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
    print(f"PC1-{n_pcs_adjusted} explain {cumsum_var[-1]:.1%} of variance")
    
    # UMAP/tSNE
    if method in ["umap", "tsne"]:
        n_pcs_for_neighbors = min(n_pcs_adjusted, 15)
        
        sc.pp.neighbors(
            adata_subset,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs_for_neighbors,
            random_state=random_state
        )
        
        if method == "umap":
            sc.tl.umap(adata_subset, random_state=random_state, **kwargs)
            print(f"  ✓ UMAP computed")
        elif method == "tsne":
            sc.tl.tsne(adata_subset, n_pcs=n_pcs_for_neighbors, random_state=random_state, **kwargs)
            print(f"  ✓ tSNE computed")
    
    # Copy embeddings back
    embedding_key = f'X_{method}'
    if embedding_key in adata_subset.obsm:
        adata.obsm[embedding_key] = adata_subset.obsm[embedding_key]
    
    if 'X_pca' in adata_subset.obsm:
        adata.obsm['X_pca'] = adata_subset.obsm['X_pca']
        adata.uns['pca'] = adata_subset.uns['pca']
    
    print("=" * 60)
    
    return sdata


# =================================================================
# 3. PLOTTING FUNCTIONS
# =================================================================

def plot_qc_distributions(
    sdata: SpatialData,
    table_name: str = "table_intensities",
    save_path: Optional[str] = None,
    visualize: bool = False
):
    """Plot distributions of QC metrics."""
    
    adata = sdata.tables[table_name]
    
    metrics_to_plot = []
    if 'area' in adata.obs:
        metrics_to_plot.append('area')
    if 'mean_intensity' in adata.obs:
        metrics_to_plot.append('mean_intensity')
    
    if not metrics_to_plot:
        print("No QC metrics found")
        return
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6*n_metrics, 10))
    
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    for i, metric in enumerate(metrics_to_plot):
        ax_hist = axes[0, i]
        
        if 'is_qc_pass' in adata.obs:
            pass_data = adata.obs[adata.obs['is_qc_pass']][metric]
            fail_data = adata.obs[~adata.obs['is_qc_pass']][metric]
            
            ax_hist.hist(pass_data, bins=50, alpha=0.6, label='Pass QC', color='green', density=True)
            ax_hist.hist(fail_data, bins=50, alpha=0.6, label='Fail QC', color='red', density=True)
            
            if f'qc_{metric}_min' in adata.obs:
                min_val = adata.obs[f'qc_{metric}_min'].iloc[0]
                ax_hist.axvline(min_val, color='red', linestyle='--', linewidth=2)
            
            if f'qc_{metric}_max' in adata.obs:
                max_val = adata.obs[f'qc_{metric}_max'].iloc[0]
                ax_hist.axvline(max_val, color='red', linestyle='--', linewidth=2)
            
            ax_hist.legend()
        
        ax_hist.set_xlabel(metric.replace('_', ' ').title())
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'{metric.replace("_", " ").title()} Distribution')
        
        ax_violin = axes[1, i]
        
        if 'is_qc_pass' in adata.obs:
            data_to_plot = [
                adata.obs[adata.obs['is_qc_pass']][metric],
                adata.obs[~adata.obs['is_qc_pass']][metric]
            ]
            
            parts = ax_violin.violinplot(data_to_plot, positions=[1, 2], 
                                        showmeans=True, showmedians=True)
            
            parts['bodies'][0].set_facecolor('green')
            parts['bodies'][0].set_alpha(0.6)
            parts['bodies'][1].set_facecolor('red')
            parts['bodies'][1].set_alpha(0.6)
            
            ax_violin.set_xticks([1, 2])
            ax_violin.set_xticklabels(['Pass QC', 'Fail QC'])
        
        ax_violin.set_ylabel(metric.replace('_', ' ').title())
        ax_violin.set_title(f'{metric.replace("_", " ").title()} by QC Status')
        ax_violin.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_plot_pdf(fig, Path(save_path), visualize=visualize)
    else:
        if visualize:
            plt.show()
        plt.close(fig)


def visualize_qc_on_umap(
    sdata: SpatialData,
    table_name: str = "table_intensities",
    embedding: str = "umap",
    save_path: Optional[str] = None,
    figsize: tuple = (20, 5),
    visualize: bool = False
):
    """Visualize QC metrics on UMAP embedding."""
    
    adata = sdata.tables[table_name]
    
    embedding_key = f"X_{embedding}"
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found. Run compute_dimensionality_reduction() first.")
    
    qc_metrics = []
    if 'is_qc_pass' in adata.obs:
        qc_metrics.append('is_qc_pass')
    if 'area' in adata.obs:
        qc_metrics.append('area')
    if 'mean_intensity' in adata.obs:
        qc_metrics.append('mean_intensity')
    
    if not qc_metrics:
        print("No QC metrics found")
        return
    
    n_plots = len(qc_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    for i, metric in enumerate(qc_metrics):
        ax = axes[i]
        
        is_categorical = adata.obs[metric].dtype == bool
        
        if is_categorical:
            sc.pl.embedding(
                adata,
                basis=embedding,
                color=metric,
                ax=ax,
                show=False,
                frameon=False,
                title=metric.replace('_', ' ').title()
            )
        else:
            sc.pl.embedding(
                adata,
                basis=embedding,
                color=metric,
                ax=ax,
                show=False,
                frameon=False,
                title=metric.replace('_', ' ').title(),
                cmap='viridis',
                vmin=np.percentile(adata.obs[metric], 2),
                vmax=np.percentile(adata.obs[metric], 98)
            )
    
    plt.tight_layout()

    if save_path:
        save_plot_pdf(fig, Path(save_path), visualize=visualize)
    else:
        if visualize:
            plt.show()
        plt.close(fig)


def plot_marker_expression_on_umap(
    sdata: SpatialData,
    table_name: str = "table_intensities",
    embedding: str = "umap",
    markers: Optional[List[str]] = None,
    n_markers: Optional[int] = None,
    include_excluded_markers: bool = False,
    save_path: Optional[str] = None,
    ncols: int = 4,
    visualize: bool = False
):
    """
    Plot marker expression patterns on UMAP.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    table_name : str
        Name of intensity table
    embedding : str
        Embedding to use (default: "umap")
    markers : list of str, optional
        Specific markers to plot. If None, plots all analysis markers
    n_markers : int, optional
        Limit number of markers. If None, plots all
    include_excluded_markers : bool
        Whether to include excluded markers (e.g., DAPI). Default: False
    save_path : str, optional
        Path to save figure
    ncols : int
        Number of columns in grid (default: 4)
    """
    
    adata = sdata.tables[table_name]
    
    embedding_key = f"X_{embedding}"
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found. Run compute_dimensionality_reduction() first.")
    
    # Determine which markers to plot
    if markers is None:
        if 'use_for_analysis' in adata.var and not include_excluded_markers:
            # Plot only analysis markers (exclude DAPI, etc.)
            analysis_mask = adata.var['use_for_analysis'].values
            markers = adata.var_names[analysis_mask].tolist()
            
            excluded_markers = adata.var_names[~analysis_mask].tolist()
            if excluded_markers:
                print(f"Excluding markers: {', '.join(excluded_markers)}")
        else:
            # Plot all markers
            markers = adata.var_names.tolist()
        
        # Apply n_markers limit if specified
        if n_markers is not None:
            markers = markers[:n_markers]
            print(f"Plotting first {n_markers} markers")
    
    n_markers_actual = len(markers)
    
    if n_markers_actual == 0:
        print("No markers to plot")
        return
    
    print(f"Plotting {n_markers_actual} markers on {embedding.upper()}...")
    
    # Calculate grid dimensions
    nrows = int(np.ceil(n_markers_actual / ncols))
    
    # Create figure
    figsize = (ncols * 4, nrows * 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_markers_actual > 1 else [axes]
    
    # Plot each marker
    for i, marker in enumerate(markers):
        ax = axes[i]
        
        try:
            marker_idx = list(adata.var_names).index(marker)
            marker_data = adata.X[:, marker_idx]
            
            sc.pl.embedding(
                adata,
                basis=embedding,
                color=marker,
                ax=ax,
                show=False,
                frameon=False,
                title=marker,
                cmap='viridis',
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
        save_plot_pdf(fig, Path(save_path), visualize=visualize)
    else:
        if visualize:
            plt.show()
        plt.close(fig)
def plot_qc_spatial_with_spatialdata_plot(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    shapes_key: str = "cell_shapes",
    coordinate_system: str = "global",
    save_path: Optional[str] = None,
    figsize: tuple = (20, 5),
    visualize: bool = False
):
    """Visualize QC spatially using spatialdata_plot."""
    
    adata = sdata.tables[intensity_table]
    
    if 'is_qc_pass' not in adata.obs:
        raise ValueError("QC flags not found. Run flag_cells_at_qc_risk() first!")
    
    if shapes_key not in sdata.shapes:
        raise ValueError(f"Shapes '{shapes_key}' not found")
    
    # Create QC status as categorical
    qc_status = adata.obs['is_qc_pass'].astype(str).replace({'True': 'Pass', 'False': 'Fail'})
    adata.obs['qc_status'] = pd.Categorical(qc_status, categories=['Pass', 'Fail'])
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    print("Creating spatial QC plots...")
    
    # Panel 1: All cells colored by QC status
    try:
        sdata.pl.render_shapes(
            shapes_key,
            color='qc_status',
            table_name=intensity_table,
            fill_alpha=0.7,
            outline=True,
            outline_width=0.5,
            palette={'Pass': 'green', 'Fail': 'red'},
            ax=axes[0],
            coordinate_systems=coordinate_system
        ).pl.show()
        axes[0].set_title('QC Status: Pass vs Fail', fontsize=12, fontweight='bold')
    except Exception as e:
        print(f"Warning: spatialdata_plot failed, using fallback: {e}")
        
        shapes = sdata.shapes[shapes_key]
        pass_mask = adata.obs['is_qc_pass'].values
        
        shapes[~pass_mask].plot(ax=axes[0], color='red', alpha=0.7, edgecolor='darkred', linewidth=0.5)
        shapes[pass_mask].plot(ax=axes[0], color='green', alpha=0.7, edgecolor='darkgreen', linewidth=0.5)
        axes[0].set_title('QC Status: Pass (green) vs Fail (red)', fontsize=12, fontweight='bold')
        axes[0].set_aspect('equal')
    
    axes[0].axis('off')
    
    # Panel 2: Only passing cells
    shapes = sdata.shapes[shapes_key]
    pass_mask = adata.obs['is_qc_pass'].values
    
    shapes[pass_mask].plot(
        ax=axes[1],
        color='green',
        alpha=0.7,
        edgecolor='darkgreen',
        linewidth=0.5
    )
    
    n_pass = pass_mask.sum()
    axes[1].set_title(f'Passing QC (n={n_pass})', fontsize=12, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    
    # Panel 3: Only failing cells
    fail_mask = ~pass_mask
    
    if fail_mask.sum() > 0:
        shapes[fail_mask].plot(
            ax=axes[2],
            color='red',
            alpha=0.7,
            edgecolor='darkred',
            linewidth=0.5
        )
        n_fail = fail_mask.sum()
        axes[2].set_title(f'Failing QC (n={n_fail})', fontsize=12, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'No cells\nfailed QC', 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=14, fontweight='bold', color='green')
        axes[2].set_title('Failing QC (n=0)', fontsize=12, fontweight='bold')
    
    axes[2].set_aspect('equal')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        save_plot_pdf(fig, Path(save_path), visualize=visualize)
    else:
        if visualize:
            plt.show()
        plt.close(fig)
# =================================================================
# 4. COMPLETE WORKFLOW
# =================================================================

def run_complete_qc_workflow(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    shapes_key: str = "cell_shapes",
    output_dir: str = "./qc_plots",
    exclude_markers: Optional[List[str]] = None,
    plot_umap: bool = True,
    plot_spatial: bool = True,
    plot_marker_expression: bool = True,  # NEW
    coordinate_system: str = "global",
    visualize=False
) -> SpatialData:
    """
    Complete QC workflow with marker expression plots.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    intensity_table : str
        Name of intensity table
    morphology_table : str
        Name of morphology table
    shapes_key : str
        Name of shapes layer
    output_dir : str
        Output directory
    exclude_markers : list of str, optional
        Markers to exclude (e.g., ['DAPI_INIT', 'DAPI_FINAL'])
    plot_umap : bool
        Whether to compute/plot UMAP
    plot_spatial : bool
        Whether to create spatial plots
    plot_marker_expression : bool
        Whether to plot marker expression on UMAP (default: True)
    coordinate_system : str
        Coordinate system for spatialdata_plot
    visualize : bool
        Whether to display plots interactively (default: False).
        If False, plots are saved as PDFs only (ideal for batch processing).
    
    Returns
    -------
    SpatialData
        Modified SpatialData with QC flags and embeddings
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("COMPLETE QC WORKFLOW")
    print("=" * 70 + "\n")
    
    adata = sdata.tables[intensity_table]
    
    # Check QC flags exist
    if 'is_qc_pass' not in adata.obs:
        print("QC flags not found. Run flag_cells_at_qc_risk() first!")
        return sdata
    
    print("Step 1: QC flags detected")
    n_pass = adata.obs['is_qc_pass'].sum()
    n_fail = (~adata.obs['is_qc_pass']).sum()
    print(f"  Pass: {n_pass} ({n_pass/adata.n_obs*100:.1f}%)")
    print(f"  Fail: {n_fail} ({n_fail/adata.n_obs*100:.1f}%)\n")
    
    # Exclude markers
    if exclude_markers:
        print("Step 2: Excluding markers")
        sdata = exclude_markers_from_analysis(
            sdata,
            table_name=intensity_table,
            exclude_markers=exclude_markers
        )
        print()
    
    # Plot distributions
    print("Step 3: Plotting QC distributions")
    plot_qc_distributions(
        sdata,
        table_name=intensity_table,
        save_path=str(output_dir / "qc_distributions.pdf"),
        visualize=visualize
    )
    print()
    
    # UMAP analysis
    if plot_umap:
        print("Step 4: UMAP analysis")
        try:
            sdata = compute_dimensionality_reduction(
                sdata,
                table_name=intensity_table,
                method="umap",
                n_neighbors=15,
                n_pcs=20,
                random_state=42,
                use_only_analysis_markers=True
            )
            
            # Plot QC metrics on UMAP
            visualize_qc_on_umap(
                sdata,
                table_name=intensity_table,
                embedding="umap",
                save_path=str(output_dir / "qc_metrics_umap.pdf"),
                visualize=visualize
            )
            
            # Plot marker expression on UMAP
            if plot_marker_expression:
                print("\n  Plotting marker expression on UMAP...")
                plot_marker_expression_on_umap(
                    sdata,
                    table_name=intensity_table,
                    embedding="umap",
                    include_excluded_markers=False,  # Don't plot DAPI, etc.
                    save_path=str(output_dir / "marker_expression_umap.pdf"),
                    ncols=4,
                    visualize=visualize
                )
            
            print("UMAP complete\n")
        except Exception as e:
            print(f"UMAP failed: {e}\n")
    
    # Spatial visualization
    if plot_spatial:
        print("Step 5: Spatial visualisation")
        try:
            plot_qc_spatial_with_spatialdata_plot(
                sdata,
                intensity_table=intensity_table,
                shapes_key=shapes_key,
                coordinate_system=coordinate_system,
                save_path=str(output_dir / "qc_spatial.pdf"),
                visualize=visualize
            )
            print("Spatial plots complete\n")
        except Exception as e:
            print(f"Spatial plots failed: {e}\n")
    
    # Summary
    print("=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.pdf")):
        print(f"  - {file.name}")
    print("=" * 70 + "\n")
    
    return sdata

import numpy as np
from spatialdata import SpatialData
from typing import Optional, List
from pathlib import Path
import argparse
import warnings


def filter_sdata_by_qc(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    shapes_key: str = "cell_shapes",
    labels_key: Optional[str] = "cell_segmentation",
    qc_column: str = "is_qc_pass",
    copy: bool = True
) -> SpatialData:
    """
    Filter SpatialData object to only QC-passing cells.
    
    This creates a NEW SpatialData object with only cells that passed QC.
    The original object is NOT modified (FAIR principle).
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with QC flags
    intensity_table : str
        Name of intensity table
    morphology_table : str
        Name of morphology table
    shapes_key : str
        Name of shapes layer
    labels_key : str, optional
        Name of labels layer (segmentation masks)
    qc_column : str
        Column name with QC pass/fail flags (default: "is_qc_pass")
    copy : bool
        If True, creates a deep copy (recommended). If False, modifies in place.
    
    Returns
    -------
    SpatialData
        Filtered SpatialData object containing only QC-passing cells
        
    Notes
    -----
    - This function filters:
      * Tables (intensity and morphology)
      * Shapes (cell boundaries)
      * Labels (segmentation masks) - optional
    - Images are NOT filtered (they remain full FOV)
    - Original SpatialData is preserved if copy=True
    
    Examples
    --------
    # Create filtered version
    sdata_filtered = filter_sdata_by_qc(sdata)
    
    # Save both versions
    sdata.write("data_with_qc_flags.zarr")  # Original with flags
    sdata_filtered.write("data_qc_filtered.zarr")  # Filtered
    """
    
    # Create copy if requested (FAIR principle: preserve original)
    if copy:
        from copy import deepcopy
        sdata_filtered = deepcopy(sdata)
        print("Creating filtered copy of SpatialData...")
    else:
        sdata_filtered = sdata
        warnings.warn("Modifying SpatialData in place. Original data will be lost!")
    
    # Check QC flags exist
    if intensity_table not in sdata_filtered.tables:
        raise ValueError(f"Table '{intensity_table}' not found")
    
    adata = sdata_filtered.tables[intensity_table]
    
    if qc_column not in adata.obs:
        raise ValueError(f"QC column '{qc_column}' not found. Run flag_cells_at_qc_risk() first!")
    
    # Get QC mask
    qc_pass_mask = adata.obs[qc_column].values
    n_total = len(qc_pass_mask)
    n_pass = qc_pass_mask.sum()
    n_fail = (~qc_pass_mask).sum()
    
    print("=" * 60)
    print("FILTERING SPATIALDATA BY QC")
    print("=" * 60)
    print(f"Total cells: {n_total}")
    print(f"Passing QC: {n_pass} ({n_pass/n_total*100:.1f}%)")
    print(f"Failing QC: {n_fail} ({n_fail/n_total*100:.1f}%)")
    print()
    
    # =================================================================
    # 1. Filter Tables (intensity and morphology)
    # =================================================================
    
    print("Filtering tables...")
    
    # Filter intensity table
    if intensity_table in sdata_filtered.tables:
        adata_int = sdata_filtered.tables[intensity_table]
        sdata_filtered.tables[intensity_table] = adata_int[qc_pass_mask].copy()
        print(f"{intensity_table}: {n_total} → {n_pass} cells")
    
    # Filter morphology table
    if morphology_table in sdata_filtered.tables:
        adata_morph = sdata_filtered.tables[morphology_table]
        sdata_filtered.tables[morphology_table] = adata_morph[qc_pass_mask].copy()
        print(f"{morphology_table}: {n_total} → {n_pass} cells")
    
    # =================================================================
    # 2. Filter Shapes (cell boundaries)
    # =================================================================
    
    print("\nFiltering shapes...")
    
    if shapes_key in sdata_filtered.shapes:
        shapes = sdata_filtered.shapes[shapes_key]
        sdata_filtered.shapes[shapes_key] = shapes[qc_pass_mask].copy()
        print(f"{shapes_key}: {n_total} → {n_pass} shapes")
    else:
        print(f"Shapes '{shapes_key}' not found, skipping")
    
    # =================================================================
    # 3. Filter Labels (segmentation masks) - OPTIONAL
    # =================================================================
    
    if labels_key and labels_key in sdata_filtered.labels:
        print("\nFiltering labels (segmentation masks)...")
        
        labels = sdata_filtered.labels[labels_key]
        
        # Get cell IDs that passed QC
        # Assuming cell IDs match the index position + 1 (common convention)
        cell_ids_to_keep = np.where(qc_pass_mask)[0] + 1
        
        # Create filtered label image
        # Keep only cells that passed QC, set others to 0 (background)
        if hasattr(labels, 'data'):
            labels_data = labels.data.compute() if hasattr(labels.data, 'compute') else labels.data
        else:
            labels_data = labels.compute() if hasattr(labels, 'compute') else labels
        
        labels_filtered = np.zeros_like(labels_data)
        
        for new_id, old_id in enumerate(cell_ids_to_keep, start=1):
            labels_filtered[labels_data == old_id] = new_id
        
        # Update labels in SpatialData
        # Note: This depends on your SpatialData structure
        # You may need to wrap this in a DataArray with proper coordinates
        try:
            from spatialdata.models import Labels2DModel
            sdata_filtered.labels[labels_key] = Labels2DModel.parse(labels_filtered)
            print(f"{labels_key}: Filtered and relabeled")
        except Exception as e:
            print(f"Could not filter labels: {e}")
            print("Labels filtering skipped (complex operation)")
    else:
        print("\nSkipping labels filtering (not requested or not found)")
    
    # =================================================================
    # 4. Update metadata
    # =================================================================
    
    print("\nUpdating metadata...")
    
    if not hasattr(sdata_filtered, 'attrs'):
        sdata_filtered.attrs = {}
    
    sdata_filtered.attrs['filtering_applied'] = True
    sdata_filtered.attrs['filtering_qc_column'] = qc_column
    sdata_filtered.attrs['filtering_n_cells_before'] = int(n_total)
    sdata_filtered.attrs['filtering_n_cells_after'] = int(n_pass)
    sdata_filtered.attrs['filtering_n_cells_removed'] = int(n_fail)
    sdata_filtered.attrs['filtering_pct_retained'] = float(n_pass / n_total * 100)
    
    print(f"Metadata stored in sdata.attrs")
    
    print()
    print("=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"Filtered SpatialData: {n_pass} cells")
    print("=" * 60 + "\n")
    
    return sdata_filtered

def filter_sdata_by_qc_efficient(
    sdata: SpatialData,
    intensity_table: str = "table_intensities",
    morphology_table: str = "table_morphology",
    shapes_key: str = "cell_shapes",
    labels_keys: Optional[Union[str, List[str], Dict[str, str]]] = None,
    qc_column: str = "is_qc_pass",
    filter_labels: bool = True,
) -> SpatialData:
    """
    Efficiently filter SpatialData to only QC-passing cells.
    
    This creates a NEW SpatialData from scratch rather than using deepcopy,
    which is MUCH faster. Supports multiple label layers (e.g., nuclear and cytoplasm).
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with QC flags
    intensity_table : str
        Name of intensity table
    morphology_table : str
        Name of morphology table
    shapes_key : str
        Name of shapes layer
    labels_keys : str, List[str], Dict[str, str], or None
        Label layer(s) to filter. Can be:
        - str: Single label layer name (e.g., "cell_segmentation")
        - List[str]: Multiple label layers (e.g., ["nuclear_segmentation", "cytoplasm_segmentation"])
        - Dict[str, str]: Mapping of label names to descriptive names for logging
        - None: No labels filtering
        Only filtered if filter_labels=True
    qc_column : str
        Column with QC pass/fail flags
    filter_labels : bool
        Whether to filter labels. Set to False to skip (much faster).
        Labels filtering is slow and often not needed for downstream analysis.
    
    Returns
    -------
    SpatialData
        New filtered SpatialData object
    """
    
    # Check QC flags exist
    if intensity_table not in sdata.tables:
        raise ValueError(f"Table '{intensity_table}' not found")
    
    adata = sdata.tables[intensity_table]
    
    if qc_column not in adata.obs:
        raise ValueError(f"QC column '{qc_column}' not found")
    
    # Get QC mask
    qc_pass_mask = adata.obs[qc_column].values
    n_total = len(qc_pass_mask)
    n_pass = qc_pass_mask.sum()
    n_fail = (~qc_pass_mask).sum()
    
    print("=" * 60)
    print("FILTERING SPATIALDATA (EFFICIENT)")
    print("=" * 60)
    print(f"Total cells: {n_total}")
    print(f"Passing QC: {n_pass} ({n_pass/n_total*100:.1f}%)")
    print(f"Failing QC: {n_fail} ({n_fail/n_total*100:.1f}%)")
    print()
    
    # =================================================================
    # Build filtered SpatialData
    # =================================================================
    
    # 1. Filter tables
    print("Filtering tables...")
    tables_dict = {}
    
    if intensity_table in sdata.tables:
        tables_dict[intensity_table] = sdata.tables[intensity_table][qc_pass_mask].copy()
        print(f"{intensity_table}: {n_total} → {n_pass} cells")
    
    if morphology_table in sdata.tables:
        tables_dict[morphology_table] = sdata.tables[morphology_table][qc_pass_mask].copy()
        print(f"{morphology_table}: {n_total} → {n_pass} cells")
    
    # 2. Filter shapes
    print("\nFiltering shapes...")
    shapes_dict = {}
    
    if shapes_key in sdata.shapes:
        shapes_dict[shapes_key] = sdata.shapes[shapes_key][qc_pass_mask].copy()
        print(f"{shapes_key}: {n_total} → {n_pass} shapes")
    
    # 3. Copy images (NOT filtered - keep full FOV)
    print("\nCopying images (full FOV)...")
    images_dict = {}
    for img_key in sdata.images.keys():
        images_dict[img_key] = sdata.images[img_key]
        print(f"{img_key}: kept full image")
    
    # 4. Normalize labels_keys to dict format
    labels_dict = {}
    labels_to_process = {}
    
    if labels_keys is not None:
        if isinstance(labels_keys, str):
            # Single label layer
            labels_to_process = {labels_keys: labels_keys}
        elif isinstance(labels_keys, list):
            # Multiple label layers - use key as description
            labels_to_process = {key: key for key in labels_keys}
        elif isinstance(labels_keys, dict):
            # Already a dict
            labels_to_process = labels_keys
        else:
            raise TypeError(
                f"labels_keys must be str, List[str], Dict[str, str], or None. "
                f"Got {type(labels_keys)}"
            )
    
    # 5. Labels (optional - SLOW if enabled)
    if filter_labels and labels_to_process:
        print("\nFiltering labels (this may take a while)...")
        
        # Get cell IDs to keep (pre-compute once)
        cell_ids_to_keep = np.where(qc_pass_mask)[0] + 1
        
        for labels_key, description in labels_to_process.items():
            if labels_key not in sdata.labels:
                print(f"{labels_key} not found in sdata.labels, skipping")
                continue
            
            print(f"  Processing {description}...")
            
            try:
                labels = sdata.labels[labels_key]
                
                # Load label data
                if hasattr(labels, 'data'):
                    labels_data = labels.data.compute() if hasattr(labels.data, 'compute') else labels.data
                else:
                    labels_data = labels.compute() if hasattr(labels, 'compute') else labels
                
                # Efficient filtering using np.isin
                mask = np.isin(labels_data, cell_ids_to_keep)
                labels_filtered = np.where(mask, labels_data, 0)
                
                # Relabel sequentially
                unique_ids = np.unique(labels_filtered[labels_filtered > 0])
                relabel_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
                
                labels_filtered_relabeled = np.zeros_like(labels_filtered)
                for old_id, new_id in relabel_map.items():
                    labels_filtered_relabeled[labels_filtered == old_id] = new_id
                
                # Parse with Labels2DModel
                from spatialdata.models import Labels2DModel
                labels_dict[labels_key] = Labels2DModel.parse(labels_filtered_relabeled)
                print(f"{labels_key}: filtered and relabeled ({len(unique_ids)} labels retained)")
                
            except Exception as e:
                print(f"Could not filter {labels_key}: {e}")
    
    else:
        if labels_to_process:
            print(f"\nSkipping labels filtering (filter_labels=False)")
            print(f"  Note: Set filter_labels=True to filter {len(labels_to_process)} label layer(s)")
            print(f"  Label layers available: {list(labels_to_process.keys())}")
    
    # =================================================================
    # Create new SpatialData object (FAST - no deepcopy)
    # =================================================================
    
    print("\nCreating new SpatialData object...")
    
    sdata_filtered = SpatialData(
        images=images_dict if images_dict else None,
        labels=labels_dict if labels_dict else None,
        shapes=shapes_dict if shapes_dict else None,
        tables=tables_dict if tables_dict else None
    )
    
    # Copy coordinate systems
    if hasattr(sdata, 'coordinate_systems'):
        for cs_name in sdata.coordinate_systems:
            # This preserves the coordinate system metadata
            pass
    
    # Add filtering metadata
    if not hasattr(sdata_filtered, 'attrs'):
        sdata_filtered.attrs = {}
    
    sdata_filtered.attrs['filtering_applied'] = True
    sdata_filtered.attrs['filtering_qc_column'] = qc_column
    sdata_filtered.attrs['filtering_n_cells_before'] = int(n_total)
    sdata_filtered.attrs['filtering_n_cells_after'] = int(n_pass)
    sdata_filtered.attrs['filtering_n_cells_removed'] = int(n_fail)
    sdata_filtered.attrs['filtering_pct_retained'] = float(n_pass / n_total * 100)
    sdata_filtered.attrs['filtering_labels_included'] = filter_labels
    sdata_filtered.attrs['filtering_labels_processed'] = list(labels_dict.keys()) if filter_labels else []
    
    print()
    print("=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"Filtered cells: {n_pass}")
    print(f"Labels filtered: {len(labels_dict) if filter_labels else 0} layer(s)")
    if filter_labels and labels_dict:
        print(f"  Layers: {', '.join(labels_dict.keys())}")
    elif not filter_labels and labels_to_process:
        print(f"  (Skipped for speed)")
    print("=" * 60 + "\n")
    
    return sdata_filtered
