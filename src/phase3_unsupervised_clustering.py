"""
Phase 3: Unsupervised Clustering for Multiplexed Imaging Data
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from spatialdata import SpatialData
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import warnings


def save_plot_pdf(fig, output_path: Path, visualize: bool = False, dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """Save matplotlib figure as PDF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='pdf', dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved plot: {output_path}")
    
    if visualize:
        plt.show()
    else:
        plt.close(fig)


def run_unsupervised_clustering(
    sdata: SpatialData,
    table_key: str = "table_intensities",
    exclude_markers: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    use_layer: Optional[str] = None,  # None uses .X (arcsinh-transformed)
    n_neighbors: int = 15,
    n_pcs: int = 20,
    resolutions: List[float] = [0.3, 0.5, 0.8, 1.0, 1.5],
    random_state: int = 42,
) -> SpatialData:
    """
    Run unsupervised clustering using Leiden algorithm on arcsinh-transformed intensities.
    
    Following best practices from:
    - scverse clustering guidelines
    - CODEX/IMC clustering workflows
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with normalized intensities
    table_key : str
        Name of intensity table
    exclude_markers : List[str], optional
        Specific markers to exclude (e.g., nuclear stains)
    exclude_patterns : List[str], optional
        Patterns to exclude (e.g., ['DAPI', 'EMPTY'])
    use_layer : str, optional
        Layer to use. If None, uses .X (should be arcsinh-transformed)
    n_neighbors : int
        Number of neighbors for KNN graph (default: 15)
    n_pcs : int
        Number of PCs to compute (default: 20)
    resolutions : List[float]
        Leiden resolutions to test (default: [0.3, 0.5, 0.8, 1.0, 1.5])
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    SpatialData
        Modified SpatialData with clustering results in .obs
        
    Notes
    -----
    - Operates on arcsinh-transformed, area-normalized data in .X
    - Computes PCA for dimensionality reduction
    - Builds KNN graph on PC space
    - Runs Leiden at multiple resolutions for exploration
    - Stores results as leiden_res_X in .obs
    """
    
    print("=" * 70)
    print("UNSUPERVISED CLUSTERING - Leiden Algorithm")
    print("=" * 70)
    
    adata = sdata.tables[table_key]
    print(f"Cells: {adata.n_obs}")
    print(f"Markers: {adata.n_vars}")
    
    # Step 1: Marker selection
    use_for_clustering = np.ones(adata.n_vars, dtype=bool)
    
    if exclude_markers:
        for marker in exclude_markers:
            if marker in adata.var_names:
                idx = list(adata.var_names).index(marker)
                use_for_clustering[idx] = False
                
    if exclude_patterns:
        for pattern in exclude_patterns:
            pattern_lower = pattern.lower()
            for i, marker in enumerate(adata.var_names):
                if pattern_lower in marker.lower():
                    use_for_clustering[i] = False
    
    # Store marker selection
    adata.var['use_for_clustering'] = use_for_clustering
    
    n_included = use_for_clustering.sum()
    n_excluded = (~use_for_clustering).sum()
    
    print(f"\nMarker selection:")
    print(f"Analysis markers: {n_included}")
    print(f"Excluded markers: {n_excluded}")
    if n_excluded > 0:
        excluded_names = adata.var_names[~use_for_clustering].tolist()
        print(f"    {', '.join(excluded_names)}")
    
    # Create subset for clustering
    adata_subset = adata[:, use_for_clustering].copy()
    
    # Step 2: PCA
    print(f"\nStep 1: PCA")
    print(f"  Computing {n_pcs} PCs on {'layer: ' + use_layer if use_layer else 'arcsinh-transformed data (.X)'}...")
    
    n_pcs_adjusted = min(n_pcs, n_included - 1, adata_subset.n_obs - 1)
    
    sc.tl.pca(
        adata_subset,
        n_comps=n_pcs_adjusted,
        layer=use_layer,
        random_state=random_state,
        svd_solver='arpack'
    )
    
    var_explained = adata_subset.uns['pca']['variance_ratio']
    cumsum_var = np.cumsum(var_explained)
    
    print(f"PC1-{n_pcs_adjusted} explain {cumsum_var[-1]*100:.1f}% of variance")
    # Only print PC1-10 variance if we have at least 10 PCs
    if n_pcs_adjusted >= 10:
        print(f"PC1-10 explain {cumsum_var[9]*100:.1f}% of variance")
    else:
        # Print the first few PCs if less than 10
        n_to_show = min(5, n_pcs_adjusted)
        if n_to_show > 0:
            print(f"PC1-{n_to_show} explain {cumsum_var[n_to_show-1]*100:.1f}% of variance")
    
    # Step 3: KNN graph
    print(f"\nStep 2: Building KNN graph")
    print(f"  Neighbors: {n_neighbors}")
    print(f"  Using {min(n_pcs_adjusted, 15)} PCs for neighborhood calculation...")
    
    n_pcs_for_neighbors = min(n_pcs_adjusted, 15)
    
    sc.pp.neighbors(
        adata_subset,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_for_neighbors,
        random_state=random_state
    )
    
    print(f"KNN graph constructed")
    
    # Step 4: UMAP
    print(f"\nStep 3: Computing UMAP embedding")
    sc.tl.umap(adata_subset, random_state=random_state)
    print(f"UMAP computed")
    
    # Step 5: Leiden clustering at multiple resolutions
    print(f"\nStep 4: Leiden clustering")
    print(f"  Testing {len(resolutions)} resolutions: {resolutions}")

    clustering_results = {}

    for res in resolutions:
        key = f'leiden_res_{res}'
    
        sc.tl.leiden(
            adata_subset,
            resolution=res,
            random_state=random_state,
            key_added=key
        )
    
        n_clusters = adata_subset.obs[key].nunique()
    
        # Store with STRING keys (required for Zarr serialization)
        clustering_results[str(res)] = {
            'n_clusters': int(n_clusters),
            'key': key,
            'resolution': float(res)  # Store the actual float value too
        }
    
        print(f"Resolution {res}: {n_clusters} clusters")

    # Copy results back to original adata
    adata.obsm['X_pca'] = adata_subset.obsm['X_pca']
    adata.obsm['X_umap'] = adata_subset.obsm['X_umap']
    adata.uns['pca'] = adata_subset.uns['pca']
    adata.uns['neighbors'] = adata_subset.uns['neighbors']
    adata.uns['umap'] = adata_subset.uns['umap']

    for res in resolutions:
        key = f'leiden_res_{res}'
        adata.obs[key] = adata_subset.obs[key]
        if key in adata_subset.uns:
            adata.uns[key] = adata_subset.uns[key]

    # Store metadata with proper types for Zarr serialization
    adata.uns['clustering'] = {
        'method': 'leiden',
        'n_neighbors': int(n_neighbors),
        'n_pcs': int(n_pcs_adjusted),
        'resolutions': [float(r) for r in resolutions],
        'random_state': int(random_state),
        'use_layer': str(use_layer if use_layer else 'X (arcsinh)'),
        'n_markers_used': int(n_included),
        'excluded_markers': [str(m) for m in adata.var_names[~use_for_clustering].tolist()],
        'clustering_results': clustering_results,  # Now has string keys
        'references': {
            'leiden': 'Traag et al. (2019) Scientific Reports',
            'scanpy': 'scverse best practices'
        }
    }      

    print(f"\nClustering metadata stored in .uns['clustering']")
    print("=" * 70)

    return sdata


def visualize_clustering_results(
    sdata: SpatialData,
    table_key: str = "table_intensities",
    resolutions: Optional[List[float]] = None,
    save_dir: Optional[Path] = None,
    visualize: bool = False,
) -> None:
    """
    Visualise clustering results at different resolutions.
    
    Creates:
    - UMAP colored by clusters at each resolution
    - Cluster size distributions
    - Marker expression heatmap by cluster
    """
    
    adata = sdata.tables[table_key]
    
    if resolutions is None:
        # Auto-detect resolutions from .obs
        resolutions = sorted([
            float(col.replace('leiden_res_', ''))
            for col in adata.obs.columns
            if col.startswith('leiden_res_')
        ])
    
    if not resolutions:
        print("No Leiden clustering results found in .obs")
        return
    
    print("=" * 70)
    print("CLUSTERING VISUALISATION")
    print("=" * 70)
    print(f"Resolutions: {resolutions}")
    
    # Plot 1: UMAP grid for all resolutions
    n_res = len(resolutions)
    ncols = min(3, n_res)
    nrows = int(np.ceil(n_res / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, res in enumerate(resolutions):
        key = f'leiden_res_{res}'
        ax = axes[i]
        
        sc.pl.umap(
            adata,
            color=key,
            ax=ax,
            show=False,
            frameon=False,
            title=f'Resolution {res}\n({adata.obs[key].nunique()} clusters)',
            legend_loc='right margin'
        )
    
    # Hide unused subplots
    for i in range(n_res, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_plot_pdf(fig, Path(save_dir) / "clustering_umap_grid.pdf", visualize=visualize)
    else:
        if visualize:
            plt.show()
        else:
            plt.close(fig)
    
    # Plot 2: Cluster size distribution for each resolution
    fig, axes = plt.subplots(1, n_res, figsize=(5*n_res, 4))
    if n_res == 1:
        axes = [axes]
    
    for i, res in enumerate(resolutions):
        key = f'leiden_res_{res}'
        cluster_sizes = adata.obs[key].value_counts().sort_index()
        
        axes[i].bar(range(len(cluster_sizes)), cluster_sizes.values, color='steelblue', edgecolor='black')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel('Number of cells')
        axes[i].set_title(f'Resolution {res}')
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_plot_pdf(fig, Path(save_dir) / "clustering_sizes.pdf", visualize=visualize)
    else:
        if visualize:
            plt.show()
        else:
            plt.close(fig)
    
    print(f"Visualisations complete")


def compute_cluster_marker_profiles(
    sdata: SpatialData,
    table_key: str = "table_intensities",
    clustering_key: str = "leiden_res_0.5",
    use_layer: Optional[str] = None,
    exclude_markers: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    visualize: bool = False,
) -> pd.DataFrame:
    """
    Compute mean marker expression per cluster for annotation.

    Returns a DataFrame with clusters x markers showing mean expression.
    This is used to manually annotate clusters based on marker profiles.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with clustering results
    table_key : str
        Name of intensity table
    clustering_key : str
        Column in .obs with cluster assignments
    use_layer : str, optional
        Layer to use. If None, uses .X (arcsinh-transformed)
    exclude_markers : List[str], optional
        Specific markers to exclude (e.g., nuclear stains)
    exclude_patterns : List[str], optional
        Patterns to exclude (e.g., ['DAPI', 'EMPTY'])
    save_dir : Path, optional
        Directory to save outputs
    visualize : bool
        Whether to display plots interactively

    Returns
    -------
    pd.DataFrame
        Cluster profiles with mean expression per marker
    """
    
    adata = sdata.tables[table_key]

    if clustering_key not in adata.obs:
        raise ValueError(f"Clustering key '{clustering_key}' not found in .obs")

    print("=" * 70)
    print("CLUSTER MARKER PROFILES")
    print("=" * 70)
    print(f"Clustering: {clustering_key}")

    # Step 1: Marker selection
    use_for_profiles = np.ones(adata.n_vars, dtype=bool)

    if exclude_markers:
        for marker in exclude_markers:
            if marker in adata.var_names:
                idx = list(adata.var_names).index(marker)
                use_for_profiles[idx] = False

    if exclude_patterns:
        for pattern in exclude_patterns:
            pattern_lower = pattern.lower()
            for i, marker in enumerate(adata.var_names):
                if pattern_lower in marker.lower():
                    use_for_profiles[i] = False

    n_included = use_for_profiles.sum()
    n_excluded = (~use_for_profiles).sum()

    if n_excluded > 0:
        excluded_names = adata.var_names[~use_for_profiles].tolist()
        print(f"\nExcluded markers: {n_excluded}")
        print(f"  {', '.join(excluded_names)}")
    print(f"Included markers: {n_included}")

    # Get data (subset to included markers)
    if use_layer and use_layer in adata.layers:
        X = adata.layers[use_layer]
    else:
        X = adata.X

    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Subset to included markers
    X_subset = X[:, use_for_profiles]
    marker_names = adata.var_names[use_for_profiles]

    # Compute mean expression per cluster
    clusters = adata.obs[clustering_key].values
    cluster_ids = sorted(adata.obs[clustering_key].unique())

    profiles = []
    for cluster_id in cluster_ids:
        mask = clusters == cluster_id
        mean_expr = X_subset[mask, :].mean(axis=0)
        profiles.append(mean_expr)

    profiles_df = pd.DataFrame(
        profiles,
        index=[f'Cluster_{cid}' for cid in cluster_ids],
        columns=marker_names
    )
    
    # Add cluster sizes
    profiles_df['n_cells'] = [
        (clusters == cid).sum() for cid in cluster_ids
    ]
    
    print(f"\nCluster sizes:")
    for cluster_id in cluster_ids:
        n_cells = (clusters == cluster_id).sum()
        pct = n_cells / len(clusters) * 100
        print(f"  Cluster {cluster_id}: {n_cells} cells ({pct:.1f}%)")
    
    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, max(6, len(cluster_ids) * 0.5)))
    
    # Exclude n_cells column for heatmap
    plot_data = profiles_df.drop(columns=['n_cells'])
    
    sns.heatmap(
        plot_data,
        cmap='viridis',
        center=None,
        cbar_kws={'label': 'Mean arcsinh intensity'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_xlabel('Marker', fontsize=12)
    ax.set_title(f'Cluster Marker Profiles ({clustering_key})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir:
        save_plot_pdf(fig, Path(save_dir) / f"cluster_profiles_{clustering_key}.pdf", visualize=visualize)
    else:
        if visualize:
            plt.show()
        else:
            plt.close(fig)
    
    # Save table
    if save_dir:
        csv_path = Path(save_dir) / f"cluster_profiles_{clustering_key}.csv"
        profiles_df.to_csv(csv_path)
        print(f"\nSaved cluster profiles: {csv_path}")
    
    print("=" * 70)
    
    return profiles_df


def annotate_clusters(
    sdata: SpatialData,
    table_key: str = "table_intensities",
    clustering_key: str = "leiden_res_0.5",
    annotations: Dict[str, str] = None,
    annotation_column: str = "cell_type",
) -> SpatialData:
    """
    Manually annotate clusters based on marker expression profiles.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object with clustering results
    table_key : str
        Name of intensity table
    clustering_key : str
        Column in .obs with cluster assignments
    annotations : Dict[str, str]
        Mapping from cluster ID to cell type name
        Example: {'0': 'Fibroblasts', '1': 'Endothelial', '2': 'Synoviocytes'}
    annotation_column : str
        Name for new column in .obs (default: 'cell_type')
        
    Returns
    -------
    SpatialData
        Modified SpatialData with annotations in .obs
        
    Examples
    --------
    >>> # After inspecting cluster_profiles, manually create annotations
    >>> annotations = {
    ...     '0': 'PI16+ Fibroblasts',
    ...     '1': 'PRG4+ Synoviocytes',
    ...     '2': 'VWF+ Endothelial',
    ...     '3': 'Mixed/Unknown'
    ... }
    >>> sdata = annotate_clusters(sdata, annotations=annotations)
    """
    
    adata = sdata.tables[table_key]
    
    if clustering_key not in adata.obs:
        raise ValueError(f"Clustering key '{clustering_key}' not found in .obs")
    
    if annotations is None:
        # Default: just copy cluster IDs
        annotations = {str(cid): f'Cluster {cid}' 
                      for cid in adata.obs[clustering_key].unique()}
    
    print("=" * 70)
    print("CLUSTER ANNOTATION")
    print("=" * 70)
    print(f"Clustering: {clustering_key}")
    print(f"Annotations: {len(annotations)} clusters")
    
    # Map cluster IDs to annotations
    cluster_to_type = {str(k): v for k, v in annotations.items()}
    adata.obs[annotation_column] = adata.obs[clustering_key].astype(str).map(cluster_to_type)
    
    # Handle unmapped clusters
    unmapped = adata.obs[annotation_column].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} cells with unmapped clusters, setting to 'Unknown'")
        adata.obs[annotation_column] = adata.obs[annotation_column].fillna('Unknown')
    
    # Make categorical
    adata.obs[annotation_column] = pd.Categorical(adata.obs[annotation_column])
    
    # Print summary
    print(f"\nCell type counts:")
    type_counts = adata.obs[annotation_column].value_counts()
    for cell_type, count in type_counts.items():
        pct = count / len(adata) * 100
        print(f"  {cell_type}: {count} cells ({pct:.1f}%)")
    
    # Store metadata
    adata.uns[f'{annotation_column}_annotations'] = {
        'source_clustering': clustering_key,
        'cluster_to_celltype': cluster_to_type,
        'n_cell_types': adata.obs[annotation_column].nunique()
    }
    
    print(f"\nAnnotations stored in .obs['{annotation_column}']")
    print("=" * 70)
    
    return sdata


def plot_annotated_clusters(
    sdata: SpatialData,
    table_key: str = "table_intensities",
    annotation_column: str = "cell_type",
    shapes_key: str = "cellshapes",
    save_dir: Optional[Path] = None,
    visualize: bool = False,
) -> None:
    """
    Visualize annotated cell types on UMAP and spatially.
    """
    
    adata = sdata.tables[table_key]
    
    if annotation_column not in adata.obs:
        raise ValueError(f"Annotation column '{annotation_column}' not found in .obs")
    
    print("=" * 70)
    print("VISUALISING ANNOTATED CELL TYPES")
    print("=" * 70)
    
    # Plot 1: UMAP
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    sc.pl.umap(
        adata,
        color=annotation_column,
        ax=ax,
        show=False,
        frameon=True,
        title=f'Cell Types (n={adata.obs[annotation_column].nunique()})',
        legend_loc='right margin',
        legend_fontsize=10
    )
    
    plt.tight_layout()
    
    if save_dir:
        save_plot_pdf(fig, Path(save_dir) / f"{annotation_column}_umap.pdf", visualize=visualize)
    else:
        if visualize:
            plt.show()
        else:
            plt.close(fig)
    
    # Plot 2: Spatial
    if shapes_key in sdata.shapes:
        print(f"Creating spatial plot with shapes...")
        
        # Copy annotations to shapes for spatial plotting
        shapes = sdata.shapes[shapes_key]
        shapes[annotation_column] = pd.Categorical(adata.obs[annotation_column].values)
        sdata.shapes[shapes_key] = shapes
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        try:
            sdata.pl.render_shapes(
                shapes_key,
                color=annotation_column,
                fill_alpha=0.8,
                outline_alpha=0.9,
                outline_width=0.5
            ).pl.show(ax=ax, title=f'Cell Types - Spatial')
            
            ax.set_xlabel('X (µm)', fontsize=12)
            ax.set_ylabel('Y (µm)', fontsize=12)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            if save_dir:
                save_plot_pdf(fig, Path(save_dir) / f"{annotation_column}_spatial.pdf", visualize=visualize)
            else:
                if visualize:
                    plt.show()
                else:
                    plt.close(fig)
        
        except Exception as e:
            print(f"Spatial plotting failed: {e}")
            plt.close(fig)
    
    print("=" * 70)
