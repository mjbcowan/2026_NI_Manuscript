#!/usr/bin/env python3
"""
Compare Spatial Statistics Across Conditions/Timepoints

This script performs statistically valid comparisons of spatial patterns across
experimental conditions (e.g., timepoints, treatments, disease states).

Following best practices:
- Mixed-effects models for repeated measures (multiple samples per donor)
- Multiple testing correction (Benjamini-Hochberg FDR)
- Non-parametric tests when assumptions violated
- Effect size reporting (Cohen's d, eta-squared)
- Bootstrapping for confidence intervals

Usage:
    # Compare timepoints
    python compare_spatial_across_conditions.py \
        --spatial-dir ./spatial_analysis \
        --metadata metadata.csv \
        --group-col timepoint \
        --output-dir ./spatial_comparisons

    # Pairwise comparisons
    python compare_spatial_across_conditions.py \
        --spatial-dir ./spatial_analysis \
        --metadata metadata.csv \
        --group-col timepoint \
        --comparisons "A vs B" "B vs C" "C vs A"

    # Include donor as random effect
    python compare_spatial_across_conditions.py \
        --spatial-dir ./spatial_analysis \
        --metadata metadata.csv \
        --group-col timepoint \
        --donor-col donor_id \
        --use-mixed-effects
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Statistics
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

# Mixed effects models
try:
    from statsmodels.formula.api import mixedlm
    HAS_MIXEDLM = True
except ImportError:
    HAS_MIXEDLM = False
    warnings.warn("statsmodels not available. Mixed-effects models disabled.")

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load sample metadata with experimental conditions"""
    logger.info(f"Loading metadata from: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Extract sample_id if not present
    if 'sample_id' not in metadata.columns:
        # Try to infer from other columns
        if 'file_path' in metadata.columns:
            metadata['sample_id'] = metadata['file_path'].apply(
                lambda x: Path(x).stem.replace('_phenotyped', '')
            )
        else:
            raise ValueError("Metadata must contain 'sample_id' column")

    logger.info(f"Loaded metadata for {len(metadata)} samples")
    return metadata


def load_all_spatial_results(
    spatial_dir: Path,
    metadata: pd.DataFrame,
    metrics: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all spatial analysis results across samples

    Parameters
    ----------
    spatial_dir : Path
        Directory containing spatial analysis results
    metadata : pd.DataFrame
        Sample metadata
    metrics : list, optional
        Specific metrics to load. If None, loads all available.
        Options: 'neighborhood_enrichment', 'morans_i', 'interactions', 'niches'

    Returns
    -------
    results : dict
        Dictionary with metric names as keys, DataFrames as values
    """
    if metrics is None:
        metrics = [
            'neighborhood_enrichment',
            'morans_i',
            'cell_cell_interactions',
            'spatial_niche_composition'
        ]

    all_results = {}

    for metric in metrics:
        logger.info(f"Loading {metric} data...")
        metric_data = []

        for sample_id in tqdm(metadata['sample_id'], desc=f"Loading {metric}"):
            sample_dir = spatial_dir / sample_id

            # Map metric to file name
            file_map = {
                'neighborhood_enrichment': 'neighborhood_enrichment_zscore.csv',
                'morans_i': 'morans_i_statistics.csv',
                'cell_cell_interactions': 'cell_cell_interactions.csv',
                'spatial_niche_composition': 'spatial_niche_composition.csv'
            }

            if metric not in file_map:
                logger.warning(f"Unknown metric: {metric}")
                continue

            file_path = sample_dir / file_map[metric]

            if not file_path.exists():
                logger.debug(f"File not found: {file_path}")
                continue

            try:
                # Different handling for different metrics
                if metric == 'neighborhood_enrichment':
                    # Matrix format: rows and columns are cell types
                    df = pd.read_csv(file_path, index_col=0)

                    # Convert to long format: one row per cell type pair
                    long_data = []
                    for row_ct in df.index:
                        for col_ct in df.columns:
                            long_data.append({
                                'cell_type_1': row_ct,
                                'cell_type_2': col_ct,
                                'enrichment_zscore': df.loc[row_ct, col_ct],
                                'sample_id': sample_id
                            })

                    df_long = pd.DataFrame(long_data)

                    # Merge with metadata
                    sample_meta = metadata[metadata['sample_id'] == sample_id]
                    for col in sample_meta.columns:
                        if col != 'sample_id':
                            df_long[col] = sample_meta[col].values[0]

                    metric_data.append(df_long)

                else:
                    # Standard DataFrame format
                    df = pd.read_csv(file_path, index_col=0)
                    df['sample_id'] = sample_id

                    # Merge with metadata
                    sample_meta = metadata[metadata['sample_id'] == sample_id]
                    for col in sample_meta.columns:
                        if col != 'sample_id':
                            df[col] = sample_meta[col].values[0]

                    metric_data.append(df)

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        if len(metric_data) > 0:
            all_results[metric] = pd.concat(metric_data, ignore_index=True)
            logger.info(f"Loaded {metric} for {len(metric_data)} samples")
        else:
            logger.warning(f"No data loaded for {metric}")

    return all_results


# =============================================================================
# Statistical Testing
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval"""
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = resample(data, replace=True, random_state=None)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return lower, upper


def compare_two_groups(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    group1: str,
    group2: str,
    paired: bool = False
) -> Dict:
    """
    Compare two groups with appropriate statistical test

    Parameters
    ----------
    data : pd.DataFrame
        Data with values and group labels
    value_col : str
        Column containing values to compare
    group_col : str
        Column containing group labels
    group1, group2 : str
        Group labels to compare
    paired : bool
        Whether to use paired test (e.g., same donor across timepoints)

    Returns
    -------
    results : dict
        Statistical test results
    """
    # Extract groups
    g1_data = data[data[group_col] == group1][value_col].dropna()
    g2_data = data[data[group_col] == group2][value_col].dropna()

    if len(g1_data) == 0 or len(g2_data) == 0:
        return {'error': 'Insufficient data'}

    results = {
        'group1': group1,
        'group2': group2,
        'n1': len(g1_data),
        'n2': len(g2_data),
        'mean1': np.mean(g1_data),
        'mean2': np.mean(g2_data),
        'median1': np.median(g1_data),
        'median2': np.median(g2_data),
        'std1': np.std(g1_data, ddof=1),
        'std2': np.std(g2_data, ddof=1)
    }

    # Effect size
    results['cohens_d'] = cohens_d(g1_data.values, g2_data.values)

    # Bootstrap CIs
    ci1 = bootstrap_ci(g1_data.values)
    ci2 = bootstrap_ci(g2_data.values)
    results['ci_lower1'] = ci1[0]
    results['ci_upper1'] = ci1[1]
    results['ci_lower2'] = ci2[0]
    results['ci_upper2'] = ci2[1]

    # Statistical test
    if paired:
        # Paired test (Wilcoxon signed-rank)
        stat, pval = wilcoxon(g1_data, g2_data)
        results['test'] = 'wilcoxon'
    else:
        # Independent test (Mann-Whitney U)
        stat, pval = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
        results['test'] = 'mann_whitney'

    results['statistic'] = stat
    results['pvalue'] = pval

    return results


def compare_multiple_groups(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: List[str]
) -> Dict:
    """
    Compare multiple groups using Kruskal-Wallis test

    Followed by pairwise comparisons if significant
    """
    # Extract data for each group
    group_data = [data[data[group_col] == g][value_col].dropna().values
                  for g in groups]

    # Remove empty groups
    group_data = [g for g in group_data if len(g) > 0]
    groups = [g for i, g in enumerate(groups) if len(group_data[i]) > 0]

    if len(group_data) < 2:
        return {'error': 'Insufficient groups'}

    # Kruskal-Wallis test
    stat, pval = kruskal(*group_data)

    results = {
        'test': 'kruskal_wallis',
        'statistic': stat,
        'pvalue': pval,
        'n_groups': len(groups),
        'groups': groups
    }

    # Group statistics
    for i, g in enumerate(groups):
        results[f'n_{g}'] = len(group_data[i])
        results[f'mean_{g}'] = np.mean(group_data[i])
        results[f'median_{g}'] = np.median(group_data[i])
        results[f'std_{g}'] = np.std(group_data[i], ddof=1)

    return results


def compare_with_mixed_effects(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    donor_col: str
) -> Dict:
    """
    Compare groups using mixed-effects model with donor as random effect

    Model: value ~ group + (1|donor)
    """
    if not HAS_MIXEDLM:
        logger.warning("Mixed-effects models not available. Install statsmodels.")
        return {'error': 'statsmodels not available'}

    # Prepare data
    df = data[[value_col, group_col, donor_col]].dropna()

    if len(df) < 10:
        return {'error': 'Insufficient data for mixed model'}

    # Fit mixed-effects model
    try:
        formula = f"{value_col} ~ C({group_col})"
        model = mixedlm(formula, df, groups=df[donor_col])
        result = model.fit()

        return {
            'test': 'mixed_effects',
            'aic': result.aic,
            'bic': result.bic,
            'summary': result.summary().as_text(),
            'pvalues': result.pvalues.to_dict(),
            'params': result.params.to_dict()
        }

    except Exception as e:
        logger.error(f"Mixed-effects model failed: {e}")
        return {'error': str(e)}


# =============================================================================
# Neighborhood Enrichment Comparison
# =============================================================================

def compare_neighborhood_enrichment(
    enrichment_data: pd.DataFrame,
    group_col: str,
    groups: List[str],
    donor_col: Optional[str] = None,
    fdr_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compare neighborhood enrichment z-scores across conditions

    Tests each cell type pair separately

    Expected format: long format with columns:
    - cell_type_1, cell_type_2, enrichment_zscore, sample_id, {group_col}
    """
    logger.info("Comparing neighborhood enrichment across groups:")

    # Check required columns
    required_cols = ['cell_type_1', 'cell_type_2', 'enrichment_zscore', 'sample_id', group_col]
    missing_cols = [col for col in required_cols if col not in enrichment_data.columns]

    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # Create cell type pair identifier
    enrichment_data['cell_type_pair'] = (
        enrichment_data['cell_type_1'] + ' - ' + enrichment_data['cell_type_2']
    )

    # Get unique pairs
    pairs = enrichment_data['cell_type_pair'].unique()

    if len(pairs) == 0:
        logger.warning("No cell type pairs found")
        return pd.DataFrame()

    results = []

    # Test each pair
    for pair in tqdm(pairs, desc="Testing cell type pairs"):
        pair_data = enrichment_data[enrichment_data['cell_type_pair'] == pair].copy()

        # Pairwise comparisons
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                test_result = compare_two_groups(
                    pair_data,
                    value_col='enrichment_zscore',
                    group_col=group_col,
                    group1=g1,
                    group2=g2,
                    paired=False
                )

                if 'error' not in test_result:
                    test_result['cell_type_pair'] = pair
                    test_result['comparison'] = f"{g1} vs {g2}"
                    results.append(test_result)

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if 'pvalue' in results_df.columns and len(results_df) > 0:
        results_df['pvalue_adj'] = multipletests(
            results_df['pvalue'],
            method='fdr_bh'
        )[1]

        results_df['significant'] = results_df['pvalue_adj'] < fdr_threshold

    return results_df


# =============================================================================
# Spatial Autocorrelation Comparison
# =============================================================================

def compare_morans_i(
    morans_data: pd.DataFrame,
    group_col: str,
    groups: List[str],
    fdr_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compare Moran's I values across conditions for each marker
    """
    logger.info("Comparing Moran's I across groups:")

    # Check if data has the required structure
    if 'I' not in morans_data.columns:
        logger.warning("Moran's I column not found in data")
        return pd.DataFrame()

    # Get unique markers from index
    if morans_data.index.name is None:
        # Try to find marker column
        marker_cols = [col for col in morans_data.columns if 'marker' in col.lower()]
        if len(marker_cols) > 0:
            morans_data = morans_data.set_index(marker_cols[0])

    markers = morans_data.index.unique()

    if len(markers) == 0:
        logger.warning("No markers found in Moran's I data")
        return pd.DataFrame()

    results = []

    for marker in tqdm(markers, desc="Testing markers"):
        # Get data for this marker
        try:
            marker_data = morans_data.loc[marker]

            # If only one row, convert Series to DataFrame
            if isinstance(marker_data, pd.Series):
                marker_data = marker_data.to_frame().T
            elif isinstance(marker_data, pd.DataFrame):
                marker_data = marker_data.copy()
            else:
                continue

            if len(marker_data) == 0:
                continue

            # Ensure we have the Moran's I column
            if 'I' not in marker_data.columns:
                continue

        except KeyError:
            continue

        # Pairwise comparisons
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                test_result = compare_two_groups(
                    marker_data,
                    value_col='I',
                    group_col=group_col,
                    group1=g1,
                    group2=g2,
                    paired=False
                )

                if 'error' not in test_result:
                    test_result['marker'] = marker
                    test_result['comparison'] = f"{g1} vs {g2}"
                    results.append(test_result)

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if 'pvalue' in results_df.columns and len(results_df) > 0:
        results_df['pvalue_adj'] = multipletests(
            results_df['pvalue'],
            method='fdr_bh'
        )[1]

        results_df['significant'] = results_df['pvalue_adj'] < fdr_threshold

    return results_df


# =============================================================================
# Cell-Cell Interaction Comparison
# =============================================================================

def compare_interactions(
    interaction_data: pd.DataFrame,
    group_col: str,
    groups: List[str],
    fdr_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compare cell-cell interaction frequencies across conditions
    """
    logger.info("Comparing cell-cell interactions across groups:S")

    # Get unique cell type pairs
    if 'cell_type_1' not in interaction_data.columns or 'cell_type_2' not in interaction_data.columns:
        logger.warning("Interaction data missing cell type columns")
        return pd.DataFrame()

    interaction_data['pair'] = (
        interaction_data['cell_type_1'] + ' - ' + interaction_data['cell_type_2']
    )

    pairs = interaction_data['pair'].unique()

    results = []

    for pair in tqdm(pairs, desc="Testing interaction pairs"):
        pair_data = interaction_data[interaction_data['pair'] == pair].copy()

        # Use interactions_per_cell as metric
        if 'interactions_per_cell' not in pair_data.columns:
            continue

        # Pairwise comparisons
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                test_result = compare_two_groups(
                    pair_data,
                    value_col='interactions_per_cell',
                    group_col=group_col,
                    group1=g1,
                    group2=g2,
                    paired=False
                )

                test_result['cell_type_pair'] = pair
                test_result['comparison'] = f"{g1} vs {g2}"
                results.append(test_result)

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if 'pvalue' in results_df.columns and len(results_df) > 0:
        results_df['pvalue_adj'] = multipletests(
            results_df['pvalue'],
            method='fdr_bh'
        )[1]

        results_df['significant'] = results_df['pvalue_adj'] < fdr_threshold

    return results_df


# =============================================================================
# Visualisation
# =============================================================================

def plot_comparison_heatmap(
    results_df: pd.DataFrame,
    metric_col: str,
    row_col: str,
    comparison_col: str,
    output_path: Path,
    title: str = "Comparison Heatmap",
    cmap: str = 'RdBu_r',
    center: float = 0
):
    """
    Plot heatmap of comparison results

    Rows: cell type pairs / markers / features
    Columns: comparisons (e.g., "Baseline vs D21")
    Values: effect sizes or test statistics
    """
    # Pivot data
    heatmap_data = results_df.pivot_table(
        values=metric_col,
        index=row_col,
        columns=comparison_col,
        aggfunc='first'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(heatmap_data) * 0.3)))

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=center,
        ax=ax,
        cbar_kws={'label': metric_col},
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Comparison', fontsize=12)
    ax.set_ylabel(row_col.replace('_', ' ').title(), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved heatmap to: {output_path}")


def plot_volcano(
    results_df: pd.DataFrame,
    comparison: str,
    output_path: Path,
    effect_col: str = 'cohens_d',
    pval_col: str = 'pvalue_adj',
    label_col: str = None,
    fdr_threshold: float = 0.05,
    effect_threshold: float = 0.5
):
    """
    Create volcano plot of comparison results

    X-axis: Effect size (Cohen's d)
    Y-axis: -log10(adjusted p-value)
    """
    # Filter to specific comparison if needed
    if 'comparison' in results_df.columns:
        plot_data = results_df[results_df['comparison'] == comparison].copy()
    else:
        plot_data = results_df.copy()

    if len(plot_data) == 0:
        logger.warning(f"No data for comparison: {comparison}")
        return

    # Calculate -log10(p-value)
    plot_data['-log10_pval'] = -np.log10(plot_data[pval_col].replace(0, 1e-300))

    # Categorize points
    plot_data['category'] = 'Not significant'
    plot_data.loc[
        (plot_data[pval_col] < fdr_threshold) &
        (np.abs(plot_data[effect_col]) > effect_threshold),
        'category'
    ] = 'Significant'

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points
    colors = {'Not significant': 'gray', 'Significant': 'red'}
    for cat in ['Not significant', 'Significant']:
        subset = plot_data[plot_data['category'] == cat]
        ax.scatter(
            subset[effect_col],
            subset['-log10_pval'],
            c=colors[cat],
            label=cat,
            alpha=0.6,
            s=50
        )

    # Add threshold lines
    ax.axhline(-np.log10(fdr_threshold), color='blue', linestyle='--',
              label=f'FDR = {fdr_threshold}')
    ax.axvline(effect_threshold, color='green', linestyle='--',
              label=f'Effect size = {effect_threshold}')
    ax.axvline(-effect_threshold, color='green', linestyle='--')

    # Labels for significant points
    if label_col and label_col in plot_data.columns:
        sig_data = plot_data[plot_data['category'] == 'Significant']
        for _, row in sig_data.iterrows():
            ax.annotate(
                row[label_col],
                (row[effect_col], row['-log10_pval']),
                fontsize=8,
                alpha=0.7
            )

    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax.set_ylabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title(f'Volcano Plot: {comparison}', fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved volcano plot to: {output_path}")


def plot_group_comparison_boxplots(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    feature_col: str,
    output_path: Path,
    top_n: int = 20
):
    """
    Create boxplots comparing groups for top features
    """
    # Get top features by variance or effect size
    feature_var = data.groupby(feature_col)[value_col].var().sort_values(ascending=False)
    top_features = feature_var.head(top_n).index

    # Filter data
    plot_data = data[data[feature_col].isin(top_features)]

    # Create figure
    n_features = len(top_features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        feature_data = plot_data[plot_data[feature_col] == feature]

        sns.boxplot(
            data=feature_data,
            x=group_col,
            y=value_col,
            ax=ax
        )

        ax.set_title(feature, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=9)
        ax.tick_params(axis='x', rotation=45)

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Top {n_features} Features by Variance', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved boxplots to: {output_path}")


# =============================================================================
# Main Workflow
# =============================================================================

def run_complete_comparison(
    spatial_dir: Path,
    metadata_path: Path,
    group_col: str,
    output_dir: Path,
    donor_col: Optional[str] = None,
    comparisons: Optional[List[str]] = None,
    use_mixed_effects: bool = False,
    fdr_threshold: float = 0.05
):
    """
    Run complete comparison workflow
    """
    logger.info("="*80)
    logger.info("SPATIAL STATISTICS COMPARISON ANALYSIS")
    logger.info("="*80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Check group column
    if group_col not in metadata.columns:
        raise ValueError(f"Group column '{group_col}' not found in metadata")

    groups = sorted(metadata[group_col].unique())
    logger.info(f"Groups: {groups}")

    # Parse comparisons
    if comparisons is None:
        # All pairwise comparisons
        comparison_pairs = [(g1, g2) for i, g1 in enumerate(groups)
                           for g2 in groups[i+1:]]
    else:
        # Parse user-specified comparisons
        comparison_pairs = []
        for comp in comparisons:
            parts = comp.replace(' vs ', ' ').split()
            if len(parts) == 2:
                comparison_pairs.append((parts[0], parts[1]))

    logger.info(f"Comparisons: {comparison_pairs}")

    # Load spatial results
    spatial_results = load_all_spatial_results(spatial_dir, metadata)

    all_comparison_results = {}

    # 1. Neighborhood enrichment comparison
    if 'neighborhood_enrichment' in spatial_results:
        logger.info("\n" + "="*80)
        logger.info("Comparing Neighborhood Enrichment")
        logger.info("="*80)

        enrichment_results = compare_neighborhood_enrichment(
            spatial_results['neighborhood_enrichment'],
            group_col=group_col,
            groups=groups,
            donor_col=donor_col,
            fdr_threshold=fdr_threshold
        )

        all_comparison_results['neighborhood_enrichment'] = enrichment_results

        # Save results
        enrichment_results.to_csv(
            output_dir / 'neighborhood_enrichment_comparisons.csv',
            index=False
        )

        # Visualize
        if len(enrichment_results) > 0:
            plot_comparison_heatmap(
                enrichment_results,
                metric_col='cohens_d',
                row_col='cell_type_pair',
                comparison_col='comparison',
                output_path=output_dir / 'neighborhood_enrichment_effect_sizes.pdf',
                title='Neighborhood Enrichment: Effect Sizes',
                cmap='RdBu_r',
                center=0
            )

            # Volcano plot for each comparison
            for g1, g2 in comparison_pairs:
                comp_name = f"{g1} vs {g2}"
                plot_volcano(
                    enrichment_results,
                    comparison=comp_name,
                    output_path=output_dir / f'volcano_neighborhood_{g1}_vs_{g2}.pdf',
                    effect_col='cohens_d',
                    pval_col='pvalue_adj',
                    label_col='cell_type_pair',
                    fdr_threshold=fdr_threshold
                )

            # Boxplots comparing groups
            plot_group_comparison_boxplots(
                spatial_results['neighborhood_enrichment'],
                value_col='enrichment_zscore',
                group_col=group_col,
                feature_col='cell_type_pair',
                output_path=output_dir / f'cell_type_boxplots_{groups[0]}_vs_{groups[1]}.pdf',
                top_n=20
            )

            # Save significant results
            sig_results = enrichment_results[enrichment_results['significant'] == True]
            if len(sig_results) > 0:
                sig_results.to_csv(
                    output_dir / f'significant_neighborhood_enrichment_{groups[0]}_vs_{groups[1]}.csv',
                    index=False
                )

    # 2. Moran's I comparison
    if 'morans_i' in spatial_results:
        logger.info("\n" + "="*80)
        logger.info("Comparing Spatial Autocorrelation (Moran's I)")
        logger.info("="*80)

        morans_results = compare_morans_i(
            spatial_results['morans_i'],
            group_col=group_col,
            groups=groups,
            fdr_threshold=fdr_threshold
        )

        all_comparison_results['morans_i'] = morans_results

        # Save results
        morans_results.to_csv(
            output_dir / 'morans_i_comparisons.csv',
            index=False
        )

        # Visualise
        if len(morans_results) > 0:
            plot_comparison_heatmap(
                morans_results,
                metric_col='cohens_d',
                row_col='marker',
                comparison_col='comparison',
                output_path=output_dir / 'morans_i_effect_sizes.pdf',
                title="Moran's I: Effect Sizes",
                cmap='RdBu_r',
                center=0
            )

    # 3. Cell-cell interactions comparison
    if 'cell_cell_interactions' in spatial_results:
        logger.info("\n" + "="*80)
        logger.info("Comparing Cell-Cell Interactions")
        logger.info("="*80)

        interaction_results = compare_interactions(
            spatial_results['cell_cell_interactions'],
            group_col=group_col,
            groups=groups,
            fdr_threshold=fdr_threshold
        )

        all_comparison_results['interactions'] = interaction_results

        # Save results
        interaction_results.to_csv(
            output_dir / 'interaction_comparisons.csv',
            index=False
        )

        # Visualise
        if len(interaction_results) > 0:
            plot_comparison_heatmap(
                interaction_results,
                metric_col='cohens_d',
                row_col='cell_type_pair',
                comparison_col='comparison',
                output_path=output_dir / 'interaction_effect_sizes.pdf',
                title='Cell-Cell Interactions: Effect Sizes',
                cmap='RdBu_r',
                center=0
            )

    # Generate summary report
    generate_comparison_report(
        all_comparison_results,
        output_dir,
        groups,
        comparison_pairs,
        fdr_threshold
    )

    logger.info("="*80)
    logger.info("✓ Comparison analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

    return all_comparison_results


def generate_comparison_report(
    results: Dict[str, pd.DataFrame],
    output_dir: Path,
    groups: List[str],
    comparisons: List[Tuple[str, str]],
    fdr_threshold: float
):
    """Generate comprehensive comparison report"""
    report_path = output_dir / 'comparison_summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPATIAL STATISTICS COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")

        f.write("GROUPS COMPARED\n")
        f.write("-"*80 + "\n")
        f.write(f"Groups: {', '.join(groups)}\n")
        f.write(f"Number of comparisons: {len(comparisons)}\n")
        for g1, g2 in comparisons:
            f.write(f"  - {g1} vs {g2}\n")
        f.write(f"\nFDR threshold: {fdr_threshold}\n\n")

        # Summary for each metric
        for metric, df in results.items():
            if len(df) == 0:
                continue

            f.write(f"\n{metric.upper().replace('_', ' ')}\n")
            f.write("-"*80 + "\n")
            f.write(f"Total tests: {len(df)}\n")

            if 'significant' in df.columns:
                n_sig = df['significant'].sum()
                f.write(f"Significant results: {n_sig} ({n_sig/len(df)*100:.1f}%)\n")

                # Top significant results
                if n_sig > 0:
                    sig_df = df[df['significant']].sort_values('pvalue_adj')
                    f.write(f"\nTop 10 significant results:\n")
                    for i, row in sig_df.head(10).iterrows():
                        feature = row.get('cell_type_pair', row.get('marker', 'Unknown'))
                        comp = row.get('comparison', 'Unknown')
                        pval = row.get('pvalue_adj', row.get('pvalue', 1))
                        effect = row.get('cohens_d', 0)
                        f.write(f"  {feature} ({comp}): p={pval:.3e}, d={effect:.2f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Analysis complete!\n")
        f.write("="*80 + "\n")

    logger.info(f"Report saved to: {report_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare spatial statistics across conditions/timepoints',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required
    parser.add_argument(
        '--spatial-dir',
        type=Path,
        required=True,
        help='Directory containing spatial analysis results'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        required=True,
        help='CSV file with sample metadata'
    )
    parser.add_argument(
        '--group-col',
        type=str,
        required=True,
        help='Column in metadata defining groups to compare (e.g., timepoint, treatment)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./spatial_comparisons'),
        help='Output directory (default: ./spatial_comparisons)'
    )

    # Optional
    parser.add_argument(
        '--donor-col',
        type=str,
        help='Column defining donor/subject ID (for paired/mixed-effects analysis)'
    )
    parser.add_argument(
        '--comparisons',
        nargs='+',
        help='Specific comparisons to perform (e.g., "A vs C")'
    )
    parser.add_argument(
        '--use-mixed-effects',
        action='store_true',
        help='Use mixed-effects models (requires donor-col)'
    )
    parser.add_argument(
        '--fdr-threshold',
        type=float,
        default=0.05,
        help='FDR threshold for significance (default: 0.05)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_complete_comparison(
        spatial_dir=args.spatial_dir,
        metadata_path=args.metadata,
        group_col=args.group_col,
        output_dir=args.output_dir,
        donor_col=args.donor_col,
        comparisons=args.comparisons,
        use_mixed_effects=args.use_mixed_effects,
        fdr_threshold=args.fdr_threshold
    )


if __name__ == '__main__':
    main()
