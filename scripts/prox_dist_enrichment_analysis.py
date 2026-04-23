#!/usr/bin/env python3
"""
Cell Type Enrichment Comparison: Proximal vs Distal Analysis

Compares the proportion of PI16 Fibroblast and Vascular Endothelium cells
between proximal and distal sites using mixed effects models with:
- Donor as random effect
- Paired sample structure accounted for
- FDR correction for multiple comparisons

Usage:
    python prox_dist_enrichment_analysis.py --data-dir ./data/processed/prox_dist
    python prox_dist_enrichment_analysis.py --data-dir /path/to/prox_dist --output-dir ./results
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spatialdata as sd
from scipy.stats import wilcoxon
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PI16 Fibroblast and Vascular Endothelium enrichment: proximal vs distal"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/prox_dist"),
        help="Directory containing processed zarr files and metadata.csv (default: data/processed/prox_dist)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: <data-dir>/enrichment_analysis)",
    )
    return parser.parse_args()


args = parse_args()
DATA_DIR = args.data_dir
METADATA_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = args.output_dir if args.output_dir is not None else DATA_DIR / "enrichment_analysis"

CELL_TYPES = ["PI16 Fibroblast", "Vascular Endothelium"]
TISSUE_REGIONS = [
    "pooled",
    "bone tissue",
    "dense regular connective tissue",
    "loose connective tissue",
    "skin epithelium",
]


def logit_transform(p: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Logit transform with boundary correction for proportions.

    Parameters
    ----------
    p : array-like
        Proportions (values between 0 and 1)
    epsilon : float
        Small value to prevent log(0) or log(1)

    Returns
    -------
    array-like
        Logit-transformed values
    """
    p_clipped = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p_clipped / (1 - p_clipped))


def extract_cell_proportions(zarr_path: Path) -> pd.DataFrame:
    """
    Extract cell type proportions per tissue region from a zarr file.

    Parameters
    ----------
    zarr_path : Path
        Path to the zarr file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: tissue_region, cell_type, n_target_cells,
        n_total_cells, proportion
    """
    sdata = sd.read_zarr(zarr_path)
    adata = sdata.tables["table_intensities"]

    # Filter QC-passed cells
    if "is_qc_pass" in adata.obs.columns:
        cells = adata.obs[adata.obs["is_qc_pass"] == True].copy()
    else:
        cells = adata.obs.copy()

    results = []

    # Get all tissue regions (excluding 'unassigned')
    tissue_col = "tissue_region_uberon_label"
    all_tissues = cells[tissue_col].unique()
    tissues_to_analyze = [t for t in all_tissues if t != "unassigned"]

    # Per-tissue region analysis
    for tissue in tissues_to_analyze:
        tissue_mask = cells[tissue_col] == tissue
        tissue_cells = cells[tissue_mask]
        n_total = len(tissue_cells)

        if n_total == 0:
            continue

        for cell_type in CELL_TYPES:
            n_target = (tissue_cells["leiden_cell_type"] == cell_type).sum()
            proportion = n_target / n_total if n_total > 0 else 0

            results.append(
                {
                    "tissue_region": tissue,
                    "cell_type": cell_type,
                    "n_target_cells": int(n_target),
                    "n_total_cells": int(n_total),
                    "proportion": proportion,
                }
            )

    # Pooled analysis (all regions combined, excluding unassigned)
    pooled_cells = cells[cells[tissue_col] != "unassigned"]
    n_total_pooled = len(pooled_cells)

    for cell_type in CELL_TYPES:
        n_target = (pooled_cells["leiden_cell_type"] == cell_type).sum()
        results.append(
            {
                "tissue_region": "pooled",
                "cell_type": cell_type,
                "n_target_cells": int(n_target),
                "n_total_cells": int(n_total_pooled),
                "proportion": n_target / n_total_pooled if n_total_pooled > 0 else 0,
            }
        )

    return pd.DataFrame(results)


def build_analysis_dataframe(data_dir: Path, metadata_path: Path) -> pd.DataFrame:
    """
    Build complete analysis DataFrame from all zarr files.

    Parameters
    ----------
    data_dir : Path
        Directory containing zarr files
    metadata_path : Path
        Path to metadata CSV file

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all samples and metadata
    """
    metadata = pd.read_csv(metadata_path)
    all_data = []

    for _, row in metadata.iterrows():
        sample_id = row["sample_id"]
        zarr_path = data_dir / f"{sample_id}.zarr"

        if not zarr_path.exists():
            print(f"Warning: {zarr_path} not found, skipping...")
            continue

        print(f"Processing: {sample_id}")

        # Extract proportions
        try:
            props_df = extract_cell_proportions(zarr_path)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue

        # Add metadata
        props_df["sample_id"] = sample_id
        props_df["donor_id"] = str(row["donor_id"])
        props_df["location"] = row["location"]

        # Create sample_pair_id by removing location suffix
        sample_pair_id = sample_id.replace("_proximal", "").replace("_distal", "")
        props_df["sample_pair_id"] = sample_pair_id

        all_data.append(props_df)

    return pd.concat(all_data, ignore_index=True)


def run_mixed_effects_model(
    df: pd.DataFrame, cell_type: str, tissue_region: str
) -> dict:
    """
    Run mixed effects model for a specific cell type and tissue region.

    Model: logit(proportion) ~ location + (1|donor_id) + (1|sample_pair_id)

    Parameters
    ----------
    df : pd.DataFrame
        Analysis DataFrame
    cell_type : str
        Cell type to analyze
    tissue_region : str
        Tissue region to analyze

    Returns
    -------
    dict
        Results dictionary with estimates, p-values, etc.
    """
    # Filter data
    subset = df[
        (df["cell_type"] == cell_type) & (df["tissue_region"] == tissue_region)
    ].copy()

    if len(subset) < 6:
        return {"error": "Insufficient data", "n_samples": len(subset)}

    # Check for both locations
    if len(subset["location"].unique()) < 2:
        return {"error": "Only one location present", "n_samples": len(subset)}

    # Apply logit transformation
    subset["logit_prop"] = logit_transform(subset["proportion"].values)

    # Check for sufficient variance
    if subset["logit_prop"].std() < 1e-6:
        return {"error": "Insufficient variance", "n_samples": len(subset)}

    # Try fitting mixed model with nested random effects
    try:
        # Model with donor and sample_pair as random effects
        # Using variance components for sample_pair nested within donor
        model = mixedlm(
            "logit_prop ~ C(location, Treatment('proximal'))",
            data=subset,
            groups=subset["donor_id"],
            vc_formula={"sample_pair_id": "0 + C(sample_pair_id)"},
        )
        result = model.fit(method="powell", maxiter=1000)

        # Extract coefficient for distal (relative to proximal)
        coef_name = "C(location, Treatment('proximal'))[T.distal]"

        return {
            "cell_type": cell_type,
            "tissue_region": tissue_region,
            "n_samples": len(subset),
            "n_proximal": (subset["location"] == "proximal").sum(),
            "n_distal": (subset["location"] == "distal").sum(),
            "mean_proximal": subset[subset["location"] == "proximal"][
                "proportion"
            ].mean(),
            "mean_distal": subset[subset["location"] == "distal"]["proportion"].mean(),
            "log_odds_ratio": result.params.get(coef_name, np.nan),
            "se": result.bse.get(coef_name, np.nan),
            "pvalue": result.pvalues.get(coef_name, np.nan),
            "converged": result.converged,
            "model_type": "nested_lmm",
        }

    except Exception as e:
        # Fallback to simpler model without nested random effects
        try:
            model = mixedlm(
                "logit_prop ~ C(location, Treatment('proximal'))",
                data=subset,
                groups=subset["donor_id"],
            )
            result = model.fit(method="powell", maxiter=1000)

            coef_name = "C(location, Treatment('proximal'))[T.distal]"

            return {
                "cell_type": cell_type,
                "tissue_region": tissue_region,
                "n_samples": len(subset),
                "n_proximal": (subset["location"] == "proximal").sum(),
                "n_distal": (subset["location"] == "distal").sum(),
                "mean_proximal": subset[subset["location"] == "proximal"][
                    "proportion"
                ].mean(),
                "mean_distal": subset[subset["location"] == "distal"][
                    "proportion"
                ].mean(),
                "log_odds_ratio": result.params.get(coef_name, np.nan),
                "se": result.bse.get(coef_name, np.nan),
                "pvalue": result.pvalues.get(coef_name, np.nan),
                "converged": result.converged,
                "model_type": "simple_lmm",
            }
        except Exception as e2:
            return {"error": str(e2), "n_samples": len(subset)}


def run_paired_wilcoxon(df: pd.DataFrame, cell_type: str, tissue_region: str) -> dict:
    """
    Paired Wilcoxon signed-rank test as non-parametric alternative.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis DataFrame
    cell_type : str
        Cell type to analyze
    tissue_region : str
        Tissue region to analyze

    Returns
    -------
    dict
        Results dictionary
    """
    subset = df[
        (df["cell_type"] == cell_type) & (df["tissue_region"] == tissue_region)
    ].copy()

    # Pivot to get paired data
    try:
        paired = subset.pivot(
            index="sample_pair_id", columns="location", values="proportion"
        ).dropna()
    except Exception:
        return {"error": "Could not pivot data"}

    if len(paired) < 5:
        return {"error": "Insufficient pairs", "n_pairs": len(paired)}

    if "proximal" not in paired.columns or "distal" not in paired.columns:
        return {"error": "Missing location columns"}

    proximal = paired["proximal"].values
    distal = paired["distal"].values

    # Wilcoxon signed-rank test
    try:
        stat, pvalue = wilcoxon(proximal, distal, alternative="two-sided")
    except Exception as e:
        return {"error": str(e)}

    return {
        "cell_type": cell_type,
        "tissue_region": tissue_region,
        "n_pairs": len(paired),
        "median_proximal": np.median(proximal),
        "median_distal": np.median(distal),
        "median_diff": np.median(distal - proximal),
        "statistic": stat,
        "pvalue": pvalue,
    }


def run_complete_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run analysis for all cell types and tissue regions with FDR correction.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis DataFrame

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Mixed effects results and Wilcoxon results
    """
    # Get tissue regions present in data
    tissue_regions = ["pooled"] + [
        t for t in df["tissue_region"].unique() if t != "pooled" and t != "unassigned"
    ]

    lmm_results = []
    wilcoxon_results = []

    for cell_type in CELL_TYPES:
        for tissue in tissue_regions:
            print(f"Analyzing: {cell_type} in {tissue}")

            # Mixed effects model
            lmm_result = run_mixed_effects_model(df, cell_type, tissue)
            if "error" not in lmm_result:
                lmm_results.append(lmm_result)
            else:
                print(f"  LMM error: {lmm_result.get('error')}")

            # Wilcoxon test
            wilcoxon_result = run_paired_wilcoxon(df, cell_type, tissue)
            if "error" not in wilcoxon_result:
                wilcoxon_results.append(wilcoxon_result)
            else:
                print(f"  Wilcoxon error: {wilcoxon_result.get('error')}")

    lmm_df = pd.DataFrame(lmm_results)
    wilcoxon_df = pd.DataFrame(wilcoxon_results)

    # Apply FDR correction to LMM results
    if len(lmm_df) > 0 and "pvalue" in lmm_df.columns:
        valid_pvals = lmm_df["pvalue"].notna()
        if valid_pvals.sum() > 0:
            _, pvals_adj, _, _ = multipletests(
                lmm_df.loc[valid_pvals, "pvalue"], method="fdr_bh", alpha=0.05
            )
            lmm_df.loc[valid_pvals, "pvalue_fdr"] = pvals_adj
            lmm_df["significant"] = lmm_df["pvalue_fdr"] < 0.05

    # Apply FDR correction to Wilcoxon results
    if len(wilcoxon_df) > 0 and "pvalue" in wilcoxon_df.columns:
        valid_pvals = wilcoxon_df["pvalue"].notna()
        if valid_pvals.sum() > 0:
            _, pvals_adj, _, _ = multipletests(
                wilcoxon_df.loc[valid_pvals, "pvalue"], method="fdr_bh", alpha=0.05
            )
            wilcoxon_df.loc[valid_pvals, "pvalue_fdr"] = pvals_adj
            wilcoxon_df["significant"] = wilcoxon_df["pvalue_fdr"] < 0.05

    return lmm_df, wilcoxon_df


def get_significance_label(pvalue: float) -> str:
    """Convert p-value to significance label."""
    if pd.isna(pvalue):
        return ""
    if pvalue <= 0.0001:
        return "****"
    if pvalue <= 0.001:
        return "***"
    if pvalue <= 0.01:
        return "**"
    if pvalue <= 0.05:
        return "*"
    return "ns"


def plot_cell_type_comparison(
    df: pd.DataFrame,
    cell_type: str,
    results_df: pd.DataFrame,
    output_path: Path,
):
    """
    Create publication-quality box plot with statistical annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis DataFrame
    cell_type : str
        Cell type to plot
    results_df : pd.DataFrame
        Statistical results DataFrame with FDR-corrected p-values
    output_path : Path
        Output file path
    """
    # Filter data
    plot_df = df[df["cell_type"] == cell_type].copy()

    # Order tissue regions
    tissue_order = [
        t for t in TISSUE_REGIONS if t in plot_df["tissue_region"].unique()
    ]

    if len(tissue_order) == 0:
        print(f"No data for {cell_type}")
        return

    # Set up figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Color palette
    palette = {"proximal": "#3498db", "distal": "#e74c3c"}

    # Box plot
    sns.boxplot(
        data=plot_df,
        x="tissue_region",
        y="proportion",
        hue="location",
        order=tissue_order,
        hue_order=["proximal", "distal"],
        palette=palette,
        ax=ax,
        width=0.6,
        showfliers=False,
        linewidth=1.5,
    )

    # Strip plot overlay
    sns.stripplot(
        data=plot_df,
        x="tissue_region",
        y="proportion",
        hue="location",
        order=tissue_order,
        hue_order=["proximal", "distal"],
        palette=palette,
        dodge=True,
        ax=ax,
        alpha=0.7,
        size=6,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    # Add pairing lines
    for i, tissue in enumerate(tissue_order):
        tissue_data = plot_df[plot_df["tissue_region"] == tissue]
        try:
            paired = tissue_data.pivot(
                index="sample_pair_id", columns="location", values="proportion"
            ).dropna()

            if len(paired) > 0:
                x_prox = i - 0.15
                x_dist = i + 0.15

                for _, row in paired.iterrows():
                    ax.plot(
                        [x_prox, x_dist],
                        [row["proximal"], row["distal"]],
                        "k-",
                        alpha=0.3,
                        linewidth=0.8,
                    )
        except Exception:
            continue

    # Add significance annotations
    # First, calculate the global y_max for proper axis scaling
    global_y_max = plot_df["proportion"].max()
    annotation_y_positions = []

    if results_df is not None and len(results_df) > 0:
        cell_results = results_df[results_df["cell_type"] == cell_type]

        for i, tissue in enumerate(tissue_order):
            tissue_result = cell_results[cell_results["tissue_region"] == tissue]
            if len(tissue_result) > 0:
                pval_fdr = tissue_result["pvalue_fdr"].values[0]
                sig_label = get_significance_label(pval_fdr)

                if sig_label:
                    # Get y position for annotation - use per-tissue max
                    tissue_data = plot_df[plot_df["tissue_region"] == tissue]
                    y_max = tissue_data["proportion"].max()
                    # Use a proportion of the global max for consistent offset
                    y_offset = global_y_max * 0.05
                    annotation_y = y_max + y_offset
                    annotation_y_positions.append(annotation_y)

                    ax.text(
                        i,
                        annotation_y,
                        sig_label,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                    )

    # Set y-axis limits to accommodate annotations
    y_min = 0
    if annotation_y_positions:
        # Add padding above the highest annotation
        y_upper = max(annotation_y_positions) + global_y_max * 0.08
    else:
        y_upper = global_y_max * 1.1
    ax.set_ylim(y_min, y_upper)

    # Formatting
    ax.set_xlabel("Tissue Region", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{cell_type} Proportion", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{cell_type}: Proximal vs Distal Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        ["Proximal", "Distal"],
        title="Location",
        loc="upper right",
        framealpha=0.9,
    )

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def generate_summary_report(
    lmm_results: pd.DataFrame, wilcoxon_results: pd.DataFrame, output_path: Path
):
    """Generate text summary report."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CELL TYPE ENRICHMENT ANALYSIS: PROXIMAL vs DISTAL\n")
        f.write("=" * 70 + "\n\n")

        f.write("STATISTICAL METHODS\n")
        f.write("-" * 40 + "\n")
        f.write("Primary: Linear Mixed Effects Model\n")
        f.write("  - Formula: logit(proportion) ~ location + (1|donor_id)\n")
        f.write("  - Multiple testing: Benjamini-Hochberg FDR\n")
        f.write("Sensitivity: Paired Wilcoxon signed-rank test\n\n")

        f.write("MIXED EFFECTS MODEL RESULTS\n")
        f.write("-" * 40 + "\n")
        if len(lmm_results) > 0:
            for _, row in lmm_results.iterrows():
                sig_marker = "*" if row.get("significant", False) else ""
                f.write(f"\n{row['cell_type']} in {row['tissue_region']}:\n")
                f.write(f"  Mean Proximal: {row['mean_proximal']*100:.2f}%\n")
                f.write(f"  Mean Distal:   {row['mean_distal']*100:.2f}%\n")
                f.write(f"  Log Odds Ratio: {row['log_odds_ratio']:.3f}\n")
                f.write(f"  P-value (raw):  {row['pvalue']:.4f}\n")
                f.write(f"  P-value (FDR):  {row.get('pvalue_fdr', np.nan):.4f} {sig_marker}\n")
        else:
            f.write("No results available.\n")

        f.write("\n\nWILCOXON SIGNED-RANK TEST RESULTS (Sensitivity Analysis)\n")
        f.write("-" * 40 + "\n")
        if len(wilcoxon_results) > 0:
            for _, row in wilcoxon_results.iterrows():
                sig_marker = "*" if row.get("significant", False) else ""
                f.write(f"\n{row['cell_type']} in {row['tissue_region']}:\n")
                f.write(f"  N pairs: {row['n_pairs']}\n")
                f.write(f"  Median Proximal: {row['median_proximal']*100:.2f}%\n")
                f.write(f"  Median Distal:   {row['median_distal']*100:.2f}%\n")
                f.write(f"  Median Diff:     {row['median_diff']*100:.2f}%\n")
                f.write(f"  P-value (raw):   {row['pvalue']:.4f}\n")
                f.write(f"  P-value (FDR):   {row.get('pvalue_fdr', np.nan):.4f} {sig_marker}\n")
        else:
            f.write("No results available.\n")

        f.write("\n\n* indicates FDR-corrected p-value < 0.05\n")

    print(f"Saved: {output_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Cell Type Enrichment Analysis: Proximal vs Distal")
    print("=" * 60 + "\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build analysis dataframe
    print("Step 1: Extracting cell proportions from zarr files...")
    df = build_analysis_dataframe(DATA_DIR, METADATA_PATH)
    print(f"  Total records: {len(df)}")
    print(f"  Unique samples: {df['sample_id'].nunique()}")
    print(f"  Unique donors: {df['donor_id'].nunique()}")
    print(f"  Unique tissue regions: {df['tissue_region'].nunique()}")

    # Save raw data
    df.to_csv(OUTPUT_DIR / "cell_proportions_data.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'cell_proportions_data.csv'}")

    # Step 2: Run statistical analysis
    print("\nStep 2: Running statistical analysis...")
    lmm_results, wilcoxon_results = run_complete_analysis(df)

    # Save results
    lmm_results.to_csv(OUTPUT_DIR / "mixed_effects_results.csv", index=False)
    wilcoxon_results.to_csv(OUTPUT_DIR / "wilcoxon_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'mixed_effects_results.csv'}")
    print(f"  Saved: {OUTPUT_DIR / 'wilcoxon_results.csv'}")

    # Step 3: Generate visualizations
    print("\nStep 3: Generating visualizations...")
    for cell_type in CELL_TYPES:
        output_path = OUTPUT_DIR / f"{cell_type.replace(' ', '_')}_comparison.pdf"
        plot_cell_type_comparison(df, cell_type, lmm_results, output_path)

    # Step 4: Generate summary report
    print("\nStep 4: Generating summary report...")
    generate_summary_report(
        lmm_results, wilcoxon_results, OUTPUT_DIR / "summary_report.txt"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nSignificant findings (FDR < 0.05):")

    if len(lmm_results) > 0 and "significant" in lmm_results.columns:
        sig_results = lmm_results[lmm_results["significant"] == True]
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                direction = "higher" if row["log_odds_ratio"] > 0 else "lower"
                print(
                    f"  - {row['cell_type']} in {row['tissue_region']}: "
                    f"Distal {direction} (FDR p={row['pvalue_fdr']:.4f})"
                )
        else:
            print("  No significant differences found.")
    else:
        print("  No results to report.")


if __name__ == "__main__":
    main()
