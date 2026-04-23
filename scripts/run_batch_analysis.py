"""
run_batch_analysis.py - Process multiple samples with the spatial analysis pipeline
Usage: python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml --n-jobs 4

Examples:
    # Process all samples in a directory with 4 parallel jobs
    python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml --n-jobs 4

    # Process specific samples matching a pattern
    python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml --pattern "sample_*"

    # Serial processing with automatic segmentation for samples without masks
    python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml --n-jobs 1
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from run_spatial_analysis import run_analysis, extract_sample_id_from_tiff

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def discover_samples(batch_dir, pattern="*"):
    """
    Discover all samples in batch folder.
    Now supports samples with OR without pre-existing masks/tables.

    For each .ome.tiff file found:
    - If mask and table exist: will use them
    - If missing: will trigger automatic segmentation

    Parameters
    ----------
    batch_dir : Path
        Directory containing sample TIFF files
    pattern : str
        Glob pattern to match sample files (default: "*")

    Returns
    -------
    list of dict
        Each dict contains: sample_id, tiff_path, mask_path (optional), table_path (optional)
    """
    batch_dir = Path(batch_dir)

    # Find all .ome.tiff files
    tiff_files = list(batch_dir.glob(f"{pattern}.ome.tiff"))
    tiff_files += list(batch_dir.glob(f"{pattern}.ome.tif"))

    if not tiff_files:
        logger.warning(f"No .ome.tiff files found matching pattern '{pattern}' in {batch_dir}")
        return []

    samples = []
    for tiff_file in sorted(tiff_files):
        sample_id = extract_sample_id_from_tiff(tiff_file)

        # Check for optional mask and table files
        mask_path = batch_dir / f"{sample_id}_whole_cell.tiff"
        if not mask_path.exists():
            mask_path = batch_dir / f"{sample_id}_whole_cell.tif"
        if not mask_path.exists():
            mask_path = None

        table_path = batch_dir / f"cell_table_arcsinh_transformed_{sample_id}.csv"
        if not table_path.exists():
            table_path = None

        sample_info = {
            'sample_id': sample_id,
            'tiff_path': tiff_file,
            'mask_path': mask_path,
            'table_path': table_path
        }
        samples.append(sample_info)

        status = "complete" if (mask_path and table_path) else "will segment"
        logger.info(f"Found sample: {sample_id} ({status})")

    logger.info(f"Discovered {len(samples)} samples in {batch_dir}")
    return samples


def process_single_sample(sample_info, batch_output_dir, signatures_path, **kwargs):
    """
    Process a single sample and return status.

    Parameters
    ----------
    sample_info : dict
        Dictionary with 'sample_id', 'tiff_path', 'mask_path', 'table_path'
    batch_output_dir : Path
        Batch output directory (each sample gets its own subdirectory)
    signatures_path : Path
        Path to signatures file
    **kwargs : dict
        Additional arguments to pass to run_analysis

    Returns
    -------
    dict
        Result dictionary with status and error info
    """
    sample_id = sample_info['sample_id']
    try:
        logger.info(f"Starting processing for {sample_id}")
        logger.info(f"  TIFF: {sample_info['tiff_path']}")
        logger.info(f"  Mask: {sample_info['mask_path'] if sample_info['mask_path'] else 'None (will segment)'}")
        logger.info(f"  Table: {sample_info['table_path'] if sample_info['table_path'] else 'None (will segment)'}")

        # Create sample-specific output directory
        sample_output_dir = batch_output_dir / sample_id
        logger.info(f"  Output: {sample_output_dir}")

        run_analysis(
            sample_id=sample_info['sample_id'],
            tiff_path=sample_info['tiff_path'],
            mask_path=sample_info['mask_path'],
            table_path=sample_info['table_path'],
            output_base_dir=sample_output_dir,  # Each sample gets its own directory
            signatures_path=signatures_path,
            **kwargs
        )

        return {"sample": sample_id, "status": "success", "error": None}
    except Exception as e:
        logger.error(f"Failed to process {sample_id}: {e}", exc_info=True)
        return {"sample": sample_id, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple spatial transcriptomics/proteomics samples",
        epilog="""
Examples:
  # Process all samples with 4 parallel jobs
  python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml --n-jobs 4

  # Process samples matching a pattern
  python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml --pattern "experiment1_*"

  # Serial processing with automatic segmentation
  python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml --n-jobs 1

  # Custom segmentation parameters
  python run_batch_analysis.py --batch-dir ./batch --signatures signatures.yaml \\
      --nucleus-channels 0 --cytoplasm-channels 1 2 3 --nucleus-diameter 25
        """
    )

    # Required arguments
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Directory containing all sample TIFF files"
    )
    parser.add_argument(
        "--signatures",
        type=Path,
        required=True,
        help="Path to YAML or JSON file with cell type signatures"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd().parent / "output",
        help="Base output directory (default: ../output)"
    )

    # Sample selection
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern to match sample files (default: * for all samples)"
    )

    # Parallel processing
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1 for serial processing)"
    )

    # Pipeline parameters
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

    # QC parameters
    parser.add_argument(
        "--n-mads",
        type=float,
        default=3.0,
        help="Number of MADs for QC outlier detection (default: 3.0)"
    )

    # Phenotyping parameters
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

    # Segmentation parameters (for samples without pre-existing masks)
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

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display plots interactively (default: save only)"
    )

    args = parser.parse_args()

    # Discover samples
    logger.info(f"Searching for samples in {args.batch_dir} with pattern '{args.pattern}'")
    samples = discover_samples(args.batch_dir, args.pattern)

    if not samples:
        logger.error(f"No samples found in {args.batch_dir}")
        return

    logger.info(f"Found {len(samples)} samples to process")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Parallel jobs: {args.n_jobs}")

    # Prepare kwargs for run_analysis
    analysis_kwargs = {
        "nuclear_stains": args.nuclear_stains,
        "pixel_size_um": args.pixel_size,
        "microscope": args.microscope,
        "objective": args.objective,
        "created_by": args.created_by,
        "pipeline_version": args.pipeline_version,
        "n_mads": args.n_mads,
        "gmm_probability_threshold": args.gmm_prob_threshold,
        "gmm_min_combined_prob": args.gmm_min_prob,
        "nucleus_channels": args.nucleus_channels,
        "cytoplasm_channels": args.cytoplasm_channels,
        "nucleus_diameter": args.nucleus_diameter,
        "cytoplasm_diameter": args.cytoplasm_diameter,
        "visualize": args.visualize,
    }

    # Process samples
    results = []
    if args.n_jobs == 1:
        # Serial processing
        logger.info("Starting serial processing...")
        for i, sample_info in enumerate(samples, 1):
            logger.info(f"Processing sample {i}/{len(samples)}: {sample_info['sample_id']}")
            result = process_single_sample(
                sample_info,
                args.output_dir,
                args.signatures,
                **analysis_kwargs
            )
            results.append(result)
            logger.info(f"Completed {i}/{len(samples)}: {result['status']}")
    else:
        # Parallel processing
        logger.info(f"Starting parallel processing with {args.n_jobs} workers...")
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {
                executor.submit(
                    process_single_sample,
                    sample_info,
                    args.output_dir,
                    args.signatures,
                    **analysis_kwargs
                ): sample_info['sample_id']
                for sample_info in samples
            }

            completed = 0
            for future in as_completed(futures):
                sample_id = futures[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {completed}/{len(samples)}: {sample_id} - {result['status']}")
                except Exception as e:
                    logger.error(f"Exception processing {sample_id}: {e}")
                    results.append({"sample": sample_id, "status": "failed", "error": str(e)})

    # Summary
    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "failed")

    logger.info("=" * 80)
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"Total: {len(results)} | Success: {successes} | Failed: {failures}")
    logger.info("=" * 80)

    if failures > 0:
        logger.warning("Failed samples:")
        for r in results:
            if r["status"] == "failed":
                logger.warning(f"  {r['sample']}: {r['error']}")

    # Write summary to file
    summary_file = args.output_dir / "batch_processing_summary.txt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Batch directory: {args.batch_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Total samples: {len(results)}\n")
        f.write(f"Successful: {successes}\n")
        f.write(f"Failed: {failures}\n")
        f.write(f"{'='*80}\n\n")

        f.write("Detailed Results:\n")
        for r in results:
            f.write(f"\n{r['sample']}: {r['status']}\n")
            if r['error']:
                f.write(f"  Error: {r['error']}\n")

    logger.info(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
