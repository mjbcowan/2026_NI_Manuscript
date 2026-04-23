"""
This module defines the contract that all pipeline phases must implement,
enabling plug-and-play modularity and testability.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging

from spatialdata import SpatialData
from pydantic import BaseModel, Field


# ============================================================================
# Configuration Models (Pydantic for validation)
# ============================================================================

class PhaseConfig(BaseModel):
    """Base configuration for any pipeline phase."""

    output_dir: Path = Field(..., description="Output directory for phase results")
    overwrite: bool = Field(True, description="Whether to overwrite existing outputs")
    save_intermediate: bool = Field(True, description="Save intermediate results")
    log_level: str = Field("INFO", description="Logging level")

    class Config:
        extra = 'forbid'  # Catch configuration typos
        arbitrary_types_allowed = True  # Allow Path objects


@dataclass
class PhaseResult:
    """
    Standardised result from a phase execution.

    Attributes
    ----------
    sdata : SpatialData
        Modified SpatialData object
    metadata : dict
        Execution metadata (timing, counts, parameters used)
    outputs : dict
        Paths to output files generated
    qc_metrics : dict, optional
        Quality control metrics from this phase
    """
    sdata: SpatialData
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Path] = field(default_factory=dict)
    qc_metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Add timestamp to metadata."""
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()


# ============================================================================
# Abstract Base Phase
# ============================================================================

class Phase(ABC):
    """
    Abstract base class for all pipeline phases.

    Each phase must implement:
    1. Input validation
    2. Execution logic
    3. Output generation

    This design enables:
    - Unit testing with mock inputs
    - Phase composition and reordering
    - Alternative implementations (e.g., different QC strategies)
    - Progress tracking and failure recovery

    Parameters
    ----------
    config : PhaseConfig
        Configuration object for this phase
    logger : logging.Logger, optional
        Logger instance (created if not provided)

    Examples
    --------
    >>> class MyPhase(Phase):
    ...     def validate_inputs(self, sdata):
    ...         return 'table_intensities' in sdata.tables
    ...     
    ...     def _execute_impl(self, sdata, **kwargs):
    ...         # Phase logic here
    ...         return sdata
    ...     
    ...     def get_required_tables(self):
    ...         return ['table_intensities']
    """

    def __init__(self, config: PhaseConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or self._setup_logger()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def _setup_logger(self) -> logging.Logger:
        """Create logger for this phase."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        return logger

    @abstractmethod
    def validate_inputs(self, sdata: SpatialData) -> bool:
        """
        Validate that input SpatialData meets phase requirements.

        Parameters
        ----------
        sdata : SpatialData
            Input SpatialData object

        Returns
        -------
        bool
            True if inputs are valid

        Raises
        ------
        ValueError
            If inputs are invalid with detailed error message
        """
        pass

    @abstractmethod
    def _execute_impl(self, sdata: SpatialData, **kwargs) -> SpatialData:
        """
        Core execution logic for the phase (to be implemented by subclasses).

        Parameters
        ----------
        sdata : SpatialData
            Input SpatialData object
        **kwargs
            Additional arguments for phase execution

        Returns
        -------
        SpatialData
            Modified SpatialData object
        """
        pass

    @abstractmethod
    def get_required_tables(self) -> List[str]:
        """Return list of required table names in SpatialData."""
        pass

    @abstractmethod
    def get_phase_name(self) -> str:
        """Return human-readable phase name."""
        pass

    def execute(self, sdata: SpatialData, **kwargs) -> PhaseResult:
        """
        Execute the phase with validation, timing, and error handling.

        This template method orchestrates:
        1. Input validation
        2. Timing capture
        3. Core execution
        4. Metadata generation
        5. Output persistence

        Parameters
        ----------
        sdata : SpatialData
            Input SpatialData object
        **kwargs
            Additional execution arguments

        Returns
        -------
        PhaseResult
            Result object with modified SpatialData and metadata
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting {self.get_phase_name()}")
        self.logger.info(f"=" * 80)

        # Validate inputs
        self.start_time = datetime.now()
        if not self.validate_inputs(sdata):
            raise ValueError(f"{self.get_phase_name()} validation failed")

        self.logger.info("✓ Input validation passed")

        # Execute core logic
        try:
            sdata_out = self._execute_impl(sdata, **kwargs)
        except Exception as e:
            self.logger.error(f"✗ {self.get_phase_name()} failed: {e}")
            raise

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Build metadata
        metadata = {
            'phase_name': self.get_phase_name(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': duration,
            'config': self.config.dict(),
        }

        # Generate outputs
        outputs = self._save_outputs(sdata_out)

        self.logger.info(f"✓ {self.get_phase_name()} completed in {duration:.1f}s")
        self.logger.info(f"=" * 80)

        return PhaseResult(
            sdata=sdata_out,
            metadata=metadata,
            outputs=outputs,
            qc_metrics=self._compute_qc_metrics(sdata_out)
        )

    def _save_outputs(self, sdata: SpatialData) -> Dict[str, Path]:
        """
        Save phase outputs (to be optionally overridden by subclasses).

        Parameters
        ----------
        sdata : SpatialData
            SpatialData to save

        Returns
        -------
        Dict[str, Path]
            Mapping of output names to paths
        """
        outputs = {}
        if self.config.save_intermediate:
            output_path = self.config.output_dir / f"{self.get_phase_name()}.zarr"
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            sdata.write(output_path, overwrite=self.config.overwrite)
            outputs['zarr'] = output_path
            self.logger.info(f"Saved output to {output_path}")
        return outputs

    def _compute_qc_metrics(self, sdata: SpatialData) -> Optional[Dict[str, Any]]:
        """
        Compute QC metrics (to be optionally overridden by subclasses).

        Parameters
        ----------
        sdata : SpatialData
            SpatialData to compute metrics from

        Returns
        -------
        Dict[str, Any] or None
            QC metrics dictionary
        """
        return None
