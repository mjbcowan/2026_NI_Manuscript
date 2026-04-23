"""
Shared pytest fixtures for the 2026_NI_Manuscript test suite.

All fixtures are CPU-only (no GPU required) to enable CI without GPU runners.
"""

import numpy as np
import pandas as pd
import anndata as ad
import pytest
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


N_CELLS = 100
MARKERS = ["DAPI_INIT", "DAPI_FINAL", "PI16", "PRG4", "VWF"]
N_MARKERS = len(MARKERS)

# Reproducible random state
RNG = np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_intensity_matrix():
    """100-cell × 5-marker raw intensity matrix with realistic values."""
    # Simulate positive cells with higher signal for specific markers
    X = RNG.exponential(scale=500, size=(N_CELLS, N_MARKERS)).astype(np.float32)
    # Make PI16 cells: first 30 cells have elevated PI16
    X[:30, MARKERS.index("PI16")] += 2000
    # Make VWF cells: cells 30-60 have elevated VWF
    X[30:60, MARKERS.index("VWF")] += 2000
    return X


@pytest.fixture(scope="session")
def synthetic_area():
    """Cell areas in pixels (realistic range for CellDIVE 20x, ~30px nucleus diameter)."""
    return RNG.integers(low=300, high=2000, size=N_CELLS).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_intensity_adata(synthetic_intensity_matrix):
    """AnnData of raw intensities, aligned obs_names with morphology table."""
    obs_names = [f"cell_{i:04d}" for i in range(N_CELLS)]
    adata = ad.AnnData(
        X=synthetic_intensity_matrix.copy(),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=MARKERS),
    )
    return adata


@pytest.fixture(scope="session")
def synthetic_morphology_adata(synthetic_area):
    """AnnData of morphology features (area only), aligned with intensity table."""
    obs_names = [f"cell_{i:04d}" for i in range(N_CELLS)]
    adata = ad.AnnData(
        X=synthetic_area.reshape(-1, 1).copy(),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["cellarea"]),
    )
    return adata


@pytest.fixture(scope="session")
def synthetic_sdata(synthetic_intensity_adata, synthetic_morphology_adata):
    """Minimal SpatialData-like object backed by two AnnData tables."""
    from unittest.mock import MagicMock

    sdata = MagicMock()
    sdata.tables = {
        "table_intensities": synthetic_intensity_adata,
        "table_morphology": synthetic_morphology_adata,
    }
    return sdata


@pytest.fixture(scope="session")
def synthetic_ome_tiff(tmp_path_factory):
    """Write a minimal 5-channel synthetic OME-TIFF and return its path."""
    import tifffile

    tmp = tmp_path_factory.mktemp("tiff")
    tiff_path = tmp / "synthetic_100cell.ome.tiff"

    # Shape: (channels, height, width) — small 64×64 px image
    data = RNG.integers(0, 65535, size=(N_MARKERS, 64, 64), dtype=np.uint16)

    # Minimal OME-XML metadata
    ome_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0" Name="synthetic">'
        '<Pixels DimensionOrder="XYZCT" Type="uint16"'
        f' SizeX="64" SizeY="64" SizeZ="1" SizeC="{N_MARKERS}" SizeT="1"'
        ' PhysicalSizeX="0.325" PhysicalSizeXUnit="µm"'
        ' PhysicalSizeY="0.325" PhysicalSizeYUnit="µm">'
        + "".join(
            f'<Channel ID="Channel:0:{i}" Name="{name}" SamplesPerPixel="1"/>'
            for i, name in enumerate(MARKERS)
        )
        + "</Pixels></Image></OME>"
    )

    tifffile.imwrite(
        tiff_path,
        data,
        photometric="minisblack",
        description=ome_xml,
        metadata=None,
    )
    return tiff_path
