"""
Tests for phase2_v4.py — MAD-based QC flagging.
"""

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from unittest.mock import MagicMock
from scipy import stats

from phase2_v4 import flag_cells_at_qc_risk


def _make_sdata_for_qc(n_cells=100, n_markers=3, seed=0):
    """Build a mock SpatialData suitable for QC flagging."""
    rng = np.random.default_rng(seed)
    obs_names = [f"cell_{i:04d}" for i in range(n_cells)]

    # Arcsinh-transformed intensities (realistic post-normalisation range)
    X_int = rng.normal(loc=1.5, scale=0.5, size=(n_cells, n_markers)).astype(np.float32)

    # Area: normally distributed around 800 pixels
    area = rng.normal(loc=800, scale=150, size=(n_cells, 1)).astype(np.float32)
    area = np.clip(area, 100, 3000)

    adata_int = ad.AnnData(
        X=X_int,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=[f"marker_{j}" for j in range(n_markers)]),
    )
    adata_morph = ad.AnnData(
        X=area,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["cellarea"]),
    )

    sdata = MagicMock()
    sdata.tables = {"table_intensities": adata_int, "table_morphology": adata_morph}
    return sdata


class TestQCFlagging:
    def test_flag_column_created(self):
        sdata = _make_sdata_for_qc()
        result = flag_cells_at_qc_risk(sdata)
        adata_int = result.tables["table_intensities"]
        assert "is_qc_pass" in adata_int.obs.columns

    def test_flag_column_is_boolean(self):
        sdata = _make_sdata_for_qc()
        result = flag_cells_at_qc_risk(sdata)
        flags = result.tables["table_intensities"].obs["is_qc_pass"]
        assert flags.dtype == bool

    def test_no_cells_removed(self):
        """QC flagging must not drop any rows — only add columns."""
        sdata = _make_sdata_for_qc(n_cells=50)
        result = flag_cells_at_qc_risk(sdata)
        assert result.tables["table_intensities"].n_obs == 50
        assert result.tables["table_morphology"].n_obs == 50

    def test_outlier_cells_flagged(self):
        """Cells with extreme area should fail QC."""
        sdata = _make_sdata_for_qc(n_cells=100)
        # Inject obvious outliers: make cells 0 and 1 have absurdly large area
        sdata.tables["table_morphology"].X[0, 0] = 1_000_000
        sdata.tables["table_morphology"].X[1, 0] = 1_000_000

        result = flag_cells_at_qc_risk(sdata, n_mads=3.0)
        flags = result.tables["table_intensities"].obs["is_qc_pass"]
        assert not flags.iloc[0], "Extreme-area cell 0 should fail QC"
        assert not flags.iloc[1], "Extreme-area cell 1 should fail QC"

    def test_normal_cells_mostly_pass(self):
        """With clean data and 3 MADs, most cells should pass."""
        sdata = _make_sdata_for_qc(n_cells=200, seed=7)
        result = flag_cells_at_qc_risk(sdata, n_mads=3.0)
        pass_rate = result.tables["table_intensities"].obs["is_qc_pass"].mean()
        assert pass_rate > 0.80, f"Expected >80% pass rate, got {pass_rate:.2%}"

    def test_explicit_area_thresholds_respected(self):
        """If min_area and max_area are passed explicitly, they must be used."""
        sdata = _make_sdata_for_qc(n_cells=50)
        # All areas will be in [100, 3000]; restrict tightly to [700, 900]
        result = flag_cells_at_qc_risk(sdata, min_area=700, max_area=900)
        adata_int = result.tables["table_intensities"]
        # Cells with area outside [700, 900] must fail area QC
        area = sdata.tables["table_morphology"].X[:, 0]
        area_pass = adata_int.obs["qc_area_pass"]
        expected_pass = (area >= 700) & (area <= 900)
        np.testing.assert_array_equal(area_pass.values, expected_pass)

    def test_individual_flag_columns_present(self):
        """Transparency: individual QC criteria flags must be stored."""
        sdata = _make_sdata_for_qc()
        result = flag_cells_at_qc_risk(sdata)
        obs = result.tables["table_intensities"].obs
        for col in ["qc_area_pass", "qc_intensity_pass"]:
            assert col in obs.columns, f"Missing transparency column: {col}"

    def test_mad_calculation_correctness(self):
        """Verify the MAD threshold matches scipy.stats.median_abs_deviation."""
        sdata = _make_sdata_for_qc(n_cells=80, seed=3)
        area = sdata.tables["table_morphology"].X[:, 0]
        median_area = np.median(area)
        mad_area = stats.median_abs_deviation(area)
        n_mads = 3.0
        expected_min = max(0, median_area - n_mads * mad_area)
        expected_max = median_area + n_mads * mad_area

        result = flag_cells_at_qc_risk(sdata, n_mads=n_mads)
        obs = result.tables["table_intensities"].obs
        # Stored area min/max should match
        np.testing.assert_allclose(obs["qc_area_min"].iloc[0], expected_min, rtol=1e-4)
        np.testing.assert_allclose(obs["qc_area_max"].iloc[0], expected_max, rtol=1e-4)

    def test_flags_synced_across_tables(self):
        """is_qc_pass must be identical in both intensity and morphology tables."""
        sdata = _make_sdata_for_qc()
        result = flag_cells_at_qc_risk(sdata)
        flags_int = result.tables["table_intensities"].obs["is_qc_pass"]
        flags_morph = result.tables["table_morphology"].obs["is_qc_pass"]
        pd.testing.assert_series_equal(flags_int.reset_index(drop=True), flags_morph.reset_index(drop=True))
