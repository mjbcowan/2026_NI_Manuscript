"""
Tests for phase1b_normalise.py — area normalization and arcsinh transformation.
"""

import numpy as np
import pytest
import anndata as ad
import pandas as pd
from unittest.mock import MagicMock

from phase1b_normalise import normalize_intensity_by_area


def _make_sdata(X, area, area_in_obs=False, area_col="cell_size", area_var="cellarea"):
    """Helper: build a mock SpatialData with two aligned tables."""
    n_cells = X.shape[0]
    n_markers = X.shape[1]
    obs_names = [f"cell_{i:04d}" for i in range(n_cells)]

    adata_int = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=[f"marker_{j}" for j in range(n_markers)]),
    )

    if area_in_obs:
        morph_obs = pd.DataFrame({area_col: area}, index=obs_names)
        adata_morph = ad.AnnData(
            X=np.zeros((n_cells, 1)),
            obs=morph_obs,
            var=pd.DataFrame(index=["dummy"]),
        )
    else:
        adata_morph = ad.AnnData(
            X=area.reshape(-1, 1),
            obs=pd.DataFrame(index=obs_names),
            var=pd.DataFrame(index=[area_var]),
        )

    sdata = MagicMock()
    sdata.tables = {"table_intensities": adata_int, "table_morphology": adata_morph}
    return sdata


@pytest.fixture
def simple_sdata():
    rng = np.random.default_rng(0)
    X = rng.exponential(500, size=(50, 3)).astype(np.float32)
    area = rng.integers(300, 1500, size=50).astype(np.float32)
    return _make_sdata(X, area)


class TestAreaNormalization:
    def test_layers_created(self, simple_sdata):
        result = normalize_intensity_by_area(simple_sdata)
        adata = result.tables["table_intensities"]
        assert "raw" in adata.layers
        assert "area_normalized" in adata.layers
        assert "asinh" in adata.layers

    def test_raw_layer_unchanged(self, simple_sdata):
        original_X = simple_sdata.tables["table_intensities"].X.copy()
        result = normalize_intensity_by_area(simple_sdata)
        adata = result.tables["table_intensities"]
        np.testing.assert_array_equal(adata.layers["raw"], original_X)

    def test_area_normalization_formula(self):
        """area_normalized[i, j] == raw[i, j] / area[i]"""
        X = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
        area = np.array([100.0, 200.0], dtype=np.float32)
        sdata = _make_sdata(X, area)
        result = normalize_intensity_by_area(sdata, use_arcsinh=False)
        adata = result.tables["table_intensities"]
        expected = X / area[:, np.newaxis]
        np.testing.assert_allclose(adata.layers["area_normalized"], expected, rtol=1e-5)

    def test_arcsinh_formula(self):
        """arcsinh layer == arcsinh(area_normalized / cofactor)"""
        X = np.array([[500.0, 1000.0]], dtype=np.float32)
        area = np.array([500.0], dtype=np.float32)
        cofactor = 5.0
        sdata = _make_sdata(X, area)
        result = normalize_intensity_by_area(sdata, cofactor=cofactor)
        adata = result.tables["table_intensities"]
        area_norm = X / area[:, np.newaxis]
        expected_asinh = np.arcsinh(area_norm / cofactor)
        np.testing.assert_allclose(adata.layers["asinh"], expected_asinh, rtol=1e-5)

    def test_X_set_to_asinh_by_default(self, simple_sdata):
        result = normalize_intensity_by_area(simple_sdata)
        adata = result.tables["table_intensities"]
        np.testing.assert_array_equal(adata.X, adata.layers["asinh"])

    def test_area_from_obs_column(self):
        """Function should find area in morphology_table.obs when area_in_obs=True."""
        rng = np.random.default_rng(1)
        X = rng.exponential(500, size=(20, 2)).astype(np.float32)
        area = rng.integers(300, 1500, size=20).astype(np.float32)
        sdata = _make_sdata(X, area, area_in_obs=True)
        result = normalize_intensity_by_area(sdata, area_column="cell_size")
        adata = result.tables["table_intensities"]
        assert "area_normalized" in adata.layers

    def test_negative_area_raises_warning(self):
        """Cells with area <= 0 should trigger a warning, not a crash."""
        X = np.ones((5, 2), dtype=np.float32) * 100
        area = np.array([500.0, 500.0, 0.0, 500.0, 500.0], dtype=np.float32)
        sdata = _make_sdata(X, area)
        with pytest.warns(UserWarning, match="area <= 0"):
            normalize_intensity_by_area(sdata)

    def test_metadata_stored(self, simple_sdata):
        result = normalize_intensity_by_area(simple_sdata)
        adata = result.tables["table_intensities"]
        assert "normalization" in adata.uns
        assert adata.uns["normalization"]["method"] == "area_arcsinh"

    def test_misaligned_tables_raises(self):
        X = np.ones((5, 2), dtype=np.float32)
        area = np.ones(5, dtype=np.float32) * 500
        obs_names_int = [f"cell_{i}" for i in range(5)]
        obs_names_morph = [f"X_{i}" for i in range(5)]  # different names

        adata_int = ad.AnnData(
            X=X, obs=pd.DataFrame(index=obs_names_int), var=pd.DataFrame(index=["m0", "m1"])
        )
        adata_morph = ad.AnnData(
            X=area.reshape(-1, 1),
            obs=pd.DataFrame(index=obs_names_morph),
            var=pd.DataFrame(index=["cellarea"]),
        )
        sdata = MagicMock()
        sdata.tables = {"table_intensities": adata_int, "table_morphology": adata_morph}

        with pytest.raises(ValueError, match="aligned"):
            normalize_intensity_by_area(sdata)
