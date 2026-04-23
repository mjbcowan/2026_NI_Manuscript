"""
Tests for phase3_intensity_phenotype_v6.py — GMM phenotyping logic.

Focuses on the pure-Python/numpy functions that don't require GPU or plotting.
"""

import numpy as np
import pandas as pd
import pytest

squidpy = pytest.importorskip(
    "squidpy",
    reason="squidpy not installed in this environment — skipping phenotype tests",
)

from phase3_intensity_phenotype_v6 import (
    compute_thresholds_all_methods,
    validate_cell_type_signatures,
    create_example_signatures,
)


class TestThresholdComputation:
    def test_returns_dict(self):
        values = np.random.default_rng(0).exponential(500, 200)
        result = compute_thresholds_all_methods(values)
        assert isinstance(result, dict)

    def test_otsu_key_present(self):
        values = np.concatenate([
            np.random.default_rng(0).normal(200, 30, 100),
            np.random.default_rng(1).normal(800, 50, 100),
        ])
        result = compute_thresholds_all_methods(values)
        assert "otsu" in result

    def test_threshold_separates_bimodal(self):
        """Otsu threshold should fall between two clearly separated populations."""
        low = np.random.default_rng(2).normal(100, 10, 200)
        high = np.random.default_rng(3).normal(900, 10, 200)
        values = np.concatenate([low, high])
        result = compute_thresholds_all_methods(values)
        threshold = result["otsu"]
        assert 200 < threshold < 800, f"Otsu threshold {threshold} outside expected range [200, 800]"

    def test_all_same_values_does_not_crash(self):
        """Constant input should not raise an exception."""
        values = np.ones(50) * 500.0
        try:
            compute_thresholds_all_methods(values)
        except Exception as e:
            pytest.fail(f"Unexpected exception on constant input: {e}")


class TestSignatureValidation:
    def test_valid_signatures_pass(self):
        sigs = create_example_signatures()
        # Should return without raising
        validate_cell_type_signatures(sigs)

    def test_empty_signatures_raises(self):
        with pytest.raises((ValueError, KeyError, AssertionError)):
            validate_cell_type_signatures({})

    def test_example_signatures_have_positive_markers(self):
        sigs = create_example_signatures()
        for cell_type, definition in sigs.items():
            assert "positive" in definition or "markers" in definition, (
                f"Cell type '{cell_type}' missing 'positive' or 'markers' key"
            )


class TestExampleSignatures:
    def test_returns_dict(self):
        result = create_example_signatures()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_known_cell_types_present(self):
        sigs = create_example_signatures()
        known = {"PI16 Fibroblast", "Vascular Endothelium", "PRG4 Fibroblast"}
        found = set(sigs.keys())
        overlap = known & found
        assert len(overlap) > 0, (
            f"Expected at least one of {known} in example signatures, got {found}"
        )

    def test_marker_values_are_strings(self):
        sigs = create_example_signatures()
        for cell_type, definition in sigs.items():
            for key in ("positive", "negative"):
                if key in definition:
                    markers = definition[key]
                    assert all(isinstance(m, str) for m in markers), (
                        f"Markers in '{cell_type}'.{key} must be strings"
                    )
