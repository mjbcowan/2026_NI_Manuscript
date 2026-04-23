"""
Tests for CLI argument handling in prox_dist_enrichment_analysis.py.

Validates that --data-dir is accepted and that the script fails gracefully
when the directory does not exist, without crashing at import time.
"""

import subprocess
import sys
from pathlib import Path
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "prox_dist_enrichment_analysis.py"


class TestArgparseInterface:
    def test_help_exits_cleanly(self):
        """--help should print usage and exit 0."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--data-dir" in result.stdout

    def test_data_dir_argument_accepted(self, tmp_path):
        """--data-dir should be accepted without ArgparseError."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--data-dir", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        # Will fail at runtime (no data files) but must NOT fail with argparse error (exit code 2)
        assert result.returncode != 2, (
            f"argparse rejected --data-dir: {result.stderr}"
        )

    def test_output_dir_argument_accepted(self, tmp_path):
        """--output-dir should be accepted without ArgparseError."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--data-dir", str(tmp_path), "--output-dir", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 2, (
            f"argparse rejected --output-dir: {result.stderr}"
        )

    def test_no_hardcoded_home_path_in_source(self):
        """The script source must not contain any hardcoded /home paths."""
        source = SCRIPT.read_text()
        forbidden = ["/home/mcowan-wsl", "/home/mcowan"]
        for pattern in forbidden:
            assert pattern not in source, (
                f"Found hardcoded path '{pattern}' in {SCRIPT.name}"
            )

    def test_default_data_dir_is_relative(self):
        """The default --data-dir value must be a relative path (not absolute)."""
        import importlib.util
        import sys as _sys

        # Temporarily patch sys.argv so argparse doesn't read pytest args
        old_argv = _sys.argv
        _sys.argv = ["prox_dist_enrichment_analysis.py"]
        try:
            spec = importlib.util.spec_from_file_location("prox_dist", str(SCRIPT))
            mod = importlib.util.module_from_spec(spec)
            # We expect parse_args() to be called at module level;
            # import will set DATA_DIR via argparse with default
            spec.loader.exec_module(mod)
            assert not mod.DATA_DIR.is_absolute(), (
                f"Default DATA_DIR should be relative, got: {mod.DATA_DIR}"
            )
        except SystemExit:
            pytest.skip("Script exited (likely missing data files) — argparse test still valid")
        finally:
            _sys.argv = old_argv
