"""Regression checks for the local inference path."""
from __future__ import annotations

import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import analyze, load_model, run_bandit  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
import common as ui  # noqa: E402


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_model()
        cls.sample = ROOT / "tests" / "fixtures" / "unsafe_exec.py"

    def test_bundle_has_required_model_fields(self) -> None:
        self.assertIn("model", self.bundle)
        self.assertIn("feature_columns", self.bundle)
        self.assertGreater(len(self.bundle["feature_columns"]), 0)

    def test_sample_scan_returns_bounded_probabilities(self) -> None:
        result = analyze(self.sample, bundle=self.bundle)
        self.assertFalse(result.empty)
        self.assertTrue(result["p_true_positive"].between(0.0, 1.0).all())
        self.assertTrue(result["p_true_positive"].is_monotonic_decreasing)

    def test_uploaded_filename_is_restricted_to_a_basename(self) -> None:
        with patch("common._analyze", return_value=self.bundle["feature_columns"]) as analyze_mock:
            ui.analyze_bytes("..\\outside.py", b"pass\n")
        target = analyze_mock.call_args.args[0]
        self.assertEqual(target.name, "outside.py")

    def test_uploaded_source_rejects_invalid_extensions_and_large_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "ending in .py"):
            ui.analyze_bytes("source.txt", b"pass\n")
        with self.assertRaisesRegex(ValueError, "1 MB"):
            ui.analyze_bytes("source.py", b"x" * 1_000_001)

    def test_bandit_timeout_is_reported_cleanly(self) -> None:
        with patch("predict.subprocess.run", side_effect=subprocess.TimeoutExpired(["bandit"], 60)):
            with self.assertRaisesRegex(RuntimeError, "60-second"):
                run_bandit(self.sample)


if __name__ == "__main__":
    unittest.main()
