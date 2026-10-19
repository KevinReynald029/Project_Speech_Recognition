from pathlib import Path
import unittest

from phoneme_scoring.boosting import load_calibrator
from phoneme_scoring.common import load_json, sha256_file


class SavedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.metrics = load_json(self.root/"reports/gop/metrics.json")

    def test_saved_experiment_coverage_and_model(self):
        self.assertEqual(self.metrics["model_sha256"], sha256_file(self.root/"models/calibrator.json"))
        self.assertEqual(sum(c["utterances_total"] for c in self.metrics["coverage"].values()), 5000)
        self.assertEqual(sum(c["utterances_scored"] for c in self.metrics["coverage"].values()), 5000)
        for coverage in self.metrics["coverage"].values():
            self.assertEqual(coverage["phones_scored"], coverage["phones_total"])

    def test_saved_retry_analysis_matches_acoustic_evidence(self):
        recovery = self.metrics.get("alignment_recovery")
        if recovery is None:
            self.skipTest("This experiment does not include historical retry analysis")
        model = load_calibrator(self.root/"models/calibrator.json")
        self.assertEqual(recovery["model_sha256"], self.metrics["model_sha256"])
        self.assertEqual(recovery["extractor_id"], model.extractor_id)
        self.assertEqual(len(recovery["recordings"]), 4)
        for row in recovery["recordings"]:
            with self.subTest(recording=row["utt_id"]):
                self.assertTrue(row["live_matches_cache"])
                self.assertTrue(row["pcm_unchanged"])
                self.assertTrue(row["reference_unchanged"])
