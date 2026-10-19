"""Check public scoring responses directly, without saved JSON fixtures."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from phoneme_scoring import PronunciationScorer
from phoneme_scoring.audio import validate_audio
from phoneme_scoring.boosting import load_calibrator
from phoneme_scoring.common import ScoringError, load_json
from phoneme_scoring.evidence import CachedBackend
from phoneme_scoring.model import RidgeCalibrator
from test_scoring import WORDS, fixture_artifact, fixture_evidence, tone


class EvidenceFixture:
    extractor_id = "synthetic-test-only"

    def __init__(self, error=None):
        self.error = error

    def extract(self, audio, words):
        if self.error:
            raise ScoringError(self.error, "Synthetic extraction failure for contract testing")
        return fixture_evidence(audio, words)


class ContractTests(unittest.TestCase):
    def assert_envelope(self, response):
        self.assertEqual(set(response), {"status", "method_version", "phoneme_accuracy_summary", "words", "preprocessing", "warnings", "error"})
        self.assertIsInstance(response["method_version"], str)
        self.assertTrue(response["method_version"])
        self.assertIsInstance(response["warnings"], list)
        self.assertTrue(all(isinstance(note, str) for note in response["warnings"]))
        json.dumps(response, allow_nan=False)

    def assert_success(self, response, reference):
        self.assert_envelope(response)
        self.assertEqual(response["status"], "ok")
        self.assertIsNone(response["error"])
        self.assertEqual(len(response["words"]), len(reference))
        self.assertEqual(response["preprocessing"]["sample_rate"], 16000)
        scores, end = [], 0.0
        for wi, (word, target) in enumerate(zip(response["words"], reference)):
            self.assertEqual(word["index"], wi)
            self.assertEqual(word["word"], target["word"])
            self.assertEqual([p["target"] for p in word["phones"]], target["phones"])
            for pi, phone in enumerate(word["phones"]):
                self.assertEqual(phone["index"], len(scores))
                self.assertEqual(phone["index_in_word"], pi)
                self.assertGreaterEqual(phone["start_seconds"], end)
                self.assertLess(phone["start_seconds"], phone["end_seconds"])
                self.assertLessEqual(phone["end_seconds"], response["preprocessing"]["duration_seconds"] + .025)
                self.assertGreaterEqual(phone["score"], 0)
                self.assertLessEqual(phone["score"], 100)
                self.assertAlmostEqual(phone["score_raw_0_2"], phone["score"]/50)
                self.assertIsInstance(phone["needs_practice"], bool)
                end = phone["end_seconds"]
                scores.append(phone["score"])
            self.assertAlmostEqual(word["phoneme_accuracy_summary"], np.mean([p["score"] for p in word["phones"]]))
        self.assertAlmostEqual(response["phoneme_accuracy_summary"], np.mean(scores))

    def assert_failure(self, response, code):
        self.assert_envelope(response)
        self.assertEqual(response["status"], "error")
        self.assertIsNone(response["phoneme_accuracy_summary"])
        self.assertEqual(response["words"], [])
        self.assertEqual(response["error"]["code"], code)
        self.assertIsInstance(response["error"]["message"], str)
        self.assertTrue(response["error"]["message"])

    def test_success_response_contract(self):
        audio, _, _ = validate_audio(tone(), 16000)
        scorer = PronunciationScorer(EvidenceFixture(), RidgeCalibrator(fixture_artifact()))
        self.assert_success(scorer.score_pronunciation(audio, 16000, WORDS), WORDS)

    def test_failure_responses_have_null_score(self):
        model = RidgeCalibrator(fixture_artifact())
        scorer = PronunciationScorer(EvidenceFixture(), model)
        cases = [
            ("no_speech", scorer, np.zeros(16000), WORDS),
            ("invalid_audio", scorer, np.zeros((16000, 2)), WORDS),
            ("invalid_reference", scorer, tone(), [{"word": "X", "phones": ["TH1"]}]),
            ("model_not_ready", PronunciationScorer(None, None), tone(), WORDS),
        ]
        for code in ("alignment_failed", "backend_timeout", "backend_unavailable", "evidence_missing"):
            cases.append((code, PronunciationScorer(EvidenceFixture(code), model), tone(), WORDS))
        for code, instance, audio, words in cases:
            with self.subTest(code=code):
                self.assert_failure(instance.score_pronunciation(audio, 16000, words), code)

    def test_failure_contract_rejects_numeric_score(self):
        response = PronunciationScorer(None, None).score_pronunciation(np.zeros(16000), 16000, WORDS)
        invalid = copy.deepcopy(response)
        invalid["phoneme_accuracy_summary"] = 0
        with self.assertRaises(AssertionError):
            self.assert_failure(invalid, "no_speech")

    def test_real_sample_response_contract(self):
        root = Path(__file__).resolve().parents[1]
        model = load_calibrator(root/"models/calibrator.json")
        words = load_json(root/"examples/reference.json")["reference_words"]
        scorer = PronunciationScorer(CachedBackend(root/"examples/evidence", model.extractor_id), model)
        self.assert_success(scorer.score_pronunciation(root/"examples/real_sample.wav", None, words), words)
