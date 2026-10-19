from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from phoneme_scoring import PronunciationScorer, configure_default, score_pronunciation
from phoneme_scoring.audio import validate_audio
from phoneme_scoring.common import FEATURE_DIM, FEATURE_VERSION, METHOD_VERSION, PHONES, PREPROCESS_VERSION, ScoringError, audio_hash, dump_json, normalize_reference, reference_hash
from phoneme_scoring.data import read_dataset, dataset_directory
from phoneme_scoring.evaluation import correlation, evaluate, metrics, select_threshold
from phoneme_scoring.evidence import CachedBackend, EVIDENCE_VERSION, save_evidence, segment_features, validate_evidence
from phoneme_scoring.kaldi import choose_phone, collapse_posteriors, parse_alignment, parse_matrix, pdf_phone_mapping
from phoneme_scoring.model import RidgeCalibrator, fit_ridge


WORDS = [{"word": "TEST", "phones": ["T", "EH", "S", "T"]}]


def tone(seconds=1, sr=16000):
    t = np.arange(int(seconds*sr))/sr
    return .1*np.sin(2*np.pi*220*t)


def fixture_artifact():
    return {"mean": [0.]*FEATURE_DIM, "scale": [1.]*FEATURE_DIM, "coef": [0.]*FEATURE_DIM, "intercept": .8, "method_version": METHOD_VERSION, "feature_version": FEATURE_VERSION, "preprocess_version": PREPROCESS_VERSION, "phone_order": list(PHONES), "extractor_id": "synthetic-test-only", "practice_threshold_norm": .85, "threshold_locked_on_validation": True}


def fixture_evidence(audio, words=WORDS):
    phones = [p for w in words for p in w["phones"]]
    return {"schema_version": EVIDENCE_VERSION, "preprocess_version": PREPROCESS_VERSION, "extractor_id": "synthetic-test-only", "phone_order": list(PHONES), "audio_sha256": audio_hash(audio), "reference_sha256": reference_hash(words), "segments": [{"phone": p, "start_seconds": i*.1, "end_seconds": (i+1)*.1, "lpp": [-4.]*39} for i, p in enumerate(phones)], "warnings": ["synthetic_integration_fixture_not_speech_assessment"]}


class AudioTests(unittest.TestCase):
    def test_silence_and_dc_are_not_scored(self):
        for x in (np.zeros(16000), np.full(16000, .3)):
            with self.assertRaises(ScoringError) as ctx:
                validate_audio(x, 16000)
            self.assertEqual(ctx.exception.code, "no_speech")

    def test_bad_audio(self):
        cases = [(np.array([]), 16000), (np.array([np.nan]), 16000), (np.zeros((2, 16000)), 16000), (np.array([2.]), 16000), (tone(), 0), (tone(), 16000.5), (["bad"], 16000)]
        for x, sr in cases:
            with self.subTest(sr=sr, shape=np.asarray(x).shape):
                with self.assertRaises(ScoringError) as ctx:
                    validate_audio(x, sr)
                self.assertEqual(ctx.exception.code, "invalid_audio")

    def test_prepared_audio_keeps_gain_silence_and_timeline(self):
        raw = np.r_[np.zeros(32000), tone(), np.zeros(32000)]
        actual, metadata, notes = validate_audio(raw, 16000)
        from phoneme_scoring.common import pcm16
        np.testing.assert_array_equal(actual, pcm16(raw).astype(float) / 32768)
        self.assertEqual(len(actual), len(raw))
        self.assertEqual(metadata["leading_trim_seconds"], 0)
        self.assertEqual(metadata["gain_db"], 0)
        self.assertEqual(notes, [])

    def test_unprepared_stereo_and_other_rates_are_rejected(self):
        for audio, rate in [(np.stack([tone()]*2, axis=1), 16000), (tone(sr=48000), 48000)]:
            with self.subTest(rate=rate, shape=audio.shape):
                with self.assertRaises(ScoringError) as ctx:
                    validate_audio(audio, rate)
                self.assertEqual(ctx.exception.code, "invalid_audio")

    def test_prepared_pcm_file_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"sample.wav"
            sf.write(path, tone(), 16000, subtype="PCM_16")
            expected, _ = sf.read(path)
            actual, metadata, _ = validate_audio(path)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(metadata["already_preprocessed"])

    def test_quiet_recording_is_not_amplified(self):
        audio, _, _ = validate_audio(tone()*.03, 16000)
        self.assertLess(np.std(audio), .003)

    def test_pcm_array_and_wav_decode_to_same_signal(self):
        signed = (tone()*32767).astype(np.int16)
        actual, _, _ = validate_audio(signed, 16000)
        np.testing.assert_array_equal(actual, signed.astype(float)/32768)


class EvidenceAndServiceTests(unittest.TestCase):
    def setUp(self):
        self.audio, _, _ = validate_audio(tone(), 16000)
        self.evidence = fixture_evidence(self.audio)

    def test_feature_formula(self):
        segment = self.evidence["segments"][0]
        segment["lpp"][PHONES.index("T")] = -3.
        segment["lpp"][PHONES.index("S")] = -1.
        features = segment_features(segment)
        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertEqual(features[78], -2.)
        self.assertEqual(features[39+PHONES.index("S")], -2.)
        self.assertEqual(features[80+PHONES.index("T")], 1.)

    def test_cache_binds_audio_reference_and_extractor(self):
        with tempfile.TemporaryDirectory() as temp:
            save_evidence(temp, self.evidence, self.audio, WORDS)
            backend = CachedBackend(temp, "synthetic-test-only")
            self.assertEqual(backend.extract(self.audio, WORDS)["schema_version"], EVIDENCE_VERSION)
            with self.assertRaises(ScoringError):
                backend.extract(self.audio*.9, WORDS)
            with self.assertRaises(ScoringError):
                CachedBackend(temp, "different-model").extract(self.audio, WORDS)

    def test_rejects_wrong_nonmonotonic_or_nonfinite_alignment(self):
        mutations = [lambda e: e["segments"].pop(), lambda e: e["segments"][1].update(start_seconds=0), lambda e: e["segments"][0].update(phone="TH"), lambda e: e["segments"][0]["lpp"].__setitem__(0, float("nan")), lambda e: e.update(audio_sha256="wrong")]
        for change in mutations:
            e = json.loads(json.dumps(self.evidence))
            change(e)
            with self.assertRaises(ScoringError):
                validate_evidence(e, self.audio, WORDS)

    def test_scoring_result_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            save_evidence(temp, self.evidence, self.audio, WORDS)
            scorer = PronunciationScorer(CachedBackend(temp, "synthetic-test-only"), RidgeCalibrator(fixture_artifact()))
            out = scorer.score_pronunciation(self.audio, 16000, WORDS)
            self.assertEqual(out["status"], "ok")
            self.assertAlmostEqual(out["phoneme_accuracy_summary"], 80.)
            self.assertEqual(len(out["words"][0]["phones"]), 4)
            self.assertTrue(all(p["needs_practice"] for p in out["words"][0]["phones"]))
            json.dumps(out, allow_nan=False)

    def test_failure_never_gets_zero_score(self):
        for code, x, words in [("no_speech", np.zeros(16000), WORDS), ("invalid_reference", self.audio, [{"word": "X", "phones": ["TH1"]}]), ("model_not_ready", self.audio, WORDS)]:
            out = PronunciationScorer(None, None).score_pronunciation(x, 16000, words)
            self.assertEqual(out["error"]["code"], code)
            self.assertIsNone(out["phoneme_accuracy_summary"])
            self.assertEqual(out["words"], [])

    def test_reference_rejects_special_tokens_and_empty(self):
        for words in ([], [{"word": "A", "phones": ["<unk>"]}], [{"word": "A", "phones": []}], [{"word": "A", "phones": [None]}]):
            with self.assertRaises(ScoringError):
                normalize_reference(words)

    def test_model_mismatch_returns_not_ready(self):
        class OtherBackend:
            extractor_id = "other"
        result = PronunciationScorer(OtherBackend(), RidgeCalibrator(fixture_artifact())).score_pronunciation(self.audio, 16000, WORDS)
        self.assertEqual(result["error"]["code"], "model_not_ready")

    def test_default_api(self):
        configure_default(PronunciationScorer(None, None))
        self.assertEqual(score_pronunciation(np.zeros(1000), 16000, WORDS)["error"]["code"], "no_speech")


class ModelAndMetricTests(unittest.TestCase):
    def test_ridge_fits_signal_and_roundtrips_json(self):
        rng = np.random.default_rng(106)
        x = rng.normal(size=(500, FEATURE_DIM))
        y = np.clip(.5+.15*x[:, 0]-.1*x[:, 1], 0, 1)
        a = {**fixture_artifact(), **fit_ridge(x[:400], y[:400], .1)}
        model = RidgeCalibrator(a)
        self.assertLess(np.mean(np.abs(model.predict(x[400:])-y[400:])), .04)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"ridge.json"
            model.save(path)
            np.testing.assert_allclose(model.predict(x[400:]), RidgeCalibrator(path).predict(x[400:]))

    def test_model_rejects_incompatible_or_unlocked(self):
        for field, value in [("feature_version", "old"), ("threshold_locked_on_validation", False), ("scale", [0.]*FEATURE_DIM), ("intercept", float("nan"))]:
            a = fixture_artifact()
            a[field] = value
            with self.assertRaises(ScoringError):
                RidgeCalibrator(a)

    def test_missing_or_corrupt_model_is_explicit_not_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"ridge.json"
            for content in (None, "not-json"):
                if content:
                    path.write_text(content)
                with self.assertRaises(ScoringError) as ctx:
                    RidgeCalibrator(path)
                self.assertEqual(ctx.exception.code, "model_not_ready")

    def test_perfect_detection_threshold_and_scale(self):
        y, p = np.array([0., .2, .9, 1.]), np.array([.1, .2, .8, .9])
        t = select_threshold(y, p)
        result = metrics(y*2, p, t)
        self.assertEqual(result["mispronunciation"]["f1"], 1.)
        self.assertAlmostEqual(result["mae_raw_0_2"], .15)
        self.assertEqual(result["mispronunciation"]["confusion_matrix"], {"tp": 2, "fp": 0, "fn": 0, "tn": 2})

    def test_threshold_ties_and_constant_predictor(self):
        y = np.array([0., 1., 1., 1.])
        p = np.ones(4)
        t = select_threshold(y, p)
        self.assertGreater(t, 1.)
        self.assertAlmostEqual(metrics(y*2, p, t)["mispronunciation"]["pr_auc_average_precision"], .25)
        self.assertIsNone(correlation(y, p))

    def test_no_positive_labels_and_exact_boundary(self):
        self.assertEqual(select_threshold(np.ones(3), np.ones(3)), 0.)
        m = metrics([.5, 2.], [.25, 1.], .25)
        self.assertIsNone(m["mispronunciation"]["pr_auc_average_precision"])
        self.assertEqual(m["mispronunciation"]["confusion_matrix"]["tn"], 2)

    def test_summary_is_separate_from_holistic_target(self):
        records = [{"utt_id": "1", "word_index": 0, "word_accuracy_raw": 6, "sentence_accuracy_raw": 5, "target_raw": 2, "phone": "T", "age_group": "adult", "gender": "m"}]*2
        m = evaluate(records, [.8, 1.], .25)
        self.assertAlmostEqual(m["word_summary_vs_expert"]["mae_raw_0_10"], 3.)
        self.assertAlmostEqual(m["sentence_summary_vs_expert"]["mae_raw_0_10"], 4.)


class KaldiParserTests(unittest.TestCase):
    def test_repeated_adjacent_phones_keep_boundaries(self):
        self.assertEqual(parse_alignment("sample 12 3 ; 12 4 ; 1 10 \n"), [(12, 3), (12, 4), (1, 10)])

    def test_malformed_or_multiple_alignment_keys_rejected(self):
        for text in ("sample 1 0", "sample 1 2 ; bad", "sample 1 2\nother 1 2", "sample [ 1 2 ]"):
            with self.assertRaises(ScoringError):
                parse_alignment(text)

    def test_phone_stress_and_position_mapping(self):
        self.assertEqual(choose_phone({"EH0_B": 7, "EH1_B": 8}, "EH", "B"), 7)
        with self.assertRaises(ScoringError):
            choose_phone({"AA": 1}, "TH", "S")

    def test_matrix_and_pdf_phone_collapse(self):
        logs = np.log(np.full((2, 39), 1/39))
        text = "sample [\n" + "\n".join(" ".join(map(str, row)) for row in logs) + " ]"
        parsed = parse_matrix(text)
        mapping_text = "\n".join(f"Transition-state {i}: phone = {p}0_B hmm-state = 0 forward-pdf = {i} self-loop-pdf = {i}" for i, p in enumerate(PHONES))
        mapping = pdf_phone_mapping(mapping_text)
        np.testing.assert_allclose(collapse_posteriors(parsed, mapping), logs)

    def test_rejects_likelihoods_not_posteriors(self):
        with self.assertRaises(ScoringError):
            collapse_posteriors(np.ones((2, 39)), {i: {p} for i, p in enumerate(PHONES)})


class DatasetTests(unittest.TestCase):
    def test_actual_handoff_has_no_speaker_leakage(self):
        try:
            root = dataset_directory(Path(__file__).resolve().parents[1])
        except FileNotFoundError:
            self.skipTest("Optional handoff unavailable")
        data = read_dataset(root)
        self.assertEqual([len(data[s]) for s in ("train", "val", "test")], [2100, 400, 2500])
        self.assertEqual(sum(len(w["phones"]) for rows in data.values() for r in rows for w in r["words"]), 94445)


if __name__ == "__main__":
    unittest.main()
