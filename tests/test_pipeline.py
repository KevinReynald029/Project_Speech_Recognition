"""Tes alur dengan audio sintetis."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from phoneme_scoring.common import dump_json, sha256_file
from phoneme_scoring.data import read_dataset
from phoneme_scoring.experiments import extract_dataset, train_experiment
from test_scoring import fixture_evidence


class SyntheticBackend:
    extractor_id = "synthetic-test-only"

    def extract(self, audio, words):
        e = fixture_evidence(audio, words)
        crossings = np.count_nonzero(np.diff(np.signbit(audio)))
        for s in e["segments"]:
            s["lpp"] = [-7.]*39
            from phoneme_scoring.common import PHONES
            s["lpp"][PHONES.index(s["phone"])] = -max(.05, 4 - crossings/200)
        return e


class PipelineTests(unittest.TestCase):
    def make_dataset(self, root):
        (root/"wav16k").mkdir()
        splits = []
        for split in ("train", "val", "test"):
            rows = []
            for i in range(6):
                uid = split + str(i)
                frequency = 100 + i*100
                t = np.arange(16000)/16000
                sf.write(root/"wav16k"/(uid+".wav"), .1*np.sin(2*np.pi*frequency*t), 16000, subtype="PCM_16")
                score = i*.4
                rows.append({"utt_id": uid, "split": split, "speaker": uid, "age": 10 if i%2 else 25, "gender": "m", "text": "TEST", "audio_proc": "wav16k/"+uid+".wav", "sentence_accuracy": score*5, "words": [{"word": "TEST", "phones": ["T", "EH", "S", "T"], "phone_scores": [score]*4, "phone_scores_norm": [score/2]*4, "word_accuracy": score*5}]})
                splits.append({"utt_id": uid, "split": split})
            (root/f"manifest_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        with (root/"splits.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["utt_id", "split"])
            writer.writeheader()
            writer.writerows(splits)

    def test_extraction_training_evaluation_and_test_label_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, cache = root/"data", root/"cache"
            data.mkdir()
            self.make_dataset(data)
            def fixture_reader(d):
                return read_dataset(d, official_counts=False)
            with patch("phoneme_scoring.experiments.read_dataset", side_effect=fixture_reader):
                summary = extract_dataset(data, cache, SyntheticBackend())
                self.assertEqual(summary["test"]["success"], 6)
                report = train_experiment(data, cache, root/"experiment1")
                self.assertEqual(report["coverage"]["test"]["utterance_coverage"], 1.)
                before = sha256_file(root/"experiment1"/"calibrator.json")
                # Ubah label test; model dan ambang harus tetap sama.
                file = data/"manifest_test.jsonl"
                rows = [json.loads(line) for line in file.read_text().splitlines()]
                for r in rows:
                    r["words"][0]["phone_scores"] = [2.]*4
                    r["words"][0]["phone_scores_norm"] = [1.]*4
                file.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
                index = json.loads((cache/"index_test.json").read_text())
                index["manifest_sha256"] = sha256_file(file)
                dump_json(cache/"index_test.json", index)
                train_experiment(data, cache, root/"experiment2")
                self.assertEqual(before, sha256_file(root/"experiment2"/"calibrator.json"))

    def test_speaker_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            path = root/"manifest_test.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["speaker"] = "train0"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "Speaker leakage"):
                read_dataset(root, official_counts=False)
