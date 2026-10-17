"""Angel: initialize once at server startup; call after recording completes.

Run from the package root after extract/train, using a ready native-English model.
This example intentionally has no server framework dependency.
"""
import json
from pathlib import Path

from phoneme_scoring import PronunciationScorer, configure_default, score_pronunciation
from phoneme_scoring.backend import create_backend
from phoneme_scoring.boosting import load_calibrator


def create_scorer(config_path, model_path):
    scorer = PronunciationScorer(create_backend(config_path), load_calibrator(model_path))
    configure_default(scorer)
    return scorer


if __name__ == "__main__":
    create_scorer("config/kaldi.native.json", "models/calibrator.json")
    reference = json.loads(Path("examples/reference.json").read_text())["reference_words"]
    result = score_pronunciation("recording.wav", None, reference)
    print(json.dumps(result, indent=2, allow_nan=False))
