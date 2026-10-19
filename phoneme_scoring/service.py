from __future__ import annotations

import numpy as np

from .audio import validate_audio
from .common import METHOD_VERSION, ScoringError, normalize_reference
from .evidence import segment_features, validate_evidence


def failure(code, message, *, preprocessing=None, warnings=None):
    return {"status": "error", "method_version": METHOD_VERSION, "phoneme_accuracy_summary": None, "words": [], "preprocessing": preprocessing, "warnings": warnings or [], "error": {"code": code, "message": message}}


class PronunciationScorer:
    def __init__(self, backend, calibrator):
        self.backend = backend
        self.calibrator = calibrator

    def score_pronunciation(self, audio, sample_rate, reference_words):
        metadata, warnings = None, []
        try:
            words = normalize_reference(reference_words)
            x, metadata, warnings = validate_audio(audio, sample_rate)
            if self.calibrator is None or self.backend is None:
                raise ScoringError("model_not_ready", "Configure a real acoustic backend and a fitted GOP calibrator")
            if self.backend.extractor_id != self.calibrator.extractor_id:
                raise ScoringError("model_not_ready", "Acoustic model identity differs from calibrator")
            evidence = validate_evidence(self.backend.extract(x, words), x, words, self.calibrator.extractor_id)
            warnings += evidence.get("warnings", [])
            scores = self.calibrator.predict(np.stack([segment_features(s) for s in evidence["segments"]]))
            output, index = [], 0
            for wi, word in enumerate(words):
                phones = []
                for pi, phone in enumerate(word["phones"]):
                    s = evidence["segments"][index]
                    phones.append({"index": index, "index_in_word": pi, "target": phone, "start_seconds": float(s["start_seconds"]), "end_seconds": float(s["end_seconds"]), "score": float(scores[index] * 100), "score_raw_0_2": float(scores[index] * 2), "needs_practice": bool(scores[index] < self.calibrator.threshold)})
                    index += 1
                output.append({"index": wi, "word": word["word"], "phoneme_accuracy_summary": float(np.mean([p["score"] for p in phones])), "phones": phones})
            return {"status": "ok", "method_version": self.calibrator.artifact["method_version"], "phoneme_accuracy_summary": float(scores.mean() * 100), "words": output, "preprocessing": metadata, "warnings": list(dict.fromkeys(warnings)), "error": None}
        except ScoringError as error:
            return failure(error.code, str(error), preprocessing=metadata, warnings=warnings)


_default = None


def configure_default(scorer):
    if not isinstance(scorer, PronunciationScorer):
        raise TypeError("Expected PronunciationScorer")
    global _default
    _default = scorer


def score_pronunciation(audio, sample_rate, reference_words):
    return (_default or PronunciationScorer(None, None)).score_pronunciation(audio, sample_rate, reference_words)
