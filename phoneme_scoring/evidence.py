"""Validated acoustic evidence. Labels are deliberately excluded from this schema."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import PHONES, PHONE_INDEX, PREPROCESS_VERSION, ScoringError, audio_hash, dump_json, load_json, reference_hash

EVIDENCE_VERSION = "phone-lpp-v1"


def evidence_key(audio, words):
    return hashlib.sha256((audio_hash(audio) + reference_hash(words)).encode()).hexdigest()


def validate_evidence(e, audio, words, extractor_id=None):
    try:
        if e["schema_version"] != EVIDENCE_VERSION or e["preprocess_version"] != PREPROCESS_VERSION:
            raise ValueError("Evidence/preprocessing version mismatch")
        if e["phone_order"] != list(PHONES) or not isinstance(e["extractor_id"], str) or not e["extractor_id"]:
            raise ValueError("Missing extractor identity or incompatible phone order")
        if extractor_id is not None and e["extractor_id"] != extractor_id:
            raise ValueError("Evidence acoustic model differs from calibrator")
        if e["audio_sha256"] != audio_hash(audio) or e["reference_sha256"] != reference_hash(words):
            raise ValueError("Evidence does not belong to this audio/reference")
        phones = [p for w in words for p in w["phones"]]
        segments = e["segments"]
        if len(segments) != len(phones):
            raise ValueError("Alignment phone count differs from reference")
        end = 0.0
        for segment, phone in zip(segments, phones):
            start, stop = float(segment["start_seconds"]), float(segment["end_seconds"])
            lpp = np.asarray(segment["lpp"], dtype=float)
            if segment["phone"] != phone or not np.isfinite([start, stop]).all() or not (end - 1e-7 <= start < stop <= len(audio) / 16000 + .025):
                raise ValueError("Invalid/nonmonotonic alignment or wrong canonical phone")
            if lpp.shape != (39,) or not np.isfinite(lpp).all() or (lpp > 1e-5).any():
                raise ValueError("LPP must contain 39 finite log posterior means <= 0")
            end = stop
        if not isinstance(e.get("warnings", []), list) or any(not isinstance(w, str) for w in e.get("warnings", [])):
            raise ValueError("Evidence warnings must be strings")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ScoringError("alignment_failed", str(error)) from error
    return e


def segment_features(segment):
    lpp = np.asarray(segment["lpp"], dtype=float)
    k = PHONE_INDEX[segment["phone"]]
    identity = np.zeros(39)
    identity[k] = 1
    # 39 LPP, 39 LPR (target minus each competitor), GOP, log duration, 39 one-hot.
    return np.r_[lpp, lpp[k] - lpp, lpp[k] - lpp.max(), np.log(segment["end_seconds"] - segment["start_seconds"]), identity]


class CachedBackend:
    def __init__(self, directory, extractor_id):
        self.directory = Path(directory)
        self.extractor_id = extractor_id

    def extract(self, audio, words):
        file = self.directory / (evidence_key(audio, words) + ".json")
        try:
            e = load_json(file)
        except FileNotFoundError as error:
            raise ScoringError("evidence_missing", "No cached acoustic evidence for this audio and reference") from error
        except (ValueError, OSError) as error:
            raise ScoringError("alignment_failed", "Corrupt evidence cache") from error
        return validate_evidence(e, audio, words, self.extractor_id)


def save_evidence(directory, evidence, audio, words):
    validate_evidence(evidence, audio, words)
    file = Path(directory) / (evidence_key(audio, words) + ".json")
    dump_json(file, evidence)
    return file
