from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PHONES = tuple("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split())
PHONE_INDEX = {p: i for i, p in enumerate(PHONES)}
METHOD_VERSION = "gop-hgb-mae-v2"
FEATURE_VERSION = "lpp39-lpr39-gop-duration-onehot39-v1"
PREPROCESS_VERSION = "handoff-16k-rms25-peak1-edge500-v1"
FEATURE_DIM = 119


class ScoringError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_reference(words):
    if not isinstance(words, (list, tuple)) or not words:
        raise ScoringError("invalid_reference", "reference_words must be a nonempty list")
    result = []
    for w in words:
        if not isinstance(w, dict) or not isinstance(w.get("word"), str) or not w["word"].strip():
            raise ScoringError("invalid_reference", "Each word needs a nonempty 'word'")
        phones = w.get("phones")
        if not isinstance(phones, (list, tuple)) or not phones or any(p not in PHONE_INDEX for p in phones if isinstance(p, str)) or any(not isinstance(p, str) for p in phones):
            raise ScoringError("invalid_reference", "Use the 39 uppercase ARPAbet phones without stress digits")
        result.append({"word": w["word"].strip().upper(), "phones": list(phones)})
    if sum(len(w["phones"]) for w in result) > 500:
        raise ScoringError("invalid_reference", "At most 500 reference phones are supported")
    return result


def reference_hash(words):
    return hashlib.sha256(json.dumps(normalize_reference(words), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audio_hash(audio):
    # Canonical little-endian PCM16, same quantization as soundfile PCM_16.
    return hashlib.sha256(pcm16(audio).tobytes()).hexdigest()


def pcm16(audio):
    return np.floor(np.clip(np.asarray(audio, dtype=float), -1, 32767 / 32768) * 32768).astype("<i2")
