"""Validate prepared scoring audio without resampling, trimming, or gain changes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .common import PREPROCESS_VERSION, ScoringError, pcm16


def validate_audio(audio, sample_rate=None):
    """Accept prepared mono 16 kHz PCM; preserve its timeline and amplitude."""
    try:
        if isinstance(audio, (str, Path)):
            info = sf.info(str(audio))
            if info.duration > 60 or info.frames == 0:
                raise ValueError("Audio must contain between 0 and 60 seconds")
            x, file_rate = sf.read(str(audio), dtype="float64", always_2d=False)
            if sample_rate is not None and sample_rate != file_rate:
                raise ValueError("sample_rate differs from file metadata")
            sample_rate = file_rate
        else:
            a = np.asarray(audio)
            if a.dtype.kind not in "fiu":
                raise ValueError("Audio must be a numeric PCM array")
            if a.dtype.kind == "u":
                if a.dtype != np.uint8:
                    raise ValueError("Only unsigned PCM8 is supported")
                x = (a.astype(float) - 128) / 128
            elif a.dtype.kind == "i":
                x = a.astype(float) / (float(np.iinfo(a.dtype).max) + 1)
            else:
                x = a.astype(float)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate != 16000:
            raise ValueError("Scoring requires prepared 16 kHz audio; resample upstream")
        if x.ndim != 1 or not x.size or x.size > 60 * sample_rate or not np.isfinite(x).all() or np.max(np.abs(x)) > 1.001:
            raise ValueError("Use finite, normalized mono PCM, maximum 60 seconds; prepare channels upstream")
    except (ValueError, TypeError, OSError, RuntimeError, OverflowError) as error:
        raise ScoringError("invalid_audio", str(error)) from error

    # Energy validation only: DC and near-silence cannot produce a score.
    if np.std(x) < 10 ** (-65 / 20):
        raise ScoringError("no_speech", "No usable signal energy")
    # Match the PCM16 representation consumed by the acoustic extractor.
    x = pcm16(x).astype(float) / 32768
    duration = len(x) / 16000
    return x, {
        "version": PREPROCESS_VERSION, "sample_rate": 16000,
        "duration_seconds": duration, "original_duration_seconds": duration,
        "leading_trim_seconds": 0.0, "trailing_trim_seconds": 0.0,
        "gain_db": 0.0, "already_preprocessed": True,
    }, []
