"""Kevin's preprocessing, plus explicit input validation and resampling."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import PREPROCESS_VERSION, ScoringError, pcm16


def preprocess(audio, sample_rate=None, *, already_preprocessed=False):
    try:
        if isinstance(audio, (str, Path)):
            info = sf.info(str(audio))
            if info.duration > 60 or info.frames == 0:
                raise ScoringError("invalid_audio", "Audio must contain between 0 and 60 seconds")
            x, file_rate = sf.read(str(audio), dtype="float64", always_2d=False)
            if sample_rate is not None and sample_rate != file_rate:
                raise ScoringError("invalid_audio", "sample_rate differs from file metadata")
            sample_rate = file_rate
        else:
            a = np.asarray(audio)
            if a.dtype.kind not in "fiu":
                raise ValueError("Audio must be a numeric PCM array")
            # Unsigned PCM8 is centered at 128; signed PCM is full-scale normalized.
            if a.dtype.kind == "u":
                if a.dtype != np.uint8:
                    raise ValueError("Only unsigned PCM8 is supported")
                x = (a.astype(float) - 128) / 128
            elif a.dtype.kind == "i":
                x = a.astype(float) / (float(np.iinfo(a.dtype).max) + 1)
            else:
                x = a.astype(float)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or not 8000 <= sample_rate <= 192000:
            raise ValueError("sample_rate must be an integer between 8000 and 192000")
        if x.ndim == 2 and 1 <= x.shape[1] <= 8:
            x = x.mean(axis=1)
        if x.ndim != 1 or not x.size or x.size > 60 * sample_rate or not np.isfinite(x).all() or np.max(np.abs(x)) > 1.001:
            raise ValueError("Use finite, normalized PCM with shape samples or samples x channels, maximum 60 seconds")
    except ScoringError:
        raise
    except (ValueError, TypeError, OSError, RuntimeError, OverflowError) as e:
        raise ScoringError("invalid_audio", str(e)) from e

    warnings = []
    if sample_rate != 16000:
        if already_preprocessed:
            raise ScoringError("invalid_audio", "Preprocessed dataset audio must already be 16 kHz")
        g = gcd(int(sample_rate), 16000)
        x = resample_poly(x, 16000 // g, int(sample_rate) // g)
        warnings.append("resampled_to_16khz")
    original_duration = len(x) / 16000
    x = np.asarray(x, dtype=float)
    centered = x - x.mean()
    # Absolute floor prevents normalization of silence/DC/near numerical noise.
    # This is an energy guard, not a speech/noise classifier.
    if np.sqrt(np.mean(centered ** 2)) < 10 ** (-65 / 20):
        raise ScoringError("no_speech", "No usable signal energy after DC removal")
    start, end, gain_db = 0, len(x), 0.0
    if not already_preprocessed:
        x = centered
        window, hop = 400, 160
        padded = np.pad(x, (0, max(0, window - len(x))))
        frames = np.lib.stride_tricks.sliding_window_view(padded, window)[::hop]
        db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
        voiced = np.flatnonzero(db > db.max() - 35)
        start = max(0, int((voiced[0] * .010 - .5) * 16000))
        end = min(len(x), int((voiced[-1] * .010 + .025 + .5) * 16000))
        x = x[start:end]
        rms, peak = np.sqrt(np.mean(x ** 2)), np.max(np.abs(x))
        gain_db = min(-25 - 20 * np.log10(rms + 1e-10), -1 - 20 * np.log10(peak + 1e-10))
        x = x * 10 ** (gain_db / 20)
        if 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10) < -26:
            warnings.append("peak_limited_normalization")
    # Align and hash the same samples that the acoustic backend will read.
    x = pcm16(x).astype(float) / 32768
    return x, {
        "version": PREPROCESS_VERSION, "sample_rate": 16000,
        "duration_seconds": len(x) / 16000, "original_duration_seconds": original_duration,
        "leading_trim_seconds": start / 16000, "trailing_trim_seconds": (round(original_duration * 16000) - end) / 16000,
        "gain_db": float(gain_db), "already_preprocessed": already_preprocessed,
    }, warnings
