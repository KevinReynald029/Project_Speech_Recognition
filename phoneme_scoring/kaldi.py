"""Phone mapping and posterior parsers used by GOP forced alignment."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .common import PHONES, ScoringError


def base_phone(symbol):
    return re.sub(r"[012]$", "", re.sub(r"_[BIES]$", "", symbol))


def read_symbols(path):
    return {parts[0]: int(parts[1]) for line in Path(path).read_text().splitlines() if (parts := line.split())}


def choose_phone(symbols, phone, position):
    # The handoff removed stress. Prefer neutral stress 0, then 1/2. No variant
    # is selected by looking at test labels or acoustic scores.
    suffixes = ["_" + position, ""]
    for suffix in suffixes:
        for stem in [phone, phone + "0", phone + "1", phone + "2"]:
            if stem + suffix in symbols:
                return symbols[stem + suffix]
    raise ScoringError("invalid_reference", f"Acoustic phone inventory cannot represent {phone}")


def parse_alignment(text):
    """Parse ali-to-phones --write-lengths=true, keeping repeated phones separate."""
    # BasicPairVectorHolder's ark,t format is `key 12 3 ; 12 4 ; 1 10`,
    # NOT the bracketed PosteriorHolder format used by compute-gop.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0].split(maxsplit=1)) != 2:
        raise ScoringError("alignment_failed", "Kaldi returned no valid phone spans")
    key, body = lines[0].split(maxsplit=1)
    matches = [re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", item) for item in body.split(";")]
    if not matches or any(m is None or int(m[2]) <= 0 for m in matches):
        raise ScoringError("alignment_failed", "Invalid Kaldi phone-length pairs")
    return [(int(m[1]), int(m[2])) for m in matches]


def parse_matrix(text):
    try:
        body = text[text.index("[") + 1:text.rindex("]")]
        rows = [np.fromstring(line, sep=" ") for line in body.splitlines() if line.strip()]
        result = np.stack(rows)
        if result.ndim != 2 or not result.size or not np.isfinite(result).all():
            raise ValueError("Empty/nonfinite matrix")
        return result
    except (ValueError, IndexError) as error:
        raise ScoringError("alignment_failed", "Invalid Kaldi posterior matrix") from error


def pdf_phone_mapping(text):
    mapping = {}
    for line in text.splitlines():
        match = re.search(r"phone\s*=\s*(\S+)", line)
        if match:
            phone = base_phone(match.group(1))
            # Includes ordinary pdf= and separate forward/self-loop pdf IDs.
            for number in re.findall(r"(?:forward-pdf|self-loop-pdf|\bpdf)\s*=\s*(\d+)", line):
                mapping.setdefault(int(number), set()).add(phone)
    if not mapping:
        raise ScoringError("backend_unavailable", "Could not read Kaldi transition-to-phone mapping")
    return mapping


def collapse_posteriors(pdf_logs, mapping):
    # Tied pdfs may belong to more than one phone, as in Kaldi compute-gop.
    if set(range(pdf_logs.shape[1])) - set(mapping):
        raise ScoringError("alignment_failed", "Unmapped acoustic pdf columns")
    if not np.allclose(logsumexp(pdf_logs, axis=1), 0, atol=.02):
        raise ScoringError("backend_unavailable", "Expected nnet3 log-softmax posteriors; likelihood/chain outputs are unsupported")
    output = []
    for phone in PHONES:
        ids = [i for i in range(pdf_logs.shape[1]) if phone in mapping.get(i, set())]
        if not ids:
            raise ScoringError("backend_unavailable", f"No acoustic posterior for {phone}")
        output.append(logsumexp(pdf_logs[:, ids], axis=1))
    return np.maximum(np.stack(output, axis=1), np.log(1e-10))
