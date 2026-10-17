"""Linux Kaldi adapter using the official SpeechOcean762 nnet3 TDNN recipe.

Requires a ready Kaldi installation + native-English TDNN, language directory,
and matching i-vector extractor. Does not download, train, or fake that model.
"""
from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.special import logsumexp

from .common import PHONES, PREPROCESS_VERSION, ScoringError, audio_hash, load_json, reference_hash, sha256_file
from .evidence import EVIDENCE_VERSION, validate_evidence


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


class KaldiBackend:
    def __init__(self, config):
        if os.name != "posix":
            raise ScoringError("backend_unavailable", "The Kaldi adapter requires a Linux/POSIX worker; use cached evidence for replay on Windows")
        config = load_json(config) if isinstance(config, (str, Path)) else config
        try:
            self.paths = {k: Path(config[k]).resolve() for k in ("recipe", "model", "lang", "ivector_extractor")}
            if any(re.search(r"[\s|'\";]", str(p)) for p in self.paths.values()):
                raise ValueError("Kaldi paths must not contain whitespace, quotes, or shell metacharacters")
            required = {"model": ["tree", "final.mdl", "cmvn_opts", "phones.txt"], "lang": ["phones.txt", "words.txt", "oov.int", "phones/disambig.int"], "ivector_extractor": ["final.ie", "final.dubm", "final.mat"]}
            h = hashlib.sha256()
            for group, names in required.items():
                for name in names:
                    f = self.paths[group] / name
                    if not f.is_file():
                        raise ValueError(f"Missing Kaldi asset: {f}")
            # Fingerprint every file consumed from the model/extractor plus the
            # language tables, so cached evidence cannot cross acoustic models.
            for group in ("model", "ivector_extractor"):
                for f in sorted(self.paths[group].glob("*")):
                    if f.is_file() and (f.suffix in {".mdl", ".raw", ".ie", ".dubm", ".mat", ".stats"} or f.name in {"tree", "phones.txt", "cmvn_opts", "splice_opts", "online_cmvn.conf", "global_cmvn.stats"}):
                        h.update((group + "/" + f.name + sha256_file(f)).encode())
            for name in required["lang"]:
                h.update((name + sha256_file(self.paths["lang"] / name)).encode())
            recipe = self.paths["recipe"]
            for name in ("path.sh", "conf/mfcc_hires.conf", "steps/make_mfcc.sh", "steps/online/nnet2/extract_ivectors_online.sh", "steps/nnet3/compute_output.sh", "steps/align_mapped.sh"):
                h.update((name + sha256_file(recipe / name)).encode())
            h.update(sha256_file(Path(__file__).with_name("kaldi_evidence.sh")).encode())
            h.update(sha256_file(__file__).encode())
            self.extractor_id = "kaldi-nnet3-" + h.hexdigest()
            self.timeout = float(config.get("timeout_seconds", 300))
            if not np.isfinite(self.timeout) or self.timeout <= 0:
                raise ValueError("timeout_seconds must be positive")
            self.phones = read_symbols(self.paths["lang"] / "phones.txt")
            self.words = read_symbols(self.paths["lang"] / "words.txt")
            self.reverse_phones = {v: k for k, v in self.phones.items()}
            if self.phones != read_symbols(self.paths["model"] / "phones.txt"):
                raise ValueError("Model/lang phone symbol tables differ")
            if base_phone(self.reverse_phones.get(1, "")) != "SIL":
                raise ValueError("Official graph compiler requires SIL with ID 1")
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise ScoringError("backend_unavailable", str(error)) from error

    def extract(self, audio, words):
        try:
            with tempfile.TemporaryDirectory(prefix="nihals-gop-") as temp:
                root = Path(temp)
                # Kaldi archive and script conventions cannot represent spaces.
                if re.search(r"[\s|'\";]", str(root)):
                    raise ScoringError("backend_unavailable", "Set TMPDIR to a path without spaces")
                data = root / "data"
                data.mkdir()
                sf.write(root / "input.wav", audio, 16000, subtype="PCM_16")
                (data / "wav.scp").write_text(f"sample {root.as_posix()}/input.wav\n")
                (data / "utt2spk").write_text("sample sample\n")
                (data / "spk2utt").write_text("sample sample\n")
                unk = self.words.get("<UNK>", self.words.get("<unk>"))
                if unk is None:
                    raise ScoringError("backend_unavailable", "Language words.txt has no UNK symbol")
                (data / "text").write_text("sample " + " ".join("<UNK>" if "<UNK>" in self.words else "<unk>" for _ in words) + "\n")
                (root / "text.int").write_text("sample " + " ".join(str(unk) for _ in words) + "\n")
                phone_lines = []
                for wi, word in enumerate(words):
                    phones = word["phones"]
                    ids = [choose_phone(self.phones, p, "S" if len(phones) == 1 else "B" if j == 0 else "E" if j == len(phones)-1 else "I") for j, p in enumerate(phones)]
                    phone_lines.append(f"sample.{wi} " + " ".join(map(str, ids)))
                (root / "text-phone.int").write_text("\n".join(phone_lines) + "\n")
                script = Path(__file__).with_name("kaldi_evidence.sh")
                command = ["bash", str(script), str(root)] + [str(self.paths[k]) for k in ("recipe", "model", "lang", "ivector_extractor")]
                # Kill the whole Kaldi process group on timeout, including run.pl
                # children, before TemporaryDirectory removes their inputs.
                run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
                try:
                    stdout, stderr = run.communicate(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    _stop_process_group(run)
                    raise ScoringError("backend_timeout", "Kaldi extraction timed out")
                except BaseException:
                    _stop_process_group(run)
                    raise
                if run.returncode:
                    logs = sorted(root.rglob("*.log"))
                    tail = "\n".join(p.read_text(errors="replace")[-700:] for p in logs[-3:])
                    raise ScoringError("alignment_failed", "Kaldi extraction failed; check installation/model compatibility. " + (stdout + stderr + tail)[-2400:])
                matrix = parse_matrix((root / "posteriors.txt").read_text())
                mapping = pdf_phone_mapping((root / "transitions.txt").read_text())
                lpps = collapse_posteriors(matrix, mapping)
                spans = parse_alignment((root / "lengths.txt").read_text())
                if sum(n for _, n in spans) != len(lpps):
                    raise ScoringError("alignment_failed", "Alignment/posterior frame count mismatch")
                segments, frame = [], 0
                for pid, n in spans:
                    phone = base_phone(self.reverse_phones.get(pid, ""))
                    if phone in PHONES:
                        segments.append({"phone": phone, "start_seconds": frame * .010, "end_seconds": (frame+n) * .010, "lpp": lpps[frame:frame+n].mean(axis=0).tolist()})
                    elif phone not in {"SIL", "SPN", "NSN"}:
                        raise ScoringError("alignment_failed", f"Unexpected phone in alignment: {phone}")
                    frame += n
                e = {"schema_version": EVIDENCE_VERSION, "preprocess_version": PREPROCESS_VERSION, "extractor_id": self.extractor_id, "phone_order": list(PHONES), "audio_sha256": audio_hash(audio), "reference_sha256": reference_hash(words), "segments": segments, "warnings": ["stress_neutral_acoustic_mapping", "forced_alignment_does_not_prove_phone_was_spoken"]}
                return validate_evidence(e, audio, words, self.extractor_id)
        except ScoringError:
            raise
        except subprocess.TimeoutExpired as error:
            raise ScoringError("backend_timeout", "Kaldi extraction timed out") from error
        except FileNotFoundError as error:
            raise ScoringError("backend_unavailable", "Kaldi/Bash or an acoustic output is missing") from error
        except (OSError, ValueError) as error:
            raise ScoringError("alignment_failed", str(error)) from error


def _stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
