"""Direct Kaldi binary adapter (Windows or POSIX), no Bash or recipe mutation.

Uses the same nnet3/MFCC/online i-vector/graph/mapped-alignment operations as the
official recipe. Each utterance has an independent i-vector state, matching app
inference rather than carrying state across a known corpus speaker.
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.special import logsumexp

from .common import PHONES, PREPROCESS_VERSION, ScoringError, audio_hash, load_json, reference_hash, sha256_file
from .evidence import EVIDENCE_VERSION, validate_evidence
from .kaldi import base_phone, choose_phone, parse_alignment, pdf_phone_mapping, read_symbols


def read_matrix_archive(path):
    """Stream standard Kaldi FM/DM matrices, avoiding huge textual posteriors."""
    with Path(path).open("rb") as stream:
        while True:
            key = bytearray()
            while (char := stream.read(1)) and char.isspace():
                pass
            if not char:
                return
            key.extend(char)
            while (char := stream.read(1)) != b" ":
                if not char:
                    raise ValueError("Truncated Kaldi archive key")
                key.extend(char)
            if stream.read(2) != b"\0B":
                raise ValueError("Expected binary Kaldi matrix")
            token = bytearray()
            while (char := stream.read(1)) != b" ":
                if not char:
                    raise ValueError("Truncated Kaldi matrix token")
                token.extend(char)
            if bytes(token) not in (b"FM", b"DM"):
                raise ValueError(f"Unsupported Kaldi matrix type {bytes(token)!r}")
            if stream.read(1) != b"\4":
                raise ValueError("Invalid Kaldi row dimension")
            rows = struct.unpack("<i", stream.read(4))[0]
            if stream.read(1) != b"\4":
                raise ValueError("Invalid Kaldi column dimension")
            cols = struct.unpack("<i", stream.read(4))[0]
            if not 0 < rows <= 100000 or not 0 < cols <= 100000:
                raise ValueError("Invalid Kaldi matrix shape")
            dtype = np.dtype("<f4" if bytes(token) == b"FM" else "<f8")
            raw = stream.read(rows * cols * dtype.itemsize)
            if len(raw) != rows * cols * dtype.itemsize:
                raise ValueError("Truncated Kaldi matrix payload")
            yield key.decode("ascii"), np.frombuffer(raw, dtype=dtype).reshape(rows, cols)


class NativeKaldiBackend:
    def __init__(self, config):
        config = load_json(config) if isinstance(config, (str, Path)) else config
        self.config = config
        self.paths = {k: Path(config[k]).resolve() for k in ("bin", "model", "lang", "ivector_extractor")}
        self.timeout = float(config.get("timeout_seconds", 600))
        self.scratch = Path(config.get("scratch", tempfile.gettempdir())).resolve()
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env["PATH"] = str(self.paths["bin"]) + os.pathsep + self.env.get("PATH", "")
        self.env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.phones = read_symbols(self.paths["lang"] / "phones.txt")
        self.reverse_phones = {v: k for k, v in self.phones.items()}
        self.words = read_symbols(self.paths["lang"] / "words.txt")
        if base_phone(self.reverse_phones.get(1, "")) != "SIL":
            raise ScoringError("backend_unavailable", "Graph compiler requires SIL phone ID 1")
        cmvn = (self.paths["model"] / "cmvn_opts").read_text().strip().split()
        if sorted(cmvn) != ["--norm-means=false", "--norm-vars=false"]:
            raise ScoringError("backend_unavailable", "Native adapter currently requires model CMVN means=false, vars=false")
        h = hashlib.sha256(b"native-kaldi-mfcc40-dither0-perutt-ivector10-v1")
        for group in ("model", "ivector_extractor"):
            names = ["final.mdl", "tree", "cmvn_opts"] if group == "model" else ["final.ie", "final.dubm", "final.mat", "global_cmvn.stats", "online_cmvn.conf", "splice_opts"]
            for name in names:
                h.update((group+"/"+name+sha256_file(self.paths[group]/name)).encode())
        for name in ("phones.txt", "words.txt", "phones/disambig.int"):
            h.update((name+sha256_file(self.paths["lang"]/name)).encode())
        for name in ("compute-mfcc-feats", "ivector-extract-online2", "nnet3-compute", "compile-train-graphs-without-lexicon", "align-compiled-mapped", "ali-to-phones", "show-transitions"):
            h.update((name+sha256_file(self.executable(name))).encode())
        h.update(sha256_file(__file__).encode())
        h.update(sha256_file(Path(__file__).with_name("kaldi.py")).encode())
        self.extractor_id = "kaldi-native-nnet3-" + h.hexdigest()
        result = subprocess.run([str(self.executable("show-transitions")), str(self.paths["lang"] / "phones.txt"), str(self.paths["model"] / "final.mdl")], env=self.env, capture_output=True, timeout=self.timeout)
        if result.returncode:
            raise ScoringError("backend_unavailable", result.stderr.decode(errors="replace")[-2000:])
        self.pdf_mapping = pdf_phone_mapping(result.stdout.decode())
        self.phone_pdf_ids = [[i for i, phones in self.pdf_mapping.items() if p in phones] for p in PHONES]
        if any(not ids for ids in self.phone_pdf_ids):
            raise ScoringError("backend_unavailable", "Acoustic model does not cover all 39 phones")

    def executable(self, name):
        return self.paths["bin"] / (name + (".exe" if os.name == "nt" else ""))

    def run(self, name, args, root):
        command = [str(self.executable(name))] + list(map(str, args))
        try:
            result = subprocess.run(command, env=self.env, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            raise ScoringError("backend_timeout", f"Kaldi {name} exceeded {self.timeout}s") from error
        (root/(name+".log")).write_bytes(result.stderr)
        if result.returncode:
            raise ScoringError("alignment_failed", name + ": " + result.stderr.decode(errors="replace")[-3000:])
        return result

    def align_with_retry(self, root, kp, utterance_ids):
        """Keep initial paths; retry missing paths with a bounded wider search.

        This changes search only: same PCM, references and acoustic posteriors.
        No human labels are read. An absent path remains a scoring failure.
        """
        spans, recovered = {}, set()
        for attempt, beam, retry_beam in (("initial", 10, 40), ("wide", 100, 1000)):
            if len(spans) == len(utterance_ids):
                break
            alignment = root / ("ali_" + attempt + ".ark")
            lengths = root / ("lengths_" + attempt + ".txt")
            try:
                self.run("align-compiled-mapped", ["--transition-scale=1", "--acoustic-scale=0.1", "--self-loop-scale=0.1", f"--beam={beam}", f"--retry-beam={retry_beam}", kp(self.paths["model"] / "final.mdl"), "ark:" + kp(root / "graphs.ark"), "ark:" + kp(root / "posteriors.ark"), "ark:" + kp(alignment)], root)
            except ScoringError as error:
                if error.code != "alignment_failed":
                    raise
            finally:
                log = root / "align-compiled-mapped.log"
                if log.exists():
                    log.replace(root / ("align_" + attempt + ".log"))
            if not alignment.exists() or not alignment.stat().st_size:
                continue
            try:
                self.run("ali-to-phones", ["--write-lengths=true", kp(self.paths["model"] / "final.mdl"), "ark:" + kp(alignment), "ark,t:" + kp(lengths)], root)
            except ScoringError as error:
                if error.code != "alignment_failed":
                    raise
                continue
            for line in lengths.read_text().splitlines():
                if not line.strip():
                    continue
                uid = line.split()[0]
                if uid not in utterance_ids or uid in spans:
                    continue
                spans[uid] = parse_alignment(line)
                if attempt == "wide":
                    recovered.add(uid)
        return spans, recovered

    def extract(self, audio, words):
        result = self.extract_many([("sample", audio, words)])
        value = result["sample"]
        if isinstance(value, ScoringError):
            raise value
        return value

    def extract_many(self, items):
        """Batch setup/model load, retaining separate state and failure per utterance."""
        if not items:
            return {}
        with tempfile.TemporaryDirectory(prefix="kaldi-native-", dir=self.scratch) as temp:
            root = Path(temp)
            def kp(p):
                text = Path(p).as_posix()
                if re.search(r"[\s|]", text):
                    raise ScoringError("backend_unavailable", "Kaldi archive paths must not contain spaces or pipes")
                return text
            wav_lines, spk_lines, word_lines, phone_lines = [], [], [], []
            for uid, audio, words in items:
                if not re.fullmatch(r"[A-Za-z0-9_-]+", uid):
                    raise ValueError("Invalid utterance ID")
                sf.write(root/(uid+".wav"), audio, 16000, subtype="PCM_16")
                wav_lines.append(f"{uid} {kp(root/(uid+'.wav'))}")
                spk_lines.append(f"{uid} {uid}")
                unk = self.words.get("<UNK>", self.words.get("<unk>"))
                if unk is None:
                    raise ScoringError("backend_unavailable", "Missing UNK word ID")
                word_lines.append(uid + " " + " ".join(str(unk) for _ in words))
                for wi, word in enumerate(words):
                    ps = word["phones"]
                    ids = [choose_phone(self.phones, p, "S" if len(ps)==1 else "B" if pi==0 else "E" if pi==len(ps)-1 else "I") for pi,p in enumerate(ps)]
                    phone_lines.append(f"{uid}.{wi} " + " ".join(map(str, ids)))
            for name, lines in (("wav.scp",wav_lines),("spk2utt",spk_lines),("text.int",word_lines),("text-phone.int",phone_lines)):
                (root/name).write_text("\n".join(lines)+"\n", encoding="ascii")
            self.run("compute-mfcc-feats", ["--use-energy=false", "--num-mel-bins=40", "--num-ceps=40", "--low-freq=20", "--high-freq=-400", "--sample-frequency=16000", "--dither=0", "scp:"+kp(root/"wav.scp"), "ark,scp:"+kp(root/"mfcc.ark")+","+kp(root/"mfcc.scp")], root)
            extractor = self.paths["ivector_extractor"]
            (root/"splice.conf").write_text("\n".join((extractor/"splice_opts").read_text().split())+"\n")
            conf = [f"--cmvn-config={kp(extractor/'online_cmvn.conf')}", f"--splice-config={kp(root/'splice.conf')}", f"--lda-matrix={kp(extractor/'final.mat')}", f"--global-cmvn-stats={kp(extractor/'global_cmvn.stats')}", f"--diag-ubm={kp(extractor/'final.dubm')}", f"--ivector-extractor={kp(extractor/'final.ie')}", "--ivector-period=10", "--num-gselect=5", "--min-post=0.025", "--posterior-scale=0.1", "--max-remembered-frames=1000", "--max-count=0"]
            (root/"ivector.conf").write_text("\n".join(conf)+"\n")
            self.run("ivector-extract-online2", ["--config="+kp(root/"ivector.conf"), "ark:"+kp(root/"spk2utt"), "scp:"+kp(root/"mfcc.scp"), "ark,scp:"+kp(root/"ivectors.ark")+","+kp(root/"ivectors.scp")], root)
            self.run("nnet3-compute", ["--use-gpu=no", "--use-priors=false", "--frame-subsampling-factor=1", "--online-ivectors=scp:"+kp(root/"ivectors.scp"), "--online-ivector-period=10", kp(self.paths["model"]/"final.mdl"), "scp:"+kp(root/"mfcc.scp"), "ark:"+kp(root/"posteriors.ark")], root)
            self.run("compile-train-graphs-without-lexicon", ["--read-disambig-syms="+kp(self.paths["lang"]/"phones/disambig.int"), kp(self.paths["model"]/"tree"), kp(self.paths["model"]/"final.mdl"), "ark,t:"+kp(root/"text.int"), "ark,t:"+kp(root/"text-phone.int"), "ark:"+kp(root/"graphs.ark")], root)
            spans, recovered = self.align_with_retry(root, kp, [uid for uid, _, _ in items])
            originals = {uid:(audio,words) for uid,audio,words in items}
            output = {}
            for uid, pdf_logs in read_matrix_archive(root/"posteriors.ark"):
                audio, words = originals[uid]
                try:
                    if uid not in spans:
                        raise ScoringError("alignment_failed", "No alignment for this utterance")
                    if sum(n for _,n in spans[uid]) != len(pdf_logs):
                        raise ScoringError("alignment_failed", "Alignment/posterior frame counts differ")
                    if pdf_logs.shape[1] != len(self.pdf_mapping) or not np.allclose(logsumexp(pdf_logs,axis=1),0,atol=.02):
                        raise ScoringError("backend_unavailable", "Acoustic output is not the expected log-softmax posterior")
                    lpp = np.stack([logsumexp(pdf_logs[:,ids],axis=1) for ids in self.phone_pdf_ids],axis=1)
                    lpp = np.maximum(lpp, np.log(1e-10))
                    segments, frame = [], 0
                    for pid,n in spans[uid]:
                        phone = base_phone(self.reverse_phones.get(pid,""))
                        if phone in PHONES:
                            segments.append({"phone":phone,"start_seconds":frame*.010,"end_seconds":(frame+n)*.010,"lpp":lpp[frame:frame+n].mean(axis=0).astype(float).tolist()})
                        elif phone not in {"SIL","SPN","NSN"}:
                            raise ScoringError("alignment_failed", "Unexpected aligned phone: "+phone)
                        frame += n
                    e={"schema_version":EVIDENCE_VERSION,"preprocess_version":PREPROCESS_VERSION,"extractor_id":self.extractor_id,"phone_order":list(PHONES),"audio_sha256":audio_hash(audio),"reference_sha256":reference_hash(words),"segments":segments,"warnings":(["alignment_recovered_with_wide_beam"] if uid in recovered else [])+["independent_utterance_ivector_state","mfcc_dither_disabled_for_reproducibility","forced_alignment_does_not_prove_phone_was_spoken"]}
                    output[uid] = validate_evidence(e,audio,words,self.extractor_id)
                except ScoringError as error:
                    output[uid] = error
            for uid,_,_ in items:
                output.setdefault(uid,ScoringError("alignment_failed","Missing acoustic output"))
            return output
