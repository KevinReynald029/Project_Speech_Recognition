from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .common import normalize_reference, sha256_file


def read_dataset(directory, *, official_counts=True):
    root = Path(directory)
    datasets, seen, speakers = {}, set(), {}
    split_map = {}
    with (root / "splits.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["utt_id"] in split_map:
                raise ValueError("Duplicate utterance in splits.csv")
            split_map[row["utt_id"]] = row["split"]
    for split in ("train", "val", "test"):
        rows = []
        file = root / f"manifest_{split}.jsonl"
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            r = json.loads(line)
            uid = r["utt_id"]
            if uid in seen or r["split"] != split or split_map.get(uid) != split:
                raise ValueError(f"Duplicate/incorrect split for {uid}")
            speaker = str(r["speaker"])
            if speaker in speakers and speakers[speaker] != split:
                raise ValueError("Speaker leakage across splits")
            speakers[speaker] = split
            seen.add(uid)
            ref = normalize_reference(r["words"])
            if r["text"] != " ".join(w["word"] for w in ref):
                raise ValueError(f"Reference text mismatch for {uid}")
            for w in r["words"]:
                y = np.asarray(w["phone_scores"], float)
                yn = np.asarray(w["phone_scores_norm"], float)
                if y.shape != (len(w["phones"]),) or yn.shape != y.shape or not np.isfinite(y).all() or ((y < 0) | (y > 2)).any() or not np.allclose(y/2, yn):
                    raise ValueError(f"Invalid phone labels for {uid}")
                if not 0 <= float(w["word_accuracy"]) <= 10:
                    raise ValueError("Invalid word accuracy")
            if not np.isfinite(float(r["age"])) or float(r["age"]) < 0 or not 0 <= float(r["sentence_accuracy"]) <= 10:
                raise ValueError("Invalid age/sentence score")
            rows.append(r)
        if official_counts and len(rows) != {"train": 2100, "val": 400, "test": 2500}[split]:
            raise ValueError("Unexpected official split counts")
        datasets[split] = rows
    if seen != set(split_map):
        raise ValueError("Manifests/splits.csv do not cover the same utterances")
    return datasets


def flatten(rows):
    phones = []
    for r in rows:
        index = 0
        for wi, w in enumerate(r["words"]):
            for pi, (phone, score) in enumerate(zip(w["phones"], w["phone_scores"])):
                phones.append({"utt_id": r["utt_id"], "speaker": r["speaker"], "age": r["age"], "age_group": "child" if r["age"] < 18 else "adult", "gender": r["gender"], "word_index": wi, "word": w["word"], "phone_index": index, "phone_index_in_word": pi, "phone": phone, "target_raw": float(score), "word_accuracy_raw": float(w["word_accuracy"]), "sentence_accuracy_raw": float(r["sentence_accuracy"])})
                index += 1
    return phones


def audit_dataset(root, datasets):
    root = Path(root)
    result = {"speaker_disjoint": True, "splits_csv_matches_manifests": True, "splits": {}, "manifest_sha256": {}}
    for split, rows in datasets.items():
        labels = flatten(rows)
        result["manifest_sha256"][split] = sha256_file(root / f"manifest_{split}.jsonl")
        result["splits"][split] = {"utterances": len(rows), "speakers": len({r["speaker"] for r in rows}), "phones": len(labels), "mispronunciation_rate": float(np.mean([r["target_raw"] < .5 for r in labels])), "audio_files_present": sum((root / r["audio_proc"]).is_file() for r in rows)}
    return result


def dataset_directory(project):
    """Find the local extracted handoff or the metadata already in the Git repo."""
    project = Path(project).resolve()
    for directory in (project / "data/handoff", project / "handoff"):
        if all((directory / f"manifest_{split}.jsonl").is_file() for split in ("train", "val", "test")):
            return directory
    raise FileNotFoundError("Expected handoff manifests in handoff/ or data/handoff/")
