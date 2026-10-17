from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .audio import preprocess
from .common import FEATURE_VERSION, METHOD_VERSION, PHONES, PREPROCESS_VERSION, ScoringError, dump_json, load_json, normalize_reference, sha256_file
from .data import audit_dataset, flatten, read_dataset
from .evaluation import evaluate, select_threshold, write_predictions, write_report
from .evidence import save_evidence, segment_features, validate_evidence
from .model import RidgeCalibrator
from .boosting import BoostedCalibrator, fit_candidates


def baseline_experiment(data, output):
    datasets = read_dataset(data)
    train = flatten(datasets["train"])
    mean = float(np.mean([r["target_raw"]/2 for r in train]))
    report = {"kind": "label_only_baselines", "status_note": "Eksperimen nyata pada label handoff. Ini baseline konstan; model GOP/Ridge belum dijalankan dan tidak diwakili oleh angka di bawah.", "data_audit": audit_dataset(data, datasets), "train_mean_raw_0_2": mean*2, "thresholds_selected_on": "val", "thresholds": {}, "results": {}}
    plots = []
    output = Path(output)
    for name, value in (("always_2", 1.), ("train_mean", mean)):
        val = flatten(datasets["val"])
        threshold = select_threshold(np.array([r["target_raw"]/2 for r in val]), np.full(len(val), value))
        report["thresholds"][name] = threshold
        for split in ("val", "test"):
            records = flatten(datasets[split])
            predictions = np.full(len(records), value)
            report["results"].setdefault(split, {})[name] = evaluate(records, predictions, threshold)
            write_predictions(output / f"predictions_{split}_{name}.csv", records, predictions, threshold)
            plots.append((split, name, records, predictions, threshold))
    write_report(output, report, plots, title="Evaluasi Baseline pada Label Asli")
    return report


def extract_dataset(data, cache, backend, splits=("train", "val", "test")):
    data, cache = Path(data), Path(cache)
    datasets = read_dataset(data)
    cache.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in splits:
        entries = []
        for row in datasets[split]:
            uid = row["utt_id"]
            try:
                words = normalize_reference(row["words"])
                audio, _, _ = preprocess(data / row["audio_proc"], None, already_preprocessed=True)
                evidence = validate_evidence(backend.extract(audio, words), audio, words, backend.extractor_id)
                file = save_evidence(cache, evidence, audio, words)
                entries.append({"utt_id": uid, "status": "ok", "file": file.name, "sha256": sha256_file(file)})
            except ScoringError as error:
                entries.append({"utt_id": uid, "status": "error", "code": error.code, "message": str(error)})
        index = {"schema_version": "evidence-index-v1", "extractor_id": backend.extractor_id, "manifest_sha256": sha256_file(data / f"manifest_{split}.jsonl"), "split": split, "entries": entries}
        dump_json(cache / f"index_{split}.json", index)
        summaries[split] = {"success": sum(e["status"] == "ok" for e in entries), "failure": sum(e["status"] != "ok" for e in entries)}
    return summaries


def load_feature_split(data, cache, split, rows, expected_extractor=None):
    data, cache = Path(data), Path(cache)
    index_file = cache / f"index_{split}.json"
    index = load_json(index_file)
    if index["schema_version"] != "evidence-index-v1" or index["split"] != split or index["manifest_sha256"] != sha256_file(data / f"manifest_{split}.jsonl"):
        raise ValueError("Evidence index does not match this exact manifest/split")
    extractor = index["extractor_id"]
    if expected_extractor is not None and extractor != expected_extractor:
        raise ValueError("Mixed acoustic models across splits")
    by_id = {e["utt_id"]: e for e in index["entries"]}
    if len(by_id) != len(index["entries"]) or set(by_id) != {r["utt_id"] for r in rows}:
        raise ValueError("Evidence index must cover every manifest ID exactly once, including failures")
    good, features, failures = [], [], []
    for row in rows:
        entry = by_id[row["utt_id"]]
        if entry["status"] != "ok":
            failures.append({"utt_id": row["utt_id"], "age_group": "child" if row["age"] < 18 else "adult", "code": entry.get("code", "unknown")})
            continue
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Evidence index must contain a basename, not an arbitrary path")
        file = cache / entry["file"]
        if sha256_file(file) != entry["sha256"]:
            raise ValueError("Evidence checksum mismatch")
        words = normalize_reference(row["words"])
        # Reverify actual PCM + reference at training time, not only cache metadata.
        audio, _, _ = preprocess(data / row["audio_proc"], None, already_preprocessed=True)
        evidence = validate_evidence(load_json(file), audio, words, extractor)
        features.extend(segment_features(s) for s in evidence["segments"])
        good.append(row)
    if not good:
        raise ValueError(f"No successful acoustic evidence for {split}; extract with a real Kaldi model first")
    coverage = {"utterances_total": len(rows), "utterances_scored": len(good), "utterance_coverage": len(good)/len(rows), "phones_total": len(flatten(rows)), "phones_scored": len(features), "failure_codes": dict(Counter(f["code"] for f in failures)), "failures_by_age_group": dict(Counter(f["age_group"] for f in failures)), "failures": failures}
    return np.stack(features), flatten(good), extractor, coverage


def train_experiment(data, cache, output, model_path=None, legacy_model=None):
    data, cache, output = Path(data), Path(cache), Path(output)
    datasets = read_dataset(data)
    x_train, train, extractor, train_coverage = load_feature_split(data, cache, "train", datasets["train"])
    x_val, val, _, val_coverage = load_feature_split(data, cache, "val", datasets["val"], extractor)
    y_train = np.array([r["target_raw"]/2 for r in train])
    y_val = np.array([r["target_raw"]/2 for r in val])
    metadata = {"method_version": METHOD_VERSION, "feature_version": FEATURE_VERSION, "preprocess_version": PREPROCESS_VERSION, "phone_order": list(PHONES), "extractor_id": extractor, "practice_threshold_norm": .25, "threshold_locked_on_validation": True}
    artifact, val_predictions = fit_candidates(x_train, y_train, x_val, y_val, metadata)
    threshold = select_threshold(y_val, val_predictions)
    artifact.update({"practice_threshold_norm": threshold, "trained_on": "train_only", "selection": "lowest validation MAE among a fixed 24-candidate grid; first candidate breaks exact ties", "train_manifest_sha256": sha256_file(data / "manifest_train.jsonl"), "validation_manifest_sha256": sha256_file(data / "manifest_val.jsonl"), "train_evidence_index_sha256": sha256_file(cache / "index_train.json"), "validation_evidence_index_sha256": sha256_file(cache / "index_val.json"), "train_phone_count": len(train), "validation_phone_count": len(val), "train_mean_norm": float(y_train.mean()), "baseline_mean_norm_all_train": float(np.mean([r["target_raw"]/2 for r in flatten(datasets["train"])])), "warning": "This artifact estimates phoneme accuracy, not holistic pronunciation/fluency/prosody."})
    calibrator = BoostedCalibrator(artifact)
    model_path = Path(model_path) if model_path else output / "calibrator.json"
    # Model + decision threshold are frozen before test evidence is loaded.
    calibrator.save(model_path)
    # Training predictions are explicitly in-sample diagnostics, not a benchmark.
    write_predictions(output / "predictions_train_gop_calibrated.csv", train, calibrator.predict(x_train), threshold)
    x_test, test, _, test_coverage = load_feature_split(data, cache, "test", datasets["test"], extractor)
    predictions_test = calibrator.predict(x_test)
    val_all = flatten(datasets["val"])
    baseline_values = {"always_2": 1., "train_mean": artifact["baseline_mean_norm_all_train"]}
    baseline_thresholds = {n: select_threshold(np.array([r["target_raw"]/2 for r in val_all]), np.full(len(val_all), v)) for n, v in baseline_values.items()}
    report = {"kind": "gop_hgb_mae_v2_acoustic_experiment", "status_note": "Model baru memakai fitur GOP dan gradient boosting. Kandidat dan ambang dipilih dari train/validation. Test yang sama sudah pernah dibaca pada eksperimen v1, jadi hasil ini adalah perbandingan ulang pada benchmark yang sama, bukan pengujian independen baru. Coverage dan baseline dihitung pada rekaman yang sama.", "model_sha256": sha256_file(model_path), "data_audit": audit_dataset(data, datasets), "coverage": {"train": train_coverage, "val": val_coverage, "test": test_coverage}, "thresholds_selected_on": "val", "thresholds": {"gop_calibrated": threshold, **baseline_thresholds}, "results": {}}
    legacy = RidgeCalibrator(legacy_model) if legacy_model else None
    if legacy:
        report["legacy_model_sha256"] = sha256_file(legacy_model)
        report["legacy_comparison_note"] = "Frozen v1 coefficients applied to the new complete evidence, including the four recovered utterances. Not a tuning candidate."
    plots = []
    for split, records, prediction in (("val", val, val_predictions), ("test", test, predictions_test)):
        methods = {"gop_calibrated": (prediction, threshold), **{n: (np.full(len(records), v), baseline_thresholds[n]) for n, v in baseline_values.items()}}
        if legacy:
            features = x_val if split == "val" else x_test
            methods["ridge_v1"] = (legacy.predict(features),legacy.threshold)
        report["results"][split] = {}
        for name, (predicted, cutoff) in methods.items():
            report["results"][split][name] = evaluate(records, predicted, cutoff)
            write_predictions(output / f"predictions_{split}_{name}.csv", records, predicted, cutoff)
            plots.append((split, name, records, predicted, cutoff))
    model_mae = report["results"]["test"]["gop_calibrated"]["phoneme"]["mae_raw_0_2"]
    report["beats_baseline_test_mae"] = {n: model_mae < report["results"]["test"][n]["phoneme"]["mae_raw_0_2"] for n in baseline_values}
    write_report(output, report, plots, title="Hasil GOP setelah perbaikan model")
    dump_json(output / "alignment_failures.json", report["coverage"])
    return report
