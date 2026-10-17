from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from .common import dump_json


def correlation(y, p, method="pearson"):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2 or np.ptp(y) < 1e-12 or np.ptp(p) < 1e-12:
        return None
    return float((pearsonr if method == "pearson" else spearmanr)(y, p).statistic)


def select_threshold(target_norm, predictions):
    """Max validation F1, then precision, then fewer flags. Strict pred < t.

    Groups ties before measuring; includes flag-none and flag-all operating points.
    Threshold may be nextafter(1, +inf) to express flag-all for clipped scores.
    """
    y, p = np.asarray(target_norm) < .25, np.asarray(predictions, float)
    if not len(y) or not y.any():
        return 0.0
    order = np.argsort(p, kind="stable")
    p, y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(p)) + 1, len(p)]
    positives = y.sum()
    tp = np.cumsum(y)[ends-1]
    precision = tp / ends
    f1 = 2 * tp / (ends + positives)
    best = max(range(len(ends)), key=lambda i: (f1[i], precision[i], -int(ends[i])))
    return float(np.nextafter(p[ends[best]-1], np.inf))


def metrics(y_raw, predictions_norm, threshold):
    y = np.asarray(y_raw, float)
    p = np.asarray(predictions_norm, float)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or ((y < 0) | (y > 2)).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("Metrics require nonempty finite labels (0-2) and predictions (0-1)")
    truth, predicted = y < .5, p < threshold
    tp, fp = int(np.sum(truth & predicted)), int(np.sum(~truth & predicted))
    fn, tn = int(np.sum(truth & ~predicted)), int(np.sum(~truth & ~predicted))
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    notes = []
    if not truth.any():
        notes.append("no_positive_labels_pr_auc_undefined")
    if truth.all():
        notes.append("no_negative_labels")
    return {"n": len(y), "mae_raw_0_2": float(np.mean(np.abs(y - 2*p))), "rmse_raw_0_2": float(np.sqrt(np.mean((y - 2*p)**2))), "pearson": correlation(y, p), "spearman": correlation(y, p, "spearman"), "mispronunciation": {"positive_label": "expert_score_raw < 0.5", "threshold_norm": float(threshold), "decision": "prediction_norm < threshold_norm", "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0, "pr_auc_average_precision": float(average_precision_score(truth, 1-p)) if truth.any() else None, "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "prevalence": float(truth.mean())}, "notes": notes}


def aggregate_metrics(records, predictions, grouping):
    groups = {}
    for r, p in zip(records, predictions):
        key = (r["utt_id"], r["word_index"]) if grouping == "word" else r["utt_id"]
        groups.setdefault(key, {"p": [], "y": r["word_accuracy_raw" if grouping == "word" else "sentence_accuracy_raw"]})["p"].append(float(p)*10)
    y, p = np.array([g["y"] for g in groups.values()]), np.array([np.mean(g["p"]) for g in groups.values()])
    return {"n": len(y), "prediction_name": "mean_phoneme_accuracy_summary", "target": grouping + "_accuracy", "mae_raw_0_10": float(np.mean(np.abs(y-p))), "rmse_raw_0_10": float(np.sqrt(np.mean((y-p)**2))), "pearson": correlation(y, p), "spearman": correlation(y, p, "spearman")}


def evaluate(records, predictions, threshold):
    p = np.asarray(predictions, float)
    y = np.array([r["target_raw"] for r in records])
    result = {"phoneme": metrics(y, p, threshold), "word_summary_vs_expert": aggregate_metrics(records, p, "word"), "sentence_summary_vs_expert": aggregate_metrics(records, p, "sentence"), "groups": {}}
    for field in ("phone", "age_group", "gender"):
        values = np.array([r[field] for r in records])
        result["groups"][field] = {}
        for value in sorted(set(values)):
            mask = values == value
            m = metrics(y[mask], p[mask], threshold)
            if field == "phone" and mask.sum() < 100:
                m["notes"].append("fewer_than_100_tokens_interpret_cautiously")
            result["groups"][field][str(value)] = m
    low = y < .5
    result["low_score_subset"] = metrics(y[low], p[low], threshold) if low.any() else None
    return result


def write_predictions(path, records, predictions, threshold):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["utt_id", "speaker", "age_group", "gender", "word_index", "word", "phone_index", "phone", "target_raw", "prediction_raw", "prediction_100", "true_mispronunciation", "needs_practice"]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r, p in zip(records, predictions):
            row = {k: r[k] for k in fields if k in r}
            row.update(prediction_raw=float(p)*2, prediction_100=float(p)*100, true_mispronunciation=r["target_raw"] < .5, needs_practice=float(p) < threshold)
            writer.writerow(row)


def write_report(directory, report, prediction_sets, *, title):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dump_json(directory / "metrics.json", report)
    _plots(directory, prediction_sets)


def _plots(directory, prediction_sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split, name, records, predictions, threshold in prediction_sets:
        p = np.asarray(predictions)
        y = np.array([r["target_raw"] for r in records])
        rng = np.random.default_rng(106)
        subset = rng.choice(len(y), min(len(y), 4000), replace=False)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        ax[0].scatter(y[subset], p[subset]*2, s=5, alpha=.15)
        ax[0].plot([0, 2], [0, 2], color="gray", linestyle="--")
        ax[0].set(xlabel="Expert phone score (0-2)", ylabel="Predicted phone score (0-2)", title=f"{name}: {split}", xlim=(-.05, 2.05), ylim=(-.05, 2.05))
        grouped = {}
        for r, pred in zip(records, p):
            grouped.setdefault(r["phone"], []).append(abs(r["target_raw"] - pred*2))
        keys = sorted(grouped, key=lambda k: np.mean(grouped[k]), reverse=True)[:12]
        ax[1].bar(keys, [np.mean(grouped[k]) for k in keys], color="#275e85")
        ax[1].set(ylabel="Mean absolute error (0-2)", title="12 largest errors by canonical phone")
        fig.savefig(directory / f"{split}_{name}.png", dpi=150)
        plt.close(fig)
