from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import ScoringError, dump_json, load_json
from .backend import create_backend
from .data import audit_dataset, read_dataset
from .evidence import CachedBackend
from .experiments import baseline_experiment, extract_dataset, train_experiment
from .kaldi import KaldiBackend
from .boosting import load_calibrator
from .service import PronunciationScorer, failure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pronunciation scoring: audit, baselines, Kaldi evidence, train/evaluate, infer")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "baselines", "extract", "train"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, required=True, help="Folder with the three manifests, splits.csv and wav16k/")
        if command in ("audit", "baselines", "train"):
            p.add_argument("--output", type=Path, required=True)
        if command in ("extract", "train"):
            p.add_argument("--cache", type=Path, required=True)
        if command == "extract":
            p.add_argument("--kaldi-config", type=Path, required=True)
            p.add_argument("--splits", choices=["train", "val", "test"], nargs="+", default=["train", "val", "test"])
        if command == "train":
            p.add_argument("--model", type=Path)
            p.add_argument("--legacy-model", type=Path)
    p = sub.add_parser("score")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="JSON with reference_words, no labels")
    p.add_argument("--model", type=Path, required=True)
    backend = p.add_mutually_exclusive_group(required=True)
    backend.add_argument("--kaldi-config", type=Path)
    backend.add_argument("--cache", type=Path)
    p.add_argument("--already-preprocessed", action="store_true", help="Only for Kevin's processed dataset WAVs")
    p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            result = audit_dataset(args.data, read_dataset(args.data))
            dump_json(args.output, result)
        elif args.command == "baselines":
            result = baseline_experiment(args.data, args.output)
        elif args.command == "extract":
            result = extract_dataset(args.data, args.cache, create_backend(args.kaldi_config), args.splits)
        elif args.command == "train":
            result = train_experiment(args.data, args.cache, args.output, args.model, args.legacy_model)
        else:
            try:
                reference = load_json(args.reference)["reference_words"]
                model = load_calibrator(args.model)
                backend = create_backend(args.kaldi_config) if args.kaldi_config else CachedBackend(args.cache, model.extractor_id)
                result = PronunciationScorer(backend, model).score_pronunciation(args.audio, None, reference, already_preprocessed=args.already_preprocessed)
            except ScoringError as error:
                result = failure(error.code, str(error))
            except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
                result = failure("model_not_ready" if not args.model.exists() else "invalid_reference", str(error))
            if args.output:
                dump_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
            return 0 if result["status"] == "ok" else 2
        print(json.dumps({"status": "completed", "command": args.command, "output": str(getattr(args, "output", getattr(args, "cache", "")))}, ensure_ascii=False))
        return 0
    except (ScoringError, ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "error", "code": getattr(error, "code", "experiment_failed"), "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
