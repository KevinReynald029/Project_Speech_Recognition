import csv
import json
import re
import time
from pathlib import Path
import jiwer
from asr.recognizer import LocalASR

MODEL_PATH = "models/asr/vosk-model-small-en-us-0.15"
MANIFEST_PATH = Path("handoff/manifest_val.jsonl")
DATA_DIR = Path("handoff")
OUTPUT_PATH = Path("reports/asr/wer_val.csv")

LIMIT = None

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_manifest(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records

def main():
    records = load_manifest(MANIFEST_PATH)

    if LIMIT is not None:
        records = records[:LIMIT]

    print(f"Validation records: {len(records)}")

    asr = LocalASR(MODEL_PATH)

    references = []
    predictions = []
    results = []
    total_audio_duration = 0
    total_processing_time = 0

    for index, record in enumerate(records, start=1):
        audio_path = DATA_DIR / record["audio_proc"]

        if not audio_path.exists():
            print(f"Missing audio: {audio_path}")
            continue

        reference = normalize_text(record["text"])
        result = asr.transcribe_file(audio_path)
        prediction = normalize_text(result["text"])

        references.append(reference)
        predictions.append(prediction)
        total_audio_duration += result["duration_seconds"]
        total_processing_time += result["processing_seconds"]
        utterance_wer = jiwer.wer(reference, prediction)

        results.append({
            "audio": record["audio_proc"],
            "reference": reference,
            "prediction": prediction,
            "wer": utterance_wer,
            "duration_seconds": result["duration_seconds"],
            "processing_seconds": result["processing_seconds"],
            "rtf": result["rtf"]
        })

        print(
            f"[{index}/{len(records)}] "
            f"WER: {utterance_wer:.2%} | "
            f"Reference: {reference} | "
            f"Prediction: {prediction}"
        )

    if not results:
        print("No audio records")
        return

    corpus_wer = jiwer.wer(references, predictions)
    average_utterance_wer = sum(item["wer"] for item in results) / len(results)
    overall_rtf = (total_processing_time / total_audio_duration if total_audio_duration > 0 else 0)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Validation Result")
    print(f"Evaluated audio: {len(results)}")
    print(f"Corpus WER: {corpus_wer:.2%}")
    print(f"Average utterance WER: {average_utterance_wer:.2%}")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Total processing time: {total_processing_time:.2f}s")
    print(f"Overall RTF: {overall_rtf:.3f}")
    print(f"Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()