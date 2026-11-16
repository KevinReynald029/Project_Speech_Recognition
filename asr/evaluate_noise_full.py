import csv
import json
import re
from pathlib import Path
import numpy as np
import jiwer
import soundfile as sf
from asr.recognizer import LocalASR
from asr.evaluate_noise import add_white_noise

MODEL_PATH = "models/asr/vosk-model-small-en-us-0.15"
MANIFEST_PATH = Path("handoff/manifest_val.jsonl")
DATA_DIR = Path("handoff")
OUTPUT_DETAIL = Path("reports/asr/noise_detail.csv")
OUTPUT_SUMMARY = Path("reports/asr/noise_summary.csv")

LIMIT = None
SNR_LEVELS = [None, 20, 10, 5]

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

    print(f"Evaluating {len(records)} validation audio files")

    asr = LocalASR(MODEL_PATH)
    all_details = []
    summaries = []

    for snr_db in SNR_LEVELS:
        condition = "clean" if snr_db is None else f"{snr_db}dB"

        references = []
        predictions = []
        total_duration = 0
        total_processing = 0

        print(f"Condition: {condition}")

        for index, record in enumerate(records, start=1):
            audio_path = DATA_DIR / record["audio_proc"]

            if not audio_path.exists():
                print(f"Missing audio: {audio_path}")
                continue

            audio, sample_rate = sf.read(audio_path)

            if sample_rate != 16000:
                raise ValueError(f"Expected 16000Hz: {audio_path}")

            if audio.ndim != 1:
                raise ValueError(f"Expected mono audio: {audio_path}")

            if snr_db is not None:
                rng = np.random.default_rng(42 + index)
                signal_power = np.mean(audio ** 2)

                if signal_power > 0:
                    noise = rng.normal(0, 1, len(audio))
                    noise_power = np.mean(noise ** 2)
                    target_noise_power = (signal_power / (10 ** (snr_db / 10)))
                    noise *= np.sqrt(target_noise_power / noise_power)
                    processed_audio = audio + noise
                    peak = np.max(np.abs(processed_audio))

                    if peak > 0.99:
                        processed_audio *= 0.99 / peak
                else:
                    processed_audio = audio.copy()
            else:
                processed_audio = audio

            result = asr.transcribe_array(processed_audio)

            reference = normalize_text(record["text"])
            prediction = normalize_text(result["text"])

            utterance_wer = jiwer.wer(reference, prediction)

            references.append(reference)
            predictions.append(prediction)
            total_duration += result["duration_seconds"]
            total_processing += result["processing_seconds"]

            all_details.append({
                "audio": record["audio_proc"],
                "condition": condition,
                "reference": reference,
                "prediction": prediction,
                "wer": utterance_wer,
                "duration_seconds": result["duration_seconds"],
                "processing_seconds": result["processing_seconds"],
                "rtf": result["rtf"]
            })

            print(
                f"[{index}/{len(records)}] "
                f"WER: {utterance_wer:.2%}"
            )

        if not references:
            continue

        corpus_wer = jiwer.wer(references, predictions)
        average_wer = sum(item["wer"] for item in all_details if item["condition"] == condition) / len(references)
        overall_rtf = (total_processing / total_duration if total_duration > 0 else 0)

        summaries.append({
            "condition": condition,
            "samples": len(references),
            "corpus_wer": corpus_wer,
            "average_utterance_wer": average_wer,
            "total_audio_duration": total_duration,
            "total_processing_time": total_processing,
            "overall_rtf": overall_rtf
        })

        print(f"Corpus WER: {corpus_wer:.2%}")
        print(f"Average WER: {average_wer:.2%}")
        print(f"Overall RTF: {overall_rtf:.3f}")

    OUTPUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_DETAIL,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=all_details[0].keys())
        writer.writeheader()
        writer.writerows(all_details)

    with open(
        OUTPUT_SUMMARY,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("Noise Experiment Result Summary")

    for summary in summaries:
        print(
            f"{summary['condition']}: "
            f"WER = {summary['corpus_wer']:.2%}, "
            f"RTF = {summary['overall_rtf']:.3f}"
        )

    print(f"\nDetails saved: {OUTPUT_DETAIL}")
    print(f"Summary saved: {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    main()