import csv
from collections import Counter
import jiwer

CSV_PATH = "reports/asr/wer_val.csv"

substitutions = Counter()
deletions = Counter()
insertions = Counter()

total_hits = 0
total_substitutions = 0
total_deletions = 0
total_insertions = 0

with open(CSV_PATH, encoding="utf-8") as file:
    records = list(csv.DictReader(file))

for record in records:
    reference = record["reference"]
    prediction = record["prediction"]

    result = jiwer.process_words(reference, prediction)

    total_hits += result.hits
    total_substitutions += result.substitutions
    total_deletions += result.deletions
    total_insertions += result.insertions

    ref_words = reference.split()
    pred_words = prediction.split()

    # individual substitution/deletion/insertion
    for alignment in result.alignments[0]:
        ref_part = ref_words[
            alignment.ref_start_idx:alignment.ref_end_idx
        ]

        hyp_part = pred_words[
            alignment.hyp_start_idx:alignment.hyp_end_idx
        ]

        if alignment.type == "substitute":
            substitutions.update(
                zip(ref_part, hyp_part)
            )
        elif alignment.type == "delete":
            deletions.update(ref_part)
        elif alignment.type == "insert":
            insertions.update(hyp_part)

print("Error Analysis")
print("Total audio:", len(records))
print("Correct words (hits):", total_hits)
print("Substitutions:", total_substitutions)
print("Deletions:", total_deletions)
print("Insertions:", total_insertions)

print("Common Substitutions:")
for (ref, hyp), count in substitutions.most_common(10):
    print(f"{ref} -> {hyp}: {count}")

print("Common Deletions:")
for word, count in deletions.most_common(10):
    print(f"{word}: {count}")

print("Common Insertions:")
for word, count in insertions.most_common(10):
    print(f"{word}: {count}")