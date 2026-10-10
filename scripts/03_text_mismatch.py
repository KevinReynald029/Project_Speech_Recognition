import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\speechocean762")
BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")

df = pd.read_csv(BASE / "data" / "processed" / "utterances_meta.csv", dtype={"utt_id": str, "speaker": str})
sc = json.loads((ROOT / "resource" / "scores.json").read_text(encoding="utf-8"))

def norm_space(s): return re.sub(r"\s+", " ", s.strip())
def norm_alnum(s): return re.sub(r"[^A-Z0-9 ]", "", norm_space(s).upper())

cats = {"identik": 0, "beda spasi": 0, "beda huruf besar/kecil": 0,
        "beda tanda baca": 0, "beda kata": 0}
examples = []
words_vs_text = 0

for _, r in df.iterrows():
    a = r.text
    b = sc[r.utt_id]["text"]
    joined = " ".join(w["text"] for w in sc[r.utt_id]["words"])
    if norm_space(b) != norm_space(joined):
        words_vs_text += 1
    if a == b:
        cats["identik"] += 1
    elif norm_space(a) == norm_space(b):
        cats["beda spasi"] += 1
    elif norm_space(a).upper() == norm_space(b).upper():
        cats["beda huruf besar/kecil"] += 1
    elif norm_alnum(a) == norm_alnum(b):
        cats["beda tanda baca"] += 1
    else:
        cats["beda kata"] += 1
        examples.append((r.utt_id, r.split, a, b, joined))

print("=== KATEGORI (text Kaldi vs scores.json) ===")
for k, v in cats.items():
    print(f"{k}: {v}")
print(f"\nscores.json 'text' != gabungan teks 'words': {words_vs_text}")

print("\n=== 15 CONTOH 'beda kata' ===")
for utt, split, a, b, j in examples[:15]:
    print(f"{utt} [{split}]")
    print(f"  text      : {a!r}")
    print(f"  scores    : {b!r}")
    print(f"  words join: {j!r}")

mism = pd.DataFrame(examples, columns=["utt_id", "split", "text_kaldi", "text_scores", "words_join"])
mism.to_csv(BASE / "reports" / "03_text_mismatch.csv", index=False)
print(f"\nDisimpan: {len(mism)} baris ke reports\\03_text_mismatch.csv")