import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\speechocean762")
BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
PROC = BASE / "data" / "processed"
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

meta = pd.read_csv(PROC / "utterances_meta.csv", dtype={"utt_id": str, "speaker": str})
sc = json.loads((ROOT / "resource" / "scores.json").read_text(encoding="utf-8"))

sent_rows, word_rows, phone_rows, mis_rows = [], [], [], []
text_mismatch = len_mismatch = 0

for _, m in meta.iterrows():
    u = m.utt_id
    e = sc[u]
    words = e["words"]
    if " ".join(w["text"] for w in words) != m.text.strip():
        text_mismatch += 1
    p_in_utt = 0
    for wi, w in enumerate(words):
        phones = w["phones"].split() if isinstance(w["phones"], str) else w["phones"]
        accs = w["phones-accuracy"]
        if len(phones) != len(accs):
            len_mismatch += 1
        word_rows.append(dict(utt_id=u, word_idx=wi, word=w["text"], accuracy=w["accuracy"],
                              stress=w["stress"], total=w["total"], n_phones=len(phones)))
        for pi, (ph, a) in enumerate(zip(phones, accs)):
            base = re.sub(r"\d", "", ph)
            st = re.findall(r"\d", ph)
            phone_rows.append(dict(utt_id=u, word_idx=wi, word=w["text"], phone_idx_in_word=pi,
                                   phone_idx_in_utt=p_in_utt, phone=ph, phone_base=base,
                                   stress_digit=st[0] if st else "", score=a))
            p_in_utt += 1
        for mi in w.get("mispronunciations", []):
            mis_rows.append(dict(utt_id=u, word_idx=wi, word=w["text"],
                                 canonical=mi["canonical-phone"], index=mi["index"],
                                 pronounced=mi["pronounced-phone"]))
    sent_rows.append(dict(utt_id=u, split=m.split, speaker=m.speaker, text=m.text,
                          accuracy=e["accuracy"], fluency=e["fluency"],
                          completeness=e["completeness"], prosodic=e["prosodic"],
                          total=e["total"], n_words=len(words), n_phones=p_in_utt))

sent = pd.DataFrame(sent_rows)
word = pd.DataFrame(word_rows)
phone = pd.DataFrame(phone_rows)
mis = pd.DataFrame(mis_rows)
sent.to_csv(PROC / "sentences.csv", index=False)
word.to_csv(PROC / "words.csv", index=False)
phone.to_csv(PROC / "phones.csv", index=False)
mis.to_csv(PROC / "mispronunciations.csv", index=False)

log("=== VALIDASI ===")
log(f"Teks Kaldi != gabungan words scores.json: {text_mismatch}")
log(f"Jumlah phones != jumlah phones-accuracy: {len_mismatch}")
log(f"Baris: sentence {len(sent)} | word {len(word)} | phone {len(phone)} | mispron {len(mis)}")

log("\n=== SKALA SKOR LEVEL KALIMAT ===")
log(sent[["accuracy", "fluency", "completeness", "prosodic", "total"]].describe().loc[["min", "max", "mean"]].round(2).to_string())
log("\ncompleteness nilai unik: " + str(sorted(sent.completeness.unique())[:15]))

log("\n=== SKALA SKOR LEVEL KATA ===")
log(word[["accuracy", "stress", "total"]].describe().loc[["min", "max", "mean"]].round(2).to_string())
log("stress nilai: " + str(word.stress.value_counts().to_dict()))

log("\n=== INVENTARIS FONEM ===")
log(f"Fonem unik dengan stress digit: {phone.phone.nunique()}")
log(f"Fonem unik tanpa stress digit: {phone.phone_base.nunique()}")
log("Frekuensi fonem dasar:\n" + phone.phone_base.value_counts().to_string())
log("Distribusi stress digit: " + str(phone.stress_digit.value_counts().to_dict()))

log("\n=== SKOR FONEM ===")
s = phone.score
log(f"min {s.min()} | max {s.max()} | mean {s.mean():.3f} | nilai unik: {s.nunique()}")
bins = pd.cut(s, [-0.01, 0.49, 0.99, 1.49, 1.99, 2.0],
              labels=["<0.5", "0.5-0.99", "1.0-1.49", "1.5-1.99", "=2.0"])
log("Sebaran:\n" + (bins.value_counts(normalize=True).sort_index() * 100).round(2).to_string())
log("Skor paling sering: " + str(s.value_counts().head(8).to_dict()))

log("\n=== MISPRONUNCIATIONS ===")
log(f"Total blok: {len(mis)} | fonem skor <0.5: {(s < 0.5).sum()}")
if len(mis):
    log("pronounced-phone terbanyak: " + str(mis.pronounced.value_counts().head(10).to_dict()))
    log("Pasangan canonical->pronounced terbanyak:\n" +
        mis.groupby(["canonical", "pronounced"]).size().sort_values(ascending=False).head(10).to_string())

(BASE / "reports" / "04_parse_scores.txt").write_text("\n".join(lines), encoding="utf-8")