import json
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
PROC = BASE / "data" / "processed"

phone = pd.read_csv(PROC / "phones.csv", dtype=str, keep_default_na=False)
for c in ["word_idx", "phone_idx_in_word", "phone_idx_in_utt"]:
    phone[c] = phone[c].astype(int)
phone["score"] = phone.score.astype(float)
phone = phone.sort_values(["utt_id", "word_idx", "phone_idx_in_word"]).reset_index(drop=True)

word = pd.read_csv(PROC / "words.csv", dtype={"utt_id": str})
sent = pd.read_csv(PROC / "sentences.csv", dtype={"utt_id": str, "speaker": str})
meta = pd.read_csv(PROC / "utterances_meta.csv", dtype={"utt_id": str, "speaker": str})
mis = pd.read_csv(PROC / "mispronunciations.csv", dtype={"utt_id": str}, keep_default_na=False)

# 1. Vocab fonem (39 fonem dasar, tanpa digit stress)
VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
base_set = sorted(phone.phone_base.unique())
assert len(base_set) == 39, len(base_set)
assert set(VOWELS) <= set(base_set)
tokens = ["<pad>", "<unk>"] + base_set
phone2id = {t: i for i, t in enumerate(tokens)}
vocab = {
    "tokens": tokens,
    "phone2id": phone2id,
    "vowels": VOWELS,
    "consonants": [p for p in base_set if p not in VOWELS],
    "notes": "ARPAbet 39 fonem, digit stress dibuang. <pad>=0, <unk>=1.",
}
(PROC / "phoneme_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
print(f"Vocab: {len(tokens)} token (2 khusus + {len(base_set)} fonem)")

# 2. Tabel fonem datar
phone["phone_id"] = phone.phone_base.map(phone2id)
phone["score_norm"] = (phone.score / 2).round(3)
phone["is_mispron"] = (phone.score < 0.5).astype(int)

mis = mis.rename(columns={"index": "phone_idx_in_word", "pronounced": "pronounced_phone"})
mis["phone_idx_in_word"] = mis.phone_idx_in_word.astype(int)
mis = mis[["utt_id", "word_idx", "phone_idx_in_word", "pronounced_phone"]]
mis["word_idx"] = mis.word_idx.astype(int)
dup = mis.duplicated(["utt_id", "word_idx", "phone_idx_in_word"]).sum()
mis = mis.drop_duplicates(["utt_id", "word_idx", "phone_idx_in_word"])

phone = phone.merge(meta[["utt_id", "split", "speaker"]], on="utt_id", how="left")
phone = phone.merge(mis, on=["utt_id", "word_idx", "phone_idx_in_word"], how="left")
phone["pronounced_phone"] = phone.pronounced_phone.fillna("")

cols = ["utt_id", "split", "speaker", "word_idx", "word", "phone_idx_in_word", "phone_idx_in_utt",
        "phone", "phone_base", "phone_id", "stress_digit", "score", "score_norm", "is_mispron",
        "pronounced_phone"]
phone[cols].to_csv(PROC / "phones_labels.csv", index=False)

# 3. JSONL per utterance
wd = {(r.utt_id, r.word_idx): r for r in word.itertuples()}
meta_i = meta.set_index("utt_id")
sent_i = sent.set_index("utt_id")
n_lines = n_phones = 0
sample = None
with open(PROC / "labels.jsonl", "w", encoding="utf-8") as f:
    for utt, g in phone.groupby("utt_id", sort=False):
        m, s = meta_i.loc[utt], sent_i.loc[utt]
        words = []
        for wi, gw in g.groupby("word_idx", sort=True):
            wr = wd[(utt, wi)]
            words.append({
                "word": gw.word.iloc[0],
                "word_accuracy": float(wr.accuracy),
                "word_stress": float(wr.stress),
                "word_total": float(wr.total),
                "phones": list(gw.phone_base),
                "phone_ids": [int(x) for x in gw.phone_id],
                "phone_scores": [float(x) for x in gw.score],
                "phone_scores_norm": [float(x) for x in gw.score_norm],
            })
        rec = {
            "utt_id": utt, "split": m.split, "speaker": m.speaker,
            "age": int(m.age), "gender": m.gender,
            "audio_path": m.wav_path, "text": m.text,
            "sentence_accuracy": float(s.accuracy), "sentence_fluency": float(s.fluency),
            "sentence_prosodic": float(s.prosodic), "sentence_completeness": float(s.completeness),
            "sentence_total": float(s.total), "sentence_total_100": float(s.total) * 10,
            "words": words,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_lines += 1
        n_phones += sum(len(w["phones"]) for w in words)
        if sample is None:
            sample = rec

(BASE / "reports" / "06_sample_label.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")

# 4. Validasi
print("\n=== VALIDASI ===")
print(f"labels.jsonl: {n_lines} baris | total fonem: {n_phones}")
print(f"phones_labels.csv: {len(phone)} baris | unk: {(phone.phone_id == 1).sum()} | NaN id: {phone.phone_id.isna().sum()}")
print(f"score_norm min/max: {phone.score_norm.min()} / {phone.score_norm.max()}")
print(f"is_mispron=1: {phone.is_mispron.sum()} | punya pronounced_phone: {(phone.pronounced_phone != '').sum()} | duplikat mispron dibuang: {dup}")
print("\nFonem per split:\n" + phone.groupby("split").size().to_string())
print("\nIs_mispron per split (%):\n" + (phone.groupby("split").is_mispron.mean() * 100).round(2).to_string())