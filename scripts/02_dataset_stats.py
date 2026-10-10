import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\speechocean762")
BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
(BASE / "data" / "processed").mkdir(parents=True, exist_ok=True)
(BASE / "reports").mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def read_kv(p):
    d = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            k, v = ln.split(maxsplit=1)
            d[k] = v.strip()
    return d

# 1. Bangun tabel metadata dari file Kaldi
rows, missing_age = [], 0
for split in ["train", "test"]:
    scp = read_kv(ROOT / split / "wav.scp")
    text = read_kv(ROOT / split / "text")
    u2s = read_kv(ROOT / split / "utt2spk")
    age = read_kv(ROOT / split / "spk2age")
    gender = read_kv(ROOT / split / "spk2gender")
    for utt, path in scp.items():
        spk = u2s[utt]
        if spk not in age:
            missing_age += 1
        dur = sf.info(str(ROOT / path)).duration
        rows.append(dict(
            utt_id=utt, split=split, speaker=spk,
            age=int(age[spk]) if spk in age else None,
            gender=gender.get(spk), wav_path=path, text=text[utt],
            duration=round(dur, 3), n_words=len(text[utt].split()),
        ))
df = pd.DataFrame(rows)
df.to_csv(BASE / "data" / "processed" / "utterances_meta.csv", index=False)
log(f"Total utterance di tabel: {len(df)} | speaker unik: {df.speaker.nunique()} | umur hilang: {missing_age}")

# 2. File WAV ekstra (ada di folder WAVE tapi tidak ada di wav.scp)
all_wavs = [w for w in (ROOT / "WAVE").rglob("*") if w.suffix.lower() == ".wav"]
known = set(df.utt_id)
extras = [w for w in all_wavs if w.stem not in known]
log(f"\n=== FILE EKSTRA === jumlah WAV: {len(all_wavs)} | ekstra: {len(extras)}")
by_spk = pd.Series([w.parent.name for w in extras]).value_counts()
log("Ekstra per speaker (top 10):\n" + by_spk.head(10).to_string())
log("Contoh nama file ekstra: " + ", ".join(w.name for w in extras[:10]))
if extras:
    d = [sf.info(str(w)).duration for w in extras]
    log(f"Durasi ekstra: min {min(d):.2f} | max {max(d):.2f} | mean {np.mean(d):.2f}")

# 3. Konsistensi dengan scores.json
sc = json.loads((ROOT / "resource" / "scores.json").read_text(encoding="utf-8"))
log("\n=== KONSISTENSI ===")
log(f"ID wav.scp == ID scores.json: {set(sc) == known}")
txt_mismatch = [u for u, t in zip(df.utt_id, df.text) if sc[u]["text"].strip() != t.strip()]
log(f"Teks beda antara 'text' dan scores.json: {len(txt_mismatch)}")
tr_spk = set(df[df.split == "train"].speaker)
te_spk = set(df[df.split == "test"].speaker)
log(f"Speaker bocor train/test (irisan): {len(tr_spk & te_spk)}")

# 4. Statistik per split (hanya 5000 utterance resmi)
log("\n=== STATISTIK PER SPLIT ===")
for split, g in df.groupby("split"):
    spk = g.drop_duplicates("speaker")
    log(f"\n[{split}] utt: {len(g)} | speaker: {g.speaker.nunique()} | "
        f"total durasi: {g.duration.sum()/3600:.2f} jam")
    log(f"  durasi (s): min {g.duration.min():.2f} | max {g.duration.max():.2f} | "
        f"mean {g.duration.mean():.2f} | median {g.duration.median():.2f}")
    log(f"  jumlah kata: min {g.n_words.min()} | max {g.n_words.max()} | mean {g.n_words.mean():.2f}")
    log(f"  gender speaker: {spk.gender.value_counts().to_dict()}")
    log(f"  umur speaker: min {spk.age.min()} | max {spk.age.max()} | mean {spk.age.mean():.1f}")
    log(f"  anak (<18): {(spk.age < 18).sum()} | dewasa (>=18): {(spk.age >= 18).sum()}")
    log(f"  utt per speaker: {g.groupby('speaker').size().describe()[['min','max','mean']].round(1).to_dict()}")

log("\n=== SEBARAN UMUR (semua speaker) ===")
spk_all = df.drop_duplicates("speaker")
log(pd.cut(spk_all.age, [0, 6, 9, 12, 17, 25, 40, 100]).value_counts().sort_index().to_string())

(BASE / "reports" / "02_dataset_stats.txt").write_text("\n".join(lines), encoding="utf-8")