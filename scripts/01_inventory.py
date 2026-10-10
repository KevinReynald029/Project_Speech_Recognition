import json
from pathlib import Path
import soundfile as sf
import numpy as np

ROOT = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\speechocean762")
OUT = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1\reports\01_inventory.txt")
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def read_kv(path):
    d = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            k, v = ln.split(maxsplit=1)
            d[k] = v
    return d

# 1. Cuplikan isi file teks (supaya format tiap file jelas)
log("=== CUPLIKAN FILE ===")
for rel in ["train/wav.scp", "train/text", "train/utt2spk", "train/spk2age",
            "train/spk2gender", "resource/lexicon.txt", "resource/text-phone"]:
    p = ROOT / rel
    head = p.read_text(encoding="utf-8").splitlines()[:3]
    log(f"\n[{rel}] total baris: {len(p.read_text(encoding='utf-8').splitlines())}")
    for h in head:
        log("  " + h[:200])

# 2. Hitung file audio
wavs = list((ROOT / "WAVE").rglob("*.WAV")) + list((ROOT / "WAVE").rglob("*.wav"))
wavs = list({w.resolve(): w for w in wavs}.values())
log(f"\n=== AUDIO ===\nJumlah file WAV: {len(wavs)}")
log(f"Jumlah folder speaker: {len(list((ROOT / 'WAVE').iterdir()))}")

# 3. Per split: jumlah utt, speaker, dan cek file ada
for split in ["train", "test"]:
    scp = read_kv(ROOT / split / "wav.scp")
    spk = read_kv(ROOT / split / "utt2spk")
    missing = [u for u, p in scp.items() if not (ROOT / p).exists()]
    log(f"\n[{split}] utterance: {len(scp)} | speaker: {len(set(spk.values()))} | file hilang: {len(missing)}")

# 4. Cek properti audio (sample rate, channel, durasi)
srs, chs, durs, bad = {}, {}, [], []
for w in wavs:
    try:
        info = sf.info(str(w))
        srs[info.samplerate] = srs.get(info.samplerate, 0) + 1
        chs[info.channels] = chs.get(info.channels, 0) + 1
        durs.append(info.duration)
    except Exception as e:
        bad.append((w.name, str(e)))
durs = np.array(durs)
log(f"\nSample rate: {srs}")
log(f"Channels: {chs}")
log(f"File rusak: {len(bad)}")
log(f"Durasi (detik) -> min {durs.min():.2f} | max {durs.max():.2f} | "
    f"mean {durs.mean():.2f} | median {np.median(durs):.2f}")
log(f"Total durasi: {durs.sum()/3600:.2f} jam")

# 5. Cek scores.json
sc = json.loads((ROOT / "resource" / "scores.json").read_text(encoding="utf-8"))
log(f"\n=== scores.json ===\nJumlah entri: {len(sc)}")
k = next(iter(sc))
log(f"Contoh key: {k}")
log(json.dumps(sc[k], indent=2)[:1500])

OUT.write_text("\n".join(lines), encoding="utf-8")