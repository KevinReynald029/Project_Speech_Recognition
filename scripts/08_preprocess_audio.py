from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\speechocean762")
BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
PROC = BASE / "data" / "processed"
OUT_WAV = PROC / "wav16k"
OUT_WAV.mkdir(parents=True, exist_ok=True)

TARGET_RMS_DB = -25.0     # target RMS (dBFS)
PEAK_CEIL_DB = -1.0       # batas peak setelah gain
MAX_EDGE_SIL = 0.5        # hening tepi maksimal (detik)
VAD_DB = 35               # ambang suara: 35 dB di bawah frame terkeras
WIN, HOP = 0.025, 0.010
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def frame_db(x, sr):
    n, h = int(WIN * sr), int(HOP * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    nf = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(nf)[:, None]
    return 20 * np.log10(np.sqrt((x[idx] ** 2).mean(1)) + 1e-12)

def db(v):
    return 20 * np.log10(v + 1e-10)

meta = pd.read_csv(PROC / "utterances_meta.csv", dtype={"utt_id": str, "speaker": str})
rows = []
for r in meta.itertuples():
    x, sr = sf.read(str(ROOT / r.wav_path), dtype="float32")
    assert sr == 16000, (r.utt_id, sr)
    if x.ndim > 1:
        x = x.mean(1)
    x = x - x.mean()
    rms_orig = db(np.sqrt((x ** 2).mean()))

    # 1. batasi hening awal/akhir
    fdb = frame_db(x, sr)
    voiced = np.where(fdb > fdb.max() - VAD_DB)[0]
    start_s = voiced[0] * HOP
    end_s = voiced[-1] * HOP + WIN
    start = max(0, int((start_s - MAX_EDGE_SIL) * sr))
    end = min(len(x), int((end_s + MAX_EDGE_SIL) * sr))
    y = x[start:end]

    # 2. normalisasi RMS dengan batas peak
    rms, peak = np.sqrt((y ** 2).mean()), np.abs(y).max()
    gain_db = min(TARGET_RMS_DB - db(rms), PEAK_CEIL_DB - db(peak))
    y = (y * 10 ** (gain_db / 20)).astype(np.float32)

    sf.write(str(OUT_WAV / f"{r.utt_id}.wav"), y, sr, subtype="PCM_16")
    rows.append(dict(
        utt_id=r.utt_id, split=r.split, speaker=r.speaker, orig_path=r.wav_path,
        proc_path=f"wav16k/{r.utt_id}.wav", n_words=r.n_words,
        dur_orig=round(len(x) / sr, 3), dur_proc=round(len(y) / sr, 3),
        lead_cut=round(start / sr, 3), trail_cut=round((len(x) - end) / sr, 3),
        gain_db=round(float(gain_db), 2), rms_orig_db=round(float(rms_orig), 2),
        rms_proc_db=round(float(db(np.sqrt((y ** 2).mean()))), 2),
        peak_proc_db=round(float(db(np.abs(y).max())), 2),
        peak_limited=int(PEAK_CEIL_DB - db(peak) < TARGET_RMS_DB - db(rms)),
    ))

m = pd.DataFrame(rows)
m["sec_per_word"] = m.dur_orig / m.n_words
thr = m.sec_per_word.mean() + 3 * m.sec_per_word.std()
m["flag_long"] = (m.dur_orig > 12).astype(int)
m["flag_slow"] = (m.sec_per_word > thr).astype(int)
m["flag_quiet"] = (m.rms_orig_db < -40).astype(int)
m.drop(columns="sec_per_word").to_csv(PROC / "audio_manifest.csv", index=False)

log(f"=== HASIL ({len(m)} file ditulis ke data\\processed\\wav16k) ===")
log(f"File dengan hening awal dipotong: {(m.lead_cut > 0).sum()} | akhir: {(m.trail_cut > 0).sum()}")
log(f"Potongan awal (s): mean {m.lead_cut.mean():.2f} | max {m.lead_cut.max():.2f}")
log(f"Potongan akhir (s): mean {m.trail_cut.mean():.2f} | max {m.trail_cut.max():.2f}")
log(f"Durasi total: asli {m.dur_orig.sum()/3600:.2f} jam -> olahan {m.dur_proc.sum()/3600:.2f} jam")
log(f"Gain (dB): min {m.gain_db.min():.1f} | median {m.gain_db.median():.1f} | max {m.gain_db.max():.1f}")
log(f"Gain dibatasi peak ceiling: {m.peak_limited.sum()} file")
log(f"RMS asli  : std {m.rms_orig_db.std():.2f} dB | min {m.rms_orig_db.min():.1f} | max {m.rms_orig_db.max():.1f}")
log(f"RMS olahan: std {m.rms_proc_db.std():.2f} dB | min {m.rms_proc_db.min():.1f} | max {m.rms_proc_db.max():.1f}")
log(f"Peak olahan maksimum: {m.peak_proc_db.max():.2f} dB")

# verifikasi baca ulang
bad = 0
for u in m.utt_id.sample(300, random_state=0):
    info = sf.info(str(OUT_WAV / f"{u}.wav"))
    if info.samplerate != 16000 or info.channels != 1:
        bad += 1
log(f"Verifikasi 300 file acak (16 kHz mono): bermasalah {bad}")

# flag vs skor fluency
sent = pd.read_csv(PROC / "sentences.csv", dtype={"utt_id": str})
mm = m.merge(sent[["utt_id", "fluency", "accuracy", "total"]], on="utt_id")
log("\n=== FLAG vs SKOR KALIMAT (mean) ===")
log(f"flag_long={mm.flag_long.sum()} | flag_slow={mm.flag_slow.sum()} | flag_quiet={mm.flag_quiet.sum()}")
for f in ["flag_long", "flag_slow", "flag_quiet"]:
    g = mm.groupby(f)[["fluency", "accuracy", "total"]].mean().round(2)
    log(f"\n{f}:\n{g.to_string()}")

(BASE / "reports" / "08_preprocess_audio.txt").write_text("\n".join(lines), encoding="utf-8")