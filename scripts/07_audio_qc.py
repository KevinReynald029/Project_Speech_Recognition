from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\speechocean762")
BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
PROC = BASE / "data" / "processed"
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def frame_db(x, sr, win=0.025, hop=0.010):
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    nf = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(nf)[:, None]
    e = np.sqrt((x[idx] ** 2).mean(1) + 1e-12)
    return 20 * np.log10(e)

meta = pd.read_csv(PROC / "utterances_meta.csv", dtype={"utt_id": str, "speaker": str})
rows = []
for r in meta.itertuples():
    x, sr = sf.read(str(ROOT / r.wav_path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()))
    fdb = frame_db(x, sr)
    thr = fdb.max() - 35                      # ambang "ada suara": 35 dB di bawah frame terkeras
    voiced = np.where(fdb > thr)[0]
    lead = voiced[0] * 0.010 if len(voiced) else np.nan
    trail = (len(fdb) - 1 - voiced[-1]) * 0.010 if len(voiced) else np.nan
    rows.append(dict(
        utt_id=r.utt_id, split=r.split, speaker=r.speaker, age=r.age, duration=r.duration,
        n_words=r.n_words, sec_per_word=round(r.duration / r.n_words, 3),
        peak_db=round(20 * np.log10(peak + 1e-10), 2),
        rms_db=round(20 * np.log10(rms + 1e-10), 2),
        clip_ratio=float((np.abs(x) >= 0.999).mean()),
        dc_offset=round(float(x.mean()), 5),
        lead_sil=round(lead, 2), trail_sil=round(trail, 2),
        dyn_range_db=round(float(np.percentile(fdb, 95) - np.percentile(fdb, 10)), 1),
    ))
qc = pd.DataFrame(rows)
qc["group"] = np.where(qc.age < 18, "anak", "dewasa")
qc.to_csv(PROC / "audio_qc.csv", index=False)

log("=== RINGKASAN (5000 utterance) ===")
cols = ["duration", "sec_per_word", "peak_db", "rms_db", "clip_ratio", "lead_sil", "trail_sil", "dyn_range_db"]
log(qc[cols].describe().loc[["min", "25%", "50%", "75%", "max", "mean"]].round(3).to_string())

log("\n=== PER KELOMPOK (anak vs dewasa) ===")
log(qc.groupby("group")[["duration", "sec_per_word", "rms_db", "peak_db", "lead_sil", "trail_sil", "dyn_range_db"]]
    .mean().round(2).to_string())

log("\n=== FLAG ===")
spw_thr = qc.sec_per_word.mean() + 3 * qc.sec_per_word.std()
flags = {
    "sangat pelan (rms < -40 dB)": (qc.rms_db < -40).sum(),
    "peak sangat rendah (< -12 dB)": (qc.peak_db < -12).sum(),
    "clipping (>0.1% sampel)": (qc.clip_ratio > 0.001).sum(),
    "peak mendekati 0 dB (> -1 dB)": (qc.peak_db > -1).sum(),
    "hening awal > 1.0 s": (qc.lead_sil > 1.0).sum(),
    "hening akhir > 1.0 s": (qc.trail_sil > 1.0).sum(),
    "durasi > 12 s": (qc.duration > 12).sum(),
    f"sec_per_word > {spw_thr:.2f} (mean+3sd)": (qc.sec_per_word > spw_thr).sum(),
    "dynamic range < 15 dB (noisy?)": (qc.dyn_range_db < 15).sum(),
    "|DC offset| > 0.01": (qc.dc_offset.abs() > 0.01).sum(),
}
for k, v in flags.items():
    log(f"{k}: {v}")

show = ["utt_id", "split", "duration", "n_words", "sec_per_word", "rms_db", "peak_db", "lead_sil", "trail_sil"]
log("\n=== 10 TERPANJANG ===\n" + qc.nlargest(10, "duration")[show].to_string(index=False))
log("\n=== 10 sec_per_word TERTINGGI ===\n" + qc.nlargest(10, "sec_per_word")[show].to_string(index=False))
log("\n=== 10 TERPELAN (rms) ===\n" + qc.nsmallest(10, "rms_db")[show].to_string(index=False))
log("\n=== 10 HENING AWAL TERPANJANG ===\n" + qc.nlargest(10, "lead_sil")[show].to_string(index=False))

(BASE / "reports" / "07_audio_qc.txt").write_text("\n".join(lines), encoding="utf-8")