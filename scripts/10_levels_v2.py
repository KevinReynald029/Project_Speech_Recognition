import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
PROC = BASE / "data" / "processed"
W_LEN, W_PHN = 0.6, 0.4        # bobot panjang vs kesulitan fonem
K_PH = 200                     # pseudo-count shrinkage fonem langka
DIGITS = {"ZERO", "OH", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"}
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

meta = pd.read_csv(PROC / "utterances_meta.csv", dtype={"utt_id": str, "speaker": str})
sent = pd.read_csv(PROC / "sentences.csv", dtype={"utt_id": str, "speaker": str})
ph = pd.read_csv(PROC / "phones_labels.csv", dtype={"utt_id": str, "speaker": str}, keep_default_na=False)
ph["score_norm"] = ph.score_norm.astype(float)
ph["word_idx"] = ph.word_idx.astype(int)
ph["phone_idx_in_word"] = ph.phone_idx_in_word.astype(int)
ph = ph.sort_values(["utt_id", "word_idx", "phone_idx_in_word"])
ph = ph.merge(meta[["utt_id", "age"]], on="utt_id")
ph["group"] = np.where(ph.age < 18, "anak", "dewasa")
ph["loss"] = 1 - ph.score_norm

# 1. Kesulitan relatif per fonem (anak & dewasa dihitung terpisah, lalu dirata-rata)
rel = []
for grp, g in ph.groupby("group"):
    overall = g.loss.mean()
    by = g.groupby("phone_base").loss.agg(["sum", "count"])
    shr = (by["sum"] + K_PH * overall) / (by["count"] + K_PH)
    rel.append((shr / overall).rename(grp))
rel = pd.concat(rel, axis=1)
phone_diff = rel.mean(axis=1).rename("phone_diff")
log("=== KESULITAN RELATIF FONEM (1.0 = rata-rata) ===")
log("Tersulit: " + ", ".join(f"{p} {v:.2f}" for p, v in phone_diff.nlargest(8).items()))
log("Termudah: " + ", ".join(f"{p} {v:.2f}" for p, v in phone_diff.nsmallest(8).items()))
log(f"Korelasi kesulitan fonem anak vs dewasa (spearman): {rel.anak.corr(rel.dewasa, method='spearman'):.3f}")

# 2. Fitur per utterance
w = ph.groupby(["utt_id", "word_idx"], sort=True).phone_base.agg(" ".join).reset_index()
ref = w.groupby("utt_id").phone_base.agg(" | ".join).rename("ref_phones")
ph["pd"] = ph.phone_base.map(phone_diff)
feat = ph.groupby("utt_id").agg(pdiff=("pd", "mean"), loss_utt=("loss", "mean")).reset_index()
u = (sent[["utt_id", "text", "total"]].merge(ref, on="utt_id").merge(feat, on="utt_id")
     .merge(meta[["utt_id", "age"]], on="utt_id"))
u["group"] = np.where(u.age < 18, "anak", "dewasa")

# 3. Agregasi per teks unik
mode = lambda s: s.value_counts().index[0]
g = u.groupby("text").agg(n_readers=("utt_id", "size"), ref_phones=("ref_phones", mode),
                          n_ref_variants=("ref_phones", "nunique"), phone_diff=("pdiff", "mean"),
                          mean_total=("total", "mean")).reset_index()
g["n_words"] = g.text.str.split().str.len()
g["n_phones"] = g.ref_phones.apply(lambda s: len(s.replace("|", " ").split()))
g["is_digits"] = g.text.apply(lambda t: all(x in DIGITS for x in t.split())).astype(int)
g["len_score"] = g.n_phones.rank(pct=True)
g["phn_score"] = g.phone_diff.rank(pct=True)
g["difficulty"] = (W_LEN * g.len_score + W_PHN * g.phn_score).round(4)
g["level"] = pd.qcut(g.difficulty.rank(method="first"), 3, labels=["Easy", "Medium", "Hard"])

log(f"\n=== KALIMAT UNIK: {len(g)} | kalimat angka: {g.is_digits.sum()} | >1 varian fonem: {(g.n_ref_variants > 1).sum()} ===")
log(f"Korelasi n_phones vs phone_diff (spearman): {g.n_phones.corr(g.phone_diff, method='spearman'):.3f}")

log("\n=== RINGKASAN PER LEVEL ===")
s = g.groupby("level", observed=True).agg(
    n_kalimat=("text", "size"), words_min=("n_words", "min"), words_max=("n_words", "max"),
    ph_min=("n_phones", "min"), ph_mean=("n_phones", "mean"), ph_max=("n_phones", "max"),
    phone_diff=("phone_diff", "mean"), digits=("is_digits", "sum")).round(2)
log(s.to_string())

# 4. Validasi independen: skor nyata per level, dipisah kelompok umur
lvl = g.set_index("text").level
u["level"] = u.text.map(lvl)
u["difficulty"] = u.text.map(g.set_index("text").difficulty)
log("\n=== VALIDASI: skor 'total' (0-10) per level & kelompok umur ===")
v = u.groupby(["group", "level"], observed=True).agg(n=("utt_id", "size"), total=("total", "mean"),
                                                      fonem_hilang_pct=("loss_utt", lambda x: x.mean() * 100)).round(2)
log(v.to_string())
for grp, gg in u.groupby("group"):
    log(f"Spearman difficulty vs total ({grp}): {gg.difficulty.corr(gg.total, method='spearman'):.3f}")

log("\n=== 5 CONTOH PER LEVEL ===")
for lv in ["Easy", "Medium", "Hard"]:
    log(f"\n[{lv}]")
    for r in g[g.level == lv].sample(5, random_state=0).itertuples():
        log(f"  {r.text}  ({r.n_words} kata, {r.n_phones} fonem)")

# 5. Simpan
g = g.sort_values(["level", "difficulty"]).reset_index(drop=True)
out = g[["text", "level", "n_words", "n_phones", "difficulty", "phone_diff", "is_digits",
         "n_readers", "n_ref_variants", "mean_total", "ref_phones"]].copy()
out["phone_diff"] = out.phone_diff.round(3)
out["mean_total"] = out.mean_total.round(2)
out.to_csv(PROC / "levels.csv", index=False)

js = {}
for lv in ["Easy", "Medium", "Hard"]:
    js[lv] = [{"id": f"{lv[0]}{i:04d}", "text": r.text, "n_words": int(r.n_words),
               "n_phones": int(r.n_phones), "is_digits": int(r.is_digits),
               "ref_phones": [x.split() for x in r.ref_phones.split(" | ")]}
              for i, r in enumerate(out[out.level == lv].itertuples(), 1)]
(PROC / "levels.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
phone_diff.round(4).to_csv(PROC / "phone_difficulty.csv")
log(f"\nTersimpan: levels.csv ({len(out)} kalimat), levels.json, phone_difficulty.csv")

(BASE / "reports" / "10_levels_v2.txt").write_text("\n".join(lines), encoding="utf-8")