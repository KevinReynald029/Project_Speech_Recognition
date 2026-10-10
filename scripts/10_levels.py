import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
PROC = BASE / "data" / "processed"
W_LEN, W_ERR = 0.6, 0.4      # bobot panjang vs kesulitan empiris
K_SHRINK = 3                 # "pembaca semu" untuk shrinkage ke rata-rata global
DIGITS = {"ZERO", "OH", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"}
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

sent = pd.read_csv(PROC / "sentences.csv", dtype={"utt_id": str, "speaker": str})
ph = pd.read_csv(PROC / "phones_labels.csv", dtype={"utt_id": str, "speaker": str}, keep_default_na=False)
ph["score_norm"] = ph.score_norm.astype(float)
ph["word_idx"] = ph.word_idx.astype(int)
ph["phone_idx_in_word"] = ph.phone_idx_in_word.astype(int)
ph = ph.sort_values(["utt_id", "word_idx", "phone_idx_in_word"])

# fonem referensi per utterance, mis. "W IY | K AO L | IH T | B EH R"
w = ph.groupby(["utt_id", "word_idx"], sort=True).phone_base.agg(" ".join).reset_index()
ref = w.groupby("utt_id").phone_base.agg(" | ".join).rename("ref_phones")
lost = ph.assign(lost=1 - ph.score_norm).groupby("utt_id").lost.sum().rename("lost_sum")

u = (sent[["utt_id", "text", "total"]]
     .merge(ref, on="utt_id").merge(lost, on="utt_id"))
u["n_ph_utt"] = u.ref_phones.apply(lambda s: len(s.replace("|", " ").split()))

def mode(s):
    return s.value_counts().index[0]

g = u.groupby("text").agg(
    n_readers=("utt_id", "size"), ref_phones=("ref_phones", mode),
    n_ref_variants=("ref_phones", "nunique"), lost_sum=("lost_sum", "sum"),
    ph_sum=("n_ph_utt", "sum"), mean_total=("total", "mean")).reset_index()
g["n_words"] = g.text.str.split().str.len()
g["n_phones"] = g.ref_phones.apply(lambda s: len(s.replace("|", " ").split()))
g["emp_loss"] = g.lost_sum / g.ph_sum
glob = g.lost_sum.sum() / g.ph_sum.sum()
g["emp_shr"] = (g.n_readers * g.emp_loss + K_SHRINK * glob) / (g.n_readers + K_SHRINK)
g["is_digits"] = g.text.apply(lambda t: all(x in DIGITS for x in t.split())).astype(int)

g["len_score"] = g.n_phones.rank(pct=True)
g["err_score"] = g.emp_shr.rank(pct=True)
g["difficulty"] = (W_LEN * g.len_score + W_ERR * g.err_score).round(4)
g["level"] = pd.qcut(g.difficulty.rank(method="first"), 3, labels=["Easy", "Medium", "Hard"])

log("=== KALIMAT UNIK ===")
log(f"Utterance: {len(u)} | teks unik: {len(g)}")
log("Jumlah pembaca per teks:\n" + g.n_readers.value_counts().sort_index().head(8).to_string())
log(f"Teks dengan >1 varian fonem referensi: {(g.n_ref_variants > 1).sum()}")
log(f"Kalimat angka (is_digits): {g.is_digits.sum()}")
log("Contoh: " + "; ".join(g[g.is_digits == 1].text.head(5)))

log("\n=== KORELASI ===")
log(f"n_phones vs emp_shr (spearman): {g.n_phones.corr(g.emp_shr, method='spearman'):.3f}")
log(f"len_score vs err_score (spearman): {g.len_score.corr(g.err_score, method='spearman'):.3f}")

log("\n=== RINGKASAN PER LEVEL ===")
s = g.groupby("level", observed=True).agg(
    n_kalimat=("text", "size"), words_min=("n_words", "min"), words_max=("n_words", "max"),
    ph_min=("n_phones", "min"), ph_mean=("n_phones", "mean"), ph_max=("n_phones", "max"),
    loss_pct=("emp_loss", lambda x: x.mean() * 100), readers=("n_readers", "mean"),
    mean_total=("mean_total", "mean"), digits=("is_digits", "sum")).round(2)
log(s.to_string())
log("\nLevel vs jumlah kata:\n" + pd.crosstab(g.n_words, g.level).to_string())

log("\n=== 5 CONTOH PER LEVEL ===")
for lv in ["Easy", "Medium", "Hard"]:
    log(f"\n[{lv}]")
    for r in g[g.level == lv].sample(5, random_state=0).itertuples():
        log(f"  {r.text}  ({r.n_words} kata, {r.n_phones} fonem, {r.n_readers} pembaca)")

g = g.sort_values(["level", "difficulty"]).reset_index(drop=True)
out = g[["text", "level", "n_words", "n_phones", "n_readers", "mean_total", "emp_loss",
         "difficulty", "is_digits", "n_ref_variants", "ref_phones"]].copy()
out["emp_loss_pct"] = (out.pop("emp_loss") * 100).round(2)
out["mean_total"] = out.mean_total.round(2)
out.to_csv(PROC / "levels.csv", index=False)

js = {}
for lv in ["Easy", "Medium", "Hard"]:
    items = []
    for i, r in enumerate(out[out.level == lv].itertuples(), 1):
        items.append({"id": f"{lv[0]}{i:04d}", "text": r.text, "n_words": int(r.n_words),
                      "n_phones": int(r.n_phones), "is_digits": int(r.is_digits),
                      "ref_phones": [x.split() for x in r.ref_phones.split(" | ")]})
    js[lv] = items
(PROC / "levels.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
log(f"\nTersimpan: levels.csv ({len(out)} kalimat) dan levels.json")

(BASE / "reports" / "10_levels.txt").write_text("\n".join(lines), encoding="utf-8")