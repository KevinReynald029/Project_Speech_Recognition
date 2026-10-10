from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\ASUS\Documents\LAB_SPEECH_RECOGNITION\jobdesk1")
PROC = BASE / "data" / "processed"
FIG = BASE / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

meta = pd.read_csv(PROC / "utterances_meta.csv", dtype={"utt_id": str, "speaker": str})
sent = pd.read_csv(PROC / "sentences.csv", dtype={"utt_id": str, "speaker": str})
ph = pd.read_csv(PROC / "phones_labels.csv", dtype={"utt_id": str, "speaker": str},
                 keep_default_na=False)
ph["score"] = ph.score.astype(float)
ph["is_mispron"] = ph.is_mispron.astype(int)

# ---------- 1. Split validasi per speaker ----------
spk = meta.drop_duplicates("speaker")[["speaker", "split", "age", "gender"]].copy()
spk["group"] = np.where(spk.age < 18, "anak", "dewasa")
mis_utt = ph.groupby("utt_id").is_mispron.mean().rename("mis").reset_index()
u = sent.merge(mis_utt, on="utt_id")
spk = spk.merge(u.groupby("speaker").agg(total=("total", "mean"), mis=("mis", "mean")).reset_index(),
                on="speaker")
tr = spk[spk.split == "train"].reset_index(drop=True)

def draw(seed, frac=0.16):
    rng = np.random.default_rng(seed)
    val = []
    for _, g in tr.groupby(["group", "gender"]):
        k = max(1, round(len(g) * frac))
        val += [str(x) for x in rng.choice(g.speaker.values, size=k, replace=False)]
    return val

def cost(val):
    v, t = tr[tr.speaker.isin(val)], tr[~tr.speaker.isin(val)]
    return sum(abs(v[c].mean() - t[c].mean()) / t[c].std() for c in ["total", "mis", "age"])

best_seed = min(range(200), key=lambda s: cost(draw(s)))
val_spk = set(draw(best_seed))

splits = meta[["utt_id", "speaker", "split"]].rename(columns={"split": "orig_split"})
splits["split"] = np.where(splits.speaker.isin(val_spk), "val", splits.orig_split)
splits.to_csv(PROC / "splits.csv", index=False)
(PROC / "val_speakers.txt").write_text("\n".join(sorted(val_spk)), encoding="utf-8")

sent = sent.drop(columns="split").merge(splits[["utt_id", "split"]], on="utt_id")
ph = ph.drop(columns="split").merge(splits[["utt_id", "split"]], on="utt_id")
ph = ph.merge(meta[["utt_id", "age"]], on="utt_id")
ph["group"] = np.where(ph.age < 18, "anak", "dewasa")
order = ["train", "val", "test"]

log(f"=== SPLIT (seed terpilih: {best_seed}) ===")
sp = splits.groupby("split").agg(utt=("utt_id", "size"), speaker=("speaker", "nunique")).loc[order]
log(sp.to_string())
sets = {s: set(splits[splits.split == s].speaker) for s in order}
log(f"Irisan speaker train&val: {len(sets['train'] & sets['val'])} | train&test: "
    f"{len(sets['train'] & sets['test'])} | val&test: {len(sets['val'] & sets['test'])}")
bal = (splits.merge(meta[["utt_id", "age"]], on="utt_id")
       .merge(u[["utt_id", "total", "mis"]], on="utt_id")
       .groupby("split").agg(umur_mean=("age", "mean"), anak_pct=("age", lambda a: (a < 18).mean() * 100),
                             total_mean=("total", "mean"), mispron_pct=("mis", lambda m: m.mean() * 100))
       .loc[order].round(2))
log("\nKeseimbangan antar split:\n" + bal.to_string())

# ---------- 2. EDA skor kalimat ----------
log("\n=== SKOR KALIMAT PER SPLIT (mean, std) ===")
log(sent.groupby("split")[["accuracy", "fluency", "prosodic", "total"]].agg(["mean", "std"])
    .loc[order].round(2).to_string())
log("\nSkor kalimat 'total' per kelompok umur:")
tmp = sent.merge(meta[["utt_id", "age"]], on="utt_id")
tmp["group"] = np.where(tmp.age < 18, "anak", "dewasa")
log(tmp.groupby("group")[["accuracy", "fluency", "prosodic", "total"]].mean().round(2).to_string())

# ---------- 3. EDA skor fonem & class imbalance ----------
log("\n=== SKOR FONEM PER SPLIT ===")
cb = ph.groupby("split").score.agg(
    n="size", mean="mean", pct_2=lambda s: (s == 2).mean() * 100,
    pct_kurang_05=lambda s: (s < 0.5).mean() * 100).loc[order].round(2)
log(cb.to_string())
log("\nPersentase fonem salah (<0.5) per kelompok umur:")
log((ph.groupby("group").is_mispron.mean() * 100).round(2).to_string())

# ---------- 4. Per fonem ----------
pp = (ph.groupby("phone_base").agg(n=("score", "size"), mean_score=("score", "mean"),
                                   mis_pct=("is_mispron", "mean")).reset_index())
pp["mis_pct"] = (pp.mis_pct * 100).round(2)
pp["mean_score"] = pp.mean_score.round(3)
pp.sort_values("n", ascending=False).to_csv(PROC / "phone_stats.csv", index=False)
log("\n=== 10 FONEM DENGAN PERSEN SALAH TERTINGGI (n >= 100) ===")
log(pp[pp.n >= 100].nlargest(10, "mis_pct").to_string(index=False))
log("\n=== FONEM LANGKA (n < 400) ===")
log(pp[pp.n < 400].sort_values("n").to_string(index=False))

# ---------- 5. Panjang kalimat ----------
log("\n=== PANJANG KALIMAT ===")
log(sent[["n_words", "n_phones"]].describe().loc[["min", "25%", "50%", "75%", "max", "mean"]].round(2).to_string())

# ---------- 6. Agregasi: skor fonem vs skor kalimat ----------
agg = (ph.groupby("utt_id").score.agg(mean_phone="mean", min_phone="min").reset_index()
       .merge(sent[["utt_id", "accuracy", "fluency", "prosodic", "total"]], on="utt_id"))
log("\n=== KORELASI SKOR FONEM (agregat per kalimat) vs SKOR KALIMAT ===")
for feat in ["mean_phone", "min_phone"]:
    for c in ["accuracy", "total", "fluency", "prosodic"]:
        log(f"{feat} vs {c}: pearson {agg[feat].corr(agg[c]):.3f} | "
            f"spearman {agg[feat].corr(agg[c], method='spearman'):.3f}")

# ---------- 7. Gambar ----------
vc = ph.score.round(1).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([str(i) for i in vc.index], vc.values)
ax.set_yscale("log"); ax.set_xlabel("Skor fonem"); ax.set_ylabel("Jumlah (log)")
ax.set_title("Distribusi skor fonem")
fig.tight_layout(); fig.savefig(FIG / "fig1_phone_score_dist.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(9, 6))
for a, c in zip(axs.ravel(), ["accuracy", "fluency", "prosodic", "total"]):
    a.hist(sent[c], bins=np.arange(-0.5, 11.5, 1), edgecolor="white")
    a.set_title(f"Skor kalimat: {c}"); a.set_xlabel("skor (0-10)")
fig.tight_layout(); fig.savefig(FIG / "fig2_sentence_scores.png", dpi=150); plt.close(fig)

p2 = pp.sort_values("n", ascending=False)
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axs[0].bar(p2.phone_base, p2.n); axs[0].set_ylabel("Frekuensi"); axs[0].set_title("Frekuensi tiap fonem")
axs[1].bar(p2.phone_base, p2.mis_pct); axs[1].set_ylabel("% fonem salah (<0.5)")
axs[1].set_title("Persen salah per fonem"); plt.xticks(rotation=90)
fig.tight_layout(); fig.savefig(FIG / "fig3_phone_freq_error.png", dpi=150); plt.close(fig)

(BASE / "reports" / "09_split_eda.txt").write_text("\n".join(lines), encoding="utf-8")
log("\nGambar tersimpan di reports\\figures\\")