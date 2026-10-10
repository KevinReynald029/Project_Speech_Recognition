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

def strip_digit(seq):
    return [re.sub(r"\d", "", p) for p in seq]

phone = pd.read_csv(PROC / "phones.csv", dtype=str, keep_default_na=False)
phone["word_idx"] = phone.word_idx.astype(int)
phone["phone_idx_in_word"] = phone.phone_idx_in_word.astype(int)
phone = phone.sort_values(["utt_id", "word_idx", "phone_idx_in_word"])
annot = (phone.groupby(["utt_id", "word_idx"])
         .agg(word=("word", "first"), phones=("phone", lambda x: " ".join(x)))
         .reset_index())
log(f"Kata teranotasi: {len(annot)}")

# 1. text-phone vs anotasi
tp = {}
tags = Counter()
for ln in (ROOT / "resource" / "text-phone").read_text(encoding="utf-8").splitlines():
    if not ln.strip():
        continue
    k, v = ln.split(maxsplit=1)
    utt, idx = k.rsplit(".", 1)
    ps = v.split()
    for p in ps:
        m = re.search(r"_([A-Z])$", p)
        tags[m.group(1) if m else "(tanpa tag)"] += 1
    tp[(utt, int(idx))] = [re.sub(r"_[BIES]$", "", p) for p in ps]

missing = mismatch = 0
ex = []
for _, r in annot.iterrows():
    t = tp.get((r.utt_id, r.word_idx))
    if t is None:
        missing += 1
    elif t != r.phones.split():
        mismatch += 1
        ex.append((r.utt_id, r.word, r.phones, " ".join(t)))
log("\n=== text-phone vs anotasi ===")
log(f"Baris text-phone: {len(tp)} | kata tidak ada di text-phone: {missing} | urutan fonem beda: {mismatch}")
log("Distribusi tag posisi: " + str(dict(tags)))
for e in ex[:5]:
    log(f"  contoh beda: {e}")

# 2. Lexicon
lex = {}
for ln in (ROOT / "resource" / "lexicon.txt").read_text(encoding="utf-8").splitlines():
    if ln.strip():
        w, pron = ln.split(maxsplit=1)
        lex.setdefault(w, []).append(pron.strip())
multi = sum(1 for v in lex.values() if len(v) > 1)
log("\n=== lexicon.txt ===")
log(f"Kata unik di lexicon: {len(lex)} | kata dengan >1 pelafalan: {multi}")

uniq = annot.drop_duplicates(["word", "phones"])
words_ann = set(annot.word)
oov = sorted(words_ann - set(lex))
log(f"Kata unik di anotasi: {len(words_ann)} | kata tidak ada di lexicon (OOV): {len(oov)}")
if oov:
    log("  OOV: " + ", ".join(oov[:30]))

exact = stripped = 0
bad = []
for _, r in annot.iterrows():
    variants = lex.get(r.word, [])
    if r.phones in variants:
        exact += 1
    if r.phones.split() and any(strip_digit(r.phones.split()) == strip_digit(v.split()) for v in variants):
        stripped += 1
    else:
        bad.append((r.word, r.phones, variants))
n = len(annot)
log(f"\nCocok persis dgn lexicon (dengan digit stress): {exact}/{n} ({exact/n*100:.2f}%)")
log(f"Cocok setelah digit stress dibuang: {stripped}/{n} ({stripped/n*100:.2f}%)")
log("Contoh tidak cocok walau digit dibuang:")
seen = set()
for w, a, v in bad:
    if (w, a) in seen:
        continue
    seen.add((w, a))
    log(f"  {w}: anotasi='{a}' | lexicon={v}")
    if len(seen) >= 10:
        break

# 3. Hipotesis digit stress: kata satu suku kata
vowel_digits = phone[phone.stress_digit != ""]
mono = annot.assign(n_vowel=annot.phones.apply(lambda s: sum(bool(re.search(r"\d", p)) for p in s.split())))
mono1 = mono[mono.n_vowel == 1]
digit_mono = Counter(re.search(r"\d", p).group() for s in mono1.phones for p in s.split() if re.search(r"\d", p))
log("\n=== Digit stress pada kata satu vokal ===")
log(f"Jumlah kata satu vokal: {len(mono1)} | distribusi digit: {dict(digit_mono)}")
multi_v = mono[mono.n_vowel > 1]
digit_multi = Counter(re.search(r"\d", p).group() for s in multi_v.phones for p in s.split() if re.search(r"\d", p))
log(f"Kata >1 vokal: {len(multi_v)} | distribusi digit: {dict(digit_multi)}")

(BASE / "reports" / "05_validate_phones.txt").write_text("\n".join(lines), encoding="utf-8")