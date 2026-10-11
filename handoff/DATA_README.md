# SpeechOcean762 - Data Siap Pakai

## Isi paket
- manifest_train.jsonl / manifest_val.jsonl / manifest_test.jsonl : satu baris JSON per utterance.
- phoneme_vocab.json : peta fonem ke index (39 fonem ARPAbet tanpa digit stress). <pad>=0, <unk>=1, fonem mulai dari 2.
- phones_labels.csv : satu baris per fonem (94.445 baris), versi tabular dari label yang sama.
- levels.json / levels.csv : kalimat latihan per level (Easy/Medium/Hard) beserta fonem referensi per kata.
- phone_stats.csv, phone_difficulty.csv : frekuensi, rata-rata skor, persen salah, dan kesulitan relatif per fonem.
- splits.csv, val_speakers.txt : pembagian data (val dipilih per speaker dari train asli).
- audio_manifest.csv : jalur audio asli dan olahan, gain, potongan hening, flag outlier.
- wav16k/ : 5.000 audio olahan (16 kHz mono) yang ikut dalam paket ini, nama file <utt_id>.wav. Jalur audio di manifest bersifat relatif terhadap folder tempat manifest berada, jadi biarkan wav16k/ satu folder dengan file manifest.

## Split
- train: 2100 utt / 105 speaker / 2.41 jam
- val  : 400 utt / 20 speaker / 0.44 jam (speaker diambil dari train asli)
- test : 2500 utt / 125 speaker / 2.67 jam (split resmi dataset)
- Tidak ada speaker yang muncul di lebih dari satu split.
- Sumber kebenaran split adalah splits.csv. Kolom split di manifest, phones_labels.csv, dan audio_manifest.csv sudah sama dengannya.
- Test = split resmi SpeechOcean762 (agar sebanding dengan benchmark). Val = 20 speaker yang diambil dari train asli, dipilih per kelompok umur dan gender (seed 106) agar distribusi umur, skor, dan fonem salah mirip train. Daftar speaker ada di val_speakers.txt.
- Setelah tuning selesai, model akhir boleh dilatih ulang pada train + val (2.500 utterance). Test tidak boleh dipakai saat tuning.

## Field utama di manifest
- utt_id, split, speaker, age, gender, text (huruf kapital, tanpa tanda baca)
- audio_proc (jalur relatif audio olahan, mis. wav16k/000010011.wav, dibaca relatif terhadap folder manifest), audio_raw (jalur audio di dataset mentah, tidak termasuk dalam paket ini), duration_proc
- sentence_accuracy, sentence_fluency, sentence_prosodic, sentence_completeness, sentence_total (skala 0-10), sentence_total_100 (total x 10)
- words: daftar kata. Tiap kata punya word, word_accuracy, word_stress, word_total, phones, phone_ids, phone_scores (0-2), phone_scores_norm (0-1)
- flags: long (durasi >12 s), slow (detik per kata > mean+3sd), quiet (RMS asli < -40 dB). Hanya penanda, tidak ada data yang dibuang.

## Keputusan label
- Skor fonem berasal dari scores.json: rata-rata lima ahli, kontinu 0-2 (kelipatan 0,2). Target regresi: phone_scores_norm = skor / 2.
- Fonem dianggap salah bila skor < 0,5 (sama dengan batas blok mispronunciations di dataset). Kolom is_mispron dan pronounced_phone ada di phones_labels.csv.
- Kelas sangat tidak seimbang: sekitar 80% fonem bernilai 2,0 dan hanya 3-4% bernilai < 0,5. Gunakan pembobotan atau metrik selain akurasi.
- Fonem langka: ZH (25 kemunculan) dan OY (56). Skor per fonem untuk keduanya tidak stabil.
- Digit stress dibuang (kata satu vokal hampir selalu bertanda 0, jadi tidak informatif).
- Skor completeness hampir konstan (rata-rata 9,98 dari 10), tidak berguna untuk rumus skor kalimat.
- Korelasi rata-rata skor fonem per kalimat dengan skor total kalimat: pearson 0,787. Dengan accuracy, min skor fonem lebih baik (0,807). Angka ini memakai label asli (batas atas).
- Fluency dan prosodic hanya berkorelasi 0,60-0,71 (spearman) dengan skor fonem, jadi perlu fitur tambahan bila ingin ikut dinilai.
- Anak dinilai lebih tinggi dari dewasa (total 7,59 vs 6,87; fonem salah 1,26% vs 5,37%). Laporkan evaluasi per kelompok umur.
- Proporsi fonem salah berbeda antar split: train 4,23%, val 4,22%, test 2,98%.

## Preprocessing audio
- Semua audio 16 kHz mono. DC offset dibuang, normalisasi RMS ke -25 dBFS dengan batas peak -1 dBFS (43 file sangat pelan tidak mencapai target penuh).
- Hening awal dan akhir dibatasi maksimal 0,5 detik. Jeda di tengah kalimat tidak disentuh.
- Rekaman dari app harus mendapat perlakuan sama pada rekaman utuh sebelum penilaian fonem.
- Audio asli tidak diubah; hasil olahan disimpan sebagai salinan di wav16k/. Folder WAVE asli berisi 5.245 file, 245 di antaranya (speaker 96xx, tanpa anotasi) diabaikan.

## Level latihan
- Heuristik: 60% panjang fonem + 40% kesulitan fonem rata-rata. Validasi empiris lemah (spearman -0,12 anak, -0,10 dewasa), jadi gunakan label level, bukan batas panjang kalimat.
- Kolom is_digits menandai 165 kalimat berisi angka.

## Catatan penggunaan
- phone_stats.csv dan phone_difficulty.csv dihitung dari seluruh 5.000 utterance (termasuk test). Untuk bobot kelas atau statistik apa pun yang dipakai saat training, hitung ulang hanya dari split train.
- levels.json dibuat dari seluruh 5.000 utterance dan hanya untuk isi layar latihan di app, bukan untuk training model.
- Fonem ZH (25 kemunculan) dan OY (56) terlalu jarang, jangan menafsirkan skor per fonem untuk keduanya.
- File labels.jsonl dan sentences.csv adalah hasil antara, tidak termasuk dalam paket ini.

## Contoh memuat data
```python
import json
from pathlib import Path
import soundfile as sf

DATA = Path(r"D:\handoff")   # folder hasil ekstrak zip (berisi manifest + wav16k)
with open(DATA / "manifest_train.jsonl", encoding="utf-8") as f:
    rec = json.loads(f.readline())
audio, sr = sf.read(DATA / rec["audio_proc"])   # sr = 16000
print(rec["text"], rec["words"][0]["phones"], rec["words"][0]["phone_scores_norm"])
```