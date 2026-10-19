Repo ini buat scoring pelafalan fonem: GOP, rumus, prediksi skor, dan evaluasi terhadap ahli. Penjelasannya ada di `Pronunciation_Scoring.ipynb`. Input wajib audio mono 16 kHz yang sudah mengikuti preprocessing handoff. Scoring tidak mengulang preprocessing.

Dataset [SpeechOcean762](https://github.com/jimbozhang/speechocean762), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): 5.000 rekaman dan 94.445 fonem. Split 2.100 train, 400 validation, 2.500 test tanpa percampuran penutur.

Model akustik: LibriSpeech 960h nnet3 TDNN pretrained, Kaldi 5.5.1172 CPU, hanya untuk alignment dan posterior. Yang dilatih adalah kalibrator skor GOP berupa gradient boosting; `models/calibrator.json` ikut repo. Library: NumPy, SciPy, scikit-learn, SoundFile, Matplotlib, jsonschema, ipykernel, nbformat, nbclient. Pakai Python 3.10+.

Install dari folder repo:

```powershell
python -m pip install -e ".[notebook,test,runtime]"
```

Buka notebook di VS Code, pilih kernel Python yang sama, lalu Run All. Default memakai satu WAV asli dan evidence tersimpan; evaluasi lengkap dibaca dari hasil eksperimen yang ikut repo.

Untuk mengulang dari audio: jalankan `python tools/bootstrap_native_runtime.py` (Windows), ekstrak [ZIP handoff](https://drive.google.com/drive/folders/163B6YyeoYBNt9B6cBVCPLD2k5kxeDMBd?usp=sharing) ke `handoff/wav16k`, lalu aktifkan `RUN_FULL_EXTRACTION` dan `RUN_TRAINING`. Runtime, audio lengkap, dan cache tidak ikut Git. File persiapan dataset bawaan tetap utuh.

MAE test skala 0-2: 0,11985; baseline konstan 2: 0,12526. Deteksi kesalahan masih terbatas; empat alignment yang pulih masih memberi skor terlalu tinggi. Test pernah dipakai sebelumnya. Hasil lengkap di `reports/gop`; sumber model dan checksum di `PROVENANCE.json`.
