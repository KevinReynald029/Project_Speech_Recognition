Tiap fonem dikasih skor, lalu hasilnya dibandingin sama penilaian ahli. Cara hitung dan hasil eksperimennya ada di `Pronunciation_Scoring.ipynb`. Audio masuknya harus sudah mono 16 kHz, sesuai preprocessing handoff.

Datanya [SpeechOcean762](https://github.com/jimbozhang/speechocean762), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): 5.000 rekaman dan 94.445 fonem. Dibagi jadi 2.100 train, 400 validation, dan 2.500 test. Penutur tiap split terpisah.

Alignment dan posterior fonem pakai LibriSpeech 960h nnet3 TDNN pretrained dari [kaldi-dnn-ali-gop](https://github.com/tbright17/kaldi-dnn-ali-gop), lewat Kaldi 5.5.1172 CPU. Kalibrator skor dilatih pakai gradient boosting dan disimpan di `models/calibrator.json`.

Pakai Python 3.10+. Library: NumPy, SciPy, scikit-learn, SoundFile, Matplotlib, ipykernel, nbformat, dan nbclient. Install dari folder repo:

```powershell
python -m pip install -e ".[notebook,runtime]"
```

Buka notebook di VS Code, pilih kernel Python yang sama, lalu Run All. Pengaturan awal memakai satu WAV asli beserta hasil alignment dan posterior yang sudah disimpan. Angka evaluasi dibaca dari laporan eksperimen. `RUN_TESTS` tetap `False`; kalau perlu cek tesnya, jalankan `python -m unittest discover -s tests -v`.

Kalau mau ulang dari audio lengkap, jalankan `python tools/bootstrap_native_runtime.py` di Windows. Ekstrak [ZIP handoff](https://drive.google.com/drive/folders/163B6YyeoYBNt9B6cBVCPLD2k5kxeDMBd?usp=sharing) ke `handoff/wav16k`, lalu aktifkan `RUN_FULL_EXTRACTION` dan `RUN_TRAINING`. Audio lengkap, runtime, dan cache nggak ikut Git.

MAE test skala 0-2: 0,11985. Baseline yang selalu menebak 2 dapat 0,12526. Deteksi kesalahan masih terbatas, dan empat rekaman yang berhasil di-align ulang masih dapat skor terlalu tinggi. Test ini pernah dipakai di eksperimen sebelumnya. Rinciannya ada di `reports/gop/metrics.json`.
