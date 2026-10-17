Pronunciation scoring untuk menilai ketepatan fonem dari rekaman dan kalimat target. Output berupa skor fonem 0â€“100, penanda perlu latihan, dan rata-rata skor per kata/kalimat. Fluency, intonasi, dan stress belum dinilai.

Dataset: SpeechOcean762 dari [handoff Kevin](https://github.com/KevinReynald029/Project_Speech_Recognition), berisi 5.000 rekaman dan 94.445 fonem. Split: 2.100 train (42%), 400 validation (8%), dan 2.500 test (50%), tanpa penutur yang bercampur. Model dilatih pada train; pemilihan parameter dan ambang memakai validation.

Model akustik: LibriSpeech 960h nnet3 TDNN dari [kaldi-dnn-ali-gop](https://github.com/tbright17/kaldi-dnn-ali-gop), memakai Kaldi 5.5.1172 CPU. Fitur GOP dipetakan ke skor ahli dengan HistGradientBoostingRegressor, loss absolute error, 250 iterasi, maksimum 31 leaf. Kalibrator terlatih tersedia di `models/calibrator.json`.

Library utama: NumPy, SciPy, scikit-learn, SoundFile, dan Matplotlib. Notebook/pengujian memakai ipykernel, nbformat, nbclient, dan jsonschema. Dari folder hasil clone, install dependency:

```powershell
python -m pip install -e ".[notebook,test,runtime]"
```

Buka `Pronunciation_Scoring.ipynb` di VS Code, pilih kernel Python tempat dependency dipasang, lalu Run All. Mode bawaan membaca evaluasi yang disimpan dan menjalankan kalibrator pada evidence Kaldi dari sampel WAV asli di `examples`. Seluruh file untuk mode ini ikut dalam repo; audio dataset lengkap, runtime, dan cache tidak diperlukan.

Scoring rekaman baru memakai runtime Windows 64-bit. Setup berikut mengunduh binary dan model akustik dengan checksum terkunci; Python 3.14 adalah versi yang sudah diuji. Jalankan dari folder proyek:

```powershell
python tools/bootstrap_native_runtime.py
python -m phoneme_scoring score --audio recording.wav --reference examples/reference.json --model models/calibrator.json --kaldi-config config/kaldi.native.json --output result.json
```

Referensi harus sesuai kalimat rekaman. Konfigurasi bersama memakai path relatif terhadap proyek. Override pribadi boleh disimpan di `config/*.local.json` dan tidak ikut Git. Untuk mencoba inferensi langsung di notebook setelah setup, aktifkan `RUN_LIVE_INFERENCE`.

Untuk ekstraksi/training ulang, ambil [ZIP audio handoff](https://drive.google.com/drive/folders/163B6YyeoYBNt9B6cBVCPLD2k5kxeDMBd?usp=sharing), lalu letakkan folder `wav16k` di dalam `handoff`, sejajar dengan manifest. Notebook juga mendukung lokasi lokal `data/handoff`. Aktifkan `RUN_FULL_EXTRACTION` dan `RUN_TRAINING`. Cache fitur akan dibuat ulang. WAV handoff memakai `already_preprocessed=True`; rekaman baru diproses seperti biasa.

Hasil test: MAE 0,11985 pada skala 0â€“2, dibanding baseline konstan 2 sebesar 0,12526. F1 0,3441, turun dari Ridge lama 0,3611. Seluruh rekaman berhasil alignment, tetapi empat kasus retry masih mendapat skor terlalu tinggi. Test dipakai ulang dari eksperimen sebelumnya. Metrik/grafik ada di `reports/gop`; analisis kasus sulit ada di `reports/gop/recovery_summary.json`.

Tes dapat dijalankan dengan `python -m unittest discover -s tests -v`. Model kalibrator, konfigurasi bersama, kode, notebook, dan hasil ringkas masuk Git. Audio, runtime, cache, pengaturan editor, serta konfigurasi pribadi tetap lokal.

Dataset dibuat oleh Junbo Zhang dkk., *speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment*, Interspeech 2021 ([sumber](https://github.com/jimbozhang/speechocean762), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Handoff Kevin menambahkan split validation, preprocessing audio, penghapusan digit stress, dan normalisasi label. Riwayat sumber/checksum tersedia di `PROVENANCE.json`.
