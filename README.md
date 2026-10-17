Pronunciation scoring untuk menilai ketepatan fonem dari rekaman dan kalimat target. Output berupa skor fonem, penanda perlu latihan, dan rata-rata skor per kata/kalimat. Fluency, intonasi, dan stress belum dinilai.

Dataset: SpeechOcean762 dari [handoff](https://github.com/KevinReynald029/Project_Speech_Recognition), 5.000 rekaman dan 94.445 fonem. Split: 2.100 train (42%), 400 validation (8%), dan 2.500 test (50%). Model dilatih pada train, pemilihan parameter dan ambang memakai validation.

Model akustik: LibriSpeech 960h nnet3 TDNN dari [kaldi-dnn-ali-gop](https://github.com/tbright17/kaldi-dnn-ali-gop), using Kaldi 5.5.1172 CPU.
Library : NumPy, SciPy, scikit-learn, SoundFile, dan Matplotlib. Notebook/pengujian memakai ipykernel, nbformat, nbclient, dan jsonschema. Dari folder hasil clone, install dependency:

```powershell
python -m pip install -e ".[notebook,test,runtime]"
```
