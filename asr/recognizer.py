import json
import time
import wave
from pathlib import Path
from vosk import Model, KaldiRecognizer
import numpy as np

class LocalASR:
    def __init__(self, model_path, sample_rate=16000):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate

        if not self.model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {self.model_path}")

        print("Loading Vosk model")
        self.model = Model(str(self.model_path))
        print("Vosk model loaded successfully")

    def transcribe_file(self, audio_path):
        audio_path = Path(audio_path)

        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()

            if channels != 1:
                raise ValueError("Audio must be mono")

            if sample_width != 2:
                raise ValueError("Audio must be PCM 16-bit")

            if sample_rate != self.sample_rate:
                raise ValueError("Audio must be 16kHz")

            duration = wav.getnframes() / sample_rate
            recognizer = KaldiRecognizer(self.model, self.sample_rate)
            start_time = time.perf_counter()

            while True:
                chunk = wav.readframes(4000)

                if not chunk:
                    break

                recognizer.AcceptWaveform(chunk)

            result = json.loads(recognizer.FinalResult())
            processing_time = time.perf_counter() - start_time

        rtf = processing_time / duration if duration > 0 else None

        return {
            "text": result.get("text", ""),
            "duration_seconds": duration,
            "processing_seconds": processing_time,
            "rtf": rtf
        }

    
    def transcribe_array(self, audio):
        audio = np.asarray(audio, dtype=np.float32)

        # convert float audio -> PCM 16-bit bytes
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        start_time = time.perf_counter()
        recognizer.AcceptWaveform(audio_int16.tobytes())

        result = json.loads(recognizer.FinalResult())
        processing_time = time.perf_counter() - start_time
        duration = len(audio) / self.sample_rate

        return {
            "text": result.get("text", ""),
            "duration_seconds": duration,
            "processing_seconds": processing_time,
            "rtf": (
                processing_time / duration
                if duration > 0 else None
            )
        }