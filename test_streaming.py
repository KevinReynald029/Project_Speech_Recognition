import wave
from asr.recognizer import LocalASR
from asr.streaming import StreamingASRSession

MODEL_PATH = "models/asr/vosk-model-small-en-us-0.15"
AUDIO_PATH = "examples/real_sample.wav"

asr = LocalASR(MODEL_PATH)
session = StreamingASRSession(asr)

with wave.open(AUDIO_PATH, "rb") as wav:
    while True:
        chunk = wav.readframes(4000)

        if not chunk:
            break

        result = session.process_chunk(chunk)
        print(result)

final_result = session.finish()

print("Transcript Final")
print(final_result["text"])