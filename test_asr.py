from asr.recognizer import LocalASR

MODEL_PATH = "models/asr/vosk-model-small-en-us-0.15"
AUDIO_PATH = "examples/real_sample.wav"

asr = LocalASR(MODEL_PATH)

result = asr.transcribe_file(AUDIO_PATH)

print("ASR Result")
print("Transcript:", result["text"])
print("Audio duration:", result["duration_seconds"])
print("Processing time:", result["processing_seconds"])
print("RTF:", result["rtf"])