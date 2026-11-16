import json
from vosk import KaldiRecognizer

class StreamingASRSession:
    def __init__(self, local_asr):
        self.recognizer = KaldiRecognizer(local_asr.model, local_asr.sample_rate)
        self.final_segments = []

    def process_chunk(self, audio_chunk):
        accepted = self.recognizer.AcceptWaveform(audio_chunk)

        if accepted:
            result = json.loads(self.recognizer.Result())
            segment = result.get("text", "")

            if segment:
                self.final_segments.append(segment)

            return {
                "type": "final_segment",
                "text": " ".join(self.final_segments),
                "segment": segment
            }

        result = json.loads(self.recognizer.PartialResult())
        partial = result.get("partial", "")
        current_text = " ".join(self.final_segments + [partial]).strip()

        return {
            "type": "partial",
            "text": current_text
        }

    def finish(self):
        result = json.loads(self.recognizer.FinalResult())
        segment = result.get("text", "")

        if segment:
            self.final_segments.append(segment)

        return {
            "type": "final",
            "text": " ".join(self.final_segments)
        }