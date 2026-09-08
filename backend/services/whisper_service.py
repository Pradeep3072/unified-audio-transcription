import whisper
import tempfile
import os
from dotenv import load_dotenv
from backend.services.metrics_service import get_audio_duration_sec

load_dotenv()


class WhisperService:
    """Handles file-based transcription using local openai-whisper model."""

    def __init__(self):
        model_size = os.getenv("WHISPER_MODEL", "small")
        print(f"[WhisperService] Loading Whisper '{model_size}' model...")
        self.model = whisper.load_model(model_size)
        print("[WhisperService] Model loaded.")

    def transcribe(self, audio_bytes: bytes, language: str = None, filename: str = None) -> tuple[str, float]:
        temp_file = None
        try:
            suffix = os.path.splitext(filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                temp_file = f.name

            print(f"[WhisperService] Transcribing {len(audio_bytes)} bytes. Language: {language}")

            if language:
                result = self.model.transcribe(temp_file, fp16=False, language=language)
            else:
                result = self.model.transcribe(temp_file, fp16=False)

            text = result["text"].strip()
            print(f"[WhisperService] Result: {text}")
            return text, get_audio_duration_sec(temp_file)

        except Exception as e:
            print(f"[WhisperService] Error: {e}")
            return "", 0.0

        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
