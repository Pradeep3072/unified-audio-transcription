import os
import tempfile
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqWhisperService:
    """Handles file-based transcription using Groq cloud whisper-large-v3."""

    def __init__(self):
        print("[GroqWhisperService] Initializing...")
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("[GroqWhisperService] WARNING: GROQ_API_KEY not set. Cloud transcription disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        print("[GroqWhisperService] Ready.")

    def transcribe(self, audio_bytes: bytes, language: str = None, filename: str = None) -> str:
        if not self.client:
            print("[GroqWhisperService] Error: API key missing.")
            return ""

        temp_file = None
        try:
            ext = os.path.splitext(filename)[1] if filename else ".wav"
            if not ext:
                ext = ".wav"

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_bytes)
                temp_file = f.name

            with open(temp_file, "rb") as audio_file:
                kwargs = {
                    "file": (os.path.basename(temp_file), audio_file),
                    "model": "whisper-large-v3",
                }
                if language:
                    kwargs["language"] = language

                transcription = self.client.audio.transcriptions.create(**kwargs)
                return transcription.text

        except Exception as e:
            print(f"[GroqWhisperService] Error: {e}")
            return ""

        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
