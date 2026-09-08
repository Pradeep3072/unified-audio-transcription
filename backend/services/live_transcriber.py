import os
import io
import numpy as np
import scipy.io.wavfile as wavfile
import logging
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger("live_transcriber")


class LiveTranscriber:
    """
    Handles real-time transcription of live audio chunks (numpy arrays).
    Uses faster-whisper locally (low latency) or Groq cloud as fallback.
    """

    def __init__(self):
        logger.info("[LiveTranscriber] Loading faster-whisper 'base' model...")
        self.local_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("[LiveTranscriber] Local model loaded.")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            self.groq_client = None
            logger.warning("[LiveTranscriber] GROQ_API_KEY not set. Cloud transcription disabled.")
        else:
            self.groq_client = Groq(api_key=api_key)

    def transcribe_local(self, audio_chunk_np: np.ndarray) -> str:
        """Transcribe a numpy float32 audio chunk using local faster-whisper."""
        segments, _ = self.local_model.transcribe(
            audio_chunk_np,
            beam_size=5,
            language="en",
            initial_prompt="Transcribe the speech accurately with proper punctuation and capitalization."
        )
        text = "".join([seg.text for seg in segments])
        return text.strip()

    def transcribe_cloud(self, audio_chunk_np: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy float32 audio chunk using Groq cloud whisper-large-v3."""
        if not self.groq_client:
            logger.error("[LiveTranscriber] Cloud transcription requested but GROQ_API_KEY is not set.")
            return "[Error: GROQ_API_KEY not configured in .env]"

        # Convert float32 → int16 for WAV compatibility
        audio_int16 = (audio_chunk_np * 32767).astype(np.int16)
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sample_rate, audio_int16)
        byte_io.seek(0)
        file_bytes = byte_io.read()

        logger.info(f"[LiveTranscriber] Sending {len(file_bytes)} bytes to Groq...")

        try:
            transcription = self.groq_client.audio.transcriptions.create(
                file=("audio.wav", file_bytes, "audio/wav"),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                prompt="Transcribe the speech accurately with proper punctuation and capitalization."
            )
            text = transcription.text.strip()
            logger.info(f"[LiveTranscriber] Groq result ({len(text)} chars): {text}")
            return text
        except Exception as e:
            logger.error(f"[LiveTranscriber] Groq error: {e}")
            return f"[Cloud Error: {e}]"
