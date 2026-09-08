"""Audio-transcription metric helpers shared by file and live endpoints."""

from __future__ import annotations

import re
import subprocess
import wave
from typing import Optional

from jiwer import cer, wer

from backend.models.schemas import MetricsResult


def get_audio_duration_sec(audio_path: str) -> float:
    """Return an audio file's duration, using WAV metadata or ffprobe as a fallback."""
    try:
        with wave.open(audio_path, "rb") as audio:
            return round(audio.getnframes() / audio.getframerate(), 3)
    except (wave.Error, EOFError, ZeroDivisionError):
        pass

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        return round(float(result.stdout.strip()), 3) if result.returncode == 0 else 0.0
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def _normalize_for_error_rate(text: str) -> str:
    """Avoid treating casing and punctuation differences as transcription errors."""
    return re.sub(r"[^\w\s]", "", text.casefold(), flags=re.UNICODE).strip()


def build_metrics(
    text: str,
    audio_duration_sec: float,
    transcription_time_ms: float,
    reference_text: Optional[str] = None,
) -> MetricsResult:
    """Calculate output metrics; WER/CER are omitted without ground truth."""
    duration = max(float(audio_duration_sec), 0.0)
    latency = max(float(transcription_time_ms), 0.0)
    word_count = len(text.split())
    reference = _normalize_for_error_rate(reference_text or "")
    hypothesis = _normalize_for_error_rate(text)

    return MetricsResult(
        audio_duration_sec=round(duration, 3),
        transcription_time_ms=round(latency, 2),
        real_time_factor=round(latency / (duration * 1000), 4) if duration else 0.0,
        word_count=word_count,
        words_per_minute=round(word_count / (duration / 60), 2) if duration else 0.0,
        wer=round(wer(reference, hypothesis), 4) if reference else None,
        cer=round(cer(reference, hypothesis), 4) if reference else None,
    )
