import asyncio
import json
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from backend.services.whisper_service import WhisperService
from backend.services.groq_whisper_service import GroqWhisperService
from backend.services.live_transcriber import LiveTranscriber
from backend.services.audio_capture import AudioCapture
from backend.services.metrics_service import build_metrics
from backend.agents.transcription_agent import TranscriptionCorrectionAgent

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="Transcription API",
    description=(
        "Unified transcription service with two endpoints:\n\n"
        "- **POST /transcribe/file** — Upload an audio file for transcription + AI correction\n"
        "- **WS  /ws/transcribe/live** — Stream live microphone audio for real-time transcription"
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD SERVICES  (loaded once at startup)
# ==========================================

whisper_service      = WhisperService()               # openai-whisper for file upload
groq_whisper_service = GroqWhisperService()           # Groq cloud for file upload
correction_agent     = TranscriptionCorrectionAgent() # LangChain agent for correction

live_transcriber = LiveTranscriber()                  # faster-whisper for live mic
audio_capture    = AudioCapture(chunk_duration_sec=5) # Mic capture for live

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Transcription API"}


# ==========================================
# ENDPOINT 1 — FILE UPLOAD TRANSCRIPTION
# POST /transcribe/file
# ==========================================

@app.post("/transcribe/file", tags=["File Transcription"])
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form(None),
    use_cloud: str = Form("false"),
    reference_text: str = Form(None),
):
    """
    Upload an audio file to transcribe it.

    - Supports: wav, mp3, m4a, ogg, flac
    - Uses local openai-whisper by default; pass `use_cloud=true` to use Groq cloud
    - The raw transcript is then corrected and summarized by an AI agent
    - Provide optional `reference_text` to calculate WER and CER
    """
    logger.info(f"[File] Received: {file.filename} | language={language} | cloud={use_cloud}")

    audio_bytes = await file.read()

    # Filter 'tanglish' — Whisper doesn't support it as a language code
    whisper_lang = None if language == "tanglish" else language

    # Choose local or cloud whisper
    active_service = groq_whisper_service if use_cloud.lower() == "true" else whisper_service

    # Transcribe (run in thread to avoid blocking the event loop)
    started_at = time.perf_counter()
    raw_text, audio_duration_sec = await asyncio.to_thread(
        active_service.transcribe,
        audio_bytes,
        whisper_lang,
        file.filename
    )
    transcription_time_ms = (time.perf_counter() - started_at) * 1000
    metrics = build_metrics(raw_text, audio_duration_sec, transcription_time_ms, reference_text)

    # Run AI correction agent
    agent_result = await asyncio.to_thread(
        correction_agent.correct,
        raw_text,
        language
    )

    return {
        "filename":       file.filename,
        "raw_text":       raw_text,
        "corrected_text": agent_result.get("corrected_text", ""),
        "summary":        agent_result.get("summary", ""),
        "evaluation":     agent_result.get("evaluation"),
        "metrics":        metrics.model_dump(),
    }


# ==========================================
# ENDPOINT 2 — LIVE MICROPHONE TRANSCRIPTION
# WS /ws/transcribe/live
# ==========================================

@app.get("/devices", tags=["Live Transcription"])
def get_devices():
    """List available microphone input devices on the server machine."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [
            {"index": i, "name": dev.get("name")}
            for i, dev in enumerate(devices)
            if dev.get("max_input_channels", 0) > 0
        ]
        default_idx = None
        try:
            default_idx = sd.default.device[0]
        except Exception:
            pass
        return {"devices": input_devices, "default": default_idx}
    except Exception as e:
        logger.error(f"[Devices] Error: {e}")
        return {"devices": [], "default": None}


@app.websocket("/ws/transcribe/live")
async def live_transcription(websocket: WebSocket):
    """
    WebSocket endpoint for real-time microphone transcription.

    Send JSON commands:
    - `{"action": "start", "model": "local"|"cloud", "device": <int|null>}`
    - `{"action": "stop"}`
    - `{"action": "switch_model", "model": "local"|"cloud"}`

    Receives JSON messages:
    - `{"text": "...", "model": "local"|"cloud"}` — transcription chunk
    - `{"status": "..."}` — status updates
    """
    await websocket.accept()
    logger.info("[Live] WebSocket connected.")

    is_transcribing = False
    model_choice = "local"
    transcription_task = None

    async def process_audio():
        while is_transcribing:
            chunk = audio_capture.get_chunk()
            if chunk is not None:
                try:
                    started_at = time.perf_counter()
                    if model_choice == "cloud":
                        text = await asyncio.to_thread(live_transcriber.transcribe_cloud, chunk)
                    else:
                        text = await asyncio.to_thread(live_transcriber.transcribe_local, chunk)
                    latency_ms = (time.perf_counter() - started_at) * 1000

                    if text:
                        logger.info(f"[Live] [{model_choice}] {text}")
                        metrics = build_metrics(
                            text,
                            audio_capture.chunk_duration_sec,
                            latency_ms,
                        )
                        await websocket.send_json({
                            "text": text,
                            "model": model_choice,
                            "metrics": metrics.model_dump(),
                        })
                except Exception as e:
                    logger.error(f"[Live] process_audio error: {e}")
                    break
            await asyncio.sleep(0.05)

    try:
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            action = command.get("action")

            if action == "start":
                model_choice = command.get("model", "local")
                device_choice = command.get("device")
                logger.info(f"[Live] start | model={model_choice} | device={device_choice}")

                if not is_transcribing:
                    is_transcribing = True
                    audio_capture.start(device=device_choice)
                    transcription_task = asyncio.create_task(process_audio())
                    await websocket.send_json({"status": f"Started recording using {model_choice} model"})
                else:
                    model_choice = command.get("model", "local")
                    await websocket.send_json({"status": f"Already recording — switched to {model_choice} model"})

            elif action == "stop":
                logger.info("[Live] stop")
                if is_transcribing:
                    is_transcribing = False
                    audio_capture.stop()
                    if transcription_task:
                        transcription_task.cancel()
                    await websocket.send_json({"status": "Stopped recording"})

            elif action == "switch_model":
                model_choice = command.get("model", "local")
                logger.info(f"[Live] switch_model → {model_choice}")
                await websocket.send_json({"status": f"Switched to {model_choice} model"})

    except WebSocketDisconnect:
        logger.info("[Live] WebSocket disconnected.")
    finally:
        is_transcribing = False
        audio_capture.stop()
        if transcription_task:
            transcription_task.cancel()
