import streamlit as st
import requests
import websocket
import json
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==========================================
# CONFIG
# ==========================================

BACKEND_URL = "http://localhost:8000"
WS_URL      = "ws://localhost:8000"

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Transcription App",
    page_icon="🎙️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        padding: 10px 28px;
        border-radius: 8px;
    }
    .endpoint-badge {
        display: inline-block;
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 12px;
        font-family: monospace;
        color: #60a5fa;
        margin-bottom: 16px;
    }
    .status-idle    { color: #94a3b8; font-weight: 600; }
    .status-active  { color: #22c55e; font-weight: 600; }
    .status-stopped { color: #f87171; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.title("🎙️ Transcription App")
st.markdown(
    "One app, two transcription modes — **file upload** with AI correction "
    "and **live microphone** streaming."
)
st.divider()

# ==========================================
# LANGUAGE MAP
# ==========================================

LANGUAGES = {
    "Auto-detect": None,
    "English":     "en",
    "Hindi":       "hi",
    "Tamil":       "ta",
    "Tanglish":    "tanglish",
    "Telugu":      "te",
    "Malayalam":   "ml",
    "Spanish":     "es",
    "French":      "fr",
    "German":      "de",
    "Chinese":     "zh",
    "Japanese":    "ja",
    "Korean":      "ko",
    "Russian":     "ru",
    "Portuguese":  "pt",
}

# ==========================================
# TABS
# ==========================================

tab1, tab2 = st.tabs(["🎵  File Transcription", "🔴  Live Transcription"])


# ==================================================================
# TAB 1 — FILE TRANSCRIPTION
# Endpoint: POST /transcribe/file
# ==================================================================

with tab1:

    st.markdown(
        '<div class="endpoint-badge">POST  /transcribe/file</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Upload an audio file. It will be transcribed by Whisper and then "
        "corrected, summarized, and evaluated by an AI agent."
    )

    settings_col, upload_col = st.columns([1, 2])

    with settings_col:
        st.markdown("#### ⚙️ Settings")

        lang_name = st.selectbox("Language", list(LANGUAGES.keys()), key="file_lang")
        lang_code = LANGUAGES[lang_name]

        use_cloud = st.toggle(
            "☁️ Use Cloud Whisper (Groq)",
            value=False,
            key="file_cloud",
            help="ON → Groq cloud (fast). OFF → local openai-whisper."
        )

        reference_text = st.text_area(
            "Reference text (optional)",
            help="Provide the expected transcript to calculate WER and CER.",
            key="file_reference_text",
            placeholder="Paste the ground-truth transcript here…",
        )

        if st.button("🔍 Check Backend", key="file_health_btn"):
            try:
                r = requests.get(f"{BACKEND_URL}/", timeout=3)
                if r.status_code == 200:
                    st.success("Backend online ✅")
                else:
                    st.warning(f"Status {r.status_code}")
            except Exception:
                st.error(f"Cannot reach `{BACKEND_URL}` — is the server running?")

    with upload_col:
        st.markdown("#### 📂 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Supported: wav, mp3, m4a, ogg, flac",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="file_uploader"
        )

        if uploaded_file:
            st.audio(uploaded_file)

            if st.button("🚀 Transcribe", type="primary", key="file_transcribe_btn"):
                with st.spinner("Transcribing and correcting with AI…"):
                    try:
                        form_data = {}
                        if lang_code:
                            form_data["language"] = lang_code
                        if use_cloud:
                            form_data["use_cloud"] = "true"
                        if reference_text.strip():
                            form_data["reference_text"] = reference_text.strip()

                        response = requests.post(
                            f"{BACKEND_URL}/transcribe/file",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            data=form_data,
                            timeout=120
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Done!")

                            col_raw, col_corrected = st.columns(2)
                            with col_raw:
                                st.markdown("##### 🗣️ Raw Transcript")
                                st.info(result.get("raw_text", "—"))
                            with col_corrected:
                                st.markdown("##### ✨ AI Corrected")
                                st.success(result.get("corrected_text", "—"))

                            if result.get("summary"):
                                st.markdown("##### 📝 Summary")
                                st.info(result["summary"])

                            metrics = result.get("metrics")
                            if metrics:
                                st.markdown("##### 📊 Metrics")
                                metric_cols = st.columns(6)
                                metric_cols[0].metric("Audio duration", f"{metrics.get('audio_duration_sec', 0):.2f}s")
                                metric_cols[1].metric("Latency", f"{metrics.get('transcription_time_ms', 0):.0f} ms")
                                metric_cols[2].metric("RTF", f"{metrics.get('real_time_factor', 0):.2f}")
                                metric_cols[3].metric("WPM", f"{metrics.get('words_per_minute', 0):.1f}")
                                metric_cols[4].metric("WER", f"{metrics['wer']:.2%}" if metrics.get("wer") is not None else "N/A")
                                metric_cols[5].metric("CER", f"{metrics['cer']:.2%}" if metrics.get("cer") is not None else "N/A")

                            eval_data = result.get("evaluation")
                            if eval_data:
                                st.markdown("##### 📊 Evaluation")
                                score_col, detail_col = st.columns([1, 2])
                                with score_col:
                                    st.metric("Grammar & Clarity", f"{eval_data.get('grammar_score')}/10")
                                with detail_col:
                                    st.write(f"**Improvement:** {eval_data.get('readability_improvement', '')}")
                                    changes = eval_data.get("changes_made", [])
                                    if changes:
                                        st.write("**Changes:**")
                                        for c in changes:
                                            st.write(f"- {c}")
                        else:
                            st.error(f"Backend error {response.status_code}: {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error(
                            f"❌ Cannot reach backend at `{BACKEND_URL}`.\n\n"
                            "Run: `uvicorn backend.main:app --port 8000 --reload`"
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")


# ==================================================================
# TAB 2 — LIVE MIC TRANSCRIPTION
# Endpoint: WS /ws/transcribe/live
# ==================================================================

with tab2:

    # --- Session state init ---
    for key, default in {
        "live_transcript":      "",
        "live_is_recording":    False,
        "live_ws":              None,
        "live_status":          "Idle",
        "live_msg_buffer":      [],
        "live_devices":         [],
        "live_default_device":  None,
        "live_model":           "local",
        "live_device_index":    None,
        "live_chunk_metrics":   [],
        "live_word_count":      0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    def fetch_devices():
        try:
            r = requests.get(f"{BACKEND_URL}/devices", timeout=3)
            if r.status_code == 200:
                data = r.json()
                st.session_state.live_devices = data.get("devices", [])
                st.session_state.live_default_device = data.get("default")
        except Exception:
            st.session_state.live_devices = []

    if not st.session_state.live_devices:
        fetch_devices()

    def on_ws_message(ws, message):
        try:
            st.session_state.live_msg_buffer.append(json.loads(message))
        except Exception:
            pass

    def on_ws_open(ws):
        if st.session_state.live_is_recording:
            ws.send(json.dumps({
                "action": "start",
                "model": st.session_state.live_model,
                "device": st.session_state.live_device_index
            }))

    def start_ws():
        ws = websocket.WebSocketApp(
            f"{WS_URL}/ws/transcribe/live",
            on_message=on_ws_message,
            on_open=on_ws_open
        )
        st.session_state.live_ws = ws
        ws.run_forever()

    # --- UI ---
    st.markdown(
        '<div class="endpoint-badge">WS  /ws/transcribe/live</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "Transcribe speech in real-time from your microphone. "
        "Text streams live as you speak."
    )

    ctrl_col, output_col = st.columns([1, 2])

    with ctrl_col:
        st.markdown("#### ⚙️ Controls")

        # Mic device selector
        dev_col, refresh_col = st.columns([4, 1])
        with dev_col:
            if st.session_state.live_devices:
                device_options = {d["index"]: d["name"] for d in st.session_state.live_devices}
                default_idx = st.session_state.live_default_device
                if default_idx not in device_options:
                    default_idx = st.session_state.live_devices[0]["index"]
                try:
                    default_ui_idx = list(device_options.keys()).index(default_idx)
                except ValueError:
                    default_ui_idx = 0

                selected_device = st.selectbox(
                    "🎤 Microphone",
                    options=list(device_options.keys()),
                    format_func=lambda x: device_options[x],
                    index=default_ui_idx,
                    disabled=st.session_state.live_is_recording,
                    key="live_device_select"
                )
                st.session_state.live_device_index = selected_device
            else:
                st.selectbox("🎤 Microphone", ["Default (backend offline)"], disabled=True, key="live_device_offline")
                st.session_state.live_device_index = None

        with refresh_col:
            st.write("")
            st.write("")
            if st.button("🔄", help="Refresh mic list", disabled=st.session_state.live_is_recording, key="live_refresh"):
                fetch_devices()
                st.rerun()

        # Model selector
        model_choice = st.selectbox(
            "🤖 Model",
            ["local", "cloud"],
            index=0 if st.session_state.live_model == "local" else 1,
            help="local = faster-whisper (no internet). cloud = Groq (needs API key).",
            key="live_model_select"
        )

        # Switch model live if already recording
        if st.session_state.live_is_recording and st.session_state.live_model != model_choice:
            st.session_state.live_model = model_choice
            ws = st.session_state.live_ws
            if ws and getattr(ws, "sock", None) and getattr(ws.sock, "connected", False):
                try:
                    ws.send(json.dumps({"action": "switch_model", "model": model_choice}))
                except Exception:
                    pass

        # Start / Stop
        is_rec = st.session_state.live_is_recording
        btn_label = "⏹ Stop Recording" if is_rec else "▶ Start Recording"
        btn_type  = "secondary"         if is_rec else "primary"

        if st.button(btn_label, type=btn_type, key="live_record_btn"):
            st.session_state.live_is_recording = not is_rec

            if st.session_state.live_is_recording:
                st.session_state.live_transcript = ""
                st.session_state.live_chunk_metrics = []
                st.session_state.live_word_count = 0
                st.session_state.live_model = model_choice
                ws = st.session_state.live_ws
                ws_ok = ws and getattr(ws, "sock", None) and getattr(ws.sock, "connected", False)
                if not ws_ok:
                    t = threading.Thread(target=start_ws, daemon=True)
                    add_script_run_ctx(t)
                    t.start()
                    st.session_state.live_status = "Connecting…"
                else:
                    try:
                        ws.send(json.dumps({"action": "start", "model": model_choice}))
                    except Exception:
                        pass
            else:
                ws = st.session_state.live_ws
                if ws and getattr(ws, "sock", None) and getattr(ws.sock, "connected", False):
                    try:
                        ws.send(json.dumps({"action": "stop"}))
                    except Exception:
                        pass
                st.session_state.live_status = "Stopped"

            st.rerun()

        # Status
        status = st.session_state.live_status
        if st.session_state.live_is_recording:
            st.markdown(f'<p class="status-active">● {status}</p>', unsafe_allow_html=True)
            with st.spinner("Listening…"):
                st.caption("Capturing audio from microphone…")
        else:
            css = "status-stopped" if status == "Stopped" else "status-idle"
            st.markdown(f'<p class="{css}">● {status}</p>', unsafe_allow_html=True)

        if st.button("🔍 Check Backend", key="live_health_btn"):
            try:
                r = requests.get(f"{BACKEND_URL}/", timeout=3)
                if r.status_code == 200:
                    st.success("Backend online ✅")
                else:
                    st.warning(f"Status {r.status_code}")
            except Exception:
                st.error(f"Cannot reach `{BACKEND_URL}`")

    with output_col:
        st.markdown("#### 📝 Live Transcript")
        transcript_box = st.empty()

        if st.session_state.live_is_recording:
            # Drain the message buffer
            while st.session_state.live_msg_buffer:
                msg = st.session_state.live_msg_buffer.pop(0)
                if "text" in msg:
                    st.session_state.live_transcript += msg["text"] + " "
                    st.session_state.live_word_count += len(msg["text"].split())
                if "metrics" in msg:
                    st.session_state.live_chunk_metrics.append(msg["metrics"])
                if "status" in msg:
                    st.session_state.live_status = msg["status"]

            transcript_box.info(
                st.session_state.live_transcript or "🎤 Listening… start speaking!"
            )
            recent_metrics = st.session_state.live_chunk_metrics[-1] if st.session_state.live_chunk_metrics else None
            with st.container(border=True):
                st.caption("Live metrics")
                live_metrics_cols = st.columns(3)
                live_metrics_cols[0].metric("Chunks", len(st.session_state.live_chunk_metrics))
                live_metrics_cols[1].metric("Words", st.session_state.live_word_count)
                live_metrics_cols[2].metric(
                    "Last chunk latency",
                    f"{recent_metrics.get('transcription_time_ms', 0):.0f} ms" if recent_metrics else "—",
                )
            time.sleep(0.5)
            st.rerun()
        else:
            if st.session_state.live_transcript:
                transcript_box.success(st.session_state.live_transcript)
                if st.session_state.live_chunk_metrics:
                    average_latency = sum(
                        metric.get("transcription_time_ms", 0)
                        for metric in st.session_state.live_chunk_metrics
                    ) / len(st.session_state.live_chunk_metrics)
                    st.caption(
                        f"{len(st.session_state.live_chunk_metrics)} chunks • "
                        f"{st.session_state.live_word_count} words • "
                        f"average chunk latency {average_latency:.0f} ms"
                    )
                if st.button("🗑️ Clear", key="live_clear"):
                    st.session_state.live_transcript = ""
                    st.session_state.live_chunk_metrics = []
                    st.session_state.live_word_count = 0
                    st.rerun()
            else:
                transcript_box.write("No transcription yet. Press **▶ Start Recording** to begin.")
