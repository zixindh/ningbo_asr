"""Ningbo dialect to Mandarin text with Alibaba Fun-ASR realtime."""
import base64
import html
import io
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dashscope
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult


MODEL_NAME = "fun-asr-realtime"
MAINLAND_WEBSOCKET_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE_BYTES = 3200
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAINLAND_PRICE_USD_PER_SECOND = 0.000047
MIN_BROWSER_AUDIO_SECONDS = 1.0
MIN_BROWSER_AUDIO_PEAK = 200
SUPPORTED_UPLOAD_TYPES = ["pcm", "wav", "mp3", "aac", "amr", "opus", "speex"]
RECORDER_COMPONENT = components.declare_component(
    "chunk_recorder",
    path=str(Path(__file__).parent / "chunk_recorder"),
)


@dataclass
class PreparedAudio:
    path: str
    audio_format: str
    duration_seconds: float
    normalized: bool


@dataclass
class Transcript:
    text: str
    segments: list[str]
    request_id: str
    duration_seconds: float
    first_package_delay_ms: int | None
    last_package_delay_ms: int | None
    estimated_cost_usd: float


class FunAsrCallback(RecognitionCallback):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.final_sentences: list[str] = []
        self.error_message: str | None = None

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        text = str(sentence.get("text", "")).strip()
        if not text:
            return

        is_sentence_end = RecognitionResult.is_sentence_end(sentence)
        with self.lock:
            self.events.append({"text": text, "is_sentence_end": is_sentence_end})
            if is_sentence_end:
                self.final_sentences.append(text)

    def on_error(self, result: RecognitionResult) -> None:
        with self.lock:
            self.error_message = getattr(result, "message", str(result))

    def final_text(self) -> str:
        with self.lock:
            if self.final_sentences:
                return " ".join(self.final_sentences).strip()
            if self.events:
                return str(self.events[-1]["text"]).strip()
            return ""

    def segment_texts(self) -> list[str]:
        with self.lock:
            if self.final_sentences:
                return self.final_sentences.copy()
            return [str(event["text"]) for event in self.events]

    def error(self) -> str | None:
        with self.lock:
            return self.error_message


def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.environ.get(name, "")).strip()


def wav_to_mono_16k(raw_audio: bytes) -> tuple[bytes, float]:
    with wave.open(io.BytesIO(raw_audio), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frame_rate = source.getframerate()
        frame_count = source.getnframes()
        frames = source.readframes(source.getnframes())

    if sample_width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128
        max_value = 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        max_value = 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        samples = ((signed << 8) >> 8).astype(np.float32)
        max_value = 8388608.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        max_value = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    samples = samples.reshape(-1, channels)
    if channels > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples[:, 0]

    duration_seconds = frame_count / frame_rate
    samples = samples / max_value

    if frame_rate != TARGET_SAMPLE_RATE:
        target_count = max(1, int(round(duration_seconds * TARGET_SAMPLE_RATE)))
        source_positions = np.linspace(0.0, duration_seconds, num=len(samples), endpoint=False)
        target_positions = np.linspace(0.0, duration_seconds, num=target_count, endpoint=False)
        samples = np.interp(target_positions, source_positions, samples).astype(np.float32)

    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(TARGET_SAMPLE_RATE)
        target.writeframes(pcm16)

    return output.getvalue(), duration_seconds


def write_temp_audio(raw_audio: bytes, suffix: str) -> str:
    # SECURITY-REVIEW: Uploaded audio bytes are isolated in a temp file and deleted after ASR.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(raw_audio)
        return temp_audio.name


def pcm_audio_stats(raw_audio: bytes) -> tuple[float, int]:
    if len(raw_audio) < 2:
        return 0.0, 0

    even_length = len(raw_audio) - (len(raw_audio) % 2)
    samples = np.frombuffer(raw_audio[:even_length], dtype="<i2")
    if not samples.size:
        return 0.0, 0

    duration_seconds = samples.size / TARGET_SAMPLE_RATE
    peak = int(np.max(np.abs(samples.astype(np.int32))))
    return duration_seconds, peak


def prepare_audio(uploaded_file: Any) -> PreparedAudio:
    raw_audio = uploaded_file.getvalue()
    if not raw_audio:
        raise ValueError("Audio file is empty.")
    if len(raw_audio) > MAX_AUDIO_BYTES:
        raise ValueError("Audio file is larger than the 25 MB app limit.")

    suffix = Path(getattr(uploaded_file, "name", "") or "recording.wav").suffix.lower()
    if suffix not in {f".{audio_type}" for audio_type in SUPPORTED_UPLOAD_TYPES}:
        suffix = ".wav"

    if suffix == ".pcm":
        duration_seconds, _ = pcm_audio_stats(raw_audio)
        return PreparedAudio(
            path=write_temp_audio(raw_audio, ".pcm"),
            audio_format="pcm",
            duration_seconds=duration_seconds,
            normalized=False,
        )

    if suffix == ".wav":
        try:
            normalized_audio, duration_seconds = wav_to_mono_16k(raw_audio)
            return PreparedAudio(
                path=write_temp_audio(normalized_audio, ".wav"),
                audio_format="wav",
                duration_seconds=duration_seconds,
                normalized=True,
            )
        except wave.Error as error:
            raise ValueError("WAV audio must be PCM encoded.") from error

    duration_seconds = 0.0
    return PreparedAudio(
        path=write_temp_audio(raw_audio, suffix),
        audio_format=suffix.removeprefix("."),
        duration_seconds=duration_seconds,
        normalized=False,
    )


def recognize_with_fun_asr(
    prepared_audio: PreparedAudio,
    api_key: str,
    semantic_punctuation_enabled: bool,
    max_sentence_silence: int,
    throttle_stream: bool,
) -> Transcript:
    dashscope.api_key = api_key
    dashscope.base_websocket_api_url = MAINLAND_WEBSOCKET_URL

    callback = FunAsrCallback()
    recognition = Recognition(
        model=MODEL_NAME,
        format=prepared_audio.audio_format,
        sample_rate=TARGET_SAMPLE_RATE,
        semantic_punctuation_enabled=semantic_punctuation_enabled,
        max_sentence_silence=max_sentence_silence,
        callback=callback,
    )

    with open(prepared_audio.path, "rb") as audio_file:
        audio_data = audio_file.read()

    if not audio_data:
        raise ValueError("Prepared audio file is empty.")

    progress = st.progress(0, text="Connecting to Fun-ASR Beijing endpoint...")
    try:
        recognition.start()
        total_bytes = len(audio_data)

        for offset in range(0, total_bytes, CHUNK_SIZE_BYTES):
            chunk = audio_data[offset : offset + CHUNK_SIZE_BYTES]
            recognition.send_audio_frame(chunk)
            progress.progress(
                min(1.0, (offset + len(chunk)) / total_bytes),
                text="Streaming audio to Fun-ASR...",
            )
            if throttle_stream:
                time.sleep(0.1)

        progress.progress(1.0, text="Finalizing transcript...")
        recognition.stop()
    finally:
        progress.empty()

    if callback.error():
        raise RuntimeError(callback.error())

    text = callback.final_text()
    if not text:
        request_id = recognition.get_last_request_id() or "N/A"
        raise RuntimeError(
            "No Mandarin transcript was returned. Check microphone permission and try speaking for 3-5 seconds. "
            f"Request ID: {request_id}"
        )

    duration_seconds = prepared_audio.duration_seconds
    if not duration_seconds and throttle_stream:
        duration_seconds = max(1.0, total_bytes / CHUNK_SIZE_BYTES * 0.1)

    return Transcript(
        text=text,
        segments=callback.segment_texts(),
        request_id=recognition.get_last_request_id(),
        duration_seconds=duration_seconds,
        first_package_delay_ms=recognition.get_first_package_delay(),
        last_package_delay_ms=recognition.get_last_package_delay(),
        estimated_cost_usd=duration_seconds * MAINLAND_PRICE_USD_PER_SECOND,
    )


def transcribe(uploaded_file: Any, api_key: str, throttle_stream: bool = False) -> Transcript:
    prepared_audio = prepare_audio(uploaded_file)
    try:
        return recognize_with_fun_asr(
            prepared_audio=prepared_audio,
            api_key=api_key,
            semantic_punctuation_enabled=False,
            max_sentence_silence=900,
            throttle_stream=throttle_stream,
        )
    finally:
        if os.path.exists(prepared_audio.path):
            os.unlink(prepared_audio.path)


class BrowserAudioFile:
    def __init__(self, audio_bytes: bytes, name: str) -> None:
        self._audio_bytes = audio_bytes
        self.name = name

    def getvalue(self) -> bytes:
        return self._audio_bytes


def render_stitched_transcript() -> None:
    stitched_text = " ".join(st.session_state.chunk_texts).strip()
    visible_text = stitched_text or "Mandarin transcript will appear here."

    st.markdown(
        f"""
        <div class="transcript">
            <div class="transcript-label">Mandarin text</div>
            <div class="result-text">{html.escape(visible_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def transcribe_browser_chunk(chunk_data: dict[str, Any], api_key: str) -> None:
    session_id = str(chunk_data.get("sessionId") or "")
    chunk_id = int(chunk_data.get("chunkId") or 0)
    pcm_base64 = str(chunk_data.get("pcmBase64") or "")
    wav_base64 = str(chunk_data.get("wavBase64") or "")
    audio_base64 = pcm_base64 or wav_base64
    audio_format = "pcm" if pcm_base64 else "wav"

    if chunk_data.get("reset") and session_id != st.session_state.current_recorder_session:
        st.session_state.current_recorder_session = session_id
        st.session_state.processed_chunk_ids = set()
        st.session_state.chunk_texts = []
        st.session_state.total_chunk_seconds = 0.0
        if not audio_base64:
            return

    processed_key = f"{session_id}:{chunk_id}"
    if not chunk_id or not audio_base64 or processed_key in st.session_state.processed_chunk_ids:
        return

    audio_bytes = base64.b64decode(audio_base64)
    if audio_format == "pcm":
        duration_seconds, peak = pcm_audio_stats(audio_bytes)
        if duration_seconds < MIN_BROWSER_AUDIO_SECONDS:
            raise RuntimeError("Recording was too short. Press Start, speak for a few seconds, then press Stop.")
        if peak < MIN_BROWSER_AUDIO_PEAK:
            raise RuntimeError("The browser sent near-silent audio. Check the microphone permission/input and try again.")

    audio_file = BrowserAudioFile(audio_bytes, f"chunk-{chunk_id}.{audio_format}")

    with st.spinner("Transcribing..."):
        transcript = transcribe(audio_file, api_key, throttle_stream=True)

    st.session_state.processed_chunk_ids.add(processed_key)
    if transcript.text:
        st.session_state.chunk_texts.append(transcript.text)
        st.session_state.total_chunk_seconds += transcript.duration_seconds


def init_state() -> None:
    defaults = {
        "current_recorder_session": "",
        "processed_chunk_ids": set(),
        "chunk_texts": [],
        "total_chunk_seconds": 0.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


st.set_page_config(page_title="Ningbo to Mandarin", page_icon="🎙️", layout="centered")
init_state()

st.markdown(
    """
    <style>
        .stApp {
            background: #fafafa;
        }
        .block-container {
            max-width: 760px;
            padding-top: 2.75rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #111827;
            font-size: 2.35rem !important;
            line-height: 1.1 !important;
            margin-bottom: 0.25rem !important;
        }
        .subtitle {
            color: #4b5563;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        .transcript {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-top: 1.25rem;
            min-height: 220px;
            padding: 1.25rem;
        }
        .transcript-label {
            color: #6b7280;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .result-text {
            color: #111827;
            font-size: 1.75rem;
            line-height: 1.7;
            margin-top: 0.75rem;
            word-break: break-word;
        }
        .clear-button {
            margin-top: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ningbo → Mandarin")
st.markdown(
    '<div class="subtitle">Press Start, speak Ningbo/Wu, then read the Mandarin text.</div>',
    unsafe_allow_html=True,
)

api_key = get_secret("FUN_ASR_KEY")
if not api_key:
    st.warning("FUN_ASR_KEY is missing in Streamlit Secrets.")

chunk_data = RECORDER_COMPONENT(key="ningbo-chunk-recorder", default=None)
if chunk_data:
    if not api_key and (chunk_data.get("pcmBase64") or chunk_data.get("wavBase64")):
        st.error("Recognition is paused until FUN_ASR_KEY is configured.")
    else:
        try:
            transcribe_browser_chunk(chunk_data, api_key)
        except Exception as error:
            st.error(f"Recognition failed: {error}")

render_stitched_transcript()

if st.session_state.chunk_texts:
    if st.button("Clear text"):
        st.session_state.processed_chunk_ids = set()
        st.session_state.chunk_texts = []
        st.session_state.total_chunk_seconds = 0.0
        st.rerun()
