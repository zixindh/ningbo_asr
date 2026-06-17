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
INTERNATIONAL_WEBSOCKET_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
SINGAPORE_MAAS_API_HOST = "ws-f6jqz1vpb4gjfvhw.ap-southeast-1.maas.aliyuncs.com"
SINGAPORE_MAAS_WEBSOCKET_URL = f"wss://{SINGAPORE_MAAS_API_HOST}/api-ws/v1/inference"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE_BYTES = 3200
FRAME_SEND_DELAY_SECONDS = 0.02
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAINLAND_PRICE_USD_PER_SECOND = 0.000047
MIN_BROWSER_AUDIO_SECONDS = 1.0
MIN_BROWSER_AUDIO_PEAK = 200
SUPPORTED_UPLOAD_TYPES = ["pcm", "wav", "mp3", "aac", "amr", "opus", "speex"]
RECORDER_COMPONENT = components.declare_component(
    "chunk_recorder",
    path=str(Path(__file__).parent / "chunk_recorder"),
)


class NoTranscriptError(RuntimeError):
    pass


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
        message = getattr(result, "message", None)
        request_id = getattr(result, "request_id", None) or getattr(result, "get_request_id", lambda: "")()
        if not message:
            sentence = result.get_sentence() if hasattr(result, "get_sentence") else {}
            message = sentence.get("message") or sentence.get("error") or type(result).__name__
        if request_id:
            message = f"{message} (request_id: {request_id})"

        with self.lock:
            self.error_message = str(message)

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


def get_fun_asr_key() -> str:
    region = get_secret("FUN_ASR_REGION").lower()
    if region in {"mainland", "china", "beijing", "cn"}:
        return get_secret("FUN_ASR_KEY")
    if region in {"international", "intl", "singapore", "sg"}:
        return get_secret("FUN_ASR_SG_KEY") or get_secret("FUN_ASR_KEY")
    return get_secret("FUN_ASR_SG_KEY") or get_secret("FUN_ASR_KEY")


def normalize_websocket_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return ""

    if value.startswith("ws://") or value.startswith("wss://"):
        return value

    if value.startswith("http://") or value.startswith("https://"):
        host = value.split("://", 1)[1].split("/", 1)[0]
    else:
        host = value.split("/", 1)[0]
    return f"wss://{host}/api-ws/v1/inference"


def get_fun_asr_endpoints() -> list[tuple[str, str]]:
    region = get_secret("FUN_ASR_REGION").lower()
    if region in {"mainland", "china", "beijing", "cn"}:
        return [("mainland", MAINLAND_WEBSOCKET_URL)]

    custom_url = normalize_websocket_url(
        get_secret("FUN_ASR_WEBSOCKET_URL")
        or get_secret("FUN_ASR_API_HOST")
        or get_secret("FUN_ASR_DASHSCOPE_URL")
    )
    if custom_url:
        return [("custom", custom_url)]

    if region in {"international", "intl", "singapore", "sg"}:
        return [
            ("singapore-maas", SINGAPORE_MAAS_WEBSOCKET_URL),
            ("international", INTERNATIONAL_WEBSOCKET_URL),
        ]
    if get_secret("FUN_ASR_SG_KEY"):
        return [
            ("singapore-maas", SINGAPORE_MAAS_WEBSOCKET_URL),
            ("international", INTERNATIONAL_WEBSOCKET_URL),
        ]

    # Streamlit Cloud Free runs outside mainland China, so try the international endpoint first.
    return [
        ("singapore-maas", SINGAPORE_MAAS_WEBSOCKET_URL),
        ("international", INTERNATIONAL_WEBSOCKET_URL),
        ("mainland", MAINLAND_WEBSOCKET_URL),
    ]


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
    endpoint_name: str,
    endpoint_url: str,
    semantic_punctuation_enabled: bool,
    max_sentence_silence: int,
    throttle_stream: bool,
) -> Transcript:
    dashscope.api_key = api_key
    dashscope.base_websocket_api_url = endpoint_url

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

    recognition.start()
    total_bytes = len(audio_data)

    for offset in range(0, total_bytes, CHUNK_SIZE_BYTES):
        chunk = audio_data[offset : offset + CHUNK_SIZE_BYTES]
        recognition.send_audio_frame(chunk)
        if throttle_stream:
            time.sleep(FRAME_SEND_DELAY_SECONDS)

    recognition.stop()

    if callback.error():
        raise RuntimeError(callback.error())

    text = callback.final_text()
    if not text:
        request_id = recognition.get_last_request_id() or "N/A"
        audio_note = ""
        if prepared_audio.audio_format == "pcm":
            duration_seconds, peak = pcm_audio_stats(audio_data)
            audio_note = f" Captured audio: {duration_seconds:.1f}s, peak level {peak}."
        raise NoTranscriptError(
            "No Mandarin transcript was returned. Check microphone permission and try speaking for 3-5 seconds. "
            f"Endpoint: {endpoint_name}. Request ID: {request_id}.{audio_note}"
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
        errors: list[str] = []
        for endpoint_name, endpoint_url in get_fun_asr_endpoints():
            try:
                return recognize_with_fun_asr(
                    prepared_audio=prepared_audio,
                    api_key=api_key,
                    endpoint_name=endpoint_name,
                    endpoint_url=endpoint_url,
                    semantic_punctuation_enabled=False,
                    max_sentence_silence=900,
                    throttle_stream=throttle_stream,
                )
            except NoTranscriptError as error:
                errors.append(str(error))
            except Exception as error:
                errors.append(f"{endpoint_name}: {error}")

        raise RuntimeError("Recognition did not return text from any configured endpoint. " + " | ".join(errors))
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
    visible_text = stitched_text or ""

    st.markdown(
        f"""
        <div class="transcript">
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
            padding-top: 0.85rem;
            padding-bottom: 1rem;
        }
        .transcript {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-top: 0.65rem;
            min-height: 320px;
            padding: 0.85rem;
        }
        .result-text {
            color: #111827;
            font-size: 1.05rem;
            line-height: 1.6;
            word-break: break-word;
            white-space: pre-wrap;
        }
        @media (max-width: 640px) {
            .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
            .result-text {
                font-size: 0.98rem;
                line-height: 1.55;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = get_fun_asr_key()
if not api_key:
    st.warning("FUN_ASR_SG_KEY or FUN_ASR_KEY is missing in Streamlit Secrets.")

chunk_data = RECORDER_COMPONENT(key="ningbo-chunk-recorder", default=None, height=132)
if chunk_data:
    if not api_key and (chunk_data.get("pcmBase64") or chunk_data.get("wavBase64")):
        st.error("Recognition is paused until FUN_ASR_KEY is configured.")
    else:
        try:
            transcribe_browser_chunk(chunk_data, api_key)
        except Exception as error:
            st.error(f"Recognition failed: {error}")

render_stitched_transcript()
