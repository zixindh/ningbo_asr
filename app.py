"""Ningbo dialect to Mandarin text with Alibaba Fun-ASR realtime."""
import audioop
import html
import io
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dashscope
import streamlit as st
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult


MODEL_NAME = "fun-asr-realtime"
MAINLAND_WEBSOCKET_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE_BYTES = 3200
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAINLAND_PRICE_USD_PER_SECOND = 0.000047
SUPPORTED_UPLOAD_TYPES = ["wav", "mp3", "aac", "amr", "opus", "speex"]


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
        self.events: list[dict[str, Any]] = []
        self.final_sentences: list[str] = []
        self.error_message: str | None = None

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        text = str(sentence.get("text", "")).strip()
        if not text:
            return

        is_sentence_end = RecognitionResult.is_sentence_end(sentence)
        self.events.append({"text": text, "is_sentence_end": is_sentence_end})
        if is_sentence_end:
            self.final_sentences.append(text)

    def on_error(self, result: RecognitionResult) -> None:
        self.error_message = getattr(result, "message", str(result))

    def final_text(self) -> str:
        if self.final_sentences:
            return " ".join(self.final_sentences).strip()
        if self.events:
            return str(self.events[-1]["text"]).strip()
        return ""

    def segment_texts(self) -> list[str]:
        if self.final_sentences:
            return self.final_sentences
        return [str(event["text"]) for event in self.events]


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
        frames = source.readframes(source.getnframes())

    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1

    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2

    if frame_rate != TARGET_SAMPLE_RATE:
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            frame_rate,
            TARGET_SAMPLE_RATE,
            None,
        )

    duration_seconds = len(frames) / (TARGET_SAMPLE_RATE * sample_width)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(TARGET_SAMPLE_RATE)
        target.writeframes(frames)

    return output.getvalue(), duration_seconds


def write_temp_audio(raw_audio: bytes, suffix: str) -> str:
    # SECURITY-REVIEW: Uploaded audio bytes are isolated in a temp file and deleted after ASR.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(raw_audio)
        return temp_audio.name


def prepare_audio(uploaded_file: Any) -> PreparedAudio:
    raw_audio = uploaded_file.getvalue()
    if not raw_audio:
        raise ValueError("Audio file is empty.")
    if len(raw_audio) > MAX_AUDIO_BYTES:
        raise ValueError("Audio file is larger than the 25 MB app limit.")

    suffix = Path(getattr(uploaded_file, "name", "") or "recording.wav").suffix.lower()
    if suffix not in {f".{audio_type}" for audio_type in SUPPORTED_UPLOAD_TYPES}:
        suffix = ".wav"

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
        language_hints=["zh"],
        semantic_punctuation_enabled=semantic_punctuation_enabled,
        max_sentence_silence=max_sentence_silence,
        callback=callback,
    )

    with open(prepared_audio.path, "rb") as audio_file:
        audio_data = audio_file.read()

    if not audio_data:
        raise ValueError("Prepared audio file is empty.")

    progress = st.progress(0, text="Connecting to Fun-ASR Beijing endpoint...")
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
    progress.empty()

    if callback.error_message:
        raise RuntimeError(callback.error_message)

    text = callback.final_text()
    if not text:
        response = recognition.get_response()
        raise RuntimeError(f"No transcript returned. Response: {response}")

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


def transcribe(uploaded_file: Any, api_key: str) -> Transcript:
    prepared_audio = prepare_audio(uploaded_file)
    try:
        return recognize_with_fun_asr(
            prepared_audio=prepared_audio,
            api_key=api_key,
            semantic_punctuation_enabled=st.session_state.semantic_punctuation_enabled,
            max_sentence_silence=st.session_state.max_sentence_silence,
            throttle_stream=st.session_state.throttle_stream,
        )
    finally:
        if os.path.exists(prepared_audio.path):
            os.unlink(prepared_audio.path)


def add_history(source: str, transcript: Transcript) -> None:
    st.session_state.history.insert(
        0,
        {
            "source": source,
            "text": transcript.text,
            "segments": transcript.segments,
            "request_id": transcript.request_id,
            "duration_seconds": transcript.duration_seconds,
            "estimated_cost_usd": transcript.estimated_cost_usd,
            "time": time.strftime("%H:%M:%S"),
        },
    )


def render_transcript(transcript: Transcript) -> None:
    escaped_text = html.escape(transcript.text)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">普通话文本</div>
            <div class="result-text">{escaped_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = [
        ("音频时长", f"{transcript.duration_seconds:.1f}s" if transcript.duration_seconds else "未知"),
        ("预估费用", f"${transcript.estimated_cost_usd:.6f}"),
        ("Request ID", transcript.request_id or "N/A"),
    ]
    columns = st.columns(3)
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)

    with st.expander("分句结果与延迟"):
        for index, segment in enumerate(transcript.segments, 1):
            st.write(f"{index}. {segment}")
        st.caption(
            f"First package: {transcript.first_package_delay_ms} ms · "
            f"Last package: {transcript.last_package_delay_ms} ms"
        )


def init_state() -> None:
    defaults = {
        "history": [],
        "semantic_punctuation_enabled": False,
        "max_sentence_silence": 900,
        "throttle_stream": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


st.set_page_config(page_title="宁波话转普通话", page_icon="🎙️", layout="centered")
init_state()

st.markdown(
    """
    <style>
        .stApp { background: #f7f2e8; }
        .hero {
            padding: 1.4rem 1.2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #14213d 0%, #0f766e 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2rem; }
        .hero p { margin: 0.35rem 0 0; color: #dbeafe; }
        .result-card {
            padding: 1.2rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            margin: 1rem 0;
        }
        .result-label {
            color: #0f766e;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .result-text {
            color: #111827;
            font-size: 1.55rem;
            line-height: 1.65;
            margin-top: 0.5rem;
        }
        .history-item {
            padding: 0.75rem 0;
            border-bottom: 1px solid #e5e7eb;
        }
        .muted { color: #64748b; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>宁波话 → 普通话</h1>
        <p>Alibaba Model Studio · Fun-ASR Realtime · 中国大陆北京部署</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = get_secret("FUN_ASR_KEY")
if not api_key:
    st.info("请在 Streamlit Secrets 中设置 `FUN_ASR_KEY`，或在环境变量中设置同名 key。")
    api_key = st.text_input("临时输入 FUN_ASR_KEY（不会保存）", type="password")
    if not api_key:
        st.stop()

with st.sidebar:
    st.subheader("识别设置")
    st.session_state.max_sentence_silence = st.slider(
        "断句静音阈值（ms）",
        min_value=200,
        max_value=3000,
        value=st.session_state.max_sentence_silence,
        step=100,
        help="宁波话句子较短时可调低；环境嘈杂或说话慢时可调高。",
    )
    st.session_state.semantic_punctuation_enabled = st.toggle(
        "语义标点（更适合长段录音）",
        value=st.session_state.semantic_punctuation_enabled,
    )
    st.session_state.throttle_stream = st.toggle(
        "按实时速度发送音频",
        value=st.session_state.throttle_stream,
        help="更贴近 Fun-ASR realtime 的推荐用法；关闭后处理更快但长音频稳定性可能下降。",
    )
    st.caption("模型：fun-asr-realtime · Endpoint：dashscope.aliyuncs.com")

record_tab, upload_tab = st.tabs(["现场录音", "上传音频"])

with record_tab:
    recorded_audio = st.audio_input("按住录音，说一段宁波话")
    if recorded_audio and st.button("识别这段录音", type="primary"):
        try:
            with st.spinner("正在识别宁波话..."):
                transcript = transcribe(recorded_audio, api_key)
            add_history("录音", transcript)
            render_transcript(transcript)
        except Exception as error:
            st.error(f"识别失败：{error}")

with upload_tab:
    uploaded_audio = st.file_uploader(
        "上传 Fun-ASR 支持的音频",
        type=SUPPORTED_UPLOAD_TYPES,
        help="推荐 WAV；录音会自动转为 16 kHz 单声道 PCM WAV。",
    )
    if uploaded_audio and st.button("识别上传文件", type="primary"):
        try:
            with st.spinner("正在上传并流式识别..."):
                transcript = transcribe(uploaded_audio, api_key)
            add_history("上传", transcript)
            render_transcript(transcript)
        except Exception as error:
            st.error(f"识别失败：{error}")

if st.session_state.history:
    st.markdown("### 最近结果")
    for item in st.session_state.history[:8]:
        st.markdown(
            f"""
            <div class="history-item">
                <strong>{html.escape(item["text"])}</strong><br>
                <span class="muted">{item["time"]} · {item["source"]} · request {item["request_id"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("清除历史"):
        st.session_state.history = []
        st.rerun()

st.caption(
    "Pricing estimate uses Chinese Mainland Fun-ASR realtime rate "
    f"(${MAINLAND_PRICE_USD_PER_SECOND}/second). Actual billing is determined by Alibaba Cloud."
)
