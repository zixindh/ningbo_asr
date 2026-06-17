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
SUPPORTED_UPLOAD_TYPES = ["wav", "mp3", "aac", "amr", "opus", "speex"]
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

    if callback.error():
        raise RuntimeError(callback.error())

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


def transcribe(uploaded_file: Any, api_key: str, throttle_stream: bool | None = None) -> Transcript:
    prepared_audio = prepare_audio(uploaded_file)
    try:
        return recognize_with_fun_asr(
            prepared_audio=prepared_audio,
            api_key=api_key,
            semantic_punctuation_enabled=st.session_state.semantic_punctuation_enabled,
            max_sentence_silence=st.session_state.max_sentence_silence,
            throttle_stream=st.session_state.throttle_stream if throttle_stream is None else throttle_stream,
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


class BrowserAudioFile:
    def __init__(self, audio_bytes: bytes, name: str) -> None:
        self._audio_bytes = audio_bytes
        self.name = name

    def getvalue(self) -> bytes:
        return self._audio_bytes


def render_stitched_transcript() -> None:
    stitched_text = " ".join(st.session_state.chunk_texts).strip()
    visible_text = stitched_text or "点击 START 后直接说宁波话。应用会每 5 秒自动识别并拼接。"
    chunk_count = len(st.session_state.chunk_texts)
    duration_seconds = st.session_state.total_chunk_seconds

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">自动分段识别 · {chunk_count} 段 · {duration_seconds:.1f}s</div>
            <div class="result-text">{html.escape(visible_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.chunk_texts:
        with st.expander("5 秒分段结果"):
            for item in st.session_state.chunk_items:
                st.write(f"{item['chunk_id']}. {item['text']}")


def transcribe_browser_chunk(chunk_data: dict[str, Any], api_key: str) -> None:
    session_id = str(chunk_data.get("sessionId") or "")
    if chunk_data.get("reset") and session_id != st.session_state.current_recorder_session:
        st.session_state.current_recorder_session = session_id
        st.session_state.processed_chunk_ids = set()
        st.session_state.chunk_texts = []
        st.session_state.chunk_items = []
        st.session_state.total_chunk_seconds = 0.0
        return

    chunk_id = int(chunk_data.get("chunkId") or 0)
    wav_base64 = str(chunk_data.get("wavBase64") or "")
    processed_key = f"{session_id}:{chunk_id}"
    if not chunk_id or not wav_base64 or processed_key in st.session_state.processed_chunk_ids:
        return

    audio_bytes = base64.b64decode(wav_base64)
    audio_file = BrowserAudioFile(audio_bytes, f"chunk-{chunk_id}.wav")

    with st.spinner(f"正在识别第 {chunk_id} 段..."):
        transcript = transcribe(audio_file, api_key, throttle_stream=False)

    st.session_state.processed_chunk_ids.add(processed_key)
    if transcript.text:
        st.session_state.chunk_texts.append(transcript.text)
        st.session_state.chunk_items.append(
            {
                "chunk_id": chunk_id,
                "text": transcript.text,
                "request_id": transcript.request_id,
                "duration_seconds": transcript.duration_seconds,
            }
        )
        st.session_state.total_chunk_seconds += transcript.duration_seconds

        stitched = Transcript(
            text=" ".join(st.session_state.chunk_texts).strip(),
            segments=st.session_state.chunk_texts.copy(),
            request_id=transcript.request_id,
            duration_seconds=st.session_state.total_chunk_seconds,
            first_package_delay_ms=transcript.first_package_delay_ms,
            last_package_delay_ms=transcript.last_package_delay_ms,
            estimated_cost_usd=st.session_state.total_chunk_seconds * MAINLAND_PRICE_USD_PER_SECOND,
        )
        add_history("自动分段", stitched)


def init_state() -> None:
    defaults = {
        "history": [],
        "current_recorder_session": "",
        "processed_chunk_ids": set(),
        "chunk_texts": [],
        "chunk_items": [],
        "total_chunk_seconds": 0.0,
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
        "上传文件按实时速度发送",
        value=st.session_state.throttle_stream,
        help="只影响上传文件；麦克风自动分段会快速识别每个 5 秒片段。",
    )
    st.caption("模型：fun-asr-realtime · Endpoint：dashscope.aliyuncs.com")

live_tab, upload_tab = st.tabs(["连续录音", "上传音频"])

with live_tab:
    st.info("点击 START 后直接说宁波话（吴语），点击 STOP 结束。应用会自动每 5 秒识别并拼接。")
    chunk_data = RECORDER_COMPONENT(key="ningbo-chunk-recorder", default=None)
    if chunk_data:
        try:
            transcribe_browser_chunk(chunk_data, api_key)
        except Exception as error:
            st.error(f"分段识别失败：{error}")

    render_stitched_transcript()
    if st.button("清除当前拼接文本"):
        st.session_state.processed_chunk_ids = set()
        st.session_state.chunk_texts = []
        st.session_state.chunk_items = []
        st.session_state.total_chunk_seconds = 0.0
        st.rerun()

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
