"""Ningbo dialect to Mandarin text with Alibaba Fun-ASR realtime."""
import html
import io
import os
import queue
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
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from streamlit_webrtc import WebRtcMode, webrtc_streamer


MODEL_NAME = "fun-asr-realtime"
MAINLAND_WEBSOCKET_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE_BYTES = 3200
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAINLAND_PRICE_USD_PER_SECOND = 0.000047
SUPPORTED_UPLOAD_TYPES = ["wav", "mp3", "aac", "amr", "opus", "speex"]
DEFAULT_ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
    {"urls": ["stun:global.stun.twilio.com:3478"]},
]


def patch_streamlit_webrtc_shutdown() -> None:
    try:
        from streamlit_webrtc import shutdown
    except Exception:
        return

    observer_class = getattr(shutdown, "SessionShutdownObserver", None)
    if not observer_class or getattr(observer_class, "_ningbo_patch_applied", False):
        return

    original_stop = observer_class.stop

    def safe_stop(self: Any) -> Any:
        polling_thread = getattr(self, "_polling_thread", None)
        if polling_thread is None:
            return None
        return original_stop(self)

    observer_class.stop = safe_stop
    observer_class._ningbo_patch_applied = True


patch_streamlit_webrtc_shutdown()


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


def split_secret_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_rtc_configuration() -> dict[str, Any]:
    ice_servers = DEFAULT_ICE_SERVERS.copy()

    turn_urls = split_secret_list(get_secret("TURN_URLS"))
    turn_username = get_secret("TURN_USERNAME")
    turn_credential = get_secret("TURN_CREDENTIAL")
    if turn_urls and turn_username and turn_credential:
        ice_servers.append(
            {
                "urls": turn_urls,
                "username": turn_username,
                "credential": turn_credential,
            }
        )

    return {"iceServers": ice_servers}


def has_turn_configuration() -> bool:
    return bool(
        split_secret_list(get_secret("TURN_URLS"))
        and get_secret("TURN_USERNAME")
        and get_secret("TURN_CREDENTIAL")
    )


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


def audio_frame_to_pcm16(frame: Any) -> tuple[bytes, float]:
    samples = np.asarray(frame.to_ndarray())
    sample_rate = int(getattr(frame, "sample_rate", TARGET_SAMPLE_RATE) or TARGET_SAMPLE_RATE)

    if samples.ndim == 2:
        if samples.shape[0] <= 8:
            samples = samples.mean(axis=0)
        elif samples.shape[1] <= 8:
            samples = samples.mean(axis=1)
        else:
            samples = samples.reshape(-1)
    else:
        samples = samples.reshape(-1)

    if samples.size == 0:
        return b"", 0.0

    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        max_value = float(max(abs(info.min), info.max))
        samples = samples.astype(np.float32) / max_value
    else:
        samples = samples.astype(np.float32)

    duration_seconds = samples.size / sample_rate
    if sample_rate != TARGET_SAMPLE_RATE:
        target_count = max(1, int(round(duration_seconds * TARGET_SAMPLE_RATE)))
        source_positions = np.linspace(0.0, duration_seconds, num=samples.size, endpoint=False)
        target_positions = np.linspace(0.0, duration_seconds, num=target_count, endpoint=False)
        samples = np.interp(target_positions, source_positions, samples).astype(np.float32)

    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype("<i2").tobytes()
    return pcm16, duration_seconds


def create_recognition(api_key: str, callback: FunAsrCallback) -> Recognition:
    dashscope.api_key = api_key
    dashscope.base_websocket_api_url = MAINLAND_WEBSOCKET_URL
    return Recognition(
        model=MODEL_NAME,
        format="pcm",
        sample_rate=TARGET_SAMPLE_RATE,
        language_hints=["zh"],
        semantic_punctuation_enabled=st.session_state.semantic_punctuation_enabled,
        max_sentence_silence=st.session_state.max_sentence_silence,
        heartbeat=True,
        callback=callback,
    )


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


def render_live_card(placeholder: Any, text: str, elapsed_seconds: float, is_active: bool) -> None:
    status = "正在实时识别" if is_active else "已停止"
    visible_text = text or "请点击 START，允许麦克风权限，然后直接说宁波话。"
    placeholder.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{status} · {elapsed_seconds:.1f}s</div>
            <div class="result-text">{html.escape(visible_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stream_microphone_to_fun_asr(ctx: Any, api_key: str) -> Transcript | None:
    audio_receiver = ctx.audio_receiver
    if not audio_receiver:
        return None

    callback = FunAsrCallback()
    recognition = create_recognition(api_key, callback)
    live_placeholder = st.empty()
    metric_placeholder = st.empty()
    segment_placeholder = st.empty()
    started_at = time.monotonic()
    streamed_seconds = 0.0

    # SECURITY-REVIEW: Microphone frames are streamed to Alibaba ASR and not persisted locally.
    recognition.start()
    render_live_card(live_placeholder, "", 0.0, True)

    try:
        while ctx.state.playing:
            try:
                frames = audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                frames = []

            for frame in frames:
                pcm16, frame_seconds = audio_frame_to_pcm16(frame)
                if not pcm16:
                    continue
                recognition.send_audio_frame(pcm16)
                streamed_seconds += frame_seconds

            error_message = callback.error()
            if error_message:
                raise RuntimeError(error_message)

            elapsed_seconds = time.monotonic() - started_at
            text = callback.final_text()
            render_live_card(live_placeholder, text, elapsed_seconds, True)
            metric_placeholder.caption(
                f"已发送 {streamed_seconds:.1f}s 音频 · "
                f"预估费用 ${streamed_seconds * MAINLAND_PRICE_USD_PER_SECOND:.6f}"
            )

            segments = callback.segment_texts()
            if segments:
                segment_placeholder.write(" / ".join(segments[-3:]))

            time.sleep(0.03)
    finally:
        recognition.stop()

    text = callback.final_text()
    if not text:
        render_live_card(live_placeholder, "", streamed_seconds, False)
        return None

    transcript = Transcript(
        text=text,
        segments=callback.segment_texts(),
        request_id=recognition.get_last_request_id(),
        duration_seconds=streamed_seconds,
        first_package_delay_ms=recognition.get_first_package_delay(),
        last_package_delay_ms=recognition.get_last_package_delay(),
        estimated_cost_usd=streamed_seconds * MAINLAND_PRICE_USD_PER_SECOND,
    )
    render_live_card(live_placeholder, transcript.text, streamed_seconds, False)
    return transcript


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
        "上传文件按实时速度发送",
        value=st.session_state.throttle_stream,
        help="只影响上传文件；实时麦克风会按真实语速发送。",
    )
    st.caption("模型：fun-asr-realtime · Endpoint：dashscope.aliyuncs.com")
    if has_turn_configuration():
        st.success("TURN relay configured for microphone connection.")
    else:
        st.warning("No TURN relay configured. Some networks may not connect with STUN only.")

live_tab, upload_tab = st.tabs(["实时麦克风", "上传音频"])

with live_tab:
    st.info("点击 START 并允许麦克风权限后，直接说宁波话（吴语）。文字会自动实时更新。")
    if not has_turn_configuration():
        st.caption("如果一直显示连接中，请在 Streamlit Secrets 添加 TURN relay 设置。上传音频仍可作为备用。")
    ctx = webrtc_streamer(
        key="ningbo-live-microphone",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=1024,
        rtc_configuration=get_rtc_configuration(),
    )

    if ctx.state.playing and ctx.audio_receiver:
        try:
            transcript = stream_microphone_to_fun_asr(ctx, api_key)
            if transcript:
                add_history("实时麦克风", transcript)
                render_transcript(transcript)
        except Exception as error:
            st.error(f"实时识别失败：{error}")

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
