"""Ningbo Dialect → Mandarin - Live Transcription with VAD."""
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from google import genai
import numpy as np
import tempfile
import os
import time
import wave
import io
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading

st.set_page_config(page_title="宁波话", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .stApp { max-width: 700px; margin: 0 auto; }
    .result { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem 1.5rem; border-radius: 10px; 
        font-size: 1.2rem; margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(102,126,234,0.3);
    }
    .history { 
        background: #f5f5f5; padding: 0.8rem 1rem; border-radius: 8px;
        margin: 0.3rem 0; border-left: 3px solid #667eea;
    }
    .status { text-align: center; color: #888; font-size: 0.9rem; }
    h1 { text-align: center; margin-bottom: 0.5rem; }
    .sub { text-align: center; color: #666; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ 宁波话实时转写")
st.markdown('<p class="sub">点击开始录音，自动检测语音并转写</p>', unsafe_allow_html=True)

# API key
def get_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return os.environ.get("GEMINI_API_KEY")

api_key = get_api_key()
if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    api_key = st.text_input("🔑 API Key", type="password")
    if not api_key:
        st.stop()

@st.cache_resource
def get_client(_key):
    return genai.Client(api_key=_key)

client = get_client(api_key)

VOCAB = "阿拉=我们,侬=你,伊=他,格=这,勒海=在,呒没=没有,晓得=知道,交关=非常"

# Initialize session state for history
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []
if "processing" not in st.session_state:
    st.session_state.processing = False

def transcribe_audio(audio_bytes: bytes) -> dict:
    """Transcribe audio bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        tmp = f.name
    
    try:
        file = client.files.upload(file=tmp)
        
        def raw():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["用汉字记录发音，只输出汉字", file]
            ).text.strip()
        
        def semantic():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[f"宁波话。{VOCAB}。输出普通话，只输出结果", file]
            ).text.strip()
        
        with ThreadPoolExecutor(2) as ex:
            r, s = ex.submit(raw), ex.submit(semantic)
        
        final = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"综合宁波话转写：音素:{r.result()} 语义:{s.result()} {VOCAB} 只输出结果"]
        ).text.strip()
        
        return {"raw": r.result(), "semantic": s.result(), "final": final}
    finally:
        os.path.exists(tmp) and os.unlink(tmp)

class AudioProcessor(AudioProcessorBase):
    """Process audio with VAD and trigger transcription."""
    
    def __init__(self):
        self.audio_buffer = []
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.silence_duration = 0
        self.speech_detected = False
        self.min_speech_duration = 0.5  # seconds
        self.max_silence_after_speech = 1.0  # seconds
        self.lock = threading.Lock()
        self.pending_audio = None
    
    def recv(self, frame):
        """Receive audio frame and detect speech."""
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        
        # Simple energy-based VAD
        energy = np.sqrt(np.mean(audio ** 2))
        is_speech = energy > self.silence_threshold
        
        with self.lock:
            if is_speech:
                self.audio_buffer.append(frame.to_ndarray())
                self.speech_detected = True
                self.silence_duration = 0
            elif self.speech_detected:
                self.audio_buffer.append(frame.to_ndarray())
                self.silence_duration += len(audio) / self.sample_rate
                
                # End of speech detected
                if self.silence_duration > self.max_silence_after_speech:
                    if len(self.audio_buffer) > 0:
                        # Combine audio frames
                        combined = np.concatenate(self.audio_buffer)
                        duration = len(combined.flatten()) / self.sample_rate
                        
                        if duration >= self.min_speech_duration:
                            # Convert to WAV bytes
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, 'wb') as wav:
                                wav.setnchannels(1)
                                wav.setsampwidth(2)
                                wav.setframerate(self.sample_rate)
                                wav.writeframes(combined.tobytes())
                            self.pending_audio = wav_buffer.getvalue()
                    
                    # Reset
                    self.audio_buffer = []
                    self.speech_detected = False
                    self.silence_duration = 0
        
        return frame
    
    def get_pending_audio(self):
        """Get pending audio for transcription."""
        with self.lock:
            audio = self.pending_audio
            self.pending_audio = None
            return audio

# WebRTC streamer for live audio
ctx = webrtc_streamer(
    key="ningbo-live",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# Status indicator
status_placeholder = st.empty()
result_placeholder = st.empty()

# Process pending audio when available
if ctx.audio_processor:
    pending = ctx.audio_processor.get_pending_audio()
    if pending:
        status_placeholder.markdown('<p class="status">🔄 转写中...</p>', unsafe_allow_html=True)
        try:
            result = transcribe_audio(pending)
            # Add to history
            st.session_state.transcriptions.insert(0, {
                "text": result["final"],
                "raw": result["raw"],
                "semantic": result["semantic"],
                "time": time.strftime("%H:%M:%S")
            })
            status_placeholder.empty()
            st.rerun()
        except Exception as e:
            status_placeholder.error(f"错误: {e}")

# Show current/latest result prominently
if st.session_state.transcriptions:
    latest = st.session_state.transcriptions[0]
    result_placeholder.markdown(f'<div class="result">{latest["text"]}</div>', unsafe_allow_html=True)

# Show history
if len(st.session_state.transcriptions) > 1:
    with st.expander(f"📜 历史记录 ({len(st.session_state.transcriptions)})", expanded=False):
        for i, t in enumerate(st.session_state.transcriptions[1:], 1):
            st.markdown(f'<div class="history"><small>{t["time"]}</small> {t["text"]}</div>', unsafe_allow_html=True)
        
        if st.button("清除历史"):
            st.session_state.transcriptions = []
            st.rerun()

# Fallback: file upload
with st.expander("📁 上传文件", expanded=False):
    uploaded = st.file_uploader("文件", type=["mp3","wav","m4a","webm"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("转写中..."):
            try:
                result = transcribe_audio(uploaded.getvalue())
                st.session_state.transcriptions.insert(0, {
                    "text": result["final"],
                    "raw": result["raw"],
                    "semantic": result["semantic"],
                    "time": time.strftime("%H:%M:%S")
                })
                st.rerun()
            except Exception as e:
                st.error(f"错误: {e}")

# Details expander
if st.session_state.transcriptions:
    with st.expander("详细分析"):
        latest = st.session_state.transcriptions[0]
        st.caption("音素")
        st.code(latest.get("raw", ""))
        st.caption("语义")
        st.code(latest.get("semantic", ""))
