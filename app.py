"""Ningbo Dialect → Mandarin - Auto Transcription with History."""
import streamlit as st
from google import genai
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="宁波话", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .stApp { max-width: 650px; margin: 0 auto; }
    [data-testid="stAudioInput"] { display: flex; justify-content: center; }
    [data-testid="stAudioInput"] > div { min-height: 120px !important; }
    [data-testid="stAudioInput"] button {
        width: 100px !important; height: 100px !important;
        border-radius: 50% !important; font-size: 1.8rem !important;
    }
    .result { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1.2rem 1.5rem; border-radius: 12px; 
        font-size: 1.3rem; text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .history { 
        background: #f8f9fa; padding: 0.7rem 1rem; border-radius: 8px;
        margin: 0.3rem 0; border-left: 3px solid #667eea; font-size: 0.95rem;
    }
    .time { color: #999; font-size: 0.75rem; margin-right: 0.5rem; }
    h1 { text-align: center; }
    .sub { text-align: center; color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ 宁波话转普通话")
st.markdown('<p class="sub">录音后自动转写 · 保留历史记录</p>', unsafe_allow_html=True)

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

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

def transcribe(audio_bytes: bytes) -> dict:
    """Fast parallel transcription."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        tmp = f.name
    
    try:
        file = client.files.upload(file=tmp)
        
        def raw():
            for _ in range(2):
                try:
                    return client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=["用汉字记录发音，只输出汉字", file]
                    ).text.strip()
                except:
                    time.sleep(0.5)
            raise Exception("API error")
        
        def semantic():
            for _ in range(2):
                try:
                    return client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[f"宁波话。{VOCAB}。输出普通话，只输出结果", file]
                    ).text.strip()
                except:
                    time.sleep(0.5)
            raise Exception("API error")
        
        with ThreadPoolExecutor(2) as ex:
            r, s = ex.submit(raw), ex.submit(semantic)
        
        final = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"综合宁波话转写：音素:{r.result()} 语义:{s.result()} {VOCAB} 只输出结果"]
        ).text.strip()
        
        return {"raw": r.result(), "semantic": s.result(), "final": final}
    finally:
        os.path.exists(tmp) and os.unlink(tmp)

# Audio input
audio = st.audio_input("录音", label_visibility="collapsed")

# Auto-transcribe when new audio is recorded
if audio:
    audio_id = id(audio)
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        
        with st.spinner("转写中..."):
            try:
                result = transcribe(audio.getvalue())
                # Add to history
                st.session_state.history.insert(0, {
                    "text": result["final"],
                    "raw": result["raw"],
                    "semantic": result["semantic"],
                    "time": time.strftime("%H:%M:%S")
                })
            except Exception as e:
                st.error(f"错误: {e}")

# Show latest result
if st.session_state.history:
    latest = st.session_state.history[0]
    st.markdown(f'<div class="result">{latest["text"]}</div>', unsafe_allow_html=True)
    
    # Details
    with st.expander("详细"):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("音素")
            st.code(latest["raw"])
        with col2:
            st.caption("语义")
            st.code(latest["semantic"])

# History
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.caption(f"📜 历史 ({len(st.session_state.history)})")
    for item in st.session_state.history[1:6]:  # Show last 5
        st.markdown(f'<div class="history"><span class="time">{item["time"]}</span>{item["text"]}</div>', unsafe_allow_html=True)
    
    if len(st.session_state.history) > 6:
        st.caption(f"... 还有 {len(st.session_state.history) - 6} 条")
    
    if st.button("清除历史", type="secondary"):
        st.session_state.history = []
        st.session_state.last_audio_id = None
        st.rerun()

# File upload fallback
with st.expander("📁 上传文件"):
    uploaded = st.file_uploader("文件", type=["mp3","wav","m4a","webm"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("转写中..."):
            try:
                result = transcribe(uploaded.getvalue())
                st.session_state.history.insert(0, {
                    "text": result["final"],
                    "raw": result["raw"],
                    "semantic": result["semantic"],
                    "time": time.strftime("%H:%M:%S")
                })
                st.rerun()
            except Exception as e:
                st.error(f"错误: {e}")
