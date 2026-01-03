"""Ningbo Dialect → Mandarin - Auto Transcription."""
import streamlit as st
from google import genai
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="宁波话", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .stApp { max-width: 600px; margin: 0 auto; padding-top: 2rem; }
    [data-testid="stAudioInput"] { display: flex; justify-content: center; }
    [data-testid="stAudioInput"] > div { min-height: 140px !important; }
    [data-testid="stAudioInput"] button {
        width: 120px !important; height: 120px !important;
        border-radius: 50% !important; font-size: 2rem !important;
    }
    .result { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1.5rem; border-radius: 12px; 
        font-size: 1.4rem; text-align: center; margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .sub { color: #888; font-size: 0.9rem; text-align: center; margin: 0.5rem 0; }
    h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ 宁波话转普通话")
st.markdown('<p class="sub">点击麦克风录音，自动转写</p>', unsafe_allow_html=True)

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

# Main mic input
audio = st.audio_input("录音", label_visibility="collapsed")

# File upload as fallback (hidden in expander)
with st.expander("📁 上传文件", expanded=False):
    uploaded = st.file_uploader("文件", type=["mp3","wav","m4a","webm"], label_visibility="collapsed")

def api_call(func, retries=2):
    for i in range(retries):
        try:
            return func()
        except Exception:
            if i < retries - 1:
                time.sleep(0.5)
            else:
                raise

def transcribe(audio_bytes: bytes, filename: str = "audio.wav"):
    """Fast transcription with parallel calls."""
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp = f.name
    
    try:
        file = client.files.upload(file=tmp)
        
        def raw():
            return api_call(lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["用汉字记录听到的发音，只输出汉字", file]
            ).text.strip())
        
        def semantic():
            return api_call(lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[f"宁波话音频。{VOCAB}。输出普通话意思，只输出结果", file]
            ).text.strip())
        
        with ThreadPoolExecutor(2) as ex:
            r = ex.submit(raw)
            s = ex.submit(semantic)
        
        # Quick synthesis
        combined = api_call(lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"综合两种宁波话转写给出准确普通话：\n音素:{r.result()}\n语义:{s.result()}\n{VOCAB}\n只输出结果"]
        ).text.strip())
        
        return {"raw": r.result(), "semantic": s.result(), "final": combined}
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

# Auto-transcribe when audio is available
audio_source = audio.getvalue() if audio else (uploaded.getvalue() if uploaded else None)
audio_name = uploaded.name if uploaded and not audio else "recording.wav"

if audio_source:
    with st.spinner(""):
        try:
            result = transcribe(audio_source, audio_name)
            st.markdown(f'<div class="result">{result["final"]}</div>', unsafe_allow_html=True)
            
            with st.expander("详细", expanded=False):
                st.caption("音素")
                st.code(result["raw"])
                st.caption("语义")
                st.code(result["semantic"])
        except Exception as e:
            st.error(f"错误: {e}")
