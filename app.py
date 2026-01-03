"""Ningbo Dialect to Mandarin - Fast Transcription."""
import streamlit as st
from google import genai
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="宁波话转普通话", page_icon="🎙️", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    [data-testid="stAudioInput"] > div { min-height: 100px !important; }
    [data-testid="stAudioInput"] button {
        width: 80px !important; height: 80px !important;
        border-radius: 50% !important; font-size: 1.5rem !important;
    }
    .bucket { background: #f8f9fa; padding: 1rem; border-radius: 8px; 
              margin: 0.5rem 0; border-left: 4px solid #ccc; }
    .bucket-primary { background: #e8f4f8; border-left-color: #1f77b4;
                      font-size: 1.2rem; font-weight: 500; }
    .bucket-raw { border-left-color: #ff7f0e; }
    .bucket-semantic { border-left-color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ 宁波话转普通话")

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

# Ningbo vocabulary
NINGBO_VOCAB = """【宁波话】阿拉=我们,侬=你,伊=他,格=这,噶=那,勒海=在,呒没=没有,晓得=知道,交关=非常,邪气=很"""

# Audio input with proper labels
st.markdown("### 🎤 点击录音")
audio_data = st.audio_input("录音", label_visibility="collapsed")

with st.expander("📁 或上传文件"):
    uploaded = st.file_uploader("上传音频", type=["mp3","wav","m4a","webm"], label_visibility="collapsed")

# Determine source
audio_source = None
audio_name = "recording.wav"
if audio_data:
    audio_source = audio_data.getvalue()
elif uploaded:
    audio_source = uploaded.getvalue()
    audio_name = uploaded.name

def call_api_with_retry(func, max_retries=2):
    """Call API with retry on failure."""
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(1)  # Brief delay before retry
            else:
                raise e

def transcribe(audio_bytes: bytes, filename: str):
    """Two-step transcription: parallel first pass, then synthesis."""
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    
    try:
        uploaded_file = client.files.upload(file=tmp_path)
        
        # Two parallel calls (reduced from 3 to avoid rate limits)
        def raw_call():
            return call_api_with_retry(lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["仔细听，用汉字记录你听到的发音。只输出汉字。", uploaded_file]
            ).text.strip())
        
        def semantic_call():
            prompt = f"这是宁波方言。{NINGBO_VOCAB}\n输出普通话意思，只输出结果。"
            return call_api_with_retry(lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded_file]
            ).text.strip())
        
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_raw = ex.submit(raw_call)
            f_semantic = ex.submit(semantic_call)
        
        return {"raw": f_raw.result(), "semantic": f_semantic.result()}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def synthesize(results: dict) -> str:
    """Final synthesis."""
    prompt = f"""综合两种宁波话转写，给出最准确的普通话：
【音素】{results['raw']}
【语义】{results['semantic']}
{NINGBO_VOCAB}
只输出结果。"""
    return call_api_with_retry(lambda: client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    ).text.strip())

# Process
if audio_source:
    if st.button("⚡ 转写", type="primary", use_container_width=True):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("🔊 音素")
            raw_ph = st.empty()
            raw_ph.info("...")
        with col2:
            st.caption("💬 语义")
            sem_ph = st.empty()
            sem_ph.info("...")
        
        final_ph = st.empty()
        
        try:
            results = transcribe(audio_source, audio_name)
            raw_ph.markdown(f'<div class="bucket bucket-raw">{results["raw"]}</div>', unsafe_allow_html=True)
            sem_ph.markdown(f'<div class="bucket bucket-semantic">{results["semantic"]}</div>', unsafe_allow_html=True)
            
            final = synthesize(results)
            final_ph.markdown(f'<div class="bucket bucket-primary">✨ {final}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"错误: {e}")

st.markdown("---")
st.caption("Gemini 2.5 Flash")
