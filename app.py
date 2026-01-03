"""Ningbo Dialect to Mandarin - Fast Live Transcription."""
import streamlit as st
from google import genai
import tempfile
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="宁波话转普通话", page_icon="🎙️", layout="centered")

# Custom CSS for big mic button and result buckets
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    
    /* Big microphone styling */
    [data-testid="stAudioInput"] > div { 
        min-height: 120px !important;
    }
    [data-testid="stAudioInput"] button {
        width: 100px !important;
        height: 100px !important;
        border-radius: 50% !important;
        font-size: 2rem !important;
    }
    
    /* Result buckets */
    .bucket { 
        background: #f8f9fa; 
        padding: 1rem; 
        border-radius: 8px; 
        margin: 0.5rem 0;
        border-left: 4px solid #ccc;
    }
    .bucket-primary { 
        background: #e8f4f8; 
        border-left-color: #1f77b4;
        font-size: 1.2rem;
        font-weight: 500;
    }
    .bucket-raw { border-left-color: #ff7f0e; }
    .bucket-semantic { border-left-color: #2ca02c; }
    .bucket-final { border-left-color: #9467bd; }
    .bucket-label { 
        font-size: 0.75rem; 
        color: #666; 
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ 宁波话转普通话")

# API key
def get_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except:
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

# Ningbo vocabulary reference
NINGBO_VOCAB = """【宁波话词汇】
代词: 阿拉=我们, 侬=你, 伊=他她, 倷=你们
指示: 格/搿=这, 噶=那, 搿个=这个
疑问: 啥=什么, 哪能=怎么, 阿是=是不是
动词: 勒海=在, 呒没=没有, 晓得=知道, 困觉=睡觉
形容词: 交关/邪气=非常, 老好=很好, 结棍=厉害
语气: 嘎=了, 个=的, 来=呢
【语音】浊音保留(婆bo头dou), 入声短促(黑hek白bak)"""

# Audio input - Recording is primary
st.markdown("### 🎤 点击录音")
audio_data = st.audio_input("", label_visibility="collapsed")

# Secondary: file upload in expander
with st.expander("📁 或上传音频文件"):
    uploaded = st.file_uploader("", type=["mp3","wav","m4a","webm"], label_visibility="collapsed")

# Determine source
audio_source = None
audio_name = "recording.wav"
if audio_data:
    audio_source = audio_data.getvalue()
elif uploaded:
    audio_source = uploaded.getvalue()
    audio_name = uploaded.name

def fast_transcribe(audio_bytes: bytes, filename: str):
    """Fast parallel transcription with progressive results."""
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    
    try:
        uploaded_file = client.files.upload(file=tmp_path)
        
        # Three parallel calls for speed
        def raw_transcribe():
            """Fastest - just record what you hear."""
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["仔细听，用汉字记录你听到的每个音节发音。只输出汉字，不解释。", uploaded_file]
            ).text.strip()
        
        def semantic_transcribe():
            """Direct semantic understanding."""
            prompt = f"这是宁波方言音频。{NINGBO_VOCAB}\n直接输出普通话意思，不解释。"
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded_file]
            ).text.strip()
        
        def context_transcribe():
            """Contextual interpretation."""
            prompt = f"""这是宁波话音频。{NINGBO_VOCAB}
作为宁波话专家，仔细听并理解说话人的意图。输出自然的普通话翻译。只输出结果。"""
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded_file]
            ).text.strip()
        
        # Run all three in parallel
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_raw = ex.submit(raw_transcribe)
            f_semantic = ex.submit(semantic_transcribe)
            f_context = ex.submit(context_transcribe)
        
        return {
            "raw": f_raw.result(),
            "semantic": f_semantic.result(),
            "context": f_context.result()
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def synthesize_final(results: dict) -> str:
    """Quick synthesis of all results."""
    prompt = f"""综合以下三种宁波话转写结果，给出最准确的普通话翻译：
【音素】{results['raw']}
【语义】{results['semantic']}  
【语境】{results['context']}
{NINGBO_VOCAB}
只输出最终翻译，不解释。"""
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    ).text.strip()

# Process
if audio_source:
    if st.button("⚡ 快速转写", type="primary", use_container_width=True):
        # Create placeholders for progressive display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="bucket-label">🔊 音素记录</div>', unsafe_allow_html=True)
            raw_placeholder = st.empty()
            raw_placeholder.info("分析中...")
        with col2:
            st.markdown('<div class="bucket-label">💬 语义理解</div>', unsafe_allow_html=True)
            sem_placeholder = st.empty()
            sem_placeholder.info("分析中...")
        with col3:
            st.markdown('<div class="bucket-label">🎯 语境推断</div>', unsafe_allow_html=True)
            ctx_placeholder = st.empty()
            ctx_placeholder.info("分析中...")
        
        final_placeholder = st.empty()
        final_placeholder.info("⏳ 正在综合分析...")
        
        try:
            # Get parallel results
            results = fast_transcribe(audio_source, audio_name)
            
            # Update buckets immediately
            raw_placeholder.markdown(f'<div class="bucket bucket-raw">{results["raw"]}</div>', unsafe_allow_html=True)
            sem_placeholder.markdown(f'<div class="bucket bucket-semantic">{results["semantic"]}</div>', unsafe_allow_html=True)
            ctx_placeholder.markdown(f'<div class="bucket bucket-context">{results["context"]}</div>', unsafe_allow_html=True)
            
            # Final synthesis
            final = synthesize_final(results)
            final_placeholder.markdown(f"""
            <div style="margin-top:1rem;">
                <div class="bucket-label">✨ 最终结果</div>
                <div class="bucket bucket-primary">{final}</div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"错误: {e}")

st.markdown("---")
st.caption("Gemini 2.5 Flash · 并行处理 · 宁波话")
