"""Ningbo Dialect → Mandarin - Two-Pass Real-Time Transcription."""
import streamlit as st
from google import genai
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="宁波话转录", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .stApp { max-width: 700px; margin: 0 auto; background: #111827; }
    [data-testid="stAudioInput"] { display: flex; justify-content: center; }
    [data-testid="stAudioInput"] > div { min-height: 100px !important; }
    [data-testid="stAudioInput"] button {
        width: 90px !important; height: 90px !important;
        border-radius: 50% !important; font-size: 1.6rem !important;
        background: #3b82f6 !important; border: none !important;
        transition: all 0.2s !important;
    }
    [data-testid="stAudioInput"] button:hover {
        background: #2563eb !important; transform: scale(1.05);
    }
    
    h1 { text-align: center; color: white; margin-bottom: 0.3rem; }
    .sub { text-align: center; color: #9ca3af; margin-bottom: 1.5rem; }
    
    /* First pass - quick result */
    .quick { 
        background: #1e3a5f; color: #93c5fd; 
        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        border-left: 4px solid #3b82f6; font-size: 1.1rem;
    }
    .quick-label { color: #60a5fa; font-size: 0.75rem; margin-bottom: 0.3rem; }
    
    /* Final result - refined */
    .final { 
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white; padding: 1.2rem; border-radius: 12px; 
        font-size: 1.3rem; text-align: center; margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }
    .final-label { color: #34d399; font-size: 0.75rem; margin-bottom: 0.3rem; text-align: center; }
    
    /* Status */
    .status { text-align: center; padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0; }
    .status-listening { background: #fef3c7; color: #92400e; }
    .status-processing { background: #dbeafe; color: #1e40af; }
    .status-refining { background: #d1fae5; color: #065f46; }
    
    /* History */
    .history { 
        background: #1f2937; padding: 0.8rem 1rem; border-radius: 8px;
        margin: 0.3rem 0; border-left: 3px solid #4b5563; color: #d1d5db;
    }
    .time { color: #6b7280; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ 宁波话转录</h1>", unsafe_allow_html=True)
st.markdown('<p class="sub">Ningbo Dialect Transcriber · 两步转写</p>', unsafe_allow_html=True)

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

VOCAB = "阿拉=我们,侬=你,伊=他她,格=这,勒海=在,呒没=没有,晓得=知道,交关/邪气=非常,老=很"

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "current_quick" not in st.session_state:
    st.session_state.current_quick = None
if "current_final" not in st.session_state:
    st.session_state.current_final = None

def quick_transcribe(file) -> str:
    """First pass: FAST transcription for immediate understanding."""
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[f"宁波话音频，快速输出大概意思，用简短普通话。{VOCAB}", file]
    ).text.strip()

def refined_transcribe(file) -> dict:
    """Second pass: Accurate parallel transcription."""
    def raw():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["仔细听宁波话发音，用汉字记录，只输出汉字", file]
        ).text.strip()
    
    def semantic():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"宁波话音频。词汇:{VOCAB}。准确翻译成普通话，只输出结果", file]
        ).text.strip()
    
    with ThreadPoolExecutor(2) as ex:
        r, s = ex.submit(raw), ex.submit(semantic)
    
    # Final synthesis
    final = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[f"综合两种宁波话转写，给出最准确的普通话翻译：\n音素记录:{r.result()}\n语义理解:{s.result()}\n参考词汇:{VOCAB}\n只输出最终翻译，不解释"]
    ).text.strip()
    
    return {"raw": r.result(), "semantic": s.result(), "final": final}

def two_pass_transcribe(audio_bytes: bytes):
    """Two-pass transcription: quick first, refined second."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        tmp = f.name
    
    try:
        file = client.files.upload(file=tmp)
        
        # PASS 1: Quick (show immediately)
        quick = quick_transcribe(file)
        st.session_state.current_quick = quick
        
        # PASS 2: Refined (more accurate)
        refined = refined_transcribe(file)
        st.session_state.current_final = refined["final"]
        
        # Add to history
        st.session_state.history.insert(0, {
            "quick": quick,
            "final": refined["final"],
            "raw": refined["raw"],
            "semantic": refined["semantic"],
            "time": time.strftime("%H:%M:%S")
        })
        
        return quick, refined
        
    finally:
        os.path.exists(tmp) and os.unlink(tmp)

# Audio input
audio = st.audio_input("录音", label_visibility="collapsed")

# Placeholders for real-time display
status_ph = st.empty()
quick_ph = st.empty()
final_ph = st.empty()

# Auto-transcribe when new audio
if audio:
    audio_id = id(audio)
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        st.session_state.current_quick = None
        st.session_state.current_final = None
        
        # Show processing status
        status_ph.markdown('<div class="status status-processing">⚡ 快速识别中...</div>', unsafe_allow_html=True)
        
        try:
            # Two-pass transcription
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio.getvalue())
                tmp = f.name
            
            file = client.files.upload(file=tmp)
            
            # PASS 1: Quick result
            quick = quick_transcribe(file)
            st.session_state.current_quick = quick
            quick_ph.markdown(f'''
                <div class="quick-label">⚡ 快速识别</div>
                <div class="quick">{quick}</div>
            ''', unsafe_allow_html=True)
            
            # Update status
            status_ph.markdown('<div class="status status-refining">🔄 精确校正中...</div>', unsafe_allow_html=True)
            
            # PASS 2: Refined result
            refined = refined_transcribe(file)
            st.session_state.current_final = refined["final"]
            
            # Clear status, show final
            status_ph.empty()
            final_ph.markdown(f'''
                <div class="final-label">✅ 精确翻译</div>
                <div class="final">{refined["final"]}</div>
            ''', unsafe_allow_html=True)
            
            # Add to history
            st.session_state.history.insert(0, {
                "quick": quick,
                "final": refined["final"],
                "raw": refined["raw"],
                "semantic": refined["semantic"],
                "time": time.strftime("%H:%M:%S")
            })
            
            # Cleanup
            os.path.exists(tmp) and os.unlink(tmp)
            
        except Exception as e:
            status_ph.error(f"错误: {e}")

# Show current results if available (after rerun)
elif st.session_state.current_quick:
    quick_ph.markdown(f'''
        <div class="quick-label">⚡ 快速识别</div>
        <div class="quick">{st.session_state.current_quick}</div>
    ''', unsafe_allow_html=True)
    
    if st.session_state.current_final:
        final_ph.markdown(f'''
            <div class="final-label">✅ 精确翻译</div>
            <div class="final">{st.session_state.current_final}</div>
        ''', unsafe_allow_html=True)

# Details expander
if st.session_state.history:
    with st.expander("📊 详细分析"):
        latest = st.session_state.history[0]
        col1, col2 = st.columns(2)
        with col1:
            st.caption("音素记录")
            st.code(latest.get("raw", ""))
        with col2:
            st.caption("语义理解")
            st.code(latest.get("semantic", ""))

# History
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.caption(f"📜 历史记录 ({len(st.session_state.history)})")
    for item in st.session_state.history[1:8]:
        st.markdown(f'''
            <div class="history">
                <span class="time">{item["time"]}</span> 
                <strong>{item["final"]}</strong>
                <br><small style="color:#6b7280">快速: {item["quick"]}</small>
            </div>
        ''', unsafe_allow_html=True)
    
    if st.button("清除历史"):
        st.session_state.history = []
        st.session_state.last_audio_id = None
        st.session_state.current_quick = None
        st.session_state.current_final = None
        st.rerun()

# File upload
with st.expander("📁 上传文件"):
    uploaded = st.file_uploader("文件", type=["mp3","wav","m4a","webm"], label_visibility="collapsed")
    if uploaded:
        st.session_state.last_audio_id = None  # Reset to trigger processing
        # Process same as audio input
        with st.spinner("处理中..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(uploaded.getvalue())
                    tmp = f.name
                
                file = client.files.upload(file=tmp)
                quick = quick_transcribe(file)
                refined = refined_transcribe(file)
                
                st.session_state.current_quick = quick
                st.session_state.current_final = refined["final"]
                st.session_state.history.insert(0, {
                    "quick": quick,
                    "final": refined["final"],
                    "raw": refined["raw"],
                    "semantic": refined["semantic"],
                    "time": time.strftime("%H:%M:%S")
                })
                
                os.path.exists(tmp) and os.unlink(tmp)
                st.rerun()
            except Exception as e:
                st.error(f"错误: {e}")

st.markdown("---")
st.markdown('<p style="text-align:center;color:#6b7280;font-size:0.8rem;">Powered by Gemini AI · 两步转写系统</p>', unsafe_allow_html=True)
