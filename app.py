"""Ningbo Dialect to Mandarin Transcription App using Gemini AI."""
import streamlit as st
from google import genai
import tempfile
import os

# Page config
st.set_page_config(page_title="宁波话转普通话", page_icon="🎙️", layout="centered")

# Minimal CSS styling
st.markdown("""
<style>
    .stApp { max-width: 700px; margin: 0 auto; }
    .result-box { 
        background: #f8f9fa; 
        padding: 1.5rem; 
        border-radius: 8px; 
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .method-label { color: #666; font-size: 0.85rem; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎙️ 宁波话转普通话")
st.caption("Ningbo Dialect → Mandarin Chinese Transcription")

# Get API key from secrets or environment
def get_api_key():
    """Get API key from Streamlit secrets, env var, or return None."""
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return os.environ.get("GEMINI_API_KEY")

api_key = get_api_key()

# Show input if no API key configured
if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    api_key = st.text_input("🔑 Gemini API Key", type="password", 
                            help="从 https://aistudio.google.com/apikey 获取")
    if not api_key:
        st.info("请输入 Gemini API Key 以继续")
        st.stop()

# Initialize Gemini client
@st.cache_resource
def get_client(_api_key: str):
    return genai.Client(api_key=_api_key)

client = get_client(api_key)

# Audio upload
audio_file = st.file_uploader(
    "上传宁波话音频 (Upload Ningbo dialect audio)", 
    type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
    help="支持 MP3, WAV, M4A, OGG, FLAC, WebM 格式"
)

def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    """Two-method transcription with final Gemini Flash evaluation."""
    
    # Save temp file for upload
    suffix = os.path.splitext(filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # Upload audio to Gemini
        uploaded_file = client.files.upload(file=tmp_path)
        
        # === METHOD 1: Direct audio transcription ===
        prompt_method1 = """
你是一个专业的宁波话（吴语宁波方言）语音转写专家。请仔细听这段音频，它是宁波话/宁波方言。

任务：
1. 仔细聆听音频中的宁波话发音
2. 识别宁波话特有的语音特征（如浊音、入声等）
3. 将听到的内容用普通话文字准确转写出来
4. 注意宁波话和普通话之间的词汇差异

请直接输出转写的普通话文字，不需要任何解释。
"""
        response1 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_method1, uploaded_file]
        )
        result_method1 = response1.text.strip()
        
        # === METHOD 2: Mandarin transcription + dialect interpretation ===
        prompt_method2a = """
请将这段音频转写成文字。直接输出你听到的内容，尽量准确地记录发音对应的汉字。
不需要任何解释，只输出转写文字。
"""
        response2a = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_method2a, uploaded_file]
        )
        raw_transcription = response2a.text.strip()
        
        # Use Gemini Flash to interpret the raw transcription as Ningbo dialect
        prompt_method2b = f"""
你是一个精通宁波话（吴语宁波方言）的语言专家。

以下文字是一段宁波话语音的初步转写：
「{raw_transcription}」

这段话是宁波人用宁波方言说的，可能包含：
- 宁波话特有词汇（如"阿拉"=我们，"侬"=你，"格"=这，"勒"=在）
- 发音导致的误听（宁波话浊音较多）
- 语法结构差异

请分析并推测说话人的实际意思，用标准普通话表达出来。
只输出最终的普通话翻译，不需要解释过程。
"""
        response2b = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_method2b]
        )
        result_method2 = response2b.text.strip()
        
        # === FINAL: Combine both methods with Gemini Flash 2.5 ===
        prompt_final = f"""
你是一个资深的宁波话（吴语宁波方言）转普通话专家，现在需要综合分析两种转写结果。

原始音频是宁波话/宁波方言，来自浙江宁波地区的吴语方言。

【方法一的转写结果】：
{result_method1}

【方法二的转写结果】（先转写后解读）：
{result_method2}

请综合考虑：
1. 宁波话的语音特点（浊音声母、入声、特殊韵母）
2. 宁波话的常用词汇和表达习惯
3. 两种方法结果的共同点和差异
4. 整体语义的连贯性和合理性

基于以上分析，请给出你认为最准确的普通话转写结果。
只输出最终结果，不需要解释。
"""
        response_final = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_final]
        )
        final_result = response_final.text.strip()
        
        return {
            "method1": result_method1,
            "method2": result_method2,
            "final": final_result,
            "raw_transcription": raw_transcription
        }
        
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Process button
if audio_file:
    if st.button("开始转写 (Transcribe)", type="primary", use_container_width=True):
        with st.spinner("🔄 正在分析音频..."):
            try:
                results = transcribe_audio(audio_file.getvalue(), audio_file.name)
                
                # Display final result prominently
                st.markdown("### 📝 转写结果")
                st.markdown(f'<div class="result-box"><b>{results["final"]}</b></div>', 
                           unsafe_allow_html=True)
                
                # Show intermediate results in expander
                with st.expander("查看详细分析过程"):
                    st.markdown('<p class="method-label">方法一：直接转写</p>', 
                               unsafe_allow_html=True)
                    st.info(results["method1"])
                    
                    st.markdown('<p class="method-label">方法二：语义解读</p>', 
                               unsafe_allow_html=True)
                    st.info(results["method2"])
                    
                    st.markdown('<p class="method-label">原始音频转写</p>', 
                               unsafe_allow_html=True)
                    st.code(results["raw_transcription"])
                    
            except Exception as e:
                st.error(f"转写失败: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by Gemini 2.5 Flash • 专为宁波话设计")

