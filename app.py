"""Ningbo Dialect to Mandarin Transcription App using Gemini AI."""
import streamlit as st
from google import genai
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

# Page config
st.set_page_config(page_title="宁波话转普通话", page_icon="🎙️", layout="centered")

# Minimal CSS
st.markdown("""
<style>
    .stApp { max-width: 700px; margin: 0 auto; }
    .result-box { 
        background: #f0f7ff; padding: 1.2rem; border-radius: 8px; 
        border-left: 4px solid #1f77b4; margin: 0.8rem 0; font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ 宁波话转普通话")
st.caption("Ningbo Dialect → Mandarin Chinese")

# API key handling
def get_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return os.environ.get("GEMINI_API_KEY")

api_key = get_api_key()
if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    api_key = st.text_input("🔑 Gemini API Key", type="password")
    if not api_key:
        st.info("请输入 Gemini API Key")
        st.stop()

@st.cache_resource
def get_client(_api_key: str):
    return genai.Client(api_key=_api_key)

client = get_client(api_key)

# Live recording as primary, upload as secondary
tab_record, tab_upload = st.tabs(["🎤 录音", "📁 上传"])

with tab_record:
    audio_recording = st.audio_input("点击录音")

with tab_upload:
    audio_file = st.file_uploader("上传音频", type=["mp3", "wav", "m4a", "ogg", "webm"])

# Prioritize recording over upload
audio_source = None
audio_name = "recording.wav"
if audio_recording:
    audio_source = audio_recording.getvalue()
elif audio_file:
    audio_source = audio_file.getvalue()
    audio_name = audio_file.name

# Comprehensive Ningbo dialect vocabulary for accurate transcription
NINGBO_VOCAB = """
【宁波话常用词汇对照】
代词: 阿拉/我伲=我们, 侬=你, 伊=他/她, 倷=你们, 俚=他们
指示: 格/搿=这, 噶=那, 搿个=这个, 噶个=那个, 搿里=这里, 噶里=那里
疑问: 啥/啥个=什么, 哪个=谁, 哪能=怎么, 几化=多少, 阿是=是不是
动词: 勒/勒海=在, 呒没=没有, 晓得=知道, 困觉=睡觉, 吃茶=喝茶, 寻=找
形容词: 老好=很好, 蛮好=挺好, 交关=非常, 邪气=很/特别, 结棍=厉害
副词: 老=很, 蛮=挺, 还要=还是, 刚刚=刚才
时间: 今朝=今天, 明朝=明天, 夜里=晚上, 天亮=早上
连词: 搭=和/跟, 拨=给/被, 朝=向
语气: 嘎=了/啊, 个=的, 来=呢, 啦=了啊

【宁波话语音特点】
1. 浊音声母保留: 婆=bo, 头=dou, 共=gong (普通话清化)
2. 入声保留: 黑hek, 白bak, 吃chik, 日nik (短促)
3. 尖团不分: 精=京, 清=轻, 心=新
4. 鼻音韵尾: 很多-n/-ng混同
5. 声调: 阴平、阳平、上声、阴去、阳去、阴入、阳入"""

def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    """Parallel two-method transcription with synthesis."""
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        uploaded_file = client.files.upload(file=tmp_path)
        
        # Prompts for parallel execution
        prompt_direct = f"""你是宁波话（吴语甬江片）转写专家。这段音频是宁波方言。

{NINGBO_VOCAB}

任务：仔细听音频，识别宁波话发音，直接输出对应的普通话意思。
注意：宁波话有浊音（如"婆"读bo）、入声（短促音节），很多词汇与普通话不同。
只输出普通话翻译，不要解释。"""

        prompt_phonetic = f"""你是语音学专家。请仔细听这段音频，用汉字尽可能准确地记录你听到的每个音节发音。
这是宁波方言，可能有浊音、入声等普通话没有的音。
只输出你听到的发音对应的汉字，不要解释。"""
        
        # Run both transcriptions in parallel
        def call_direct():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt_direct, uploaded_file]
            ).text.strip()
        
        def call_phonetic():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt_phonetic, uploaded_file]
            ).text.strip()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_direct = executor.submit(call_direct)
            future_phonetic = executor.submit(call_phonetic)
            result_direct = future_direct.result()
            result_phonetic = future_phonetic.result()
        
        # Final synthesis with comprehensive context
        prompt_final = f"""你是资深的宁波话（吴语甬江片）语言学家。请根据两种转写结果，给出最准确的普通话翻译。

{NINGBO_VOCAB}

【直接语义转写】：{result_direct}
【音素记录】：{result_phonetic}

分析要点：
1. 对照宁波话词汇表，识别方言词汇
2. 考虑浊音、入声导致的听音偏差
3. 结合语境判断最合理的意思
4. 如果两种结果冲突，优先考虑语义合理性

输出最终的普通话翻译，只输出结果，不要解释过程。"""

        final_result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_final]
        ).text.strip()
        
        return {
            "final": final_result,
            "direct": result_direct,
            "phonetic": result_phonetic
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Process
if audio_source:
    if st.button("转写", type="primary", use_container_width=True):
        with st.spinner("分析中..."):
            try:
                results = transcribe_audio(audio_source, audio_name)
                st.markdown("### 📝 结果")
                st.markdown(f'<div class="result-box">{results["final"]}</div>', unsafe_allow_html=True)
                
                with st.expander("详细分析"):
                    st.caption("直接语义")
                    st.info(results["direct"])
                    st.caption("音素记录")
                    st.code(results["phonetic"])
            except Exception as e:
                st.error(f"失败: {e}")

st.markdown("---")
st.caption("Gemini 2.5 Flash • 宁波话")
