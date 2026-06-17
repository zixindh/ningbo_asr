# Ningbo ASR

Streamlit web app for transcribing Ningbo dialect / Wu Chinese speech into Mandarin Chinese text.

## Model

- Alibaba Cloud Model Studio `fun-asr-realtime`
- Chinese Mainland deployment scope
- Beijing WebSocket endpoint: `wss://dashscope.aliyuncs.com/api-ws/v1/inference`
- Audio target: 16 kHz mono WAV for browser recordings

## Streamlit Setup

Add this secret in Streamlit Cloud:

```toml
FUN_ASR_KEY = "your-alibaba-model-studio-key"
```

You can also run locally with the same environment variable:

```powershell
$env:FUN_ASR_KEY = "your-alibaba-model-studio-key"
streamlit run app.py
```

## Local Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Use a Chinese Mainland Alibaba Model Studio API key for this app.
- Browser recordings are normalized to 16 kHz mono PCM WAV before being streamed to Fun-ASR.
- Upload support is limited to Fun-ASR realtime formats: WAV, MP3, AAC, AMR, OPUS, SPEEX.
