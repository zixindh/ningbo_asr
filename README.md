# Ningbo ASR

Streamlit web app for transcribing Ningbo dialect / Wu Chinese speech into Mandarin Chinese text.

## Model

- Alibaba Cloud Model Studio `fun-asr-realtime`
- Chinese Mainland deployment scope
- Beijing WebSocket endpoint: `wss://dashscope.aliyuncs.com/api-ws/v1/inference`
- Live microphone audio target: 16 kHz mono PCM

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

## Usage

1. Open the app.
2. Click **Start** and allow browser microphone access.
3. Speak Ningbo dialect / Wu Chinese.
4. Read the Mandarin transcript as it appears.
5. Click **Stop** when finished.

## Notes

- Use a Chinese Mainland Alibaba Model Studio API key for this app.
- The screen intentionally shows only Start, Stop, and the Mandarin transcript.
- Browser microphone chunks are converted to 16 kHz mono PCM in the browser, processed in memory by Streamlit Cloud, and sent directly to Fun-ASR realtime.
- This app does not require STUN, TURN, or WebRTC server settings.
