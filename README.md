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

1. Open the app and choose **连续录音**.
2. Click **START** and allow browser microphone access.
3. Speak Ningbo dialect directly. The browser records hidden 5-second chunks.
4. The app transcribes each chunk and stitches the Mandarin text together.
5. Click **STOP** when finished.

## Notes

- Use a Chinese Mainland Alibaba Model Studio API key for this app.
- Browser microphone chunks are processed in memory and sent directly to Fun-ASR.
- This app does not require STUN, TURN, or WebRTC server settings.
- Upload support is limited to Fun-ASR realtime formats: WAV, MP3, AAC, AMR, OPUS, SPEEX.
