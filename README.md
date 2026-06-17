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

For reliable live microphone connections, especially on restricted networks, add a TURN relay:

```toml
TURN_URLS = "turn:your-turn-host:3478,turns:your-turn-host:5349"
TURN_USERNAME = "your-turn-username"
TURN_CREDENTIAL = "your-turn-password"
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

1. Open the app and choose **实时麦克风**.
2. Click **START** and allow browser microphone access.
3. Speak Ningbo dialect directly. The Mandarin text updates while you speak.
4. Click **STOP** to close the live Fun-ASR session.

## Notes

- Use a Chinese Mainland Alibaba Model Studio API key for this app.
- Live microphone frames are streamed directly to Fun-ASR and are not saved first.
- Live microphone uses WebRTC. STUN is tried by default, but many corporate/mobile networks need TURN.
- Upload support is limited to Fun-ASR realtime formats: WAV, MP3, AAC, AMR, OPUS, SPEEX.
