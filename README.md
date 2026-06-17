# Ningbo ASR

Streamlit web app for transcribing Ningbo dialect / Wu Chinese speech into Mandarin Chinese text.

## Model

- Alibaba Cloud Model Studio `fun-asr-realtime`
- Streamlit Cloud defaults to automatic endpoint selection:
  - International/Singapore: `wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference`
  - Chinese Mainland/Beijing: `wss://dashscope.aliyuncs.com/api-ws/v1/inference`
- Live microphone audio target: 16 kHz mono PCM

## Streamlit Setup

Add this secret in Streamlit Cloud:

```toml
FUN_ASR_KEY = "your-alibaba-model-studio-key"
# Optional. Use "international", "mainland", or omit for auto.
FUN_ASR_REGION = "international"
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

- Alibaba Model Studio API keys are region-specific. On Streamlit Cloud Free, start with the International/Singapore key and `FUN_ASR_REGION = "international"`. Use `FUN_ASR_REGION = "mainland"` only with a Beijing/Chinese Mainland key.
- The screen intentionally shows only Start, Stop, and the Mandarin transcript.
- Browser microphone chunks are converted to 16 kHz mono PCM in the browser, processed in memory by Streamlit Cloud, and sent directly to Fun-ASR realtime.
- This app does not require STUN, TURN, or WebRTC server settings.
