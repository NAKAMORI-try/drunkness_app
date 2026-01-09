import streamlit as st
import numpy as np
import librosa
import av
import queue
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    AudioProcessorBase,
)

st.title("🍶 酔っ払い度判定アプリ（WebRTC＋TURN対応版）")

st.markdown("""
### 使い方
1. **START** を押して録音開始  
2. 5〜10秒ほど日本語を話す  
3. **STOP** で自動解析し、酔っ払い度を表示します
""")

# =========================
# TURN 設定（Secrets）
# =========================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{
        "urls": st.secrets["webrtc"]["turn_uri"],
        "username": st.secrets["webrtc"]["turn_username"],
        "credential": st.secrets["webrtc"]["turn_password"],
    }]
})

audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

# =========================
# Audio Processor（正規ルート）
# =========================
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().astype(np.float32)
        audio_queue.put(pcm)
        return frame

# =========================
# WebRTC 起動
# =========================
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

# =========================
# 録音終了後の解析
# =========================
if webrtc_ctx.state.playing is False and not audio_queue.empty():
    audio = np.concatenate(list(audio_queue.queue)).flatten()
    audio_queue.queue.clear()

    sr = 48000
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    rms = librosa.feature.rms(y=audio).mean()
    zcr = librosa.feature.zero_crossing_rate(audio).mean()
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()

    score = (
        min(rms / 0.2, 1.0) * 0.4 +
        (1 - min(zcr / 0.15, 1.0)) * 0.3 +
        (1 - min(centroid / 4000, 1.0)) * 0.3
    )
    drunk_score = int(score * 100)

    st.subheader("🍶 推定酔っ払い度")
    st.metric("スコア（0〜100）", drunk_score)

    with st.expander("解析詳細"):
        st.write({
            "rms": float(rms),
            "zcr": float(zcr),
            "centroid": float(centroid),
        })
