import streamlit as st
import numpy as np
import librosa
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import queue

st.title("🍶 酔っ払い度判定アプリ（WebRTC＋TURN対応版）")

st.markdown("""
### 使い方
1. ブラウザで **録音開始**
2. 5〜10秒ほど日本語を話す
3. **停止**すると自動で解析し、酔っ払い度を表示します
""")

# =========================
# TURN 設定（Secretsから取得）
# =========================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {
            "urls": st.secrets["webrtc"]["turn_uri"],
            "username": st.secrets["webrtc"]["turn_username"],
            "credential": st.secrets["webrtc"]["turn_password"],
        }
    ]
})

audio_queue = queue.Queue()

def audio_receiver(frame: av.AudioFrame):
    pcm = frame.to_ndarray().astype(np.float32)
    audio_queue.put(pcm)
    return frame

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_frame_callback=audio_receiver,
)

if webrtc_ctx.state.playing is False:
    st.warning("🎤 マイクが未接続です。ブラウザのマイク許可を確認してください。")

if not audio_queue.empty() and not webrtc_ctx.state.playing:
    audio = np.concatenate(list(audio_queue.queue)).flatten()
    audio_queue.queue.clear()

    sr = 48000
    audio = audio / np.max(np.abs(audio) + 1e-9)

    # 特徴量
    rms = librosa.feature.rms(y=audio).mean()
    zcr = librosa.feature.zero_crossing_rate(audio).mean()
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()

    # 正規化（ざっくり）
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
