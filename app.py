import streamlit as st
import numpy as np
import librosa
import av
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase

st.title("🍶 酔っ払い度判定アプリ（WebRTC＋TURN対応版）")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{
        "urls": st.secrets["webrtc"]["turn_uri"],
        "username": st.secrets["webrtc"]["turn_username"],
        "credential": st.secrets["webrtc"]["turn_password"],
    }]
})

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().astype(np.float32)
        audio_q.put(pcm)
        return frame

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

# STOP後に解析（「前回の残骸」で動かないよう state でガード）
if (webrtc_ctx.state.playing is False) and (not audio_q.empty()):
    audio = np.concatenate(list(audio_q.queue)).flatten()
    audio_q.queue.clear()

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
    st.metric("酔っ払い度（0-100）", int(score * 100))
