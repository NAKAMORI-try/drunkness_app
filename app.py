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

# セッション跨ぎで保持
if "audio_q" not in st.session_state:
    st.session_state.audio_q = queue.Queue()
if "last_analyzed_bytes" not in st.session_state:
    st.session_state.last_analyzed_bytes = 0

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().astype(np.float32)  # (channels, samples) or (samples,)
        st.session_state.audio_q.put(pcm)
        return frame

st.caption("1) START → 5〜10秒話す → 2) STOP → 3) 下の『解析』を押す")

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

# キューの状態を見える化（デバッグ用）
q_size = st.session_state.audio_q.qsize()
st.info(f"受信フレーム数（目安）: {q_size}  ※0のままならマイクが取れていません")

def drain_audio_queue():
    """キューを全部取り出して1本の波形にする"""
    chunks = []
    while not st.session_state.audio_q.empty():
        chunks.append(st.session_state.audio_q.get())

    if not chunks:
        return None

    audio = np.concatenate([c.flatten() for c in chunks]).astype(np.float32)
    return audio

# STOPを検知するだけだと環境差があるので、ユーザー操作で確実に解析させる
analyze = st.button("解析する（STOP後に押す）", type="primary")

if analyze:
    audio = drain_audio_queue()
    if audio is None:
        st.error("音声が取得できていません。START後にマイク許可が出ているか、別のマイクを選んでください。")
        st.stop()

    sr = 48000  # streamlit-webrtcの既定が48kHzのことが多い
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # 3秒未満だと特徴量が安定しないので弾く
    duration = len(audio) / sr
    st.write(f"取得音声長: {duration:.2f} 秒")
    if duration < 3.0:
        st.warning("音声が短すぎます。3秒以上話してからSTOP→解析してください。")
        st.stop()

    rms = float(librosa.feature.rms(y=audio).mean())
    zcr = float(librosa.feature.zero_crossing_rate(audio).mean())
    centroid = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())

    score = (
        min(rms / 0.2, 1.0) * 0.4 +
        (1 - min(zcr / 0.15, 1.0)) * 0.3 +
        (1 - min(centroid / 4000, 1.0)) * 0.3
    )
    drunk_score = int(max(0, min(100, round(score * 100))))

    st.subheader("🍶 推定酔っ払い度（0〜100）")
    st.metric("スコア", drunk_score)

    with st.expander("解析詳細"):
        st.json({
            "rms": rms,
            "zcr": zcr,
            "centroid": centroid,
            "duration_sec": duration,
        })
