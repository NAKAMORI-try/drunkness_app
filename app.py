import streamlit as st
from openai import OpenAI
import numpy as np
import librosa
import io
import soundfile as sf

# ============================
# OpenAI API クライアント
# ============================
client = OpenAI()

# ============================
# Streamlit UI
# ============================
st.title("🍶 Whisper酔っ払い度解析アプリ（TURN不要版）")

st.markdown("""
### 📝 手順
1. スマホ or PC で **録音アプリ（ボイスメモ等）** を使って音声を録音  
2. 録音した音声をこの画面に **アップロード**  
3. Whisper が文字起こし → 音声特徴と合わせて酔っ払い度を表示します  
""")

uploaded_file = st.file_uploader("音声ファイルをアップロード（wav/mp3/m4aなど）", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file)

    # ============================
    # 音声読み込み
    # ============================
    data, sr = librosa.load(uploaded_file, sr=None, mono=True)

    # Whisper に渡すためにバイナリ化
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, data, sr, format="WAV")
    wav_buffer.seek(0)

    # ============================
    # Whisper 文字起こし
    # ============================
    with st.spinner("Whisperで文字起こし中..."):
        transcript = client.audio.transcriptions.create(
            file=wav_buffer,
            model="gpt-4o-mini-tts",   # Whisper v3相当
            response_format="text"
        )

    st.subheader("📝 文字起こし結果")
    st.write(transcript)

    # ============================
    # 音声特徴量計算（酔っ払い度の材料）
    # ============================
    rms = librosa.feature.rms(y=data).mean()
    zcr = librosa.feature.zero_crossing_rate(y=data).mean()
    tempo = (len(transcript) / (len(data) / sr))  # 文字密度による速度の近似

    # 正規化
    def norm(v, lo, hi):
        return max(0, min(1, (v - lo) / (hi - lo)))

    rms_norm = norm(rms, 0.01, 0.2)   # 声が大きいほど酔い↑
    zcr_norm = norm(zcr, 0.01, 0.15)  # 低いほど酔い↑
    tempo_norm = 1 - norm(tempo, 2, 8)  # 遅いほど酔い↑

    # ============================
    # 総合スコア
    # ============================
    drunk_score = int((rms_norm * 0.4 + zcr_norm * 0.3 + tempo_norm * 0.3) * 100)

    st.subheader("🍶 酔っ払い度（0-100）")
    st.metric("推定スコア", f"{drunk_score}")

    # 詳細値
    with st.expander("詳細指標を表示"):
        st.write({
            "rms": float(rms),
            "zcr": float(zcr),
            "speech_speed": float(tempo),
            "rms_norm": float(rms_norm),
            "zcr_norm": float(zcr_norm),
            "tempo_norm": float(tempo_norm)
        })

