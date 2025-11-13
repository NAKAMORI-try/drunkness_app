# app.py
# -*- coding: utf-8 -*-
"""
酔っ払い度（0-100）を推定するデモアプリ。
- 表示した日本語文章を読み上げてもらい、スマホ/PCのマイクをWebRTC経由で取得
- 音量（RMS）と発話の明瞭度（ろれつ）に関係する特徴量からスコアリング
- 任意で「平常時の自分」サンプルをキャリブレーションとして保存

必要ライブラリ:
    pip install streamlit streamlit-webrtc av numpy librosa soundfile scipy

起動:
    streamlit run app.py

注意:
    * これは実験用デモ。医療・法的判定用途では使わないでください。
    * 実デバイス・環境差が大きいため、スコアは相対的な指標です。
"""

import av
import io
import math
import queue
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
import soundfile as sf
import librosa
from scipy.signal import medfilt

st.set_page_config(page_title="酔っ払い度判定デモ", page_icon="🍶", layout="centered")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ----------------------------- ユーティリティ ----------------------------- #

def rms_dbfs(y: np.ndarray) -> float:
    eps = 1e-9
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(y)) + eps))


def vad_mask(y: np.ndarray, sr: int, frame_ms: float = 30.0, hop_ms: float = 10.0,
             energy_thresh_db: float = -45.0) -> np.ndarray:
    """非常に単純なエネルギーベースVAD。True=有声。
    energy_thresh_db はRMS[dBFS]のしきい値。
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    rms_list = []
    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i:i+frame_len]
        rms_list.append(rms_dbfs(frame))
    rms_arr = np.array(rms_list)
    mask = rms_arr > energy_thresh_db
    return mask


def syllable_like_rate(y: np.ndarray, sr: int) -> float:
    """発話速度の簡易推定（シラブル/秒に相当）。
    スペクトルフラックスのピーク数を数える簡易手法。
    """
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    flux = np.diff(S, axis=1)
    flux = np.maximum(flux, 0).mean(axis=0)
    flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-9)
    flux_smooth = medfilt(flux, kernel_size=7)
    # ピーク検出（しきい値超え＆局所最大）
    thr = 0.6
    peaks = []
    for i in range(1, len(flux_smooth)-1):
        if flux_smooth[i] > thr and flux_smooth[i] > flux_smooth[i-1] and flux_smooth[i] > flux_smooth[i+1]:
            peaks.append(i)
    time = np.arange(len(flux_smooth)) * (256 / sr)
    duration = time[-1] if len(time) > 0 else 0.0
    rate = (len(peaks) / duration) if duration > 0 else 0.0
    return float(rate)


def slur_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """ろれつに関連しそうな簡易スペクトル特徴を計算。
    高スルーは一般に明瞭度↓: スペクトル平坦度↑, セントロイド↓, ZCR↓ などを仮定。
    """
    y = librosa.util.normalize(y)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-9
    cent = librosa.feature.spectral_centroid(S=S, sr=sr).flatten()
    flat = librosa.feature.spectral_flatness(S=S).flatten()
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256).flatten()

    features = {
        "centroid_mean": float(np.mean(cent)),
        "centroid_std": float(np.std(cent)),
        "flatness_mean": float(np.mean(flat)),
        "flatness_std": float(np.std(flat)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }
    return features


def normalize_feature(x: float, lo: float, hi: float, invert: bool = False) -> float:
    x_clamped = max(lo, min(hi, x))
    norm = (x_clamped - lo) / (hi - lo + 1e-9)
    return 1.0 - norm if invert else norm


@dataclass
class AnalysisResult:
    rms_db: float
    voiced_ratio: float
    speech_rate: float
    slur_score_raw: float
    drunkness: int
    details: Dict[str, float]


# --------------------------- スコアリング ロジック --------------------------- #

def score_drunkness(y: np.ndarray, sr: int, baseline: Optional[Dict[str, float]] = None) -> AnalysisResult:
    # 全体音量
    rms = rms_dbfs(y)

    # VAD
    mask = vad_mask(y, sr)
    voiced_ratio = float(mask.mean()) if mask.size > 0 else 0.0

    # 発話速度（疑似シラブル/秒）
    rate = syllable_like_rate(y, sr)

    # ろれつ関連特徴
    f = slur_features(y, sr)

    # -------- 正規化（経験的レンジ。端末差吸収のため広めに） -------- #
    loud_norm = normalize_feature(rms, -45.0, -10.0, invert=False)  # 大きいほど酔いポイント↑
    rate_norm = normalize_feature(rate, 1.5, 6.0, invert=True)      # 遅いほど酔いポイント↑
    voiced_norm = normalize_feature(voiced_ratio, 0.3, 0.95, invert=True)  # 断続的/沈黙多いほど↑
    flat_norm = normalize_feature(f["flatness_mean"], 0.05, 0.5, invert=False) # 平坦度高いほど↑
    cent_norm = normalize_feature(f["centroid_mean"], 1500, 4500, invert=True)  # セントロイド低いほど↑
    zcr_norm = normalize_feature(f["zcr_mean"], 0.02, 0.12, invert=True)        # ZCR低いほど↑

    # ベースライン補正（任意）
    baseline_boost = 0.0
    if baseline:
        # 自己対比: 平常時より音量がどれだけ大きいか、速度がどれだけ遅いか
        loud_delta = max(0.0, (rms - baseline.get("rms_db", rms)) / 10.0)  # 10dBで+1
        rate_delta = max(0.0, (baseline.get("rate", rate) - rate) / 2.0)   # 2シラブル/秒で+1
        baseline_boost = 10.0 * (loud_delta + rate_delta)

    # 重み付け（合計100に近づくよう調整）
    score = (
        30.0 * loud_norm +
        20.0 * (0.5*rate_norm + 0.5*voiced_norm) +
        40.0 * (0.5*flat_norm + 0.3*cent_norm + 0.2*zcr_norm) +
        baseline_boost
    )
    score = int(max(0, min(100, round(score))))

    details = {
        "rms_db": rms,
        "voiced_ratio": voiced_ratio,
        "speech_rate": rate,
        **f,
        "loud_norm": loud_norm,
        "rate_norm": rate_norm,
        "voiced_norm": voiced_norm,
        "flat_norm": flat_norm,
        "centroid_norm": cent_norm,
        "zcr_norm": zcr_norm,
        "baseline_boost": baseline_boost,
    }

    return AnalysisResult(rms, voiced_ratio, rate, (flat_norm+cent_norm+zcr_norm)/3.0, score, details)


# ------------------------------ UI / WebRTC ------------------------------ #

st.title("🍶 酔っ払い度判定デモ")

st.markdown(
    """
**手順**
1. 下の文章を声に出して読み上げてください（できれば一定の速さ・声量で）
2. 「録音開始」を押して10秒程度録音 → 「停止」で解析
3. 必要なら「平常時サンプルを保存」で自分専用の補正を有効化

> ⚠️ デモ用途です。端末や環境、個人差によってばらつきます。
    """
)

TEXTS = [
    "生麦生米生卵。隣の客はよく柿食う客だ。",
    "赤巻紙青巻紙黄巻紙。雨があがれば綾なす彩。",
    "東京特許許可局、今日急遽許可却下。",
]

text_choice = st.selectbox("読み上げる文章", TEXTS, index=0)
st.info(text_choice)

# セッションステート
if "audio_buffers" not in st.session_state:
    st.session_state.audio_buffers = []
if "baseline" not in st.session_state:
    st.session_state.baseline = None


class AudioRecorder:
    def __init__(self):
        self.q: "queue.Queue[av.AudioFrame]" = queue.Queue()
        self.frames: List[np.ndarray] = []
        self.sr: int = 48000  # WebRTC標準
        self.channels: int = 1

    def recv_callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        # モノラル・float32へ
        frame = frame.to_ndarray(format="s16", layout="mono")
        # int16 -> float32
        pcm = frame.astype(np.float32) / 32768.0
        self.frames.append(pcm.copy())
        return av.AudioFrame.from_ndarray((pcm * 32768.0).astype(np.int16), format="s16", layout="mono")

    def get_audio(self) -> Tuple[np.ndarray, int]:
        if not self.frames:
            return np.zeros(0, dtype=np.float32), self.sr
        y = np.concatenate(self.frames)
        return y, self.sr


recorder = AudioRecorder()

col1, col2, col3 = st.columns(3)
with col1:
    duration_sec = st.slider("録音長(秒)", 5, 20, 10, 1)
with col2:
    st.write("")
with col3:
    energy_thresh = st.slider("VADしきい値(dBFS)", -70, -20, -45, 1)

webrtc_ctx = webrtc_streamer(
    key="speech-capture",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=False,
)

if webrtc_ctx.state.playing:
    webrtc_ctx.receiver.audio_transformer = recorder.recv_callback

st.markdown("---")

# 録音コントロール
start = st.button("⏺️ 録音開始")
stop = st.button("⏹️ 停止・解析")

if start and webrtc_ctx.state.playing:
    recorder.frames = []
    st.session_state.audio_buffers = []
    st.info("録音中… 指定秒数読んだら『停止・解析』を押してください。")

if stop:
    y, sr = recorder.get_audio()
    if y.size == 0:
        st.warning("音声が取得できませんでした。マイク権限や接続を確認してください。")
    else:
        # 必要ならトリム（無音カット）
        yt, _ = librosa.effects.trim(y, top_db=40)
        if yt.size < sr * 0.8:
            yt = y  # 過度に切れたら元に戻す

        # 解析
        res = score_drunkness(yt, sr, baseline=st.session_state.baseline)

        # 保存WAV（デバッグ用）
        buf = io.BytesIO()
        sf.write(buf, yt, sr, format="WAV")
        st.session_state.audio_buffers.append(buf.getvalue())

        st.subheader("推定結果")
        st.metric("🍶 酔っ払い度", f"{res.drunkness}/100")

        with st.expander("詳細指標"):
            st.json(res.details)

        st.download_button("解析音声をWAVで保存", data=st.session_state.audio_buffers[-1], file_name="sample.wav", mime="audio/wav")

        st.caption("*スコアは相対指標です。明瞭な発話ほど低スコア、声量が大きく不明瞭なほど高スコアになりやすい設計です。*")

st.markdown("---")

# ベースライン（平常時）サンプルの保存
st.subheader("任意: 平常時サンプルを保存して補正する")
if st.button("平常時サンプルを現在音声から保存"):
    y, sr = recorder.get_audio()
    if y.size == 0:
        st.warning("まず録音してください。")
    else:
        yt, _ = librosa.effects.trim(y, top_db=40)
        f = slur_features(yt, sr)
        base = {
            "rms_db": rms_dbfs(yt),
            "rate": syllable_like_rate(yt, sr),
            "flat": f["flatness_mean"],
            "cent": f["centroid_mean"],
            "zcr": f["zcr_mean"],
        }
        st.session_state.baseline = base
        st.success("平常時サンプルを保存しました。この後の判定に自己対比補正を加えます。")
        st.json(base)

st.caption("© Demo. This is a heuristic prototype; not a diagnostic tool.")
