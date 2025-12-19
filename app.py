import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io

st.title("🍶 酔っ払い度解析アプリ（音声特徴量のみ・TURN不要版）")

st.markdown("""
### 📝 使い方
1. スマホやPCの **ボイスメモ等で録音** する  
2. 録音した音声ファイル（wav/mp3/m4a など）を下からアップロード  
3. 音量・発話のばらつき・無音の多さなどから **酔っ払い度（0〜100）** を推定します  

※ 完全に遊び用の指標です。本気の診断・評価には使わないでください。
""")

uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "m4a", "ogg"])

def normalize(x, lo, hi, invert=False):
    """値を0〜1に正規化（範囲外はクリップ）"""
    x_clamped = max(lo, min(hi, x))
    v = (x_clamped - lo) / (hi - lo + 1e-9)
    return 1.0 - v if invert else v

if uploaded:
    # 再生用
    st.audio(uploaded)

    # librosaで読み込み
    # 一度バッファに吸い上げてから読むとフォーマットの違いに強い
    buf = io.BytesIO(uploaded.read())
    y, sr = librosa.load(buf, sr=None, mono=True)

    duration = len(y) / sr
    st.caption(f"録音長: 約 {duration:.1f} 秒, サンプリングレート: {sr} Hz")

    if duration < 1.0:
        st.warning("1秒以上の音声をアップロードしてください。")
    else:
        # 無音トリム（極端に短くなったら元のまま）
        yt, _ = librosa.effects.trim(y, top_db=40)
        if len(yt) < sr * 0.8:
            yt = y

        # --- 特徴量計算 --- #
        # 全体の平均音量（RMS）
        rms_frame = librosa.feature.rms(y=yt)[0]
        rms_mean = float(rms_frame.mean())

        # ろれつ感に関連しそうな指標
        zcr = float(librosa.feature.zero_crossing_rate(yt)[0].mean())  # 雑音・子音の多さ
        flat = float(librosa.feature.spectral_flatness(y=yt).mean())   # スペクトル平坦度（こもり具合）
        cent = float(librosa.feature.spectral_centroid(y=yt, sr=sr).mean())  # 明瞭さのざっくり指標

        # 無音の多さ（喋ってない時間が多いほど酔いっぽいとみなす）
        energy = rms_frame
        voiced_ratio = float((energy > (energy.mean() * 0.5)).mean())

        # --- 正規化（0〜1） --- #
        # 経験的レンジ。かなりざっくり＆端末差を考えて広めに取る
        loud_norm   = normalize(rms_mean, 0.01, 0.2, invert=False)      # 大きいほど酔い↑
        zcr_norm    = normalize(zcr,      0.02, 0.15, invert=True)      # 低いほど酔い↑（単調・こもりぎみ）
        flat_norm   = normalize(flat,     0.1,  0.5,  invert=False)     # 平坦度高いほど酔い↑
        cent_norm   = normalize(cent,     1500, 4500, invert=True)      # セントロイド低いほど酔い↑
        voiced_norm = normalize(voiced_ratio, 0.4, 0.95, invert=True)   # 無音多いほど酔い↑

        # --- スコアリング（0〜100） --- #
        # 重みは遊び用のヒューリスティック
        score = (
            0.35 * loud_norm +
            0.2  * zcr_norm +
            0.25 * flat_norm +
            0.1  * cent_norm +
            0.1  * voiced_norm
        )
        drunk_score = int(max(0, min(100, round(score * 100))))

        st.subheader("🍶 推定酔っ払い度（0〜100）")
        st.metric("スコア", f"{drunk_score}")

        with st.expander("解析に使った指標の詳細"):
            st.json({
                "duration_sec": duration,
                "rms_mean": rms_mean,
                "zcr_mean": zcr,
                "flatness_mean": flat,
                "centroid_mean": cent,
                "voiced_ratio": voiced_ratio,
                "loud_norm": loud_norm,
                "zcr_norm": zcr_norm,
                "flat_norm": flat_norm,
                "cent_norm": cent_norm,
                "voiced_norm": voiced_norm,
            })

        st.caption(
            "※ 音量が大きく、スペクトルが平坦で、ゼロ交差率が低く、無音が多いほどスコアが上がるように設計しています。"
        )
else:
    st.info("まずは音声ファイルをアップロードしてみてください。")
