# =========================================================
# DASHBOARD PERINGATAN DINI TANAH LONGSOR
# Header & Konfigurasi Awal
# =========================================================

import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import os
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---------------------------------------------------------
# Konfigurasi Halaman
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Peringatan Dini Longsor",
    page_icon="🌋",
    layout="wide"
)

# ---------------------------------------------------------
# HEADER UTAMA
# ---------------------------------------------------------
st.title("🌋 Dashboard Peringatan Dini Tanah Longsor")

st.markdown(
    """
    **Berbasis Deep Learning (LSTM)**  
    Time Window: **7 Hari** | Lokasi Studi: **Makale**
    """
)

st.divider()

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

MODEL_PATH = "model_lstm_longsor.h5"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model tidak ditemukan")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error("❌ Scaler tidak ditemukan")
        st.stop()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False  # penting untuk hindari error keras.metrics.mse
    )
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


model, scaler = load_model_and_scaler()

st.success("✅ Model & Scaler berhasil dimuat")

# =========================================================
# INPUT DATA
# =========================================================


st.header("📊 Dashboard Prediksi Risiko Longsor")

st.markdown("""
### 🔹 Format CSV yang Wajib

CSV berisi data historis **7 hari terakhir**

**Kolom wajib:**
- `Tanggal`
- `curah_hujan`
- `kelembapan_tanah`

📌 Minimal **7 baris**  
📌 Data akan diurutkan otomatis berdasarkan tanggal
""")

uploaded_file = st.file_uploader(
    "Upload file CSV (minimal 7 hari terakhir)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Upload CSV terlebih dahulu untuk melanjutkan.")
    st.stop()

df = pd.read_csv(uploaded_file)

# -----------------------------
# Validasi kolom wajib
# -----------------------------
required_cols = ["Tanggal", "curah_hujan", "kelembapan_tanah"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Kolom CSV tidak lengkap: {missing}")
    st.stop()

# -----------------------------
# Validasi jumlah baris
# -----------------------------
if len(df) < 7:
    st.error("❌ CSV harus berisi minimal 7 baris")
    st.stop()

# -----------------------------
# Urutkan tanggal
# -----------------------------
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

# -----------------------------
# Hitung rain_3d, rain_7d, rain_14d otomatis
# -----------------------------
# Pastikan kolom numeric
df["curah_hujan"] = pd.to_numeric(df["curah_hujan"], errors="coerce").fillna(0)
df["kelembapan_tanah"] = pd.to_numeric(df["kelembapan_tanah"], errors="coerce").fillna(0)

# Hitung rain_3d / 7d / 14d
df["rain_3d"] = df["curah_hujan"].rolling(window=3, min_periods=1).sum()
df["rain_7d"] = df["curah_hujan"].rolling(window=7, min_periods=1).sum()
df["rain_14d"] = df["curah_hujan"].rolling(window=14, min_periods=1).sum()

# Ambil 7 hari terakhir
df_last7 = df.tail(7).reset_index(drop=True)
st.dataframe(df_last7)  

# =========================================================
# TOMBOL PREDIKSI & ALUR LSTM
# =========================================================

# Tombol aktif hanya jika data CSV / manual sudah valid
data_valid = 'df_last7' in locals() and len(df_last7) == 7

prediksi_btn = st.button(
    "🚀 Prediksi Risiko Longsor",
    disabled=not data_valid,
    help="Upload CSV minimal 7 hari agar tombol aktif"
)

if prediksi_btn:
    st.markdown("---")
    st.header("🟢 Hasil Prediksi & Visualisasi")

    # -----------------------------
    # BLOK C — Grafik Dinamis (Prediksi 7 hari terakhir)
    # -----------------------------

    # Sliding window untuk 7 hari terakhir
    feature_cols = ["curah_hujan","kelembapan_tanah","rain_3d","rain_7d","rain_14d"]
    X_all = df_last7[feature_cols].values
    X_scaled_all = scaler.transform(X_all)

    risk_history = []
    for i in range(len(df_last7)):
        if i+1 < 7:
            continue  # skip jika window kurang dari 7
        X_seq_window = np.expand_dims(X_scaled_all[i-6:i+1], axis=0)
        pred = model.predict(X_seq_window)[0][0]
        risk_history.append(pred)

    # Ambil risk score terakhir (H-0)
    risk_score_today = risk_history[-1]
    if risk_score_today < 0.6:
        status_today = "Aman 🟢"
    elif risk_score_today < 0.75:
        status_today = "Waspada 🟡"
    else:
        status_today = "Siaga 🔴"
    # -----------------------------
    # BLOK A — Status Peringatan (Card besar)
    # -----------------------------
    st.subheader("🔮 Nilai Index Risiko")
    
    st.markdown(f"""
    <div style="background-color: {'#4CAF50' if risk_score_today<0.6 else '#FFC107' if risk_score_today<0.75 else '#F44336'}; 
                color:white; padding:20px; border-radius:10px; font-size:24px; text-align:center;">
        Risk Score: {risk_score_today:.2f} <br>
        Status: {status_today}
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # BLOK B — Metric Ringkas
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{risk_score_today:.2f}")
    col2.metric("Time Window", "7 Hari")
    tanggal_prediksi = df_last7["Tanggal"].iloc[-1].strftime("%Y-%m-%d")
    col3.metric("Tanggal Prediksi", tanggal_prediksi)

    # -----------------------------
    # Grafik
    # -----------------------------
    st.subheader("📈 Grafik Risk Score (H-6 → H-0)")
    
    days = df_last7["Tanggal"].iloc[-len(risk_history):].dt.strftime("%Y-%m-%d").tolist()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(days, risk_history, marker='o', label="Risk Score", color='blue')
    ax.axhline(0.6, color='orange', linestyle='--', label="Aman → Waspada")
    ax.axhline(0.75, color='red', linestyle='--', label="Waspada → Siaga")
    ax.set_ylim(0,1)
    ax.set_ylabel("Risk Score")
    ax.set_title("Trend Risk Score 7 Hari Terakhir")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # BLOK D — Tabel Data Input + Risk Score
    # -----------------------------
    st.subheader("📋 Tabel Data 7 Hari Terakhir")
    df_display = df_last7.iloc[-len(risk_history):].copy()
    df_display["Risk Score"] = np.round(risk_history,2)
    st.dataframe(df_display)

