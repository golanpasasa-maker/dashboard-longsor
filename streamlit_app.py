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

# ---------------------------------------------------------
# HEADER UTAMA
# ---------------------------------------------------------
st.title("🌋 Dashboard Peringatan Dini Tanah Longsor")

st.markdown(
    """
    Berbasis Deep Learning (LSTM)
    """
    """
    Lokasi Studi: Kab.Tana Toraja Kec.Makale
    """
    """
    Model: Long Short-Term Memory (LSTM)
    """
    """
    Peneliti: Golan Prabu pasasa & Guido Arung Sallata Putra
    """
)

st.divider()

# =========================================================
# INPUT DATA
# =========================================================

st.header("📊 Input Data")

st.markdown("""
### 🔹 Format CSV yang Wajib

CSV berisi data historis **14 hari terakhir**

**Kolom wajib:**
- `Tanggal`
- `curah_hujan`
- `kelembapan_tanah`

📌 Minimal **14 baris**  
📌 Data akan diurutkan otomatis berdasarkan tanggal
""")

uploaded_file = st.file_uploader(
    "Upload file CSV (minimal 14 hari terakhir)",
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
if len(df) < 14:
    st.error("❌ CSV harus berisi minimal 14 baris")
    st.stop()

# -----------------------------
# Jumlah data historis
# -----------------------------
jumlah_data = len(df)

st.info(f"📊 Data historis tersedia: {jumlah_data} hari → maksimal prediksi {jumlah_data} hari berikutnya")

# -----------------------------
# Urutkan tanggal
# -----------------------------
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

# -----------------------------
# Pastikan numeric
# -----------------------------
df["curah_hujan"] = pd.to_numeric(df["curah_hujan"], errors="coerce").fillna(0)
df["kelembapan_tanah"] = pd.to_numeric(df["kelembapan_tanah"], errors="coerce").fillna(0)

# -----------------------------
# Hitung fitur agregasi
# -----------------------------
df["rain_3d"] = df["curah_hujan"].rolling(3, min_periods=1).sum()
df["rain_7d"] = df["curah_hujan"].rolling(7, min_periods=1).sum()
df["rain_14d"] = df["curah_hujan"].rolling(14, min_periods=1).sum()

# -----------------------------
# Ambil 7 hari terakhir untuk model
# -----------------------------
df_model_input = df.tail(jumlah_data).reset_index(drop=True)

st.subheader("📅 Data Input Model")
st.dataframe(df_model_input)

# =========================================================
# INPUT JUMLAH HARI PREDIKSI
# =========================================================

jumlah_prediksi = st.number_input(
    f"Jumlah hari prediksi ke depan (maks {jumlah_data})",
    min_value=1,
    max_value=jumlah_data,
    value=min(7, jumlah_data),
    step=1,
    help="Jumlah prediksi maksimal sama dengan jumlah data historis"
)


# =========================================================
# TOMBOL PREDIKSI & ALUR LSTM
# =========================================================

data_valid = len(df_model_input) >= 7

prediksi_btn = st.button(
    "🚀 Prediksi Risiko Longsor",
    disabled=not data_valid,
    help="Upload CSV minimal 14 hari agar tombol aktif"
)

# =========================================================
# PROSES PREDIKSI
# =========================================================

if prediksi_btn:

    st.markdown("---")
    st.header("🟢 Hasil Prediksi & Visualisasi")

    # -----------------------------
    # FEATURE MODEL
    # -----------------------------
    feature_cols = [
        "curah_hujan",
        "kelembapan_tanah",
        "rain_3d",
        "rain_7d",
        "rain_14d"
    ]

    X_all = df_model_input[feature_cols].values
    X_scaled = scaler.transform(X_all)

    # ambil 7 hari terakhir sebagai input sequence
    X_seq = X_scaled[-7:]
    risk_history = []

    # -----------------------------
    # RECURSIVE FORECASTING REALISTIS
    # -----------------------------

    # gunakan data asli untuk update rolling rainfall
    rain_history = df_model_input["curah_hujan"].tolist()
    soil_history = df_model_input["kelembapan_tanah"].tolist()
    # safety check
    jumlah_prediksi = min(jumlah_prediksi, jumlah_data)

    for _ in range(jumlah_prediksi):

        X_input = np.expand_dims(X_seq, axis=0)
        pred = model.predict(X_input, verbose=0)[0][0]
        risk_history.append(pred)

        # ====================================
        # SIMULASI KONDISI HARI BERIKUTNYA
        # ====================================

        # asumsi hujan = rata-rata 3 hari terakhir
        next_rain = np.mean(rain_history[-3:])

        # asumsi kelembapan mengikuti hari terakhir
        next_soil = soil_history[-1]

        rain_history.append(next_rain)
        soil_history.append(next_soil)

        # hitung rolling rainfall baru
        rain_3d = sum(rain_history[-3:])
        rain_7d = sum(rain_history[-7:])
        rain_14d = sum(rain_history[-14:])

        new_row = np.array([
            next_rain,
            next_soil,
            rain_3d,
            rain_7d,
            rain_14d
        ]).reshape(1, -1)

        # scaling
        new_row_scaled = scaler.transform(new_row)[0]

        # update sliding window
        X_seq = np.vstack([X_seq[1:], new_row_scaled])


    # =========================================================
    # BLOK A — STATUS H+1
    # =========================================================
    risk_score_today = risk_history[0]

    if risk_score_today < 0.5:
        status_today = "Aman 🟢"
        warna = "#4CAF50"
    elif risk_score_today > 0.5 and risk_score_today < 1.0:
        status_today = "Waspada 🟠"
        warna = "#FFAA00"

    
    last_date = df_model_input["Tanggal"].iloc[-1]

    tanggal_prediksi = [
    (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
    for i in range(jumlah_prediksi)
    ]
    st.subheader(f"🔮 Resiko Tanah Longsor Hari Ke-{jumlah_prediksi}")

    st.markdown(f"""
    <div style="background-color:{warna};
                color:white; padding:20px;
                border-radius:10px;
                font-size:24px;
                text-align:center;">
        Status: {status_today} Pada tanggal {tanggal_prediksi[-1]}
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # BLOK B — TABEL OUTPUT PREDIKSI
    # =========================================================

    st.subheader(f"📋 Hasil Prediksi Risiko ({jumlah_prediksi} Hari ke Depan)")

    last_date = df_model_input["Tanggal"].iloc[-1]

    tanggal_prediksi = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(jumlah_prediksi)
    ]

    df_output = pd.DataFrame({
        "Hari ke-": np.arange(1, jumlah_prediksi + 1),
        "Tanggal Prediksi": tanggal_prediksi,
        "Status Resiko": status_today
    })

    st.dataframe(
    df_output.style.set_table_styles([
        {"selector": "th", "props": [
            ("text-align", "center"),
            ("font-size", "24px")
        ]},
        {"selector": "td", "props": [
            ("text-align", "center"),
            ("font-size", "16px")
        ]}
    ]),
    width="stretch",
    hide_index=True
    )

    # =========================================================
