🌋 DASHBOARD PERINGATAN DINI TANAH LONGSOR
Berbasis Deep Learning (LSTM)

Dashboard ini merupakan Sistem Peringatan Dini Tanah Longsor berbasis model Long Short-Term Memory (LSTM) untuk memprediksi risiko longsor beberapa hari ke depan berdasarkan data historis curah hujan dan kelembapan tanah.

📍 Informasi Studi

Lokasi Studi: Kab. Tana Toraja, Kec. Makale

Model: Long Short-Term Memory (LSTM)

Peneliti: Golan Prabu Pasasa & Guido Arung Sallata Putra

Framework: Streamlit + TensorFlow

📂 STRUKTUR FILE
dashboard-longsor/
│
├─ streamlit_app.py       # Aplikasi utama Streamlit
├─ model_lstm_longsor.h5  # Model LSTM hasil training
├─ scaler.pkl             # Scaler preprocessing fitur
├─ requirements.txt       # Dependency Python
└─ README.md              # Dokumentasi proyek
⚙️ INSTALASI
1️⃣ Clone Repository
git clone <repo_url>
cd dashboard-longsor
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Jalankan Dashboard
streamlit run streamlit_app.py
📊 FORMAT DATA INPUT (WAJIB)
CSV minimal 14 hari terakhir
Kolom wajib:

Tanggal (format YYYY-MM-DD)

curah_hujan

kelembapan_tanah

Contoh:
Tanggal	curah_hujan	kelembapan_tanah
2026-02-01	120	0.78
2026-02-02	80	0.74
2026-02-03	95	0.76
Ketentuan:

Minimal 14 baris

Data otomatis diurutkan berdasarkan tanggal

Kolom numerik dipaksa menjadi numeric

Nilai kosong akan diisi 0

🧠 FITUR OTOMATIS YANG DIHITUNG

Sistem secara otomatis menghitung fitur agregasi:

rain_3d → total hujan 3 hari terakhir

rain_7d → total hujan 7 hari terakhir

rain_14d → total hujan 14 hari terakhir

Fitur ini digunakan sebagai input tambahan model LSTM.

🔮 MEKANISME PREDIKSI
1️⃣ Sliding Window (7 Hari)

Model menggunakan 7 hari terakhir sebagai input sequence LSTM.

[Day-6 ... Day-1] → LSTM → Risk Score H+1
2️⃣ Recursive Forecasting (Multi-step Prediction)

Untuk prediksi lebih dari 1 hari:

Prediksi hari pertama dihitung

Data hari berikutnya disimulasikan:

Curah hujan = rata-rata 3 hari terakhir

Kelembapan tanah = mengikuti hari terakhir

Rolling rainfall diperbarui

Sequence digeser (sliding window)

Proses diulang hingga N hari

Jumlah prediksi maksimal = jumlah data historis.

🎛️ FITUR DASHBOARD
🔹 BLOK A — Status Risiko Hari Ke-N

Menampilkan status risiko dalam bentuk card berwarna:

🟢 Aman → Risk score < 0.5

🟠 Waspada → 0.5 ≤ Risk score < 1.0

Status ditampilkan untuk hari terakhir dari periode prediksi.

🔹 BLOK B — Tabel Hasil Prediksi

Menampilkan:

Hari ke-

Tanggal prediksi

Status risiko

🔹 BLOK C — Data Input Model

Menampilkan seluruh data historis yang digunakan model untuk prediksi.

🧪 ALUR SISTEM

Upload CSV minimal 14 hari

Validasi kolom & jumlah data

Hitung fitur agregasi hujan

Scaling fitur menggunakan scaler.pkl

Ambil 7 hari terakhir sebagai sequence input

Lakukan recursive forecasting

Tampilkan hasil prediksi & status

🛠 TEKNOLOGI YANG DIGUNAKAN

Python 3.10+

TensorFlow / Keras

Streamlit

Pandas

NumPy

Matplotlib

Joblib

⚠️ CATATAN PENTING

File model_lstm_longsor.h5 dan scaler.pkl wajib berada di folder yang sama dengan streamlit_app.py

Model diload dengan compile=False untuk menghindari error metric Keras

CSV kurang dari 14 baris → sistem tidak berjalan

Prediksi maksimal = jumlah data historis

Model menerima input sequence 7 hari terakhir

📚 REFERENSI

LSTM untuk time series forecasting

Early Warning System berbasis data hidrometeorologi

Deep Learning untuk mitigasi bencana

Dokumentasi Streamlit
