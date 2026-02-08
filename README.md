🌋 DASHBOARD PERINGATAN DINI TANAH LONGSOR – LSTM

Deskripsi Singkat:
Dashboard ini menggunakan model LSTM untuk memprediksi risk score tanah longsor 
berdasarkan data historis curah hujan dan kelembapan tanah. Dashboard menampilkan 
status peringatan, grafik tren 7 hari terakhir, dan tabel input + prediksi.

-------------------------------------------------------------
STRUKTUR FILE

dashboard-longsor/
│
├─ streamlit_app.py       # File utama Streamlit
├─ model_lstm_longsor.h5  # Model LSTM hasil training
├─ scaler.pkl             # Scaler (preprocessing) untuk fitur input
├─ README.txt             # Dokumentasi & panduan
├─ requirements.txt       # Paket Python yang dibutuhkan
└─ contoh_data.csv        # CSV contoh minimal 7 hari

-------------------------------------------------------------
INSTALASI & PERSIAPAN

1. Clone repository ini:
   git clone <repo_url>
   cd dashboard-longsor

2. Install dependencies:
   pip install -r requirements.txt

3. Pastikan file model_lstm_longsor.h5 dan scaler.pkl ada di folder yang sama.

-------------------------------------------------------------
FORMAT CSV

- CSV minimal 7 baris (7 hari terakhir)
- Kolom wajib:
  - Tanggal (YYYY-MM-DD)
  - curah_hujan (mm)
  - kelembapan_tanah (0–1)

Contoh preview:

Tanggal     | curah_hujan | kelembapan_tanah
------------|-------------|----------------
2026-02-02  | 120         | 0.78
2026-02-03  | 80          | 0.74
...         | ...         | ...

Fitur tambahan otomatis:
- rain_3d, rain_7d, rain_14d dihitung dari curah hujan secara rolling

-------------------------------------------------------------
CARA MENJALANKAN DASHBOARD

streamlit run streamlit_app.py

- Upload CSV minimal 7 hari
- Klik "Prediksi Risiko Longsor"
- Lihat status peringatan, metric ringkas, grafik trend, dan tabel prediksi

-------------------------------------------------------------
FITUR DASHBOARD

1. Header & Info: Nama dashboard, lokasi, model, time window
2. Input Data: CSV 7 hari + optional manual override hari terakhir
3. Prediksi LSTM: Sequence 7 hari terakhir → model → risk score
4. BLOK A – Status Peringatan: Card besar dengan warna (Hijau / Kuning / Merah)
5. BLOK B – Metric Ringkas: Risk score, time window, tanggal prediksi
6. BLOK C – Grafik Dinamis: Trend risk score 7 hari terakhir dengan threshold
7. BLOK D – Tabel Input + Prediksi: 7 hari terakhir + risk score

-------------------------------------------------------------
CATATAN

- Pastikan Python 3.10+ atau sesuai kompatibilitas TensorFlow
- File CSV harus memiliki setidaknya 7 baris untuk prediksi
- Model LSTM hanya menerima sequence 7 hari → data historis diperlukan

-------------------------------------------------------------

REFERENSI & LITERATUR

- LSTM untuk time series forecasting
- Early warning system berbasis data hidrometeorologi
- Panduan Streamlit untuk dashboard ilmiah
