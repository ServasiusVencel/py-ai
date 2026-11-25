# 📊 Laporan Training Model AI & Data Pipeline IoT AWS

Proyek: **Sistem Monitoring & Prediksi Bencana Berbasis IoT Automatic Weather Station (AWS) & Tinggi Muka Air Tanah (TMAT)**  
Arsitektur: **Nimbus Fire & Peatland Environmental Intelligence Engine**  
Versi: `2.0.0` (Full History Training & Multi-Dimensional Data Moat)

---

## 1. 📋 Ringkasan Dataset & Training Scope

- **File Sumber**: `Data Dummy AWS dan TMAT.xlsx` (`data/raw/`)
- **Total Data Rekaman**: 17,567 baris (Interval waktu 5 menit, rentang 1 Juni 2026 s.d. 31 Juli 2026 / 61 hari penuh).
- **Cakupan Training**: **Seluruh data historis** dilatih ke dalam model machine learning, bukan sekadar sliding window 7 hari.
- **Variabel Sensor**:
  - `Tinggi Muka Air Tanah (cm)`: Rentang -164.83 cm s.d. -5.41 cm (Mean: -104.46 cm).
  - `Kelembaban Tanah (%)`: Rentang 75.87% s.d. 100.0% (Mean: 97.29%).
  - `Intensitas Hujan (mm/jam)`: Rentang 0.0 s.d. 93.0 mm/jam.
  - `Suhu Udara (°C)`: Rentang 18.6°C s.d. 30.9°C.
  - `Suhu Titik Embun (°C)`: Rentang 14.9°C s.d. 24.9°C.
  - `Kelembapan Udara (%)`: Rentang 53% s.d. 97%.
  - `Radiasi Matahari (W/m²)`: Rentang 0.0 s.d. 762.2 W/m².
  - `Tekanan Udara (hPa)`: Rentang 980.8 s.d. 988.3 hPa.
  - `Kecepatan Angin (m/s)`: Rentang 0.0 s.d. 5.1 m/s.
  - `Arah Angin (°)`: Rentang 0° s.d. 359°.
  - `Geolokasi`: Embedded Latitude (`-0.0263`) & Longitude (`109.3425`).

---

## 2. ⚙️ Preprocessing & Feature Engineering (53 Fitur Cerdas)

Model mengimplementasikan arsitektur fitur sesuai algoritma Nimbus Fire Intelligence Engine:

1. **Efek Jeda (Lag Effects & Temporal Dependency)**:
   - Lag WTD: $WTD_{t-1h}, WTD_{t-6h}, WTD_{t-24h}$ menangkap inersia hidrologis gambut.
   - Lag Curah Hujan: $Rain_{t-1h}, Rain_{t-6h}, Rain_{t-24h}$ menangkap riwayat pembasahan.
   - Lag VPD: $VPD_{t-1h}, VPD_{t-6h}, VPD_{t-24h}$ menangkap akumulasi laju pengeringan atmosferik.
2. **Durasi Pengeringan & Retensi Gambut (Dry Spells & Drawdown)**:
   - `dry_spell_hours`: Jam berturut-turut tanpa hujan (< 0.1 mm/jam).
   - `consecutive_low_wtd_hours`: Jam berturut-turut tinggi muka air tanah di bawah ambang batas regulasi (-40 cm).
   - `wtd_drawdown_rate_24h`: Laju perubahan muka air tanah dalam 24 jam ($cm/24h$).
   - `vpd_cum_24h`: Akumulasi kebutuhan evaporatif atmosferik 24 jam.
3. **Interaksi Non-Linear Antar Variabel (Interaction Features)**:
   - `interaction_wtd_vpd`: $WTD 	imes VPD$ (kedalaman air tanah dikombinasikan dengan daya hisap atmosferik).
   - `interaction_wtd_dryspell`: $WTD 	imes 	ext{dry\_spell}$ (kedalaman air tanah dikalikan durasi hari tanpa hujan).
   - `interaction_vpd_solar`: $VPD 	imes 	ext{radiasi surya}$ (tekanan pengeringan vegetasi puncak siang hari).
   - `interaction_wtd_trend_vpd`: Tren laju penurunan muka air tanah dikalikan VPD.
4. **Kalibrasi Lokasi Spesifik & Deteksi Anomali**:
   - `wtd_site_anomaly`: Deviasi WTD aktual terhadap rata-rata historis spesifik lokasi (-104.46 cm).
   - `is_site_wtd_depressed`: Indikator jika muka air tanah jatuh di bawah persentil 20 lokasi spesifik (-136.96 cm).
5. **Baseline Ilmiah FWI (Dokumen 1)**:
   - Formula Pseudo-FWI deterministik dipertahankan sebagai baseline ilmiah murni, berjalan berdampingan dengan probabilitas fire risk machine learning.

---

## 3. 🎯 Hasil Kinerja Model Machine Learning

Dilatih dengan Random Forest Classifier & Gradient Boosting Regressor (120 Estimators, Class Weighted Balanced, 53 Features):

| Model | Target Prediksi | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Rain Classifier** | 0=Tidak Hujan, 1=Berpotensi Hujan | **98.75%** | 0.9635 | 0.9437 | **0.9533** |
| **Fire Risk Model** | 0=Rendah, 1=Sedang, 2=Tinggi | **99.03%** | 0.9934 | 0.9772 | **0.9850** |
| **Drought Risk Model** | 0=Normal, 1=Waspada, 2=Kritis | **97.55%** | 0.9770 | 0.9787 | **0.9776** |

### Top Features Terpenting yang Dipelajari Model:
- **Fire Risk**: `interaction_vpd_solar` (17.7%), `vpd_kpa` (14.8%), `solar_radiation_wm2` (13.9%), `temp_rolling_mean_1h` (7.5%), `interaction_wtd_trend_vpd` (4.2%).
- **Drought Risk**: `rain_cum_24h` (9.2%), `wtd_lag_1h` (7.6%), `water_table_depth_cm` (7.5%), `wtd_lag_6h` (7.3%), `wtd_site_anomaly` (6.8%), `consecutive_low_wtd_hours` (5.7%).
- **Rain Prediction**: `dry_spell_hours` (17.5%), `interaction_wtd_dryspell` (13.8%), `rain_cum_1h` (11.3%), `rainfall_intensity_mmh` (10.0%).

---

## 4. 🔮 Engine Proyeksi 10 Hari ke Depan (Peatland Trained Forecaster)

- Model autoregresif dilatih pada seluruh dataset harian (61 hari penuh) menangkap hubungan antar hari, tren musim kemarau/hujan, laju penipisan cadangan air gambut, dan dinamika kelembaban.
- Menghasilkan status langsung per hari:
  - **Tingkat Kebakaran** (Rendah / Sedang / Tinggi / Ekstrim)
  - **Kekeringan** (Rendah / Sedang / Tinggi / Ekstrim)
  - **Potensi Hujan** (Rendah / Sedang / Tinggi / Ekstrim)
- Dilengkapi detail proyeksi kuantitatif: Suhu (°C), Kelembapan (%), Radiasi (W/m²), Curah Hujan (mm), TMAT Proyeksi (cm), dan Skor Pseudo-FWI.

---

## 5. 🔄 MLOps & Continuous Learning (Feedback Loop)

- **Hot Reload Otomatis**: Endpoint `/api/v1/retrain` mentrigger retraining pada seluruh dataset secara asynchronous di background tanpa downtime.
- **Data Moat**: Setiap observasi sensor baru memperkaya dataset historis lokasi, memperdalam pemahaman model atas lag effect dan karakteristik hidrologis unik lokasi tersebut.
