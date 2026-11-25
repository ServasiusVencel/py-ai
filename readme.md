# 🌦️ Nimbus AI — Climate & Environmental Prediction Engine

Platform monitoring dan kecerdasan prediktif kondisi lingkungan berbasis **IoT Automatic Weather Station (AWS)** & **Tinggi Muka Air Tanah (TMAT / WTD)** yang ditenagai oleh **Nimbus Fire & Environmental Intelligence Engine**.

Sistem ini dirancang bukan sekadar untuk menghitung indeks cuaca harian, melainkan membangun **Data Moat** proprietary di mana setiap hari data telemetri baru melatih AI untuk semakin mengenali karakteristik hidrologis dan mikroklimat spesifik lokasi (*site-specific*).

---

## 📌 1. Nilai Strategis & Data Moat Nimbus

Bagi ekosistem lahan gambut (*tropical peatland*), hubungan antara **WTD (Water Table Depth / TMAT), curah hujan, kelembaban tanah, pengeringan atmosferik (Vapor Pressure Deficit / VPD), dan kejadian kebakaran (*fire occurrence*)** bersifat **site-specific** (sangat spesifik per lokasi).

Penelitian **PFVI (Peatland Fire Vulnerability Index)** sendiri membuktikan bahwa data harian *groundwater*, *rainfall*, dan *soil moisture* sangat krusial untuk kalibrasi dan verifikasi terhadap kejadian kebakaran riil di lahan gambut.

```
                    Day 1
         Temperature, RH, Solar, WTD, Rainfall
                          ↓
                      PFWI = 35
                          ↓
                    Setelah 1 Tahun
              365 hari × multiple variables
                          ↓
            ~8,760 hourly observations atau
               365 daily observations
                          ↓
                  Nimbus Memahami:
            • Seasonal pattern
            • Drying pattern
            • Rainfall response
            • WTD response
            • Lag effect
            • Local fire conditions
```

### Prinsip: Data Moat ≠ Sekadar Punya Banyak Data
Jumlah data saja belum menjadi moat. Moat terbentuk ketika data tersebut digunakan untuk membuat model yang semakin:
1. **Site-specific**: Mengetahui kondisi normal (*baseline*) dan ambang batas khas lokasi tersebut.
2. **Predictive**: Menangkap dinamika temporal dan efek jeda (*lag effect*).
3. **Sulit direplikasi**: Kompetitor tidak dapat menyamai performa prediksi tanpa memiliki dataset historis bertahun-tahun di titik lokasi yang sama.

---

## 🧠 2. Lima Fungsi Kecerdasan AI Nimbus

AI Nimbus tidak sekadar menjalankan formula statis, melainkan menjalankan 5 fungsi kecerdasan adaptif:

### A. AI Belajar "Normal Condition" Lokasi (Site-Specific Calibration)
Awalnya sistem generic hanya memiliki aturan seragam:
$$\text{WTD} < -60\text{ cm} \rightarrow \text{Mulai Berbahaya}$$

Namun setelah memiliki rekaman data panjang, AI mempelajari sifat hidrologi & retensi air spesifik masing-masing lokasi:
- **Lokasi A**:
  - WTD -40 cm $\rightarrow$ relatif aman
  - WTD -55 cm $\rightarrow$ mulai terjadi *drying*
  - WTD -65 cm $\rightarrow$ risiko meningkat
  - WTD -75 cm + VPD tinggi + 10 hari tanpa hujan $\rightarrow$ risiko kebakaran meningkat tajam
- **Perubahan Paradigma**:
  - **Generic Model**: $\text{WTD} < -60\text{ cm} \rightarrow \text{High Risk}$
  - **Nimbus Site-Specific Model**:
    - **Lokasi A**: $\text{WTD} < -55\text{ cm} + \text{VPD} > X + \text{dry spell} > Y \rightarrow \text{High Risk}$
    - **Lokasi B**: $\text{WTD} < -70\text{ cm} + \text{VPD} > X + \text{dry spell} > Y \rightarrow \text{High Risk}$

### B. AI Belajar "Lag Effect" (Ketergantungan Temporal)
Kebakaran lahan gambut tidak langsung terjadi pada hari pertama saat cuaca mulai kering:
```
Day -10 : WTD -35 cm (Aman)
Day -7  : WTD -42 cm
Day -5  : WTD -50 cm
Day -3  : WTD -58 cm
Day -2  : WTD -63 cm (Mulai kritis)
Day -1  : WTD -67 cm (Kering ekstrem)
Day  0  : 🔥 FIRE OCCURRENCE
```
Model tidak hanya mengevaluasi satu titik waktu $t$, melainkan rangkaian temporal (*temporal dependency*):
$$\text{WTD}_t, \text{WTD}_{t-1}, \dots, \text{WTD}_{t-n} \qquad \text{Rain}_t, \dots, \text{Rain}_{t-n} \qquad \text{VPD}_t, \dots, \text{VPD}_{t-n}$$

> *"WTD -65 cm hari ini belum tentu berbahaya. Tetapi WTD < -60 cm selama 12 hari berturut-turut + tidak ada rainfall signifikan + VPD tinggi menghasilkan risiko yang jauh lebih tinggi."*

### C. AI Menemukan Interaksi Non-Linear Antar Variabel
Model sederhana hanya memetakan relasi linear tunggal: $\text{FireRisk} = f(\text{WTD})$.
AI Nimbus memetakan interaksi multi-variabel non-linear:
$$\text{FireRisk} = f(\text{WTD}, \text{VPD}, \text{Rainfall Deficit}, \text{Solar Radiation}, \text{Dry Spell Duration}, \text{Drawdown Trend}, \dots)$$

- $\text{WTD} -60\text{ cm} + \text{VPD rendah} \rightarrow \text{Moderate}$
- $\text{WTD} -60\text{ cm} + \text{VPD tinggi} + 12\text{ hari tanpa hujan} \rightarrow \text{High}$
- $\text{WTD} -60\text{ cm} + \text{VPD tinggi} + 20\text{ hari tanpa hujan} + \text{tren WTD anjlok} \rightarrow \text{Very High}$

*(Sesuai riset tropical peatland mutakhir menggunakan machine learning, temporal recurrent lag, dan kalibrasi probabilistik).*

### D. AI Belajar dari Kesalahan Prediksi (Feedback Loop)
Inti dari pembentukan data moat adalah *learning from error*:
- **Kasus False Alarm**: Prediksi risiko 82%, namun tidak terjadi kebakaran $\rightarrow$ AI menyimpan error dan menyesuaikan sensitivitas threshold.
- **Kasus Missed Detection**: Prediksi risiko 35%, namun terjadi kebakaran $\rightarrow$ AI mengidentifikasi variabel pemicu tersembunyi.
$$\text{Data} \rightarrow \text{Prediction} \rightarrow \text{Actual Outcome} \rightarrow \text{Error Analysis} \rightarrow \text{Retraining} \rightarrow \text{Better Prediction}$$

### E. Fire Labeling & Ground Truth
Nimbus mengintegrasikan telemetri lingkungan dengan label *ground truth*:
- IoT AWS + TMAT
- NASA FIRMS Hotspot (MODIS / VIIRS)
- Citra satelit burned area & satellite soil moisture
- Sejarah kejadian api dan jaringan kanal/drainase

Dataset Nimbus:
| Date | WTD (cm) | Temp (°C) | RH (%) | VPD (kPa) | Rain (mm) | Solar (W/m²) | Fire Label |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D1 | -45.0 | 30.0 | 80.0 | 0.85 | 10.0 | 300 | 0 |
| D2 | -50.0 | 31.0 | 75.0 | 1.10 | 0.0 | 450 | 0 |
| D3 | -58.0 | 32.0 | 70.0 | 1.60 | 0.0 | 500 | 0 |
| D4 | -67.0 | 33.0 | 65.0 | 2.05 | 0.0 | 550 | 1 |

---

## 🏛️ 3. Nimbus Fire Intelligence Engine (5 Arsitektur Layer)

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5 — Continuous Learning & MLOps Feedback Loop         │
│ (Prediction vs Actual Fire Outcome → Auto Retrain 01:00 AM) │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│ Layer 4 — Multi-Horizon Forecasting Engine                  │
│ (Recursive Projection: 1-Day, 3-Day, 7-Day, 10-Day Horizons)│
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│ Layer 3 — AI Predictive Models (Machine Learning)           │
│ (Temporal Lags, Drawdown Rates, VPD Interactions, Random F.)│
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│ Layer 2 — Site-Specific Hydrological Calibration            │
│ (Local WTD Baselines, Anomaly Detection, Peat Water Storage)│
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│ Layer 1 — Scientific Baseline Index (Dokumen 1 Pseudo-FWI)  │
│ Pseudo-FWI = 0.35*Temp + 0.35*RH + 0.20*Solar + 0.10*Rain   │
└─────────────────────────────────────────────────────────────┘
```

1. **Layer 1 — Scientific Index**: Pseudo-FWI deterministik Dokumen 1 sebagai fondasi ilmiah standar.
2. **Layer 2 — Local Calibration**: Kalibrasi ambang batas hidrologis lokal berdasarkan riwayat sensor AWS dan TMAT setempat.
3. **Layer 3 — AI Prediction**: Model Machine Learning terlatih memetakan interaksi multi-variabel untuk menghitung probabilitas kebakaran, kekeringan, dan hujan.
4. **Layer 4 — Forecast**: Peramalan multi-bencana jangka pendek hingga menengah (1 hari, 3 hari, 7 hari, 10 hari).
5. **Layer 5 — Continuous Learning**: Evaluasi berkala terhadap kejadian lapangan dan retraining otomatis (*daily scheduled & hot reload*).

---

## 🔄 4. Data Flywheel & 4 Dimensi Data Moat

### Data Flywheel Nimbus:
```
                 LEBIH BANYAK WAKTU
                         ↓
              LEBIH BANYAK OBSERVASI
                         ↓
              LEBIH BANYAK KASUS API
                         ↓
             LEBIH BANYAK TRAINING DATA
                         ↓
             KALIBRASI LOKASI LEBIH BAIK
                         ↓
              PREDIKSI RISIKO LEBIH AKURAT
                         ↓
             RETENSI PELANGGAN MENINGKAT
                         ↓
            PENYEBARAN SENSOR LEBIH BANYAK
                         ↓
              LEBIH BANYAK DATA PROPRIETARY
                         ↺
```

### 4 Dimensi Nimbus Environmental Data Moat:
1. **Temporal Moat**: Semakin lama sensor beroperasi (1 bulan $\rightarrow$ 1 tahun $\rightarrow$ 3 tahun $\rightarrow$ 5 tahun), AI semakin memahami dinamika musiman dan lag behavior.
2. **Spatial Moat**: Semakin banyak sensor dipasang di berbagai blok lahan gambut, AI memahami variasi retensi air antar lokasi.
3. **Outcome Moat**: Setiap kejadian kebakaran atau kondisi aman terverifikasi menjadi label training bernilai tinggi.
4. **Model Moat**: Data menghasilkan threshold spesifik lokasi, interaksi fitur teruji, dan probabilitas risiko terkalibrasi yang mustahil disaingi oleh kompetitor yang baru memasang sensor hari ini.

$$\boxed{ \text{IoT} + \text{Satellite} + \text{Climate} + \text{Hydrology} + \text{Fire History} \longrightarrow \text{AI} \longrightarrow \text{Predictive Fire Intelligence} }$$

> **Posisi Nimbus**: Pseudo-FWI/PFVI menjadi *scientific foundation*, IoT menyediakan *continuous local observation*, AI mempelajari *local dynamics*, dan Nimbus menghasilkan *site-specific predictive fire intelligence*.

---

## ⚡ 5. Training Menggunakan Seluruh Data Historis (Bukan Sekadar 7 Hari)

Sesuai arsitektur Data Moat Nimbus:
- **Bukan Jendela 7 Hari**: Model AI **tidak** dibatasi oleh jendela pengamatan 7 hari ke belakang.
- **Seluruh Data Historis Dilatih**: Model menyerap **seluruh 17.567 rekaman data (61 hari penuh interval 5 menit)** dari dataset.
- **Prediksi Memanfaatkan Seluruh Pengalaman AI**: Prediksi instan maupun peramalan multi-hari (1, 3, 7, hingga 10 hari ke depan) dieksekusi menggunakan model yang telah memahami seluruh siklus pengeringan (*drawdown*), pengisian (*recharge*), dan respons cuaca historis.

---

## 📡 6. Ringkasan Endpoint REST API

> 🛡️ **Aturan Proteksi `is_peatland: false`**:
> Seluruh endpoint algoritma inferensi & peramalan (`/predict`, `/fwi`, `/drought`, `/forecast`) diproteksi oleh dependency `require_non_peatland`. Akun dengan status lahan gambut (`is_peatland: true`) akan otomatis ditolak dengan status HTTP **`403 Forbidden`**.

| Method | Endpoint | Otorisasi & Syarat Lahan | Deskripsi & Cakupan Data |
| :--- | :--- | :---: | :--- |
| `GET` | `/health` | Publik | Cek status server, kesiapan 3 model AI, dan jadwal auto-retraining |
| `POST` | `/api/v1/predict` | Bearer Token (`is_peatland=false`) | Prediksi real-time telemetri instan (Hujan, Kebakaran, Kekeringan, EWS) |
| `POST` | `/api/v1/telemetry` | Publik (IoT Hardware) | Ingesti telemetri perangkat AWS hardware + deteksi risiko dini |
| `POST` | `/api/v1/predict/batch` | Bearer Token (`is_peatland=false`) | Inferensi batch multi-stasiun / sinkronisasi data massal |
| `POST` | `/api/v1/fwi/calculate` | Bearer Token (`is_peatland=false`) | Kalkulator deterministik Pseudo-FWI Dokumen 1 (Layer 1 Scientific Baseline) |
| `POST` | `/api/v1/drought/calculate` | Bearer Token (`is_peatland=false`) | Kalkulator ilmiah PDWI/MDWI (Google Slides Slide 13 & 14) |
| `POST` | `/api/v1/drought/forecast-7days` | Bearer Token (`is_peatland=false`) | Peramalan kekeringan 7 hari skenario iklim-hidrologi |
| `GET` | `/api/v1/drought/latest-comparison` | Bearer Token (`is_peatland=false`) | Komparasi indeks kekeringan aktual: Gambut vs Mineral |
| `POST` | `/api/v1/peatland/drought/calculate` | Bearer Token (`is_peatland=true`) | PDWI-Gambut dari 1 pembacaan sensor |
| `POST` | `/api/v1/peatland/drought/forecast` | Bearer Token (`is_peatland=true`) | Prakiraan PDWI-Gambut N hari |
| `GET` | `/api/v1/peatland/drought/latest` | Bearer Token (`is_peatland=true`) | Kondisi terakhir + forecast 7 hari + skenario |
| `GET` | `/api/v1/peatland/drought/assumptions` | Bearer Token (`is_peatland=true`) | Katalog asumsi & kalibrasi lokasi |
| `POST` | `/api/v1/peatland/fwi/calculate` | Bearer Token (`is_peatland=true`) | PFWI-Gambut (Peatland Fire Weather Index) dari 1 pembacaan sensor |
| `POST` | `/api/v1/peatland/fwi/forecast` | Bearer Token (`is_peatland=true`) | Prakiraan PFWI-Gambut N hari |
| `GET` | `/api/v1/peatland/fwi/latest` | Bearer Token (`is_peatland=true`) | PFWI terakhir + riwayat + forecast 7 hari + skenario |
| `GET` | `/api/v1/peatland/fwi/assumptions` | Bearer Token (`is_peatland=true`) | Katalog asumsi PFWI, kelas FDRS & kalibrasi |
| `POST` | `/api/v1/forecast/7-days` | Bearer Token (`is_peatland=false`) | Proyeksi 7 hari ke depan berbasis model cerdas full-history |
| `POST` | `/api/v1/forecast/10-days` | Bearer Token (`is_peatland=false`) | Proyeksi 10 hari ke depan berbasis model cerdas full-history |
| `GET` | `/api/v1/forecast/7-days` | Publik | GET Proyeksi 7 hari dari titik data sensor mutakhir dataset |
| `GET` | `/api/v1/forecast/10-days` | Publik | GET Proyeksi 10 hari dari titik data sensor mutakhir dataset |
| `GET` | `/api/v1/dataset/analysis-and-forecast` | Publik | Analisis riwayat data riil & peramalan ke depan berbasis dataset |
| `GET` | `/api/v1/scheduler/status` | Publik | Monitoring status cron scheduler retraining harian (01:00 AM) |
| `POST` | `/api/v1/retrain` | Publik / Admin | Pemicu manual training ulang model di background (hot-reload) |
| `GET` | `/api/v1/metrics` | Publik / Admin | Laporan metrik akurasi, F1-Score, dan bobot fitur model aktif |
| `POST` | `/api/v1/auth/register` | Publik | Registrasi akun baru langsung aktif (opsional `is_peatland: false`) |
| `POST` | `/api/v1/auth/login` | Publik | Login akun & penerbitan Bearer JWT Token |
| `GET` | `/api/v1/auth/me` | Bearer Token | Profil akun pengguna aktif & status tipe lahan (`is_peatland`) |
| `GET/PUT/DELETE` | `/api/v1/superadmin/users` | Bearer (`superadmin`) | Manajemen user & modifikasi status lahan (`is_peatland`) |

---

## 🚀 7. Cara Menjalankan Sistem

### 1. Menjalankan Training Model (Full Historical Dataset)
```bash
source .venv/bin/activate
python -m src.train
```

### 2. Menguji Inferensi & Evaluasi Data Aktual
```bash
source .venv/bin/activate
python test_inference.py
```

### 3. Menjalankan REST API Server
```bash
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI interaktif: **`http://localhost:8000/docs`**
- Dokumentasi API lengkap: [API_DOCUMENTATION.md](file:///Users/vcl/Desktop/python-ai/API_DOCUMENTATION.md)
- Laporan metrik training: [training_report.md](file:///Users/vcl/Desktop/python-ai/training_report.md)
