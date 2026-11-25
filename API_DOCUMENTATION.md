# 📡 Dokumentasi REST API — Nimbus Fire & Environmental Intelligence Engine

API ini menyediakan layanan integrasi inferensi AI *real-time* untuk perangkat **Automatic Weather Station (AWS)** dan sensor **Tinggi Muka Air Tanah (TMAT / WTD)** guna mendeteksi potensi **Hujan**, bahaya **Kebakaran Lahan Gambut**, dan **Kekeringan**, dilengkapi dengan sistem peringatan dini (*Early Warning System / EWS*).

Sistem ini didukung oleh **Nimbus Data Moat & 5-Layer Fire Intelligence Engine**:
- **Layer 1**: Scientific Baseline Index (Pseudo-FWI Dokumen 1).
- **Layer 2**: Site-Specific Hydrological Calibration (WTD Anomaly, Water Retention).
- **Layer 3**: AI Predictive Machine Learning (Temporal Lags, Interactions, Drawdown Trend).
- **Layer 4**: Multi-Horizon Forecasting (1-day, 3-day, 7-day, 10-day Projections).
- **Layer 5**: Continuous Learning & Automated Retraining Feedback Loop (01:00 AM Daily).

> **PENTING**: Model AI **tidak dibatasi oleh jendela sliding 7 hari**. Seluruh **17.567 data telemetri historis (61 hari penuh interval 5 menit)** dilatih ke dalam model sehingga AI memahami seluruh siklus pengeringan (*drawdown*), pengisian (*recharge*), efek jeda (*lag effects*), dan respons hidrologis gambut secara mendalam.

---

## 📌 1. Informasi Umum

- **Base URL (Lokal)**: `http://localhost:8000`
- **Content-Type**: `application/json`
- **Dokumentasi Interaktif (Swagger UI)**: `http://localhost:8000/docs`
- **Dokumentasi Alternatif (ReDoc)**: `http://localhost:8000/redoc`

---

## 📖 2. Kamus Data Parameter Sensor (Payload Input)

API mendukung format penamaan bahasa Inggris (*snake_case*) maupun alias bahasa Indonesia:

| Parameter (Standard) | Parameter (Alias ID) | Tipe Data | Satuan | Deskripsi / Contoh |
| :--- | :--- | :---: | :---: | :--- |
| `water_table_depth_cm` | `Tinggi Muka Air Tanah (cm)` | `float` | cm | Kedalaman air tanah (misal: `-45.5`) |
| `soil_moisture_pct` | `Kelembaban Tanah (%)` | `float` | % | Kelembaban tanah (0.0 - 100.0) |
| `air_temperature_c` | `Suhu Udara (°C)` | `float` | °C | Suhu udara (misal: `30.5`) |
| `relative_humidity_pct`| `Kelembapan Udara (%)` | `float` | % | Kelembapan relatif udara (0 - 100) |
| `rainfall_intensity_mmh`| `Intensitas Hujan (mm/jam)` | `float` | mm/jam | Intensitas curah hujan (default: `0.0`) |
| `solar_radiation_wm2` | `Radiasi Matahari (W/m²)` | `float` | W/m² | Radiasi matahari (default: `0.0`) |
| `dew_point_c` | `Suhu Titik Embun (°C)` | `float` | °C | Suhu titik embun (opsional, auto dihitung jika kosong) |
| `air_pressure_hpa` | `Tekanan Udara (hPa)` | `float` | hPa | Tekanan udara (default: `985.0`) |
| `wind_speed_ms` | `Kecepatan Angin (m/s)` | `float` | m/s | Kecepatan angin (default: `0.0`) |
| `wind_direction_deg` | `Arah Angin (°)` | `float` | derajat | Arah mata angin 0 - 360° |
| `latitude` | `lat` | `float` | deg | Koordinat lintang stasiun AWS |
| `longitude` | `lon` | `float` | deg | Koordinat bujur stasiun AWS |
| `timestamp` | `waktu` | `string` | ISO/WIB | Waktu pencatatan sensor (opsional) |

---

## 🚀 3. Daftar Lengkap Endpoint REST API

### 1. Cek Status Server & Model AI (Health Check)
Memeriksa kesiapan memori, model AI yang aktif, dan jadwal retraining.

- **Method**: `GET`
- **Endpoint**: `/health`

#### Response (`200 OK`):
```json
{
  "status": "healthy",
  "models_loaded": true,
  "scheduler_running": true,
  "next_auto_retrain": "2026-10-06 01:00:00+07:00",
  "models_available": {
    "rain_model": true,
    "fire_risk_model": true,
    "drought_risk_model": true
  },
  "timestamp": "2026-10-05T19:50:00.123456"
}
```

---

### 2. Prediksi Real-Time Bencana & Proyeksi Multi-Horizon (Single Record)
Menerima satu data telemetri dari stasiun AWS dan menghasilkan prediksi instan beserta proyeksi peramalan ke depan (7-day dan 10-day) yang didukung oleh model terlatih dari **seluruh data historis**.

- **Method**: `POST`
- **Endpoint**: `/api/v1/predict`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**.
  - Jika akun pengguna memiliki status `is_peatland: true`, server menolak akses dengan status **`403 Forbidden`**:
    ```json
    {
      "detail": "Akses Ditolak: Algoritma FWI, Drought, dan Forecast tidak tersedia untuk akun dengan tipe lahan gambut (is_peatland=True). Fitur ini hanya diperuntukkan bagi lahan non-gambut."
    }
    ```

> 💡 **Transmisi IoT Hardware (`/api/v1/telemetry`)**:
> Untuk transmisi otomatis langsung dari mikrokontroler/hardware AWS di lapangan tanpa token autentikasi user, gunakan endpoint `POST /api/v1/telemetry`.

#### Request Body (`JSON`):
```json
{
  "timestamp": "2026-07-20 13:00:00",
  "water_table_depth_cm": -138.5,
  "soil_moisture_pct": 80.2,
  "rainfall_intensity_mmh": 0.0,
  "air_temperature_c": 31.0,
  "dew_point_c": 18.0,
  "relative_humidity_pct": 54.0,
  "solar_radiation_wm2": 690.0,
  "air_pressure_hpa": 984.5,
  "wind_speed_ms": 2.5,
  "wind_direction_deg": 110.0,
  "latitude": -0.0263,
  "longitude": 109.3425
}
```

#### Response (`200 OK`):
```json
{
  "success": true,
  "data": {
    "ringkasan": {
      "tingkat_kebakaran": "Tinggi",
      "kekeringan": "Tinggi",
      "hujan": "Rendah",
      "status_teks": "Tingkat Kebakaran Tinggi, Kekeringan Tinggi, Hujan Rendah"
    },
    "prediksi_7_hari": [
      {
        "hari_ke": 1,
        "tanggal": "2026-07-21",
        "tingkat_kebakaran": "Tinggi",
        "kekeringan": "Tinggi",
        "hujan": "Rendah",
        "pseudo_fwi_score": 0.535
      }
    ],
    "prediksi_10_hari": [
      {
        "hari_ke": 1,
        "tanggal": "2026-07-21",
        "tingkat_kebakaran": "Tinggi",
        "kekeringan": "Tinggi",
        "hujan": "Rendah",
        "ringkasan": {
          "tingkat_kebakaran": "Tinggi",
          "kekeringan": "Tinggi",
          "hujan": "Rendah",
          "status_teks": "Tingkat Kebakaran Tinggi, Kekeringan Tinggi, Hujan Rendah"
        },
        "pseudo_fwi_score": 0.535,
        "fwi_baseline_class": 2,
        "ai_effective_class": 2,
        "detail_cuaca": {
          "suhu_c": 31.1,
          "kelembapan_persen": 56.8,
          "radiasi_wm2": 725.0,
          "curah_hujan_mm": 0.0,
          "probabilitas_hujan": 0.12,
          "tmat_cm": -144.0,
          "kelembaban_tanah_persen": 79.5,
          "vpd_kpa": 1.95,
          "dry_spell_days": 1
        }
      }
      /* ... hingga Hari 10 ... */
    ],
    "predictions": {
      "rain": {
        "class": 0,
        "label": "Tidak Hujan (No Rain)",
        "level": "Rendah",
        "probability_rain": 0.075
      },
      "fire_risk": {
        "class": 2,
        "label": "Tinggi / Bahaya (High / Danger)",
        "level": "Tinggi",
        "pseudo_fwi_score": 0.535,
        "fwi_baseline_class": 2,
        "ml_model_predicted_class": 2,
        "site_calibrated_class": 2,
        "algorithm": "Nimbus Fire Intelligence Engine (Pseudo-FWI + Site Calibration + Full-History ML)"
      },
      "drought_risk": {
        "class": 2,
        "label": "Kritis (Severe Drought)",
        "level": "Tinggi"
      }
    },
    "early_warning_system": [
      {
        "level": "DANGER",
        "category": "FIRE_HAZARD",
        "title": "🚨 WASPADA KEBAKARAN LAHAN TINGGI (NIMBUS INTELLIGENCE TINGGI)",
        "message": "Indeks Pseudo-FWI tinggi: 0.535. Vegetasi kering dan mudah terbakar. Tingkatkan patroli dan siagakan sekat kanal."
      }
    ]
  }
}
```

---

### 2b. Batch Inference Prediksi Bencana (Multi-Record)
Memproses sekumpulan rekaman sensor telemetri secara massal dalam satu payload array JSON (misal untuk sinkronisasi rekaman berkala atau multi-stasiun AWS).

- **Method**: `POST`
- **Endpoint**: `/api/v1/predict/batch`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**.
  - Jika `is_peatland: true`, server menolak akses dengan HTTP **`403 Forbidden`**.

#### Request Body (`JSON Array`):
```json
[
  {
    "timestamp": "2026-07-20 13:00:00",
    "water_table_depth_cm": -138.5,
    "soil_moisture_pct": 80.2,
    "rainfall_intensity_mmh": 0.0,
    "air_temperature_c": 31.0,
    "relative_humidity_pct": 54.0,
    "solar_radiation_wm2": 690.0,
    "latitude": -0.0263,
    "longitude": 109.3425
  }
]
```

#### Response (`200 OK`):
```json
{
  "success": true,
  "total_records": 1,
  "data": [
    {
      "ringkasan": {
        "tingkat_kebakaran": "Tinggi",
        "kekeringan": "Tinggi",
        "hujan": "Rendah",
        "status_teks": "Tingkat Kebakaran Tinggi, Kekeringan Tinggi, Hujan Rendah"
      },
      "probabilitas_kebakaran": 0.825,
      "early_warning_system": [ ... ]
    }
  ]
}
```

---

### 3. Prediksi Proyeksi 7 Hari ke Depan (7-Day Forecast)
Menghasilkan proyeksi kondisi harian selama 7 hari ke depan dengan model yang telah dilatih dari **seluruh data historis**:
- **Tingkat Kebakaran**: `Rendah` | `Sedang` | `Tinggi` | `Ekstrim`
- **Kekeringan**: `Rendah` | `Sedang` | `Tinggi` | `Ekstrim`
- **Potensi Hujan**: `Rendah` | `Sedang` | `Tinggi` | `Ekstrim`

- **Method**: `POST`
- **Endpoint**: `/api/v1/forecast/7-days`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**. Menghasilkan `403 Forbidden` jika `is_peatland: true`.

#### Request Body (`JSON`):
```json
{
  "water_table_depth_cm": -120.0,
  "soil_moisture_pct": 82.0,
  "air_temperature_c": 31.5,
  "relative_humidity_pct": 55.0,
  "solar_radiation_wm2": 670.0,
  "rainfall_intensity_mmh": 0.0
}
```

#### Response (`200 OK`):
```json
{
  "success": true,
  "training_scope": "Full Historical Dataset (All observations used for model training)",
  "ringkasan_hari_ini": {
    "tingkat_kebakaran": "Tinggi",
    "kekeringan": "Tinggi",
    "hujan": "Rendah",
    "status_teks": "Tingkat Kebakaran Tinggi, Kekeringan Tinggi, Hujan Rendah"
  },
  "prediksi_7_hari": [
    {
      "hari_ke": 1,
      "tanggal": "2026-10-06",
      "tingkat_kebakaran": "Tinggi",
      "kekeringan": "Tinggi",
      "hujan": "Rendah",
      "ringkasan": {
        "tingkat_kebakaran": "Tinggi",
        "kekeringan": "Tinggi",
        "hujan": "Rendah",
        "status_teks": "Tingkat Kebakaran Tinggi, Kekeringan Tinggi, Hujan Rendah"
      },
      "pseudo_fwi_score": 0.525,
      "detail_cuaca": {
        "suhu_c": 31.2,
        "kelembapan_persen": 57.0,
        "radiasi_wm2": 710.0,
        "curah_hujan_mm": 0.0,
        "probabilitas_hujan": 0.12,
        "tmat_cm": -128.5,
        "kelembaban_tanah_persen": 81.0,
        "vpd_kpa": 1.92,
        "dry_spell_days": 1
      }
    }
    /* ... data berlanjut sampai Hari 7 ... */
  ]
}
```

---

### 4. GET Proyeksi 7 Hari dari Titik Mutakhir Dataset
Mengambil status paling mutakhir pada file data dan langsung menghasilkan ramalan 7 hari ke depan.

- **Method**: `GET`
- **Endpoint**: `/api/v1/forecast/7-days`

#### Response (`200 OK`):
```json
{
  "success": true,
  "training_scope": "Full Historical Dataset (All 17,567 observations and 61 full days used for model training)",
  "titik_sensor_mutakhir": {
    "waktu": "2026-07-31 23:55:00",
    "suhu_c": 21.0,
    "rh_persen": 88.0,
    "tmat_cm": -161.52
  },
  "prediksi_7_hari": [ ... ]
}
```

---

### 5. Prediksi Proyeksi 10 Hari ke Depan (10-Day Forecast)
Menghasilkan proyeksi status kondisi lingkungan selama 10 hari ke depan secara lengkap:

- **Method**: `POST`
- **Endpoint**: `/api/v1/forecast/10-days`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**. Menghasilkan `403 Forbidden` jika `is_peatland: true`.

#### Response (`200 OK`):
```json
{
  "success": true,
  "training_scope": "Full Historical Dataset (All observations used for model training)",
  "ringkasan_hari_ini": {
    "tingkat_kebakaran": "Tinggi",
    "kekeringan": "Tinggi",
    "hujan": "Rendah",
    "status_teks": "Tingkat Kebakaran Tinggi, Kekeringan Tinggi, Hujan Rendah"
  },
  "prediksi_10_hari": [ ... ]
}
```

---

### 6. GET Proyeksi 10 Hari dari Titik Mutakhir Dataset
- **Method**: `GET`
- **Endpoint**: `/api/v1/forecast/10-days`

---

### 7. Analisis Riwayat Dataset Aktual & Peramalan Masa Depan (Full History Training)
Menganalisis riwayat data riil langsung dari dataset Excel (`Data Dummy AWS dan TMAT.xlsx`) di mana model AI dilatih menggunakan **seluruh rekaman data historis (17.567 rekaman / 61 hari penuh)** tanpa dibatasi jendela 7 hari.

Menyajikan:
- Riwayat kondisi harian dari dataset aktual.
- Titik mutakhir sensor pada dataset.
- Proyeksi 7 hari ke depan (`prediksi_7_hari_kedepan`).
- Proyeksi 10 hari ke depan (`prediksi_10_hari_kedepan`).
- Proyeksi milestone Layer 4 (1 hari, 3 hari, 7 hari, 10 hari ke depan).

- **Method**: `GET`
- **Endpoint**: `/api/v1/dataset/analysis-and-forecast` *(atau alias: `/api/v1/dataset/7-days`)*
- **Query Params**: `days` (*Opsional*, integer. Jika kosong, default mengevaluasi seluruh hari pada dataset).

#### Response (`200 OK`):
```json
{
  "success": true,
  "data": {
    "dataset_info": {
      "total_records": 17567,
      "start_time": "2026-06-01 00:05:00",
      "end_time": "2026-07-31 23:55:00",
      "training_scope": "Full Historical Dataset (All 17,567 observations and 61 full days used for model training)",
      "days_analyzed": 61,
      "engine_layers": "Nimbus Fire Intelligence Engine (5 Layers: Scientific Baseline + Site Calibration + ML Prediction + Multi-Horizon Forecast + Continuous Learning)"
    },
    "riwayat_kebelakang": [
      {
        "tanggal": "2026-07-25",
        "tingkat_kebakaran": "Sedang",
        "kekeringan": "Ekstrim",
        "hujan": "Tinggi",
        "ringkasan": {
          "tingkat_kebakaran": "Sedang",
          "kekeringan": "Ekstrim",
          "hujan": "Tinggi",
          "status_teks": "Tingkat Kebakaran Sedang, Kekeringan Ekstrim, Hujan Tinggi"
        },
        "pseudo_fwi_score": 0.405,
        "detail_sensor_harian": {
          "suhu_maks_c": 30.0,
          "kelembapan_min_persen": 57.0,
          "radiasi_maks_wm2": 567.0,
          "total_hujan_mm": 18.4,
          "rata_rata_tmat_cm": -137.5,
          "rata_rata_kelembaban_tanah_persen": 97.2
        }
      }
    ],
    "kondisi_terakhir": {
      "waktu": "2026-07-31 23:55:00",
      "suhu_c": 21.0,
      "rh_persen": 88.0,
      "radiasi_wm2": 0.0,
      "tmat_cm": -161.52,
      "soil_moisture_pct": 96.5,
      "hujan_mmh": 0.0,
      "ringkasan": {
        "tingkat_kebakaran": "Rendah",
        "kekeringan": "Ekstrim",
        "hujan": "Rendah",
        "status_teks": "Tingkat Kebakaran Rendah, Kekeringan Ekstrim, Hujan Rendah"
      },
      "pseudo_fwi": 0.1
    },
    "prediksi_7_hari_kedepan": [ ... ],
    "prediksi_10_hari_kedepan": [ ... ],
    "proyeksi_milestone_layer4": {
      "1_day_ahead": { "tingkat_kebakaran": "Rendah", "kekeringan": "Ekstrim", "hujan": "Ekstrim" },
      "3_days_ahead": { "tingkat_kebakaran": "Tinggi", "kekeringan": "Ekstrim", "hujan": "Ekstrim" },
      "7_days_ahead": { "tingkat_kebakaran": "Tinggi", "kekeringan": "Ekstrim", "hujan": "Ekstrim" },
      "10_days_ahead": { "tingkat_kebakaran": "Tinggi", "kekeringan": "Ekstrim", "hujan": "Ekstrim" }
    }
  }
}
```

---

### 7b. Deteksi & Analisis Model AI Seluruh Hari Dataset (Full History)
Endpoint khusus untuk menjalankan inferensi **Model AI (Rain Model, Fire Risk Model, Drought Risk Model, dan Forecaster)** pada **SETIAP HARI di seluruh data historis (61 hari penuh)**:
- Melakukan deteksi inferensi AI harian lengkap dengan status risiko kebakaran, kekeringan, hujan, status TMAT, dan EWS.
- Menyajikan rekapitulasi statistik & distribusi bahaya di seluruh dataset.
- Menyoroti kejadian ekstrem (puncak kebakaran, kekeringan terparah, dan hujan terbanyak).
- Menyajikan analisis dinamika air tanah (total laju drawdown) dan proyeksi peramalan ke depan.

- **Method**: `GET`
- **Endpoint**: `/api/v1/dataset/all-days` *(atau alias: `/api/v1/dataset/full-history`)*
- **Header**: `Authorization: Bearer <access_token>`
- **Query Params**: `days` (*Opsional*, default kosong = memproses seluruh 61 hari dataset).

#### Response (`200 OK`):
```json
{
  "status": "success",
  "dataset_scope": {
    "total_catatan_sensor": 17567,
    "total_hari_dianalisis": 61,
    "rentang_waktu_dataset": "2026-06-01 00:05:00 s/d 2026-07-31 23:55:00",
    "metode_training": "Trained on Full Historical Dataset (17,567 observations across 61 full days)",
    "ai_models_used": [
      "Rain Prediction Classifier (CatBoost/RandomForest)",
      "Peatland Fire Risk Classifier (XGBoost/LightGBM)",
      "Drought Hazard Classifier (RandomForest)",
      "Peatland Trained Forecaster (Multi-Horizon Hydrological Engine)"
    ]
  },
  "analisis_statistik_keseluruhan": {
    "distribusi_risiko_kebakaran_hari": { "Rendah": 9, "Sedang": 49, "Tinggi": 3, "Ekstrim": 0 },
    "distribusi_risiko_kekeringan_hari": { "Normal": 24, "Waspada": 30, "Kritis": 7 },
    "distribusi_potensi_hujan_hari": { "Tidak Hujan": 36, "Berpotensi Hujan": 25 },
    "kejadian_ekstrem_tercatat": {
      "hari_kebakaran_tertinggi": {
        "tanggal": "2026-07-01",
        "score": 0.535,
        "level": "Tinggi (High)",
        "suhu_maks": 28.0,
        "rh_min": 61.0
      },
      "hari_kekeringan_terparah": {
        "tanggal": "2026-07-09",
        "tmat": -163.07,
        "tmat_terendah": -164.83,
        "level": "Waspada (Moderate Drought)"
      },
      "hari_curah_hujan_terbanyak": {
        "tanggal": "2026-07-10",
        "total_mm": 110.65
      }
    },
    "dinamika_air_tanah": {
      "tmat_awal_periode_cm": -7.33,
      "tmat_akhir_periode_cm": -159.37,
      "perubahan_tmat_total_cm": -152.04,
      "tren_gambut": "Penurunan Muka Air Tanah Signifikan (Kekeringan Parah)"
    }
  },
  "deteksi_analisis_seluruh_hari": [
    {
      "hari_ke": 1,
      "tanggal": "2026-06-01",
      "deteksi_model_ai": {
        "kebakaran_lahan": { "class": 0, "label": "Rendah (Low)", "level": "Rendah", "pseudo_fwi_score": 0.088 },
        "kekeringan_gambut": { "class": 0, "label": "Normal", "level": "Normal" },
        "potensi_hujan": { "class": 1, "label": "Berpotensi Hujan (Rain Expected)", "probability_rain": 0.94 },
        "tinggi_muka_air_tanah": { "rata2_depth_cm": -7.33, "terdalam_cm": -8.1, "status": "Aman" }
      },
      "ringkasan_formal": {
        "status_teks": "Tingkat Kebakaran Rendah, Kekeringan Rendah, Hujan Tinggi"
      },
      "early_warning_alerts": [ ... ],
      "parameter_sensor_harian": {
        "suhu_maks_c": 27.8,
        "suhu_rata2_c": 23.5,
        "kelembapan_min_persen": 75.0,
        "radiasi_maks_wm2": 520.0,
        "total_curah_hujan_mm": 24.5,
        "rata2_tmat_cm": -7.33,
        "tmat_terdalam_cm": -8.1,
        "rata2_kelembaban_tanah_persen": 98.4,
        "kecepatan_angin_maks_ms": 2.1
      }
    }
  ],
  "proyeksi_peramalan_ke_depan": {
    "prediksi_7_hari": [ ... ],
    "prediksi_10_hari": [ ... ],
    "milestone_layer4": {
      "1_hari_kedepan": { "status_teks": "..." },
      "3_hari_kedepan": { "status_teks": "..." },
      "7_hari_kedepan": { "status_teks": "..." },
      "10_hari_kedepan": { "status_teks": "..." }
    }
  }
}
```

---

### 8. Kalkulator Deterministik Pseudo Fire Weather Index (Dokumen 1)
Endpoint khusus untuk menghitung indeks bahaya kebakaran lahan terbuka murni secara deterministik berdasarkan formula **Pseudo-FWI (Dokumen 1)** (Layer 1 Scientific Baseline).

$$\text{Pseudo-FWI} = (0.35 \times \text{Suhu}) + (0.35 \times \text{RH}) + (0.20 \times \text{Radiasi}) + (0.10 \times \text{Hujan})$$

- **Method**: `POST`
- **Endpoint**: `/api/v1/fwi/calculate`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**.
  - Jika akun pengguna memiliki status `is_peatland: true`, server merespons **`403 Forbidden`**.

#### Request Body (`JSON`):
```json
{
  "air_temperature_c": 33.5,
  "relative_humidity_pct": 48.0,
  "solar_radiation_wm2": 720.0,
  "rainfall_intensity_mmh": 0.0
}
```

#### Response (`200 OK`):
```json
{
  "success": true,
  "data": {
    "pseudo_fwi_score": 0.635,
    "risk_class": 2,
    "risk_label": "Tinggi (High)",
    "sub_scores": {
      "temperature_score": 0.6,
      "humidity_score": 0.6,
      "solar_radiation_score": 0.6,
      "rainfall_score": 1.0
    },
    "weights": {
      "temperature": 0.35,
      "humidity": 0.35,
      "solar_radiation": 0.2,
      "rainfall": 0.1
    },
    "algorithm_reference": "Document 1: Pseudo Fire Weather Index (FWI) adapted from Van Wagner (1987)"
  }
}
```

---

### 8b. Kalkulator Ilmiah Drought Warning Index (PDWI & MDWI — Google Slides)
Endpoint untuk menghitung indeks kekeringan ilmiah bertingkat (*Peatland & Mineral Drought Warning Index*) berdasarkan slide: **'Algoritma Prediksi kebakaran lahan gambut'** (Google Slides).

Mendukung dua tipe lahan:
1. **Lahan Gambut (`land_type="gambut"`)**:
   - Level 1 Sensor-Based: $\text{PDWI}^{L1} = 0.60 H_t + 0.25 A_{VPD} + 0.15 A_{Solar}$
   - Level 2 Climate-Hydrology: $\text{PDWI}^{L2} = 0.50 H_t + 0.20 A_{VPD} + 0.10 A_{Solar} + 0.20 RD_t$
   - Ambang batas: 0–20 (NORMAL), 20–40 (WATCH), 40–60 (MODERATE), 60–80 (HIGH), 80–100 (SEVERE)
2. **Lahan Biasa / Mineral (`land_type="lahan_biasa"`)**:
   - Menyesuaikan karakteristik tanah mineral (tidak terbakar di bawah tanah, TMAT lebih dalam, kelembaban zona perakaran dan kekeringan atmosfer/serasah lebih dominan).
   - $\text{MDWI}^{L1} = 0.35 H_t + 0.40 A_{VPD} + 0.25 A_{Solar}$
   - $\text{MDWI}^{L2} = 0.30 H_t + 0.30 A_{VPD} + 0.15 A_{Solar} + 0.25 RD_t$

- **Method**: `POST`
- **Endpoint**: `/api/v1/drought/calculate`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**. Menghasilkan `403 Forbidden` jika `is_peatland: true`.

#### Request Body (`JSON`):
```json
{
  "water_table_depth_cm": -159.37,
  "air_temperature_c": 29.2,
  "relative_humidity_pct": 75.0,
  "solar_radiation_wm2": 320.0,
  "rainfall_intensity_mmh": 0.5,
  "rain_7d_mm": 23.2,
  "soil_moisture_pct": 82.0,
  "land_type": "gambut"
}
```

#### Response (`200 OK`):
```json
{
  "success": true,
  "tipe_lahan": "gambut",
  "tmat_input_cm": -159.37,
  "wtd_konversi_cm": 159.37,
  "vpd_kpa": 1.013,
  "pet_proxy_mm": 7.14,
  "level_1_sensor_based": {
    "index_name": "PDWI-L1 (Peatland Drought Warning Index Level 1)",
    "score": 69.18,
    "drought_class": 3,
    "drought_label": "HIGH"
  },
  "level_2_climate_hydrology": {
    "index_name": "PDWI-L2 (Peatland Climate-Hydrology Drought Warning Index)",
    "score": 64.8,
    "drought_class": 3,
    "drought_label": "HIGH"
  },
  "interpretasi_ilmiah": {
    "prinsip_arsitektur": "Sensor -> PDWI -> Hydrological Forecast -> Fire Vulnerability -> AI Fire Probability",
    "catatan": "PDWI != Fire Probability. Drought adalah driver hidrologis; Fire risk adalah outcome kebakaran."
  }
}
```

---

### 8c. Proyeksi 7-Hari Skenario Iklim-Hidrologi (Slide 16 & 17)
Melakukan proyeksi peramalan kekeringan 7 hari ke depan secara berurutan:
$$\text{Climate Scenario} \longrightarrow \Delta \text{WTD} = \beta_0 + \beta_1 P_t + \beta_2 \text{PET}_t \longrightarrow \text{WTD}_{t+1} \longrightarrow \text{PDWI / MDWI Forecast}$$

- **Method**: `POST`
- **Endpoint**: `/api/v1/drought/forecast-7days`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**. Menghasilkan `403 Forbidden` jika `is_peatland: true`.

---

### 8d. Evaluasi & Komparasi Mutakhir: Gambut vs Lahan Biasa
Membandingkan status kekeringan aktual dataset secara berdampingan antara formulasi Lahan Gambut (PDWI) vs Lahan Biasa (MDWI).

- **Method**: `GET`
- **Endpoint**: `/api/v1/drought/latest-comparison`
- **Header**: `Authorization: Bearer <access_token>`
- **Batasan Akses (`is_peatland`)**:
  - **HANYA untuk akun non-gambut (`is_peatland: false`)**. Menghasilkan `403 Forbidden` jika `is_peatland: true`.

---

### 9. Trigger Manual Retraining di Background
Memaksa sistem melatih ulang model secara instan tanpa menunggu jam 01:00 pagi (*zero-downtime hot reload*).

- **Method**: `POST`
- **Endpoint**: `/api/v1/retrain`

#### Response (`200 OK`):
```json
{
  "status": "retraining_initiated",
  "message": "Model retraining job has been triggered manually in the background. Inference remains active.",
  "started_at": "2026-10-05 19:55:30"
}
```

---

### 10. Status Scheduler Retraining Otomatis
- **Method**: `GET`
- **Endpoint**: `/api/v1/scheduler/status`

#### Response (`200 OK`):
```json
{
  "scheduler_active": true,
  "job_id": "daily_retrain_job",
  "job_name": "Daily Model Retraining at 01:00 AM",
  "schedule_cron": "01:00 AM every day",
  "next_execution": "2026-10-06 01:00:00+07:00"
}
```

---

### 11. Cek Laporan Metrik Akurasi Model
- **Method**: `GET`
- **Endpoint**: `/api/v1/metrics`

---

### 12. PDWI-Gambut (khusus akun `is_peatland=true`)
Akses: Bearer Token dengan `is_peatland=true` (atau `superadmin`). Akun `is_peatland=false` → **403**.
Endpoint kekeringan/FWI lama sebaliknya hanya untuk `is_peatland=false`.

| Method | Endpoint | Fungsi |
|---|---|---|
| `POST` | `/api/v1/peatland/drought/calculate` | PDWI-Gambut dari 1 pembacaan sensor |
| `POST` | `/api/v1/peatland/drought/forecast` | Prakiraan PDWI N hari (`mode`: persistence/kering/basah atau `scenarios`) |
| `GET` | `/api/v1/peatland/drought/latest` | Kondisi terakhir dataset + riwayat + forecast 7 hari + skenario + peringatan dini |
| `GET` | `/api/v1/peatland/drought/assumptions` | Katalog asumsi parameter + ringkasan kalibrasi lokasi |

**Input wajib**: `tmat_cm` (negatif = di bawah permukaan), `temp_c`, `rh_pct`, `solar_wm2`.
**Opsional**: `wind_ms`, `rain_7d_mm`, `tmat_7d_ago_cm`, `dry_days`, `tanggal`. Komponen yang tidak ada dibuang dan bobot dinormalisasi ulang; semua asumsi dilaporkan di `asumsi_diterapkan`.

**Urutan rumus (implementasi: `src/peatland_pdwi.py`)**

| Langkah | Rumus |
|---|---|
| S1 | $WTD = -TMAT$ (cm, dibatasi ≥ 0) |
| S2 | Agregasi 5 menit → harian |
| S3 | $VPD = e_s(T)\,(1-RH/100)$, $e_s = 0.6108\,e^{17.27T/(T+237.3)}$ |
| S4 | $PET = K_c \cdot ET_0$ (FAO-56 Penman-Monteith) |
| S5 | $H_t = (1-\lambda_h)H_{abs} + \lambda_h H_{pct}$; anchor 0→0, 10→5, 40→40, 80→70, 120→90, 160→100 |
| S6 | $A_t = 0.70\,A_{VPD} + 0.30\,A_{Solar}$ |
| S7 | $RD = 100\cdot\max(0, 1 - P_7/P_{7,ref})$ |
| S8 | $L1 = 0.60H + 0.25A_{VPD} + 0.15A_{Solar}$ |
| S9 | $L2 = 0.50H + 0.20A_{VPD} + 0.10A_{Solar} + 0.20RD$ |
| S10 | Tren $\Delta WTD_7$ (Stabil / Pembasahan / Pengeringan / Pengeringan Cepat ≥ 20 cm/minggu) |
| S11 | $\Delta WTD = b_0 + b_1(P - f\cdot PET) + b_3 WTD$ (dikalibrasi `lsq_linear` dengan batas fisik) |
| S12 | Forecast: cuaca(t) → ΔWTD → WTD(t+1) → $H_t$ → PDWI |

Kelas: NORMAL / WATCH / MODERATE / HIGH / SEVERE.

**Hasil pada data simulasi** (61 hari, 31‑Jul‑2026, WTD 159.37 cm): L1 = 69.4 (HIGH), L2 = 65.5 (HIGH). MAE ΔWTD 1 hari 3.4 cm; MAE level WTD 7 hari 8.9 cm (cuaca aktual) vs 21.7 cm persistence.

**Contoh request** `POST /api/v1/peatland/drought/calculate`:
```json
{"tmat_cm": -159.37, "temp_c": 27.0, "rh_pct": 80.0, "solar_wm2": 150.0, "rain_7d_mm": 5.0}
```

> [!WARNING]
> Keterbatasan: baseline hanya satu siklus musim kemarau (61 hari), sehingga ambang regulasi (PP 57/2016, 40 cm) dominan. Prakiraan di atas WTD terkalibrasi (maks ±163 cm) adalah ekstrapolasi dan diberi flag `di_luar_rentang_kalibrasi`. Parameter yang diasumsikan (elevasi 5 m, angin 1 m/s, albedo 0.23, Kc 1.0, Sy 0.08–0.60, dll.) dapat dilihat di endpoint `assumptions`.

---

### 13. PFWI-Gambut — Peatland Fire Weather Index (khusus akun `is_peatland=true`)
Akses sama dengan PDWI-Gambut (`require_peatland`): hanya untuk akun `is_peatland=true` atau `superadmin` (akun `is_peatland=false` ditolak dengan **403 Forbidden**). Pseudo-FWI Dokumen 1 (`/api/v1/fwi/calculate`) sebaliknya tetap hanya untuk `is_peatland=false`.

| Method | Endpoint | Fungsi |
|---|---|---|
| `POST` | `/api/v1/peatland/fwi/calculate` | PFWI dari 1 pembacaan sensor siang (12:00 WIB) |
| `POST` | `/api/v1/peatland/fwi/forecast` | Prakiraan PFWI N hari (`mode` atau `scenarios`, opsional `prev_ffmc/dmc/dc`) |
| `GET` | `/api/v1/peatland/fwi/latest` | Kondisi terakhir dataset + riwayat + forecast 7 hari + skenario kering/basah + peringatan dini |
| `GET` | `/api/v1/peatland/fwi/assumptions` | Katalog asumsi, batas kelas FDRS, kalibrasi lokasi |

**Input wajib (`/calculate`)**: `tmat_cm`, `temp_c`, `rh_pct` (cuaca siang). **Opsional**: `wind_ms`, `rain_24h_mm`, `prev_ffmc`, `prev_dmc`, `prev_dc`, `tmat_7d_ago_cm`, `dry_days`, `tanggal`.
Sistem FWI memiliki **memori harian**: kirim `kode_fwi_hari_ini` dari respons hari ini sebagai `prev_*` esok hari. Tanpa itu dipakai start-up Van Wagner (FFMC 85, DMC 6) dan DC disetarakan dari WTD terukur.

**Urutan rumus (implementasi: [`src/peatland_fwi.py`](file:///Users/vcl/Desktop/python-ai/src/peatland_fwi.py))**

| Langkah | Rumus |
|---|---|
| F1 | Cuaca siang: rata-rata 11:00–13:00 WIB; hujan 24 jam berakhir 12:00 |
| F2 | Angin 10 m: $u_{10} = u_2 \ln(67.8\cdot10-5.42)/4.87$, ×3.6 → km/jam |
| F3–F6 | FFMC, DMC, DC, ISI, BUI, FWI, DSR — Van Wagner (1987); Le ekuator = 9, Lf = 1.4 |
| F7 | Subskor 0–100 dari batas kelas FDRS Indonesia: FFMC 73/78/83, ISI 2/4/6, DC 140/260/350 → 25/50/75 |
| F8 | $S_{peat} = 0.60\,H_t(WTD) + 0.40\,S_{DC}$ ($H_t$ dari engine PDWI-Gambut) |
| F9 | $S_{dry}$: hari kering beruntun 3/7/14/21 → 25/50/75/100 |
| F10 | $PFWI = 0.45\,S_{peat} + 0.30\,S_{ign} + 0.15\,S_{spread} + 0.10\,S_{dry}$ |
| F11 | Eskalasi: R1 WTD≥40 & FFMC≥83 → min. Tinggi; R2 WTD≥80 & DC≥350 & FFMC≥78 → Ekstrim; R3 ΔWTD₇≥20 & WTD≥40 & FFMC≥73 → min. Tinggi |
| F12 | Forecast: WTD dari model neraca air PDWI + kode FWI dilanjutkan dengan cuaca siang skenario (offset siang dikalibrasi dari data lokasi) |

Kelas: Rendah (<25) / Sedang (25–50) / Tinggi (50–75) / Ekstrim (≥75). Respons juga memuat FWI kanonik dan kelas FDRS tiap komponen sebagai pembanding standar BMKG.

#### 13a. Kalkulasi 1 Pembacaan Sensor (`POST /api/v1/peatland/fwi/calculate`)
- **Header**: `Authorization: Bearer <access_token>` (`is_peatland=true` / `superadmin`)
- **Request Body**:
```json
{
  "tmat_cm": -95.0,
  "temp_c": 32.5,
  "rh_pct": 55.0,
  "wind_ms": 2.5,
  "rain_24h_mm": 0.0,
  "prev_ffmc": 86.0,
  "prev_dmc": 25.0,
  "prev_dc": 280.0,
  "tmat_7d_ago_cm": -70.0,
  "dry_days": 9
}
```
- **Response (`200 OK`)**:
```json
{
  "engine": "PFWI-Gambut Nimbus v1",
  "tipe_lahan": "gambut",
  "tanggal": "2026-11-08",
  "input": {
    "tmat_cm": -95.0,
    "suhu_siang_c": 32.5,
    "rh_siang_persen": 55.0,
    "angin_ms": 2.5,
    "hujan_24j_mm": 0.0,
    "kode_kemarin": {"ffmc": 86.0, "dmc": 25.0, "dc": 280.0},
    "tmat_7_hari_lalu_cm": -70.0,
    "hari_kering_beruntun": 9
  },
  "turunan": {"wtd_cm": 95.0, "tergenang": false, "angin_10m_kmh": 12.04},
  "pfwi": {
    "score": 74.32,
    "class": 2,
    "label": "Tinggi (High)",
    "kelas_sebelum_eskalasi": 2,
    "eskalasi_gambut": [],
    "komponen_tidak_tersedia": [],
    "rekomendasi": "Tinggi: api mudah menyala dan gambut dapat ikut terbakar. Patroli harian, siagakan regu & pompa, lakukan pembasahan."
  },
  "subskor": {
    "S_peat": 67.26,
    "H_t_wtd": 73.76,
    "S_DC": 57.51,
    "S_ign_ffmc": 89.76,
    "S_spread_isi": 76.08,
    "S_dry": 57.14
  },
  "sistem_fwi_fdrs": {
    "FFMC": {"nilai": 88.31, "kelas": 3, "label": "Ekstrim / Bahaya (Extreme / Danger)"},
    "DMC": {"nilai": 27.58, "kelas": 1, "label": "Sedang (Moderate)"},
    "DC": {"nilai": 287.05, "kelas": 2, "label": "Tinggi (High)"},
    "ISI": {"nilai": 6.17, "kelas": 3, "label": "Ekstrim / Bahaya (Extreme / Danger)"},
    "BUI": {"nilai": 44.47, "kelas": 2, "label": "Tinggi (High)"},
    "FWI": {"nilai": 14.65, "kelas": 3, "label": "Ekstrim / Bahaya (Extreme / Danger)"},
    "DSR": {"nilai": 3.147}
  },
  "kode_fwi_hari_ini": {"ffmc": 88.31, "dmc": 27.58, "dc": 287.05},
  "tren_7_hari": {"delta_wtd_7_hari_cm": 25.0, "tren": "Pengeringan Cepat", "mengering": true},
  "asumsi_diterapkan": []
}
```

#### 13b. Prakiraan PFWI N Hari (`POST /api/v1/peatland/fwi/forecast`)
- **Request Body**:
```json
{
  "current_tmat_cm": -120.0,
  "mode": "kering",
  "horizon": 7,
  "prev_ffmc": 85.0,
  "prev_dmc": 6.0,
  "prev_dc": 300.0
}
```
*(Bisa juga mengirim `scenarios` berisi array cuaca harian kustom sebagai pengganti `mode`).*
- **Response (`200 OK`)**: Mengembalikan array `forecast` (`D+1` s/d `D+N`) beserta proyeksi `wtd_cm`, `kode_fwi`, `pfwi_skor`, `pfwi_label`, `status_kekeringan_pdwi`, dan ringkasan `peringatan_dini`.

#### 13c. Kondisi Terakhir Dataset & Katalog Asumsi
- **`GET /api/v1/peatland/fwi/latest`**: Mengembalikan kondisi hari terakhir dataset (`keadaan_saat_ini`), `riwayat` 7 hari terakhir, `forecast` 7 hari (`persistence`), `skenario_pembanding` (`kering` vs `basah`), `peringatan_dini`, dan `statistik_dataset`.
- **`GET /api/v1/peatland/fwi/assumptions`**: Mengembalikan katalog lengkap 12 parameter asumsi (`asumsi`), ambang batas `kelas_fdrs`, serta hasil `kalibrasi_lokasi` (offset suhu siang +3.2 °C, RH −10.11 %, faktor angin 2.414×).

> [!NOTE]
> Implementasi Van Wagner tervalidasi terhadap kasus uji standar (FFMC 87.7, DMC 8.5, DC 19.0, ISI 10.9, BUI 8.5, FWI 10.1). Bobot PFWI & aturan eskalasi adalah asumsi ahli yang perlu dikalibrasi dengan data hotspot/kejadian kebakaran riil.

---

## 🔐 4. Autentikasi Tabel Database Supabase & Role-Based Access Control (RBAC)

Sistem autentikasi menggunakan tabel langsung di database **Supabase** (`public.users`) dengan enkripsi password **Bcrypt** dan token **JWT**. **TIDAK MEMERLUKAN KONFIRMASI EMAIL SAMA SEKALI** — akun langsung aktif dan bisa login seketika setelah registrasi.

### Persiapan Tabel di Supabase
Jalankan query yang tersedia di file [`supabase_schema.sql`](file:///Users/vcl/Desktop/python-ai/supabase_schema.sql) pada menu **SQL Editor** di Dashboard Supabase Anda:
```sql
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user', -- 'user' atau 'superadmin'
    full_name TEXT,
    is_peatland BOOLEAN NOT NULL DEFAULT FALSE, -- Tipe lahan: true = lahan gambut, false = bukan lahan gambut
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tambahkan kolom is_peatland jika tabel sudah ada sebelumnya (migrasi)
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_peatland BOOLEAN NOT NULL DEFAULT FALSE;

-- Nonaktifkan RLS agar dapat diakses oleh API backend
ALTER TABLE public.users DISABLE ROW LEVEL SECURITY;
```

### Konfigurasi Environment (`.env`)
```bash
SUPABASE_URL=https://bxbpkhngcflsyotgbhqd.supabase.co
SUPABASE_KEY=sb_publishable_60eJEoSgHghjC_2mmxLGpQ_QRf1u3I5
```

---

### 🛡️ Kebijakan Batasan Akses Tipe Lahan (`is_peatland`)
Sistem menerapkan proteksi akses algoritma berbasis tipe lahan hidrologis:
- **`is_peatland: false` (Lahan Non-Gambut / Mineral)**: Memiliki hak akses penuh untuk mengeksekusi algoritma inferensi prediksi dan kalkulator bahaya kebakaran (Predict, Batch Predict, FWI Dokumen 1, Drought Warning PDWI/MDWI, dan Multi-Horizon Forecast).
- **`is_peatland: true` (Lahan Gambut Murni)**: Akses ke endpoint algoritma dibatasi secara ketat. Akun bertipe lahan gambut yang memanggil endpoint tersebut akan otomatis ditolak oleh dependency `require_non_peatland` dengan respon HTTP **`403 Forbidden`**:
  ```json
  {
    "detail": "Akses Ditolak: Algoritma FWI, Drought, dan Forecast tidak tersedia untuk akun dengan tipe lahan gambut (is_peatland=True). Fitur ini hanya diperuntukkan bagi lahan non-gambut."
  }
  ```

| Endpoint Terproteksi (Hanya `is_peatland=false`) | Endpoint Bebas / Publik / Manajemen |
| :--- | :--- |
| `POST /api/v1/predict` (Single Record AI Predict) | `POST /api/v1/telemetry` (IoT Hardware Ingestion) |
| `POST /api/v1/predict/batch` (Batch AI Predict) | `GET /api/v1/forecast/7-days` (Dataset Latest Point) |
| `POST /api/v1/forecast/7-days` (7-Day AI Forecast) | `GET /api/v1/forecast/10-days` (Dataset Latest Point) |
| `POST /api/v1/forecast/10-days` (10-Day AI Forecast) | `GET /api/v1/dataset/*` (Historical Dataset & CSV Export) |
| `POST /api/v1/fwi/calculate` (Kalkulator Pseudo-FWI) | `GET /health` & `GET /` (System Status) |
| `POST /api/v1/drought/calculate` (Kalkulator PDWI/MDWI) | `GET /api/v1/scheduler/status` (Scheduler Cron) |
| `POST /api/v1/drought/forecast-7days` (7-Day Drought Forecast) | `POST /api/v1/retrain` & `GET /api/v1/metrics` (MLOps) |
| `GET /api/v1/drought/latest-comparison` (Evaluasi PDWI vs MDWI) | Seluruh Rute Auth & Dashboard (`/api/v1/auth/*`, `/api/v1/user/*`) |

---

### 1. Registrasi Akun Langsung (`POST /api/v1/auth/register`)
Mendaftarkan akun baru ke tabel `users` dengan pilihan role (`user` atau `superadmin`) dan tipe lahan (`is_peatland`). Akun langsung aktif tanpa email konfirmasi!
- **Body**:
```json
{
  "email": "operator@perusahaan.com",
  "password": "Password123!",
  "role": "user",
  "full_name": "Andi Operator",
  "is_peatland": false
}
```
*(Field `is_peatland` bersifat opsional, nilai default adalah `false`)*.

- **Response (`201 Created`)**:
```json
{
  "success": true,
  "message": "Akun berhasil dibuat dengan role 'user'. Anda dapat langsung login tanpa konfirmasi email!",
  "user": {
    "id": "197ac669-b1b8-4fca-9d9f-2b2cf3be1862",
    "email": "operator@perusahaan.com",
    "role": "user",
    "full_name": "Andi Operator",
    "is_peatland": false
  }
}
```

---

### 2. Login Pengguna Langsung (`POST /api/v1/auth/login`)
Login memvalidasi password ke database tabel `users` dan mengembalikan JWT Bearer Token yang memuat klaim role dan status `is_peatland`.
- **Body**:
```json
{
  "email": "operator@perusahaan.com",
  "password": "Password123!"
}
```
- **Response (`200 OK`)**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "197ac669-b1b8-4fca-9d9f-2b2cf3be1862",
    "email": "operator@perusahaan.com",
    "role": "user",
    "full_name": "Andi Operator",
    "is_peatland": false
  }
}
```

---

### 3. Logout Pengguna (`POST /api/v1/auth/logout`)
- **Header**: `Authorization: Bearer <access_token>`
- **Response (`200 OK`)**:
```json
{
  "success": true,
  "message": "Logout berhasil. Sesi telah diakhiri di Supabase."
}
```

---

### 4. Profil Pengguna Saat Ini (`GET /api/v1/auth/me`)
- **Header**: `Authorization: Bearer <access_token>`
- **Response (`200 OK`)**:
```json
{
  "id": "197ac669-b1b8-4fca-9d9f-2b2cf3be1862",
  "email": "operator@perusahaan.com",
  "role": "user",
  "full_name": "Andi Operator",
  "is_peatland": false,
  "metadata": {
    "created_at": "2026-10-06T14:32:01.042425+00:00"
  }
}
```

---

### 5. Rute Terproteksi Berdasarkan Role (RBAC)
- **Akses User & Superadmin**: `GET /api/v1/user/dashboard` atau `GET /api/v1/auth/user-access`
  - Memerlukan header `Authorization: Bearer <token>`
  - Role yang diizinkan: `user`, `superadmin`
- **Akses Khusus Superadmin**: `GET /api/v1/superadmin/dashboard` atau `GET /api/v1/auth/superadmin-access`
  - Memerlukan header `Authorization: Bearer <token>`
  - Role yang diizinkan: `superadmin` saja
  - Akun dengan role `user` akan otomatis ditolak dengan status **`403 Forbidden`**:
  ```json
  {
    "detail": "Akses Ditolak: Anda tidak memiliki wewenang untuk mengakses endpoint ini. Dibutuhkan salah satu dari role: ['superadmin'], sedangkan role akun Anda: 'user'."
  }
  ```

---

### 6. Melihat Seluruh Data Pengguna (Eksklusif Superadmin)
Hanya akun dengan role `superadmin` yang memiliki wewenang melihat seluruh data pengguna database. Password hash disaring secara otomatis demi keamanan.
- **Method**: `GET`
- **Endpoint**: `/api/v1/superadmin/users` *(atau alias: `/api/v1/auth/users`)*
- **Header**: `Authorization: Bearer <token_superadmin>`
- **Response (`200 OK`)**:
```json
{
  "status": "success",
  "access": "granted",
  "message": "Daftar seluruh pengguna database (3 akun).",
  "authorized_by": {
    "email": "bos_superadmin@perusahaan.com",
    "role": "superadmin"
  },
  "total_users": 3,
  "users": [
    {
      "id": "451f3cfa-ba47-4315-9233-9dba68c76603",
      "email": "bos_superadmin@perusahaan.com",
      "role": "superadmin",
      "full_name": "Budi Superadmin",
      "is_peatland": false,
      "created_at": "2026-10-06T14:32:02.692953+00:00"
    },
    {
      "id": "bef08a82-ed7f-4871-8b7d-c5430f327c39",
      "email": "andi_operator@perusahaan.com",
      "role": "user",
      "full_name": "Andi Operator",
      "is_peatland": false,
      "created_at": "2026-10-06T14:32:01.042425+00:00"
    }
  ]
}
```
*(Jika diakses menggunakan token role `user`, server merespons `403 Forbidden`)*.

---

### 7. Mengedit / Update Data Pengguna (Eksklusif Superadmin)
Superadmin dapat mengupdate nama lengkap, email, role (`user` / `superadmin`), status tipe lahan (`is_peatland`), atau mereset password pengguna.
- **Method**: `PUT`
- **Endpoint**: `/api/v1/superadmin/users/{user_id}` *(atau alias: `/api/v1/auth/users/{user_id}`)*
- **Header**: `Authorization: Bearer <token_superadmin>`
- **Body**:
```json
{
  "full_name": "Nama Lengkap Baru",
  "role": "superadmin",
  "email": "emailbaru@perusahaan.com",
  "password": "PasswordBaru123!",
  "is_peatland": false
}
```
*(Semua field di atas bersifat opsional, isi hanya yang ingin diubah)*.

- **Response (`200 OK`)**:
```json
{
  "status": "success",
  "message": "Data pengguna 'emailbaru@perusahaan.com' (ID: ...) berhasil diperbarui.",
  "updated_by": "bos_superadmin@perusahaan.com",
  "user": {
    "id": "bef08a82-ed7f-4871-8b7d-c5430f327c39",
    "email": "emailbaru@perusahaan.com",
    "role": "superadmin",
    "full_name": "Nama Lengkap Baru",
    "is_peatland": false,
    "created_at": "2026-10-06T14:32:01.042425+00:00"
  }
}
```

---

### 8. Menghapus Pengguna (Eksklusif Superadmin)
Superadmin dapat menghapus akun pengguna dari database Supabase secara permanen.
- **Method**: `DELETE`
- **Endpoint**: `/api/v1/superadmin/users/{user_id}` *(atau alias: `/api/v1/auth/users/{user_id}`)*
- **Header**: `Authorization: Bearer <token_superadmin>`
- **Response (`200 OK`)**:
```json
{
  "status": "success",
  "message": "Pengguna dengan ID 'bef08a82-ed7f-4871-8b7d-c5430f327c39' berhasil dihapus dari database Supabase.",
  "deleted_by": "bos_superadmin@perusahaan.com"
}
```
*(Catatan Keamanan: Superadmin diproteksi dan tidak dapat menghapus akunnya sendiri demi mencegah lockout sistem).*




