"""
Evaluasi Data Riil AWS & Muka Air Gambut (TMAT)
1. Menganalisis data historis beberapa hari kebelakang langsung dari dataset Excel.
2. Menghasilkan status langsung: Tingkat Kebakaran, Kekeringan, dan Hujan.
3. Memproyeksikan prediksi 10 hari ke depan dari titik data mutakhir dataset.
"""

import sys
import pandas as pd
from pathlib import Path

from src.config import DEFAULT_RAW_DATA_PATH
from src.data_loader import load_aws_dataset
from src.predict import AWSPredictor
from src.forecast import analyze_and_forecast_from_dataset


def run_real_data_analysis():
    print("==========================================================================================")
    print("📡  SISTEM PREDIKSI BENCANA LAHAN GAMBUT - PEMROSESAN DATA RIIL AWS & TMAT  📡")
    print("==========================================================================================\n")

    # 1. Memuat dataset riil
    print(f"[*] Memuat data aktual dari: {DEFAULT_RAW_DATA_PATH}")
    df = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
    
    # 2. Inisialisasi engine prediksi
    predictor = AWSPredictor()

    # 3. Analisis Historis & Proyeksi 10 Hari ke Depan dari Data Aktual
    n_history_days = 8
    result = analyze_and_forecast_from_dataset(df, n_history_days=n_history_days, n_forecast_days=10)

    ds_info = result["dataset_info"]
    print(f"\n📂 INFORMASI DATASET AKTUAL:")
    print(f"   • Total Catatan Sensor : {ds_info['total_records']:,} baris telemetri (interval 5 menit)")
    print(f"   • Rentang Waktu        : {ds_info['start_time']} s/d {ds_info['end_time']}")

    # ---------------------------------------------------------------------------------------
    # BAGIAN 1: ANALISIS BEBERAPA HARI KEBELAKANG DARI DATA AKTUAL
    # ---------------------------------------------------------------------------------------
    print("\n" + "="*85)
    print(f"📊 1. ANALISIS KONDISI {n_history_days} HARI KEBELAKANG (DATA RIIL DARI DATASET)")
    print("="*85)
    print(f"   {'Tanggal':<12} {'Kebakaran':<12} {'Kekeringan':<12} {'Hujan':<10} {'Suhu Maks':<11} {'RH Min':<9} {'TMAT Rata2':<12} {'Total Hujan'}")
    print("   " + "-"*83)
    
    for day in result["riwayat_kebelakang"]:
        sens = day["detail_sensor_harian"]
        print(
            f"   {day['tanggal']:<12} "
            f"{day['tingkat_kebakaran']:<12} "
            f"{day['kekeringan']:<12} "
            f"{day['hujan']:<10} "
            f"{sens['suhu_maks_c']}°C       "
            f"{sens['kelembapan_min_persen']}%      "
            f"{sens['rata_rata_tmat_cm']:>6.1f} cm    "
            f"{sens['total_hujan_mm']:>5.1f} mm"
        )

    # ---------------------------------------------------------------------------------------
    # BAGIAN 2: STATUS MUTAKHIR DARI DATASET (TITIK TERAKHIR)
    # ---------------------------------------------------------------------------------------
    latest = result["kondisi_terakhir"]
    print("\n" + "="*85)
    print(f"📍 2. STATUS DATA MUTAKHIR PADA DATASET ({latest['waktu']})")
    print("="*85)
    ringkasan_terakhir = latest["ringkasan"]
    print(f"   👉 RINGKASAN STATUS: {ringkasan_terakhir['status_teks'].upper()}")
    print(f"   • Tingkat Kebakaran : {ringkasan_terakhir['tingkat_kebakaran']} (Indeks Pseudo-FWI: {latest['pseudo_fwi']})")
    print(f"   • Kekeringan        : {ringkasan_terakhir['kekeringan']} (Tinggi Muka Air Tanah: {latest['tmat_cm']} cm)")
    print(f"   • Potensi Hujan     : {ringkasan_terakhir['hujan']} (Intensitas Curah Hujan: {latest['hujan_mmh']} mm/jam)")
    print(f"   • Parameter Lingkungan : Suhu={latest['suhu_c']}°C, RH={latest['rh_persen']}%, Radiasi={latest['radiasi_wm2']} W/m²")

    # Inisialisasi EWS untuk kondisi mutakhir
    last_row_dict = df.iloc[-1].to_dict()
    pred_res = predictor.predict(last_row_dict, include_forecast=False)
    print("\n   📢 EARLY WARNING SYSTEM (EWS) AKTIF:")
    for alert in pred_res["early_warning_system"]:
        print(f"      [{alert['level']}] {alert['title']}")
        print(f"      -> {alert['message']}")

    # ---------------------------------------------------------------------------------------
    # BAGIAN 3: PREDIKSI PROYEKSI 10 HARI KE DEPAN
    # ---------------------------------------------------------------------------------------
    print("\n" + "="*85)
    print("🔮 3. PREDIKSI PROYEKSI 10 HARI KE DEPAN (DARI TITIK AKHIR DATASET)")
    print("="*85)
    print(f"   {'Hari':<8} {'Tanggal':<12} {'Kebakaran':<12} {'Kekeringan':<12} {'Hujan':<10} {'Suhu':<8} {'RH':<7} {'TMAT Proyeksi':<15} {'FWI'}")
    print("   " + "-"*83)

    for fc in result["prediksi_10_hari_kedepan"]:
        w = fc["detail_cuaca"]
        print(
            f"   Hari {fc['hari_ke']:<3} "
            f"{fc['tanggal']:<12} "
            f"{fc['tingkat_kebakaran']:<12} "
            f"{fc['kekeringan']:<12} "
            f"{fc['hujan']:<10} "
            f"{w['suhu_c']}°C   "
            f"{w['kelembapan_persen']}%   "
            f"{w['tmat_cm']:>6.1f} cm       "
            f"{fc['pseudo_fwi_score']}"
        )

    print("\n" + "="*85)
    print("✅ PENGUJIAN & EVALUASI DATA AKTUAL SELESAI")
    print("="*85)


if __name__ == "__main__":
    run_real_data_analysis()
