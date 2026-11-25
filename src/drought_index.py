"""
Drought Warning Index Engine (PDWI & MDWI)
Implements scientific formulations from Google Slides:
'Algoritma Prediksi kebakaran lahan gambut'
(Doc ID: 11k2qcM7WzzvNArG54MK-bCjurPVw2aZGQNTNoawp7Xc)

Supports:
1. Lahan Gambut (Peatland Drought Warning Index - PDWI)
2. Lahan Biasa / Lahan Mineral (Mineral Soil Drought Warning Index - MDWI)

Architecture Hierarchy (Slide 21):
Sensor -> Drought Index (PDWI/MDWI) -> Hydrological Forecast -> Fire Vulnerability -> AI Fire Probability
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge


class LandType(str, Enum):
    PEATLAND = "gambut"           # Lahan Gambut (Organic Peat Soil)
    MINERAL = "lahan_biasa"        # Lahan Biasa (Mineral Soil / Non-Gambut)


# ---------------------------------------------------------------------------
# 1. Step 1 - Konversi Data Sensor (Slide 4)
# ---------------------------------------------------------------------------
def convert_tmat_to_wtd(tmat_cm: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
    """
    Konversi TMAT (Tinggi Muka Air Tanah) negatif sensor ke WTD (Water Table Depth):
        WTD_t = - TMAT_t
    Misal: TMAT = -159.37 cm -> WTD = 159.37 cm di bawah permukaan tanah.
    Semakin besar WTD -> semakin dalam muka air -> semakin kering gambut.
    """
    if isinstance(tmat_cm, (pd.Series, np.ndarray)):
        return -tmat_cm
    return -float(tmat_cm)


def convert_wtd_to_tmat(wtd_cm: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
    """Konversi kebalikan dari WTD ke TMAT (cm): TMAT = -WTD."""
    if isinstance(wtd_cm, (pd.Series, np.ndarray)):
        return -wtd_cm
    return -float(wtd_cm)


# ---------------------------------------------------------------------------
# 2. Step 3 - Hitung Vapor Pressure Deficit / VPD (Slide 6)
# ---------------------------------------------------------------------------
def calculate_vpd(temperature_c: float, relative_humidity_pct: float) -> float:
    """
    Menghitung Vapor Pressure Deficit (VPD dalam kPa) menggunakan formula Tetens:
        e_s(T) = 0.6108 * exp( (17.27 * T) / (T + 237.3) )
        e_a    = e_s(T) * (RH / 100)
        VPD    = e_s(T) - e_a = e_s(T) * (1 - RH / 100)
    """
    temp = float(temperature_c)
    rh = float(np.clip(relative_humidity_pct, 0.0, 100.0))
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (rh / 100.0)
    vpd = max(0.0, es - ea)
    return float(round(vpd, 4))


# ---------------------------------------------------------------------------
# 3. Atmospheric Water Demand / PET Proxy (Slide 13)
# ---------------------------------------------------------------------------
def calculate_pet_proxy(
    solar_radiation_wm2: float,
    vpd_kpa: float,
    wind_speed_ms: float = 1.0,
) -> float:
    """
    Atmospheric Water Demand (AWD) / PET Proxy (Slide 13 & Chat 3):
        PET_t = f(R_s, VPD, U)
        PET_proxy = (0.20 * R_s_MJ) + (1.50 * VPD) + (0.10 * U)
    Konversi Radiasi W/m² rata-rata harian ke MJ/m²/day (faktor 0.0864).
    """
    rs_mj = float(solar_radiation_wm2) * 0.0864
    pet = (0.20 * rs_mj) + (1.50 * float(vpd_kpa)) + (0.10 * float(wind_speed_ms))
    return float(max(0.0, round(pet, 3)))


# ---------------------------------------------------------------------------
# 4. Agregasi Sensor 5-Menit Menjadi Daily State (Slide 5)
# ---------------------------------------------------------------------------
def aggregate_5min_to_daily(df_sensor: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi data IoT interval 5-menit ke daily state nyata (Slide 5):
      - WTD_t = mean(WTD_{t,i})
      - T_mean,t = mean(T_{t,i}), T_max,t = max(T_{t,i})
      - Rain harian dari intensitas mm/jam:
          Rain_i = RainIntensity_i * (5 / 60)
          P_t = sum(Rain_i)
      - VPD_mean = mean(VPD_{t,i})
      - Solar_mean = mean(Solar_{t,i})
    """
    df = df_sensor.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    if "water_table_depth_cm" in df.columns and "wtd_cm" not in df.columns:
        df["wtd_cm"] = convert_tmat_to_wtd(df["water_table_depth_cm"])

    if "vpd_kpa" not in df.columns:
        es = 0.6108 * np.exp((17.27 * df["air_temperature_c"]) / (df["air_temperature_c"] + 237.3))
        ea = es * (df["relative_humidity_pct"] / 100.0)
        df["vpd_kpa"] = np.maximum(0.0, es - ea)

    # 5-minute rain conversion: mm/h * (5 / 60)
    rain_col = "rainfall_intensity_mmh" if "rainfall_intensity_mmh" in df.columns else "rain_mm"
    df["rain_vol_5m"] = df[rain_col] * (5.0 / 60.0)

    daily = df.groupby("date").agg({
        "wtd_cm": "mean",
        "air_temperature_c": ["mean", "max"],
        "relative_humidity_pct": ["mean", "min"],
        "solar_radiation_wm2": ["mean", "max"],
        "rain_vol_5m": "sum",
        "vpd_kpa": "mean",
        "wind_speed_ms": "mean" if "wind_speed_ms" in df.columns else lambda x: 1.0,
        "soil_moisture_pct": "mean" if "soil_moisture_pct" in df.columns else lambda x: 50.0,
    })

    daily.columns = [
        "wtd_mean",
        "temp_mean", "temp_max",
        "rh_mean", "rh_min",
        "solar_mean", "solar_max",
        "rain_mm",
        "vpd_mean",
        "wind_mean",
        "soil_moisture_mean"
    ]
    daily = daily.reset_index()

    # Hitung PET proxy harian
    daily["pet_proxy"] = [
        calculate_pet_proxy(row["solar_mean"], row["vpd_mean"], row["wind_mean"])
        for _, row in daily.iterrows()
    ]

    return daily


# ---------------------------------------------------------------------------
# 5. Klasifikasi Kelas Kekeringan (Slide 9 & Chat 2)
# ---------------------------------------------------------------------------
def classify_drought_score(score: float) -> Tuple[int, str]:
    """
    Thresholds Kelas Kekeringan PDWI / MDWI (0 - 100):
      - 0  – 20 : NORMAL (Aman)
      - 20 – 40 : WATCH (Waspada)
      - 40 – 60 : MODERATE (Sedang)
      - 60 – 80 : HIGH (Tinggi)
      - 80 – 100: SEVERE (Ekstrim / Kritis)
    """
    s = float(np.clip(score, 0.0, 100.0))
    if s < 20.0:
        return 0, "NORMAL"
    elif s < 40.0:
        return 1, "WATCH"
    elif s < 60.0:
        return 2, "MODERATE"
    elif s < 80.0:
        return 3, "HIGH"
    else:
        return 4, "SEVERE"


# ---------------------------------------------------------------------------
# 6. Hydrological Stress Score (H_t) - Empirical CDF (Slide 7)
#    Dukungan Lahan Gambut vs Lahan Biasa (Mineral)
# ---------------------------------------------------------------------------
def calculate_hydrological_stress(
    wtd_cm: float,
    baseline_wtd: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    soil_moisture_pct: Optional[float] = None,
    land_type: LandType = LandType.PEATLAND,
) -> float:
    """
    Menghitung Hydrological Stress Score (H_t) skala 0 - 100:
    
    1. Lahan Gambut (Peatland):
       - State variable utama adalah WTD (Water Table Depth).
       - Menggunakan Empirical Cumulative Distribution Function (Slide 7):
           H_t = 100 * CDF_{WTD}(WTD_t)
       - Dikombinasikan dengan batas regulasi BRGM (-40 cm = 40 cm WTD, -80 cm, -120 cm).
       
    2. Lahan Biasa / Lahan Mineral (Mineral Land):
       - Tanah mineral tidak terbakar, melainkan biomassa permukaan & perakaran tanaman.
       - TMAT alami lahan mineral jauh lebih dalam (bisa 100 - 300+ cm).
       - Stress hidrologis di lahan mineral dipengaruhi oleh:
         a) Penurunan kelembaban tanah zona perakaran (Soil Moisture depletion).
         b) Deviasi muka air tanah relatif terhadap baseline lokal (P10..P90).
         c) Rasio kedalaman terhadap zona perakaran efektif (0-150 cm).
    """
    wtd = float(wtd_cm)

    # A. Empirical CDF jika baseline tersedia
    if baseline_wtd is not None and len(baseline_wtd) > 1:
        arr = np.sort(np.asarray(baseline_wtd))
        # Persentil empiris (0 - 100)
        h_cdf = float(stats.percentileofscore(arr, wtd, kind="weak"))
    else:
        # Fallback domain-based jika tanpa baseline historis
        if land_type == LandType.PEATLAND:
            # Domain gambut: 0 cm (basah total) s/d 160 cm (sangat kering/kritis parah)
            h_cdf = float(np.clip((wtd / 160.0) * 100.0, 0.0, 100.0))
        else:
            # Domain lahan biasa: 40 cm s/d 300 cm
            h_cdf = float(np.clip(((wtd - 40.0) / 260.0) * 100.0, 0.0, 100.0))

    if land_type == LandType.PEATLAND:
        return float(np.clip(h_cdf, 0.0, 100.0))

    # B. Penyesuaian Khusus Lahan Biasa (Mineral Land)
    # Pada lahan biasa, jika sensor menyediakan soil moisture, integrasikan zona perakaran
    if soil_moisture_pct is not None:
        sm = float(soil_moisture_pct)
        # Di tanah mineral: SM < 20% adalah titik layu (wilting point), > 60% adalah field capacity
        sm_stress = float(np.clip((1.0 - (sm / 60.0)) * 100.0, 0.0, 100.0))
        # Gabungan: 60% Kelembaban Tanah + 40% Kedalaman Muka Air (TMAT)
        h_mineral = 0.60 * sm_stress + 0.40 * h_cdf
        return float(np.clip(h_mineral, 0.0, 100.0))

    return float(np.clip(h_cdf, 0.0, 100.0))


# ---------------------------------------------------------------------------
# 7. Atmospheric Drying Score (A_t) (Slide 8)
# ---------------------------------------------------------------------------
def calculate_atmospheric_drying(
    vpd_kpa: float,
    solar_radiation_wm2: float,
    baseline_vpd: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    baseline_solar: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
) -> Tuple[float, float, float]:
    """
    Menghitung Atmospheric Drying Score (Slide 8):
        A_{VPD,t}   = 100 * CDF(VPD_t)
        A_{Solar,t} = 100 * CDF(Solar_t)
        A_t         = 0.70 * A_{VPD,t} + 0.30 * A_{Solar,t}
    Returns: (A_t, A_{VPD}, A_{Solar})
    """
    vpd = float(vpd_kpa)
    solar = float(solar_radiation_wm2)

    if baseline_vpd is not None and len(baseline_vpd) > 1:
        a_vpd = float(stats.percentileofscore(np.asarray(baseline_vpd), vpd, kind="weak"))
    else:
        # Domain fallback: VPD 0.0 kPa (lembab) s/d 2.0 kPa (kering)
        a_vpd = float(np.clip((vpd / 2.0) * 100.0, 0.0, 100.0))

    if baseline_solar is not None and len(baseline_solar) > 1:
        a_solar = float(stats.percentileofscore(np.asarray(baseline_solar), solar, kind="weak"))
    else:
        # Domain fallback: Radiasi 0 s/d 800 W/m²
        a_solar = float(np.clip((solar / 800.0) * 100.0, 0.0, 100.0))

    a_total = 0.70 * a_vpd + 0.30 * a_solar
    return float(round(a_total, 2)), float(round(a_vpd, 2)), float(round(a_solar, 2))


# ---------------------------------------------------------------------------
# 8. Rainfall Deficit (RD_t) - 7 Hari (Slide 12)
# ---------------------------------------------------------------------------
def calculate_rainfall_deficit_7d(
    rain_7d_mm: float,
    rain_7d_ref_mm: float,
) -> float:
    """
    Menghitung Rainfall Deficit (Slide 12):
        P_{7,t}   = sum_{i=t-6}^t P_i
        P_{7,ref} = Median(P_7)
        RD_t      = 100 * max(0, 1 - (P_{7,t} / P_{7,ref}))
    Jika P_7 >= P_ref -> RD = 0.
    Jika hujan jauh di bawah referensi -> RD mendekati 100.
    """
    p7 = max(0.0, float(rain_7d_mm))
    pref = max(0.1, float(rain_7d_ref_mm))
    deficit = 100.0 * max(0.0, 1.0 - (p7 / pref))
    return float(np.clip(deficit, 0.0, 100.0))


# ---------------------------------------------------------------------------
# 9. LEVEL 1 — Sensor Based Drought Index (Slide 9)
#    Dukungan Gambut (PDWI-L1) vs Lahan Biasa (MDWI-L1)
# ---------------------------------------------------------------------------
def calculate_drought_level_1(
    wtd_cm: float,
    vpd_kpa: float,
    solar_radiation_wm2: float,
    baseline_df: Optional[pd.DataFrame] = None,
    soil_moisture_pct: Optional[float] = None,
    land_type: LandType = LandType.PEATLAND,
) -> Dict[str, Any]:
    """
    LEVEL 1 — Sensor Based Drought Index:
    Menjawab: "Seberapa kering lokasi ini sekarang?"
    
    1. Lahan Gambut (PDWI-L1) (Slide 9):
       - Bobot: 60% Hydrological Stress (H_t)
                25% Atmospheric Drying / VPD (A_{VPD})
                15% Solar Drying (A_{Solar})
         PDWI^{L1}_t = 0.60 * H_t + 0.25 * A_{VPD,t} + 0.15 * A_{Solar,t}
         
    2. Lahan Biasa / Lahan Mineral (MDWI-L1):
       - Di lahan mineral, vegetasi atas dan serasah mengering jauh lebih cepat
         melalui permintaan atmosfer (VPD dan radiasi matahari), sementara TMAT
         berada lebih dalam.
       - Bobot: 35% Hydrological/Soil Stress
                40% Atmospheric Drying (VPD)
                25% Solar Drying
         MDWI^{L1}_t = 0.35 * H_t + 0.40 * A_{VPD,t} + 0.25 * A_{Solar,t}
    """
    baseline_wtd = baseline_df["wtd_mean"] if baseline_df is not None and "wtd_mean" in baseline_df.columns else None
    baseline_vpd = baseline_df["vpd_mean"] if baseline_df is not None and "vpd_mean" in baseline_df.columns else None
    baseline_solar = baseline_df["solar_mean"] if baseline_df is not None and "solar_mean" in baseline_df.columns else None

    # Hitung Subskor
    h_t = calculate_hydrological_stress(wtd_cm, baseline_wtd, soil_moisture_pct, land_type)
    _, a_vpd, a_solar = calculate_atmospheric_drying(vpd_kpa, solar_radiation_wm2, baseline_vpd, baseline_solar)

    if land_type == LandType.PEATLAND:
        weights = {"hydrological_stress": 0.60, "atmospheric_vpd": 0.25, "solar_radiation": 0.15}
        drought_score = (0.60 * h_t) + (0.25 * a_vpd) + (0.15 * a_solar)
        index_name = "PDWI-L1 (Peatland Drought Warning Index Level 1)"
    else:
        weights = {"hydrological_stress": 0.35, "atmospheric_vpd": 0.40, "solar_radiation": 0.25}
        drought_score = (0.35 * h_t) + (0.40 * a_vpd) + (0.25 * a_solar)
        index_name = "MDWI-L1 (Mineral Land Drought Warning Index Level 1)"

    drought_score = float(np.clip(drought_score, 0.0, 100.0))
    drought_class, drought_label = classify_drought_score(drought_score)

    return {
        "index_name": index_name,
        "land_type": land_type.value,
        "level": 1,
        "score": round(drought_score, 2),
        "drought_class": drought_class,
        "drought_label": drought_label,
        "sub_scores": {
            "hydrological_stress_ht": round(h_t, 2),
            "atmospheric_vpd_score": round(a_vpd, 2),
            "solar_radiation_score": round(a_solar, 2),
        },
        "weights": weights,
        "inputs": {
            "wtd_cm": round(float(wtd_cm), 2),
            "tmat_cm": round(-float(wtd_cm), 2),
            "vpd_kpa": round(float(vpd_kpa), 3),
            "solar_radiation_wm2": round(float(solar_radiation_wm2), 1),
            "soil_moisture_pct": round(float(soil_moisture_pct), 1) if soil_moisture_pct else None,
        }
    }


# ---------------------------------------------------------------------------
# 10. LEVEL 2 — Climate-Hydrology Drought Index (Slide 11-14)
# ---------------------------------------------------------------------------
def calculate_drought_level_2(
    wtd_cm: float,
    vpd_kpa: float,
    solar_radiation_wm2: float,
    rain_7d_mm: float,
    rain_7d_ref_mm: float,
    baseline_df: Optional[pd.DataFrame] = None,
    soil_moisture_pct: Optional[float] = None,
    land_type: LandType = LandType.PEATLAND,
) -> Dict[str, Any]:
    """
    LEVEL 2 — Climate-Hydrology Drought Index (Slide 11-14):
    Menjawab: "Bagaimana kondisi hydrological drought berkembang jika climate forcing diperhitungkan?"
    
    1. Lahan Gambut (PDWI-L2) (Slide 14):
       PDWI^{L2}_t = 0.50 * H_t + 0.20 * A_{VPD,t} + 0.10 * A_{Solar,t} + 0.20 * RD_t
       - 50% WTD / hydrological state
       - 20% atmospheric dryness (VPD)
       - 10% solar drying
       - 20% rainfall deficit
       
    2. Lahan Biasa / Lahan Mineral (MDWI-L2):
       MDWI^{L2}_t = 0.30 * H_t + 0.30 * A_{VPD,t} + 0.15 * A_{Solar,t} + 0.25 * RD_t
       - 30% hydrological state
       - 30% atmospheric dryness
       - 15% solar drying
       - 25% rainfall deficit
    """
    baseline_wtd = baseline_df["wtd_mean"] if baseline_df is not None and "wtd_mean" in baseline_df.columns else None
    baseline_vpd = baseline_df["vpd_mean"] if baseline_df is not None and "vpd_mean" in baseline_df.columns else None
    baseline_solar = baseline_df["solar_mean"] if baseline_df is not None and "solar_mean" in baseline_df.columns else None

    h_t = calculate_hydrological_stress(wtd_cm, baseline_wtd, soil_moisture_pct, land_type)
    _, a_vpd, a_solar = calculate_atmospheric_drying(vpd_kpa, solar_radiation_wm2, baseline_vpd, baseline_solar)
    rd_t = calculate_rainfall_deficit_7d(rain_7d_mm, rain_7d_ref_mm)

    if land_type == LandType.PEATLAND:
        weights = {
            "hydrological_stress": 0.50,
            "atmospheric_vpd": 0.20,
            "solar_radiation": 0.10,
            "rainfall_deficit": 0.20
        }
        score = (0.50 * h_t) + (0.20 * a_vpd) + (0.10 * a_solar) + (0.20 * rd_t)
        index_name = "PDWI-L2 (Peatland Climate-Hydrology Drought Warning Index)"
    else:
        weights = {
            "hydrological_stress": 0.30,
            "atmospheric_vpd": 0.30,
            "solar_radiation": 0.15,
            "rainfall_deficit": 0.25
        }
        score = (0.30 * h_t) + (0.30 * a_vpd) + (0.15 * a_solar) + (0.25 * rd_t)
        index_name = "MDWI-L2 (Mineral Land Climate-Hydrology Drought Warning Index)"

    score = float(np.clip(score, 0.0, 100.0))
    drought_class, drought_label = classify_drought_score(score)

    return {
        "index_name": index_name,
        "land_type": land_type.value,
        "level": 2,
        "score": round(score, 2),
        "drought_class": drought_class,
        "drought_label": drought_label,
        "sub_scores": {
            "hydrological_stress_ht": round(h_t, 2),
            "atmospheric_vpd_score": round(a_vpd, 2),
            "solar_radiation_score": round(a_solar, 2),
            "rainfall_deficit_score": round(rd_t, 2),
        },
        "weights": weights,
        "inputs": {
            "wtd_cm": round(float(wtd_cm), 2),
            "tmat_cm": round(-float(wtd_cm), 2),
            "vpd_kpa": round(float(vpd_kpa), 3),
            "solar_radiation_wm2": round(float(solar_radiation_wm2), 1),
            "rain_7d_mm": round(float(rain_7d_mm), 2),
            "rain_7d_ref_mm": round(float(rain_7d_ref_mm), 2),
            "soil_moisture_pct": round(float(soil_moisture_pct), 1) if soil_moisture_pct else None,
        }
    }


# ---------------------------------------------------------------------------
# 11. Persistence & WTD Trend (Slide 15)
# ---------------------------------------------------------------------------
def calculate_wtd_trend_7d(wtd_current: float, wtd_7d_ago: float) -> Dict[str, Any]:
    """
    Persistence / Trend (Slide 15):
        \\Delta WTD_{7,t} = WTD_t - WTD_{t-7}
    Interpretasi:
        \\Delta WTD_7 > 0 -> muka air semakin dalam -> Drying trend
        \\Delta WTD_7 < 0 -> muka air naik -> Wetting trend
    """
    delta_wtd = float(wtd_current - wtd_7d_ago)
    if delta_wtd > 2.0:
        trend = "Drying Trend (Muka air tanah surut menjauhi permukaan)"
    elif delta_wtd < -2.0:
        trend = "Wetting Trend (Muka air tanah naik terisi kembali)"
    else:
        trend = "Stable (Ketinggian muka air relatif stabil)"

    return {
        "delta_wtd_7d_cm": round(delta_wtd, 2),
        "trend_direction": trend,
        "is_drying": delta_wtd > 0.0,
    }


# ---------------------------------------------------------------------------
# 12. Model Regresi Perubahan Muka Air (\\Delta WTD) (Slide 16)
# ---------------------------------------------------------------------------
class HydrologicalTransitionModel:
    """
    Model Regresi Perkiraan Perubahan Kedalaman Muka Air (Slide 16):
        \\Delta WTD_t = \\beta_0 + \\beta_1 P_t + \\beta_2 PET_t
    Kemudian update rekursif harian:
        WTD_{t+1} = WTD_t + \\Delta WTD_t
        
    Karakteristik Fisik:
      \\beta_1 < 0 (Hujan menaikkan muka air -> kedalaman WTD turun)
      \\beta_2 > 0 (PET menarik air tanah -> kedalaman WTD bertambah dalam)
    """
    def __init__(self, beta_0: float = 0.20, beta_1: float = -0.015, beta_2: float = 0.08):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.is_fitted = False

    def fit_from_daily_history(self, daily_df: pd.DataFrame) -> "HydrologicalTransitionModel":
        """
        Melatih parameter \\beta_0, \\beta_1, \\beta_2 dari time-series aktual WTD, Hujan, dan PET (Chat 5).
        """
        df = daily_df.copy().dropna(subset=["wtd_mean", "rain_mm", "pet_proxy"])
        if len(df) < 5:
            return self

        # Perubahan harian WTD: dWTD = WTD_{t+1} - WTD_t
        y_delta = df["wtd_mean"].diff().shift(-1).iloc[:-1]
        X = df[["rain_mm", "pet_proxy"]].iloc[:-1]

        # Regresi dengan constraint fisika (Ridge)
        model = Ridge(positive=False).fit(X, y_delta)
        self.beta_0 = float(model.intercept_)
        self.beta_1 = float(model.coef_[0])
        self.beta_2 = float(model.coef_[1])
        self.is_fitted = True
        return self

    def predict_delta_wtd(self, rain_mm: float, pet_mm: float) -> float:
        """Menghitung \\Delta WTD_t = \\beta_0 + \\beta_1 P_t + \\beta_2 PET_t"""
        delta = self.beta_0 + (self.beta_1 * float(rain_mm)) + (self.beta_2 * float(pet_mm))
        return float(delta)

    def forecast_step(self, current_wtd: float, rain_mm: float, pet_mm: float) -> float:
        """Menghitung WTD_{t+1} = WTD_t + \\Delta WTD_t"""
        delta = self.predict_delta_wtd(rain_mm, pet_mm)
        next_wtd = max(0.0, current_wtd + delta)
        return float(round(next_wtd, 2))


# ---------------------------------------------------------------------------
# 13. Step-by-Step 7-Day Scenario Forecast (Slide 16 & 17)
# ---------------------------------------------------------------------------
def run_7day_climate_hydrology_forecast(
    current_wtd_cm: float,
    climate_scenarios: List[Dict[str, Any]],
    baseline_df: Optional[pd.DataFrame] = None,
    hydro_model: Optional[HydrologicalTransitionModel] = None,
    land_type: LandType = LandType.PEATLAND,
) -> List[Dict[str, Any]]:
    """
    Eksekusi Proyeksi 7-Hari Sesuai Slide 16 & 17:
      Climate scenario -> Rainfall + PET -> WTD forecast -> Hydrological Stress -> PDWI/MDWI Forecast
    """
    if hydro_model is None:
        hydro_model = HydrologicalTransitionModel()
        if baseline_df is not None:
            hydro_model.fit_from_daily_history(baseline_df)

    rain_7d_ref = (
        float(baseline_df["rain_mm"].rolling(7, min_periods=1).sum().median())
        if baseline_df is not None and "rain_mm" in baseline_df.columns
        else 15.0
    )

    wtd_running = float(current_wtd_cm)
    forecast_results = []
    accumulated_rain_window = []

    for idx, sc in enumerate(climate_scenarios, start=1):
        rain = float(sc.get("rain_mm", 0.0))
        temp = float(sc.get("temp_c", 26.0))
        rh = float(sc.get("rh_pct", 80.0))
        solar = float(sc.get("solar_wm2", 150.0))
        wind = float(sc.get("wind_ms", 1.0))

        vpd = calculate_vpd(temp, rh)
        pet = calculate_pet_proxy(solar, vpd, wind)

        # 1. Update WTD melalui model regresi hidrologis
        delta_wtd = hydro_model.predict_delta_wtd(rain, pet)
        wtd_running = max(0.0, wtd_running + delta_wtd)

        accumulated_rain_window.append(rain)
        rain_7d_curr = float(sum(accumulated_rain_window[-7:]))

        # 2. Hitung Level 1 dan Level 2
        l1 = calculate_drought_level_1(
            wtd_cm=wtd_running,
            vpd_kpa=vpd,
            solar_radiation_wm2=solar,
            baseline_df=baseline_df,
            land_type=land_type
        )

        l2 = calculate_drought_level_2(
            wtd_cm=wtd_running,
            vpd_kpa=vpd,
            solar_radiation_wm2=solar,
            rain_7d_mm=rain_7d_curr,
            rain_7d_ref_mm=rain_7d_ref,
            baseline_df=baseline_df,
            land_type=land_type
        )

        forecast_results.append({
            "hari": f"D+{idx}",
            "hari_ke": idx,
            "wtd_forecast_cm": round(wtd_running, 2),
            "tmat_forecast_cm": round(-wtd_running, 2),
            "delta_wtd_cm": round(delta_wtd, 2),
            "skenario_iklim": {
                "rain_mm": rain,
                "temp_c": temp,
                "rh_pct": rh,
                "solar_wm2": solar,
                "wind_ms": wind,
                "vpd_kpa": vpd,
                "pet_proxy_mm": pet,
            },
            "level_1": {
                "score": l1["score"],
                "class": l1["drought_class"],
                "label": l1["drought_label"],
            },
            "level_2": {
                "score": l2["score"],
                "class": l2["drought_class"],
                "label": l2["drought_label"],
            },
            "status_kekeringan": l2["drought_label"],
        })

    return forecast_results
