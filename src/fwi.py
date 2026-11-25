"""
Pseudo Fire Weather Index (Pseudo-FWI) Module
Based on Document 1: 'Peringatan Dini Bahaya Kebakaran di Lahan Terbuka'
Formula:
    Pseudo FWI = (0.35 * Temperature Score) + 
                 (0.35 * Humidity Score) + 
                 (0.20 * Solar Radiation Score) + 
                 (0.10 * Rainfall Score)

Score Normalization (0 - 1):
    1. Suhu Udara:
       - <28°C   : 0.0
       - 28–32°C : 0.3
       - 32–36°C : 0.6
       - >36°C   : 1.0

    2. Kelembapan Relatif (RH):
       - >80%    : 0.0
       - 65–80%  : 0.3
       - 45–65%  : 0.6
       - <45%    : 1.0

    3. Radiasi Sinar Matahari:
       - <300 W/m²   : 0.0
       - 300–600 W/m²: 0.3
       - 600–800 W/m²: 0.6
       - >800 W/m²   : 1.0

    4. Curah Hujan (Kumulatif / Harian):
       - >20 mm : 0.0
       - 5–20 mm: 0.3
       - 0–5 mm : 1.0

FWI Index Risk Classification (0 - 1):
    - 0.00 – 0.25 : Rendah (Low)     -> Class 0
    - 0.26 – 0.50 : Sedang (Moderate) -> Class 1
    - 0.51 – 0.75 : Tinggi (High)     -> Class 2
    - 0.76 – 1.00 : Ekstrim (Extreme) -> Class 3
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd

from src.config import FIRE_RISK_LABELS

# Weights definition according to Document 1
WEIGHT_TEMPERATURE = 0.35
WEIGHT_HUMIDITY = 0.35
WEIGHT_SOLAR = 0.20
WEIGHT_RAINFALL = 0.10


def calculate_temperature_score(temperature: float) -> float:
    """Computes risk score for air temperature (°C)."""
    if temperature < 28.0:
        return 0.0
    elif temperature <= 32.0:
        return 0.3
    elif temperature <= 36.0:
        return 0.6
    else:
        return 1.0


def calculate_humidity_score(rh: float) -> float:
    """Computes risk score for relative humidity (%)."""
    if rh > 80.0:
        return 0.0
    elif rh >= 65.0:
        return 0.3
    elif rh >= 45.0:
        return 0.6
    else:
        return 1.0


def calculate_solar_score(solar: float) -> float:
    """Computes risk score for solar radiation (W/m²)."""
    if solar < 300.0:
        return 0.0
    elif solar <= 600.0:
        return 0.3
    elif solar <= 800.0:
        return 0.6
    else:
        return 1.0


def calculate_rainfall_score(rain: float) -> float:
    """Computes risk score for rainfall (mm)."""
    if rain > 20.0:
        return 0.0
    elif rain >= 5.0:
        return 0.3
    else:
        return 1.0


def classify_fwi(score: float) -> Tuple[int, str]:
    """
    Classifies Pseudo-FWI into risk class and label:
      0: 0.00 - 0.25 (Rendah)
      1: 0.26 - 0.50 (Sedang)
      2: 0.51 - 0.75 (Tinggi)
      3: 0.76 - 1.00 (Ekstrim)
    """
    if score <= 0.25:
        risk_class = 0
    elif score <= 0.50:
        risk_class = 1
    elif score <= 0.75:
        risk_class = 2
    else:
        risk_class = 3
    
    return risk_class, FIRE_RISK_LABELS.get(risk_class, "Unknown")


def calculate_pseudo_fwi(
    temperature: float,
    relative_humidity: float,
    solar_radiation: float,
    rainfall: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculates Pseudo-FWI for a single weather observation.
    
    Args:
        temperature: Air temperature in Celsius (°C)
        relative_humidity: Relative humidity percentage (0-100%)
        solar_radiation: Solar radiation in W/m²
        rainfall: Rainfall in mm (intensity or cumulative)
        
    Returns:
        Dict containing index score, risk class, label, subscores, and weights.
    """
    t_score = calculate_temperature_score(float(temperature))
    rh_score = calculate_humidity_score(float(relative_humidity))
    solar_score = calculate_solar_score(float(solar_radiation))
    rain_score = calculate_rainfall_score(float(rainfall))

    pseudo_fwi = (
        (WEIGHT_TEMPERATURE * t_score)
        + (WEIGHT_HUMIDITY * rh_score)
        + (WEIGHT_SOLAR * solar_score)
        + (WEIGHT_RAINFALL * rain_score)
    )
    pseudo_fwi = min(max(pseudo_fwi, 0.0), 1.0)
    risk_class, risk_label = classify_fwi(pseudo_fwi)

    return {
        "pseudo_fwi_score": round(pseudo_fwi, 4),
        "risk_class": risk_class,
        "risk_label": risk_label,
        "sub_scores": {
            "temperature_score": t_score,
            "humidity_score": rh_score,
            "solar_radiation_score": solar_score,
            "rainfall_score": rain_score,
        },
        "weights": {
            "temperature": WEIGHT_TEMPERATURE,
            "humidity": WEIGHT_HUMIDITY,
            "solar_radiation": WEIGHT_SOLAR,
            "rainfall": WEIGHT_RAINFALL,
        },
        "algorithm_reference": "Document 1: Pseudo Fire Weather Index (FWI) adapted from Van Wagner (1987)",
    }


def compute_vectorized_fwi(
    df: pd.DataFrame,
    temp_col: str = "air_temperature_c",
    rh_col: str = "relative_humidity_pct",
    solar_col: str = "solar_radiation_wm2",
    rain_col: str = "rain_cum_24h",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized calculation of Pseudo-FWI and target risk classes for a DataFrame.
    """
    # 1. Temperature Score
    t = df[temp_col].values
    t_score = np.select(
        [t < 28.0, t <= 32.0, t <= 36.0],
        [0.0, 0.3, 0.6],
        default=1.0,
    )

    # 2. Humidity Score
    rh = df[rh_col].values
    rh_score = np.select(
        [rh > 80.0, rh >= 65.0, rh >= 45.0],
        [0.0, 0.3, 0.6],
        default=1.0,
    )

    # 3. Solar Radiation Score
    s = df[solar_col].values
    s_score = np.select(
        [s < 300.0, s <= 600.0, s <= 800.0],
        [0.0, 0.3, 0.6],
        default=1.0,
    )

    # 4. Rainfall Score
    # Check if rain column exists, otherwise fallback to rainfall_intensity_mmh
    if rain_col in df.columns:
        r = df[rain_col].values
    elif "rainfall_intensity_mmh" in df.columns:
        r = df["rainfall_intensity_mmh"].values
    else:
        r = np.zeros(len(df))

    r_score = np.select(
        [r > 20.0, r >= 5.0],
        [0.0, 0.3],
        default=1.0,
    )

    # Pseudo-FWI aggregate
    pseudo_fwi = (
        (WEIGHT_TEMPERATURE * t_score)
        + (WEIGHT_HUMIDITY * rh_score)
        + (WEIGHT_SOLAR * s_score)
        + (WEIGHT_RAINFALL * r_score)
    )
    pseudo_fwi = np.clip(pseudo_fwi, 0.0, 1.0)

    # Target class: 0 (Rendah), 1 (Sedang), 2 (Tinggi), 3 (Ekstrim)
    risk_classes = np.select(
        [pseudo_fwi <= 0.25, pseudo_fwi <= 0.50, pseudo_fwi <= 0.75],
        [0, 1, 2],
        default=3,
    )

    return pseudo_fwi, risk_classes
