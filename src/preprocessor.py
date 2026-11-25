import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from src.config import (
    TMAT_SAFE_THRESHOLD,
    TMAT_DANGER_THRESHOLD,
    TMAT_CRITICAL_THRESHOLD,
    PROCESSED_DATA_PATH,
)
from src.fwi import compute_vectorized_fwi

class AWSPreprocessor:
    """
    Data preprocessor and feature engineering pipeline for IoT Automatic Weather Station (AWS)
    and Peatland Water Table Depth (TMAT/WTD).
    
    Implements Nimbus Fire & Environmental Intelligence Features:
    - Lag effects (WTD, Rainfall, VPD across 1h, 6h, 24h)
    - Cumulative dry spell (hours without significant rainfall)
    - Low WTD duration (hours WTD < -40cm and < -60cm)
    - Non-linear variable interactions (WTD x VPD, WTD x dry spell, VPD x solar)
    - Site-specific hydrological baseline calibration & anomaly detection
    """
    def __init__(self):
        self.feature_columns: List[str] = []
        # Site-specific calibration parameters learned from historical observations
        self.site_wtd_mean: float = -104.46
        self.site_wtd_std: float = 38.85
        self.site_wtd_p20: float = -138.0

    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans missing or erratic sensor readings using time-series forward/backward fill.
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
        return df_clean

    def engineer_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates domain-specific physical meteorological and hydrological features.
        """
        df_feat = df.copy()

        # 1. Dew Point Depression (T - Td) - Crucial for precipitation and condensation
        if 'dew_point_c' not in df_feat.columns or df_feat['dew_point_c'].isna().all():
            # Approximation from RH and Temp (Lawrence 2005)
            df_feat['dew_point_c'] = df_feat['air_temperature_c'] - ((100.0 - df_feat['relative_humidity_pct']) / 5.0)

        df_feat["dew_point_depression_c"] = df_feat["air_temperature_c"] - df_feat["dew_point_c"]

        # 2. Saturated Vapor Pressure (es) and Actual Vapor Pressure (ea) in kPa (Tetens Formula)
        es = 0.61078 * np.exp((17.27 * df_feat["air_temperature_c"]) / (df_feat["air_temperature_c"] + 237.3))
        ea = es * (df_feat["relative_humidity_pct"] / 100.0)
        
        # Vapor Pressure Deficit (VPD in kPa) - atmospheric drying demand
        df_feat["vpd_kpa"] = np.maximum(0.0, es - ea)

        # 3. Apparent Temperature / Heat Index estimation (°C)
        df_feat["apparent_temp_c"] = (
            -2.653 
            + (0.994 * df_feat["air_temperature_c"]) 
            + (0.0153 * (df_feat["dew_point_c"] ** 2))
        )

        # 4. Cyclical Temporal Features
        if "timestamp" in df_feat.columns and pd.api.types.is_datetime64_any_dtype(df_feat["timestamp"]):
            hours = df_feat["timestamp"].dt.hour + df_feat["timestamp"].dt.minute / 60.0
            df_feat["hour"] = df_feat["timestamp"].dt.hour
            df_feat["day_of_week"] = df_feat["timestamp"].dt.dayofweek
            df_feat["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
            df_feat["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
            df_feat["is_daytime"] = (df_feat["solar_radiation_wm2"] > 5.0).astype(int)
        else:
            df_feat["hour"] = 12
            df_feat["day_of_week"] = 0
            df_feat["sin_hour"] = 0.0
            df_feat["cos_hour"] = 1.0
            df_feat["is_daytime"] = (df_feat.get("solar_radiation_wm2", 0) > 5.0).astype(int) if isinstance(df_feat.get("solar_radiation_wm2"), pd.Series) else 1

        # 5. Cyclical Wind Direction Features
        wind_deg = df_feat.get("wind_direction_deg", 0)
        wind_rad = np.radians(wind_deg)
        df_feat["sin_wind_dir"] = np.sin(wind_rad)
        df_feat["cos_wind_dir"] = np.cos(wind_rad)

        # 6. Wind Speed Components (U and V vectors)
        wind_speed = df_feat.get("wind_speed_ms", 0)
        df_feat["wind_u"] = -wind_speed * np.sin(wind_rad)
        df_feat["wind_v"] = -wind_speed * np.cos(wind_rad)

        return df_feat

    def engineer_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates rolling time-series window aggregates and temporal lag features:
        - 1 step = 5 min
        - 1h = 12 steps
        - 6h = 72 steps
        - 24h = 288 steps
        """
        df_roll = df.copy()

        w_1h = 12
        w_6h = 72
        w_24h = 288

        # --- A. Rolling Temperature & Humidity Trends ---
        df_roll["temp_rolling_mean_1h"] = df_roll["air_temperature_c"].rolling(w_1h, min_periods=1).mean()
        df_roll["humidity_rolling_mean_1h"] = df_roll["relative_humidity_pct"].rolling(w_1h, min_periods=1).mean()
        df_roll["temp_rolling_std_6h"] = df_roll["air_temperature_c"].rolling(w_6h, min_periods=1).std().fillna(0)
        df_roll["solar_rolling_mean_6h"] = df_roll["solar_radiation_wm2"].rolling(w_6h, min_periods=1).mean()

        # --- B. Rain Lags and Cumulative Precipitation ---
        df_roll["rain_cum_1h"] = df_roll["rainfall_intensity_mmh"].rolling(w_1h, min_periods=1).sum() / 12.0
        df_roll["rain_cum_24h"] = df_roll["rainfall_intensity_mmh"].rolling(w_24h, min_periods=1).sum() / 12.0
        df_roll["rain_lag_1h"] = df_roll["rainfall_intensity_mmh"].shift(w_1h).fillna(0)
        df_roll["rain_lag_6h"] = df_roll["rainfall_intensity_mmh"].shift(w_6h).fillna(0)
        df_roll["rain_lag_24h"] = df_roll["rainfall_intensity_mmh"].shift(w_24h).fillna(0)

        # --- C. WTD / TMAT Temporal Dependencies & Lag Effects ---
        df_roll["tmat_rolling_mean_24h"] = df_roll["water_table_depth_cm"].rolling(w_24h, min_periods=1).mean()
        df_roll["tmat_delta_1h"] = df_roll["water_table_depth_cm"] - df_roll["water_table_depth_cm"].shift(w_1h).fillna(df_roll["water_table_depth_cm"])
        df_roll["wtd_lag_1h"] = df_roll["water_table_depth_cm"].shift(w_1h).fillna(df_roll["water_table_depth_cm"])
        df_roll["wtd_lag_6h"] = df_roll["water_table_depth_cm"].shift(w_6h).fillna(df_roll["water_table_depth_cm"])
        df_roll["wtd_lag_24h"] = df_roll["water_table_depth_cm"].shift(w_24h).fillna(df_roll["water_table_depth_cm"])
        df_roll["wtd_drawdown_rate_24h"] = df_roll["water_table_depth_cm"] - df_roll["wtd_lag_24h"]

        # --- D. VPD Lags & Cumulative Atmospheric Drying Demand ---
        df_roll["vpd_rolling_mean_6h"] = df_roll["vpd_kpa"].rolling(w_6h, min_periods=1).mean()
        df_roll["vpd_cum_24h"] = df_roll["vpd_kpa"].rolling(w_24h, min_periods=1).sum() / 12.0
        df_roll["vpd_lag_1h"] = df_roll["vpd_kpa"].shift(w_1h).fillna(df_roll["vpd_kpa"])
        df_roll["vpd_lag_6h"] = df_roll["vpd_kpa"].shift(w_6h).fillna(df_roll["vpd_kpa"])
        df_roll["vpd_lag_24h"] = df_roll["vpd_kpa"].shift(w_24h).fillna(df_roll["vpd_kpa"])

        # --- E. Dry Spell Duration & Consecutive Low WTD ---
        # Consecutive steps with rain < 0.1 mm/h
        is_dry = (df_roll["rainfall_intensity_mmh"] < 0.1).astype(int)
        dry_groups = (is_dry != is_dry.shift()).cumsum()
        dry_steps = is_dry.groupby(dry_groups).cumsum()
        df_roll["dry_spell_hours"] = (dry_steps * (5.0 / 60.0)).round(2)

        # Consecutive hours WTD < -40 cm (Regulatory threshold)
        is_low_wtd = (df_roll["water_table_depth_cm"] < -40.0).astype(int)
        low_wtd_groups = (is_low_wtd != is_low_wtd.shift()).cumsum()
        low_wtd_steps = is_low_wtd.groupby(low_wtd_groups).cumsum()
        df_roll["consecutive_low_wtd_hours"] = (low_wtd_steps * (5.0 / 60.0)).round(2)

        # Soil moisture rolling mean
        df_roll["soil_moisture_rolling_mean_24h"] = df_roll["soil_moisture_pct"].rolling(w_24h, min_periods=1).mean()

        # --- F. Non-Linear Interactions (Nimbus Model Intelligence) ---
        df_roll["interaction_wtd_vpd"] = df_roll["water_table_depth_cm"] * df_roll["vpd_kpa"]
        df_roll["interaction_wtd_dryspell"] = df_roll["water_table_depth_cm"] * df_roll["dry_spell_hours"]
        df_roll["interaction_vpd_solar"] = df_roll["vpd_kpa"] * df_roll["solar_radiation_wm2"]
        df_roll["interaction_wtd_trend_vpd"] = df_roll["wtd_drawdown_rate_24h"] * df_roll["vpd_kpa"]

        # --- G. Site-Specific Hydrological Baseline Calibration ---
        df_roll["wtd_site_anomaly"] = df_roll["water_table_depth_cm"] - self.site_wtd_mean
        df_roll["is_site_wtd_depressed"] = (df_roll["water_table_depth_cm"] < self.site_wtd_p20).astype(int)

        return df_roll

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives ground-truth target labels for the 3 disaster / condition scenarios:
        1. Rain Prediction (Binary: 0=No Rain, 1=Rain Occurrence in near horizon)
        2. Fire Risk (3 Classes: 0=Low, 1=Moderate, 2=High/Danger) based on Dokumen 1 Pseudo-FWI
        3. Drought Risk (3 Classes: 0=Normal, 1=Moderate, 2=Severe) based on cumulative TMAT deficit & soil dryness
        """
        df_labeled = df.copy()

        # --- Target 1: Potensi Hujan (Rain Prediction) ---
        rain_now = (df_labeled["rainfall_intensity_mmh"] > 0.1).astype(int)
        rain_lead_1 = (df_labeled["rainfall_intensity_mmh"].shift(-1).fillna(0) > 0.1).astype(int)
        rain_lead_2 = (df_labeled["rainfall_intensity_mmh"].shift(-2).fillna(0) > 0.1).astype(int)
        df_labeled["target_rain"] = ((rain_now | rain_lead_1 | rain_lead_2) > 0).astype(int)

        # --- Target 2: Potensi Kebakaran (Pseudo Fire Weather Index - Dokumen 1) ---
        fwi_scores, fire_classes = compute_vectorized_fwi(df_labeled)
        df_labeled["pseudo_fwi_score"] = fwi_scores
        df_labeled["target_fire_risk"] = fire_classes

        # --- Target 3: Potensi Kekeringan (Drought Risk) ---
        drought_score = np.zeros(len(df_labeled))
        drought_score += np.where(df_labeled["tmat_rolling_mean_24h"] < TMAT_CRITICAL_THRESHOLD, 4.0,
                                  np.where(df_labeled["tmat_rolling_mean_24h"] < TMAT_DANGER_THRESHOLD, 2.5,
                                           np.where(df_labeled["tmat_rolling_mean_24h"] < TMAT_SAFE_THRESHOLD, 1.0, 0.0)))
        
        drought_score += np.where(df_labeled["soil_moisture_rolling_mean_24h"] < 90.0, 3.0,
                                  np.where(df_labeled["soil_moisture_rolling_mean_24h"] < 96.0, 1.5, 0.0))
        
        drought_score += np.where(df_labeled["rain_cum_24h"] < 0.1, 2.0,
                                  np.where(df_labeled["rain_cum_24h"] < 2.0, 1.0, -1.0))
        
        drought_score += np.where(df_labeled["vpd_kpa"] > 1.2, 1.5,
                                  np.where(df_labeled["vpd_kpa"] > 0.6, 0.5, 0.0))

        conditions_drought = [
            (drought_score >= 6.5),
            (drought_score >= 3.5) & (drought_score < 6.5),
            (drought_score < 3.5)
        ]
        choices_drought = [2, 1, 0]
        df_labeled["target_drought_risk"] = np.select(conditions_drought, choices_drought, default=0)

        return df_labeled

    def fit_transform(self, df: pd.DataFrame, save_processed: bool = True) -> pd.DataFrame:
        """
        Executes full preprocessing & feature engineering pipeline.
        """
        print("[*] Starting Data Preprocessing & Feature Engineering...")
        # Calibrate site-specific baseline from complete dataset
        if "water_table_depth_cm" in df.columns:
            self.site_wtd_mean = float(df["water_table_depth_cm"].mean())
            self.site_wtd_std = float(df["water_table_depth_cm"].std())
            self.site_wtd_p20 = float(df["water_table_depth_cm"].quantile(0.20))
            print(f"[+] Site-specific calibration learned: Mean WTD={self.site_wtd_mean:.2f}cm, Std={self.site_wtd_std:.2f}cm, P20={self.site_wtd_p20:.2f}cm")

        df_clean = self.clean_missing_values(df)
        df_physics = self.engineer_physical_features(df_clean)
        df_roll = self.engineer_rolling_features(df_physics)
        df_final = self.create_labels(df_roll)

        df_final = df_final.ffill().bfill()

        exclude_cols = [
            "timestamp", "target_rain", "target_fire_risk", "target_drought_risk", "pseudo_fwi_score"
        ]
        self.feature_columns = [col for col in df_final.columns if col not in exclude_cols]

        if save_processed:
            df_final.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"[+] Saved processed dataset ({len(df_final)} rows) to {PROCESSED_DATA_PATH}")

        print(f"[+] Feature engineering completed with {len(self.feature_columns)} features:")
        print(f"    Features: {self.feature_columns}")
        print("\n[*] Target Distribution Summary:")
        print("    Target Rain:", df_final["target_rain"].value_counts().to_dict())
        print("    Target Fire Risk:", df_final["target_fire_risk"].value_counts().to_dict())
        print("    Target Drought Risk:", df_final["target_drought_risk"].value_counts().to_dict())

        return df_final

    def transform_single_record(self, raw_input: dict) -> pd.DataFrame:
        """
        Transforms a single IoT AWS payload for online inference.
        """
        df_single = pd.DataFrame([raw_input])
        for orig, clean in {
            "Tinggi Muka Air Tanah (cm)": "water_table_depth_cm",
            "Kelembaban Tanah (%)": "soil_moisture_pct",
            "Intensitas Hujan (mm/jam)": "rainfall_intensity_mmh",
            "Suhu Udara (°C)": "air_temperature_c",
            "Suhu Titik Embun (°C)": "dew_point_c",
            "Kelembapan Udara (%)": "relative_humidity_pct",
            "Radiasi Matahari (W/m²)": "solar_radiation_wm2",
            "Tekanan Udara (hPa)": "air_pressure_hpa",
            "Kecepatan Angin (m/s)": "wind_speed_ms",
            "Arah Angin (°)": "wind_direction_deg",
        }.items():
            if orig in df_single.columns and clean not in df_single.columns:
                df_single[clean] = df_single[orig]

        df_physics = self.engineer_physical_features(df_single)
        
        tmat_val = float(df_physics.get("water_table_depth_cm", [self.site_wtd_mean])[0])
        vpd_val = float(df_physics.get("vpd_kpa", [0.5])[0])
        solar_val = float(df_physics.get("solar_radiation_wm2", [300.0])[0])
        rain_val = float(df_physics.get("rainfall_intensity_mmh", [0.0])[0])

        for col in self.feature_columns:
            if col not in df_physics.columns:
                if col == "interaction_wtd_vpd":
                    df_physics[col] = tmat_val * vpd_val
                elif col == "interaction_wtd_dryspell":
                    df_physics[col] = tmat_val * (24.0 if rain_val < 0.1 else 0.0)
                elif col == "interaction_vpd_solar":
                    df_physics[col] = vpd_val * solar_val
                elif col == "interaction_wtd_trend_vpd":
                    df_physics[col] = 0.0
                elif col == "wtd_site_anomaly":
                    df_physics[col] = tmat_val - self.site_wtd_mean
                elif col == "is_site_wtd_depressed":
                    df_physics[col] = 1 if tmat_val < self.site_wtd_p20 else 0
                elif col == "dry_spell_hours":
                    df_physics[col] = 24.0 if rain_val < 0.1 else 0.0
                elif col == "consecutive_low_wtd_hours":
                    df_physics[col] = 24.0 if tmat_val < -40.0 else 0.0
                elif "tmat" in col or "wtd" in col:
                    df_physics[col] = tmat_val
                elif "vpd" in col:
                    df_physics[col] = vpd_val
                elif "rain" in col:
                    df_physics[col] = rain_val
                elif "temp" in col:
                    df_physics[col] = df_physics.get("air_temperature_c", 28.0)
                elif "humidity" in col:
                    df_physics[col] = df_physics.get("relative_humidity_pct", 80.0)
                elif "solar" in col:
                    df_physics[col] = solar_val
                elif "soil_moisture" in col:
                    df_physics[col] = df_physics.get("soil_moisture_pct", 95.0)
                else:
                    df_physics[col] = 0.0

        return df_physics[self.feature_columns]

if __name__ == "__main__":
    from src.data_loader import load_aws_dataset
    df_raw = load_aws_dataset()
    preprocessor = AWSPreprocessor()
    df_processed = preprocessor.fit_transform(df_raw)
