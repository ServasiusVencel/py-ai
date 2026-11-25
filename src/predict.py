import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Optional
from pathlib import Path

from src.config import (
    RAIN_MODEL_PATH,
    FIRE_MODEL_PATH,
    DROUGHT_MODEL_PATH,
    RAIN_LABELS,
    FIRE_RISK_LABELS,
    DROUGHT_RISK_LABELS,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
)
from src.preprocessor import AWSPreprocessor

from src.fwi import calculate_pseudo_fwi
from src.forecast import (
    generate_10_day_forecast,
    generate_7_day_forecast,
    format_direct_summary,
    map_fire_risk_level,
    map_drought_risk_level,
    map_rain_level,
)

class AWSPredictor:
    """
    Unified Inference Engine for IoT Automatic Weather Station (AWS).
    Implements the 5 Layers of Nimbus Fire & Environmental Intelligence:
      - Layer 1: Scientific Baseline Index (Pseudo-FWI Dokumen 1)
      - Layer 2: Site-Specific Hydrological Calibration (WTD Anomaly, Water Retention)
      - Layer 3: AI Predictive Machine Learning (Lags, Interactions, Drawdown Trends)
      - Layer 4: Multi-Horizon Forecasting (1-day, 3-day, 7-day, 10-day forward projections)
      - Layer 5: Continuous Learning & Hot-Reload Retraining
    """
    def __init__(
        self,
        rain_model_path: Union[str, Path] = RAIN_MODEL_PATH,
        fire_model_path: Union[str, Path] = FIRE_MODEL_PATH,
        drought_model_path: Union[str, Path] = DROUGHT_MODEL_PATH,
    ):
        print("[*] Initializing AWS Disaster Prediction Inference Engine...")
        self.rain_bundle = joblib.load(rain_model_path)
        self.fire_bundle = joblib.load(fire_model_path)
        self.drought_bundle = joblib.load(drought_model_path)

        self.rain_model = self.rain_bundle["model"]
        self.fire_model = self.fire_bundle["model"]
        self.drought_model = self.drought_bundle["model"]
        
        self.features = self.rain_bundle["features"]
        self.preprocessor = AWSPreprocessor()
        self.preprocessor.feature_columns = self.features
        print(f"[+] All 3 AI Models loaded successfully. Model features: {len(self.features)}")

    def _generate_early_warnings(
        self,
        rain_pred: int,
        fire_pred: int,
        drought_pred: int,
        sensor_data: Dict[str, Any],
        fwi_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generates contextual Early Warning System (EWS) alerts for decision makers.
        """
        alerts = []
        effective_fire_class = max(fire_pred, fwi_result["risk_class"] if fwi_result else fire_pred)
        fwi_score = fwi_result["pseudo_fwi_score"] if fwi_result else "N/A"
        
        # Fire Risk Warnings (Document 1: Pseudo-FWI: 0=Rendah, 1=Sedang, 2=Tinggi, 3=Ekstrim)
        if effective_fire_class >= 3:
            alerts.append({
                "level": "CRITICAL",
                "category": "FIRE_HAZARD",
                "title": "🔥 BAHAYA KEBAKARAN LAHAN EKSTRIM (NIMBUS INTELLIGENCE EKSTRIM)",
                "message": (
                    f"Kondisi mikroklimat & gambut sangat ekstrem! Skor Pseudo-FWI: {fwi_score}. "
                    f"Suhu udara {sensor_data.get('air_temperature_c', 'N/A')}°C, "
                    f"RH {sensor_data.get('relative_humidity_pct', 'N/A')}%, "
                    f"radiasi matahari {sensor_data.get('solar_radiation_wm2', 'N/A')} W/m². "
                    "Bahan bakar vegetasi sangat kering dan api sangat mudah menyebar. Siagakan tim pemadam darurat!"
                )
            })
        elif effective_fire_class == 2:
            alerts.append({
                "level": "DANGER",
                "category": "FIRE_HAZARD",
                "title": "🚨 WASPADA KEBAKARAN LAHAN TINGGI (NIMBUS INTELLIGENCE TINGGI)",
                "message": (
                    f"Indeks Pseudo-FWI tinggi: {fwi_score}. "
                    f"Vegetasi kering dan mudah terbakar (Suhu {sensor_data.get('air_temperature_c', 'N/A')}°C, "
                    f"RH {sensor_data.get('relative_humidity_pct', 'N/A')}%). "
                    "Tingkatkan patroli dan siagakan sekat kanal."
                )
            })
        elif effective_fire_class == 1:
            alerts.append({
                "level": "WARNING",
                "category": "FIRE_HAZARD",
                "title": "⚠️ POTENSI KEBAKARAN LAHAN SEDANG (NIMBUS INTELLIGENCE SEDANG)",
                "message": (
                    f"Indeks Pseudo-FWI sedang: {fwi_score}. "
                    "Vegetasi mulai mengering. Lakukan pemantauan berkala pada lahan terbuka."
                )
            })

        # Drought Warnings
        if drought_pred == 2:
            alerts.append({
                "level": "DANGER",
                "category": "DROUGHT_HAZARD",
                "title": "🚨 KRISIS KEKERINGAN LAHAN & AIR TANAH",
                "message": (
                    f"Defisit air tanah parah (TMAT {sensor_data.get('water_table_depth_cm', 'N/A')} cm) "
                    f"dan kelembaban tanah {sensor_data.get('soil_moisture_pct', 'N/A')}%. "
                    "Perlu irigasi darurat / pembasahan lahan gambut segera."
                )
            })
        elif drought_pred == 1:
            alerts.append({
                "level": "WARNING",
                "category": "DROUGHT_HAZARD",
                "title": "⚠️ WASPADA KEKERINGAN",
                "message": "Terjadi tren penurunan cadangan air tanah dan kelembaban lingkungan."
            })

        # Rain Warnings
        if rain_pred == 1:
            alerts.append({
                "level": "INFO",
                "category": "RAIN_PRECIPITATION",
                "title": "🌧️ POTENSI HUJAN / PRESIPITASI",
                "message": "Saturasi kelembaban tinggi terdeteksi, berpotensi hujan dalam waktu dekat."
            })

        if not alerts:
            alerts.append({
                "level": "NORMAL",
                "category": "ENVIRONMENT_SAFE",
                "title": "✅ KONDISI LINGKUNGAN AMAN",
                "message": f"Kondisi stabil. Indeks risiko kebakaran Pseudo-FWI Rendah ({fwi_score})."
            })

        return alerts

    def predict(self, sensor_payload: Dict[str, Any], include_forecast: bool = True) -> Dict[str, Any]:
        """
        Executes real-time inference on a single IoT AWS sensor telemetry message,
        generates direct simplified risk levels (Tingkat Kebakaran, Kekeringan, Hujan),
        and projects 7-day and 10-day forward forecasts based on models trained on ALL data.
        """
        # Ensure coordinates
        lat = sensor_payload.get("latitude", sensor_payload.get("lat", DEFAULT_LATITUDE))
        lon = sensor_payload.get("longitude", sensor_payload.get("lon", DEFAULT_LONGITUDE))
        sensor_payload["latitude"] = float(lat)
        sensor_payload["longitude"] = float(lon)

        # Standardized sensor readings extraction
        sensor_readings = {
            "air_temperature_c": float(sensor_payload.get("air_temperature_c", sensor_payload.get("Suhu Udara (°C)", 0))),
            "relative_humidity_pct": float(sensor_payload.get("relative_humidity_pct", sensor_payload.get("Kelembapan Udara (%)", 0))),
            "solar_radiation_wm2": float(sensor_payload.get("solar_radiation_wm2", sensor_payload.get("Radiasi Matahari (W/m²)", 0))),
            "water_table_depth_cm": float(sensor_payload.get("water_table_depth_cm", sensor_payload.get("Tinggi Muka Air Tanah (cm)", 0))),
            "soil_moisture_pct": float(sensor_payload.get("soil_moisture_pct", sensor_payload.get("Kelembaban Tanah (%)", 0))),
            "rainfall_intensity_mmh": float(sensor_payload.get("rainfall_intensity_mmh", sensor_payload.get("Intensitas Hujan (mm/jam)", 0))),
            "wind_speed_ms": float(sensor_payload.get("wind_speed_ms", sensor_payload.get("Kecepatan Angin (m/s)", 0))),
        }

        # Layer 1: Calculate Deterministic Pseudo-FWI (Dokumen 1 Formula - Scientific Baseline)
        fwi_result = calculate_pseudo_fwi(
            temperature=sensor_readings["air_temperature_c"],
            relative_humidity=sensor_readings["relative_humidity_pct"],
            solar_radiation=sensor_readings["solar_radiation_wm2"],
            rainfall=sensor_readings["rainfall_intensity_mmh"],
        )

        # Feature Transformation with 53 engineered features (Lags, Dry Spell, Interactions)
        X_df = self.preprocessor.transform_single_record(sensor_payload)

        # 1. Rain Prediction
        rain_class = int(self.rain_model.predict(X_df)[0])
        rain_proba = float(self.rain_model.predict_proba(X_df)[0][1]) if hasattr(self.rain_model, "predict_proba") else None

        # 2. Fire Risk Prediction (Layer 3 Machine Learning)
        fire_class = int(self.fire_model.predict(X_df)[0])
        fire_probas = self.fire_model.predict_proba(X_df)[0].tolist() if hasattr(self.fire_model, "predict_proba") else []

        # 3. Drought Risk Prediction
        drought_class = int(self.drought_model.predict(X_df)[0])
        drought_probas = self.drought_model.predict_proba(X_df)[0].tolist() if hasattr(self.drought_model, "predict_proba") else []

        # Land type configuration: "lahan_biasa" (default) or "gambut"
        land_type = str(sensor_payload.get("land_type", "lahan_biasa")).lower()

        # Layer 2 & 3 Integration: Site-Specific Hydrological Calibration & Multi-Variable Interactions
        site_fire_class = fwi_result["risk_class"]
        if land_type == "gambut":
            vpd_val = float(X_df.get("vpd_kpa", [0.0])[0]) if "vpd_kpa" in X_df.columns else 0.0
            wtd_val = sensor_readings["water_table_depth_cm"]
            dry_hours = float(X_df.get("dry_spell_hours", [0.0])[0]) if "dry_spell_hours" in X_df.columns else 0.0

            if wtd_val < -75.0 and vpd_val > 1.4 and dry_hours >= 240.0:
                site_fire_class = max(site_fire_class, 3)  # Ekstrim
            elif wtd_val < -55.0 and vpd_val > 1.1 and dry_hours >= 120.0:
                site_fire_class = max(site_fire_class, 2)  # Tinggi
            elif wtd_val < -45.0 and dry_hours >= 48.0:
                site_fire_class = max(site_fire_class, 1)  # Sedang

        # Nimbus Calibrated Fire Risk (Synthesis of Baseline, ML Model, and Site Calibration)
        effective_fire_class = max(fwi_result["risk_class"], fire_class, site_fire_class)

        # Generate Early Warnings
        alerts = self._generate_early_warnings(
            rain_class,
            effective_fire_class,
            drought_class,
            sensor_readings,
            fwi_result=fwi_result
        )

        # Direct Simplified Status (User request: "tingkat kebakaran tinggi, kekeringan sedang, hujan rendah")
        tingkat_kebakaran = map_fire_risk_level(effective_fire_class)
        kekeringan = map_drought_risk_level(
            sensor_readings["water_table_depth_cm"],
            sensor_readings["soil_moisture_pct"],
            land_type=land_type
        )
        rain_prob_val = rain_proba if rain_proba is not None else (0.85 if rain_class == 1 else 0.10)
        hujan = map_rain_level(rain_prob_val, sensor_readings["rainfall_intensity_mmh"])
        ringkasan = format_direct_summary(tingkat_kebakaran, kekeringan, hujan)

        # Multi-Horizon Forward Forecasts (Trained on ALL data, projecting 7 and 10 days)
        prediksi_7_hari = []
        prediksi_10_hari = []
        if include_forecast:
            sensor_state = {
                **sensor_readings,
                "timestamp": sensor_payload.get("timestamp", str(pd.Timestamp.now()))
            }
            prediksi_10_hari = generate_10_day_forecast(sensor_state, land_type=land_type)
            prediksi_7_hari = prediksi_10_hari[:7]

        return {
            "ringkasan": ringkasan,
            "prediksi_7_hari": prediksi_7_hari,
            "prediksi_10_hari": prediksi_10_hari,
            "station_info": {
                "latitude": sensor_payload["latitude"],
                "longitude": sensor_payload["longitude"],
                "timestamp": str(sensor_payload.get("timestamp", pd.Timestamp.now())),
            },
            "sensor_readings": sensor_readings,
            "predictions": {
                "rain": {
                    "class": rain_class,
                    "label": RAIN_LABELS.get(rain_class, "Unknown"),
                    "level": hujan,
                    "probability_rain": round(rain_proba, 4) if rain_proba is not None else None,
                },
                "fire_risk": {
                    "class": effective_fire_class,
                    "label": FIRE_RISK_LABELS.get(effective_fire_class, fwi_result["risk_label"]),
                    "level": tingkat_kebakaran,
                    "pseudo_fwi_score": fwi_result["pseudo_fwi_score"],
                    "fwi_baseline_class": fwi_result["risk_class"],
                    "ml_model_predicted_class": fire_class,
                    "site_calibrated_class": site_fire_class,
                    "algorithm": "Nimbus Fire Intelligence Engine (Pseudo-FWI + Site Calibration + Full-History ML)",
                    "formula": "0.35*Temp + 0.35*RH + 0.20*Solar + 0.10*Rain (Baseline) + Site Hydrological Lags",
                    "sub_scores": fwi_result["sub_scores"],
                    "weights": fwi_result["weights"],
                    "probabilities": {
                        FIRE_RISK_LABELS[i]: round(fire_probas[i], 4) for i in range(len(fire_probas))
                    } if fire_probas else {},
                },
                "drought_risk": {
                    "class": drought_class,
                    "label": DROUGHT_RISK_LABELS.get(drought_class, "Unknown"),
                    "level": kekeringan,
                    "probabilities": {
                        DROUGHT_RISK_LABELS[i]: round(drought_probas[i], 4) for i in range(len(drought_probas))
                    } if drought_probas else {},
                },
            },
            "early_warning_system": alerts,
        }

if __name__ == "__main__":
    predictor = AWSPredictor()
    
    # Test Scenario: High Fire Risk and Drought (Dry peatland condition)
    sample_telemetry = {
        "timestamp": "2026-07-15 13:00:00",
        "Tinggi Muka Air Tanah (cm)": -135.5,
        "Kelembaban Tanah (%)": 82.3,
        "Intensitas Hujan (mm/jam)": 0.0,
        "Suhu Udara (°C)": 30.2,
        "Suhu Titik Embun (°C)": 18.5,
        "Kelembapan Udara (%)": 58,
        "Radiasi Matahari (W/m²)": 650.0,
        "Tekanan Udara (hPa)": 984.5,
        "Kecepatan Angin (m/s)": 2.4,
        "Arah Angin (°)": 120,
        "latitude": -0.0263,
        "longitude": 109.3425
    }
    
    result = predictor.predict(sample_telemetry)
    import json
    print("\n--- INFERENCE RESULT ---")
    print(json.dumps(result, indent=2))
