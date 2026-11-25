import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bxbpkhngcflsyotgbhqd.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_60eJEoSgHghjC_2mmxLGpQ_QRf1u3I5")

# JWT Authentication Config
JWT_SECRET = os.getenv("JWT_SECRET", SUPABASE_KEY)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")) # Default 24 jam

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default File Paths
DEFAULT_RAW_DATA_PATH = DATA_RAW_DIR / "Data Dummy AWS dan TMAT.xlsx"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "aws_processed_dataset.csv"

# Model File Paths
RAIN_MODEL_PATH = MODELS_DIR / "rain_prediction_model.joblib"
FIRE_MODEL_PATH = MODELS_DIR / "fire_risk_model.joblib"
DROUGHT_MODEL_PATH = MODELS_DIR / "drought_risk_model.joblib"
FORECAST_MODEL_PATH = MODELS_DIR / "peatland_forecast_engine.joblib"
METRICS_REPORT_PATH = MODELS_DIR / "metrics_summary.json"

# Column Mapping Standard (Input Raw Excel -> Clean Internal Name)
RAW_COLUMN_MAPPING = {
    "Waktu (WIB)": "timestamp",
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
}

# Default AWS Coordinates (e.g. Kalimantan/Sumatera Peatland AWS station)
DEFAULT_LATITUDE = -0.0263
DEFAULT_LONGITUDE = 109.3425

# Domain Thresholds & Business Logic
# Peatland Water Table Depth (TMAT BRGM/KLHK Standard):
# Safe (> -40 cm), Warning (-80 cm to -40 cm), Danger (< -80 cm)
TMAT_SAFE_THRESHOLD = -40.0
TMAT_DANGER_THRESHOLD = -80.0
TMAT_CRITICAL_THRESHOLD = -120.0

# Fire Danger Risk Levels (Document 1: Pseudo Fire Weather Index - FWI)
FIRE_RISK_LABELS = {
    0: "Rendah (Low)",
    1: "Sedang (Moderate)",
    2: "Tinggi (High)",
    3: "Ekstrim / Bahaya (Extreme / Danger)",
}

# Drought Risk Levels
DROUGHT_RISK_LABELS = {
    0: "Normal",
    1: "Waspada (Moderate Drought)",
    2: "Kritis (Severe Drought)",
}

# Rain Prediction Labels
RAIN_LABELS = {
    0: "Tidak Hujan (No Rain)",
    1: "Berpotensi Hujan (Rain Expected)",
}
