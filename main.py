import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    METRICS_REPORT_PATH,
    RAIN_MODEL_PATH,
    FIRE_MODEL_PATH,
    DROUGHT_MODEL_PATH,
    DEFAULT_RAW_DATA_PATH,
)
from src.data_loader import load_aws_dataset
from src.predict import AWSPredictor
from src.train import main as run_retraining_pipeline
from src.fwi import calculate_pseudo_fwi
from src.drought_index import (
    calculate_drought_level_1,
    calculate_drought_level_2,
    calculate_vpd,
    calculate_pet_proxy,
    convert_tmat_to_wtd,
    convert_wtd_to_tmat,
    aggregate_5min_to_daily,
    run_7day_climate_hydrology_forecast,
    LandType,
)
from src.forecast import (
    analyze_and_forecast_from_dataset,
    analyze_full_dataset_with_ai_models,
    generate_7_day_forecast,
    generate_10_day_forecast,
)
from src import peatland_pdwi as pdwi
from src import peatland_fwi as pfwi
from src.auth import (
    auth_router,
    RoleAuthMiddleware,
    require_user,
    require_superadmin,
    require_non_peatland,
    require_peatland,
    UserInfo,
    get_all_users,
    update_user_by_id,
    delete_user_by_id,
    UserUpdateRequest,
)

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AWS-API")

# Global Predictor and Scheduler Instances
predictor: Optional[AWSPredictor] = None
scheduler = AsyncIOScheduler()

# ---------------------------------------------------------------------------
# Background Task Function for Automated Model Retraining
# ---------------------------------------------------------------------------
def execute_background_retraining():
    """
    Executes the end-to-end retraining pipeline asynchronously and reloads models.
    """
    global predictor
    logger.info("[AUTOMATED RETRAINING] 🚀 Starting model retraining pipeline...")
    try:
        run_retraining_pipeline()
        logger.info("[AUTOMATED RETRAINING] Retraining complete. Reloading updated models into memory...")
        predictor = AWSPredictor()
        logger.info("[AUTOMATED RETRAINING] ✅ Inference engine successfully hot-reloaded with new models!")
    except Exception as e:
        logger.error(f"[AUTOMATED RETRAINING] ❌ Retraining pipeline failed: {e}")

# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    1. Loads ML models into RAM on startup.
    2. Starts the automated cron scheduler (Runs every day at 01:00 AM).
    3. Shuts down scheduler cleanly on stop.
    """
    global predictor
    logger.info("Initializing IoT AWS Machine Learning Models into memory...")
    try:
        predictor = AWSPredictor()
        logger.info("All 3 models loaded and ready for inference!")
    except Exception as e:
        logger.error(f"Failed to load AI models: {e}")
        predictor = None

    # Schedule automated retraining every day at 01:00 AM
    trigger = CronTrigger(hour=1, minute=0)
    scheduler.add_job(
        execute_background_retraining,
        trigger=trigger,
        id="daily_retrain_job",
        name="Daily Model Retraining at 01:00 AM",
        replace_existing=True
    )
    scheduler.start()
    next_run = scheduler.get_job("daily_retrain_job").next_run_time
    logger.info(f"⏰ [SCHEDULER ACTIVATED] Automatic retraining scheduled daily at 01:00 AM. Next run: {next_run}")

    yield

    logger.info("Shutting down automated scheduler and AWS API service...")
    if scheduler.running:
        scheduler.shutdown()

# Initialize FastAPI App
app = FastAPI(
    title="🌧️ Nimbus Fire & Environmental Intelligence API",
    description=(
        "Enterprise-grade REST API for IoT Automatic Weather Station (AWS) & Groundwater Level (TMAT/WTD) monitoring. "
        "Powered by the 5-layer Nimbus Fire Intelligence Engine with Full Historical Dataset Training. "
        "Provides real-time AI predictions for Rain, Peatland Fire, and Drought hazards with Early Warning System alerts. "
        "Includes multi-horizon forecasting (1-day, 3-day, 7-day, 10-day) and automated daily retraining at 01:00 AM."
    ),
    version="1.2.0",
    lifespan=lifespan
)

# Enable CORS for Frontend Dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable Role Authentication Middleware
# Melindungi SELURUH rute /api/v1/... dengan Bearer JWT token secara default.
# Kecuali endpoint publik (/api/v1/auth/login, /api/v1/auth/register, /health, /docs).
app.add_middleware(
    RoleAuthMiddleware,
    protect_all_api=True,
    api_prefix="/api/v1",
    role_protected_routes={
        "/api/v1/superadmin": ["superadmin"],
        "/api/v1/retrain": ["superadmin"],
    }
)

# Register Authentication & RBAC Router
app.include_router(auth_router)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TelemetryPayload(BaseModel):
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    water_table_depth_cm: float = Field(..., alias="Tinggi Muka Air Tanah (cm)", description="Groundwater table depth in cm (e.g. -45.0)")
    soil_moisture_pct: float = Field(..., alias="Kelembaban Tanah (%)", description="Soil moisture percentage (0-100%)")
    rainfall_intensity_mmh: float = Field(default=0.0, alias="Intensitas Hujan (mm/jam)", description="Rainfall intensity in mm/h")
    air_temperature_c: float = Field(..., alias="Suhu Udara (°C)", description="Air temperature in Celsius")
    dew_point_c: Optional[float] = Field(default=None, alias="Suhu Titik Embun (°C)", description="Dew point temperature in Celsius")
    relative_humidity_pct: float = Field(..., alias="Kelembapan Udara (%)", description="Relative humidity in percentage (0-100%)")
    solar_radiation_wm2: float = Field(default=0.0, alias="Radiasi Matahari (W/m²)", description="Solar radiation in W/m²")
    air_pressure_hpa: float = Field(default=985.0, alias="Tekanan Udara (hPa)", description="Atmospheric pressure in hPa")
    wind_speed_ms: float = Field(default=0.0, alias="Kecepatan Angin (m/s)", description="Wind speed in m/s")
    wind_direction_deg: float = Field(default=0.0, alias="Arah Angin (°)", description="Wind direction in degrees (0-360°)")
    latitude: Optional[float] = Field(default=DEFAULT_LATITUDE, description="Station Latitude")
    longitude: Optional[float] = Field(default=DEFAULT_LONGITUDE, description="Station Longitude")
    land_type: Optional[str] = Field(default="lahan_biasa", description="Tipe lahan: 'lahan_biasa' (tanah mineral) atau 'gambut'")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "timestamp": "2026-07-20 13:00:00",
                "water_table_depth_cm": -125.4,
                "soil_moisture_pct": 81.2,
                "rainfall_intensity_mmh": 0.0,
                "air_temperature_c": 31.5,
                "dew_point_c": 18.2,
                "relative_humidity_pct": 52.0,
                "solar_radiation_wm2": 680.0,
                "air_pressure_hpa": 984.2,
                "wind_speed_ms": 2.6,
                "wind_direction_deg": 115.0,
                "latitude": -0.0263,
                "longitude": 109.3425
            }
        }
    }

class RetrainResponse(BaseModel):
    status: str
    message: str
    started_at: str

class FWIRequest(BaseModel):
    air_temperature_c: float = Field(..., alias="Suhu Udara (°C)", description="Suhu udara (°C)")
    relative_humidity_pct: float = Field(..., alias="Kelembapan Udara (%)", description="Kelembapan relatif udara (%)")
    solar_radiation_wm2: float = Field(..., alias="Radiasi Matahari (W/m²)", description="Radiasi sinar matahari (W/m²)")
    rainfall_intensity_mmh: float = Field(default=0.0, alias="Curah Hujan (mm)", description="Curah hujan / intensitas hujan (mm)")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "air_temperature_c": 33.5,
                "relative_humidity_pct": 48.0,
                "solar_radiation_wm2": 720.0,
                "rainfall_intensity_mmh": 0.0
            }
        }
    }

class DroughtCalculationRequest(BaseModel):
    water_table_depth_cm: float = Field(..., alias="Tinggi Muka Air Tanah (cm)", description="TMAT dalam cm (nilai negatif dari sensor, misal -159.37 atau -45.0)")
    air_temperature_c: float = Field(..., alias="Suhu Udara (°C)", description="Suhu udara harian (°C)")
    relative_humidity_pct: float = Field(..., alias="Kelembapan Udara (%)", description="Kelembapan relatif udara (%)")
    solar_radiation_wm2: float = Field(..., alias="Radiasi Matahari (W/m²)", description="Radiasi matahari rata-rata (W/m²)")
    rainfall_intensity_mmh: float = Field(default=0.0, alias="Curah Hujan (mm)", description="Curah hujan harian (mm)")
    rain_7d_mm: Optional[float] = Field(default=None, description="Kumulatif hujan 7 hari (mm) untuk Level 2 Climate-Hydrology")
    soil_moisture_pct: Optional[float] = Field(default=None, alias="Kelembaban Tanah (%)", description="Kelembaban tanah (0-100%)")
    land_type: str = Field(default="gambut", description="Tipe lahan: 'gambut' (Peatland) atau 'lahan_biasa' (Mineral Soil)")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "water_table_depth_cm": -159.37,
                "air_temperature_c": 29.2,
                "relative_humidity_pct": 75.0,
                "solar_radiation_wm2": 320.0,
                "rainfall_intensity_mmh": 0.5,
                "rain_7d_mm": 23.2,
                "soil_moisture_pct": 82.0,
                "land_type": "gambut"
            }
        }
    }

class DroughtForecastScenarioDay(BaseModel):
    rain_mm: float = Field(default=0.0, description="Prediksi hujan harian (mm)")
    temp_c: float = Field(default=26.0, description="Suhu (°C)")
    rh_pct: float = Field(default=80.0, description="Kelembapan (%)")
    solar_wm2: float = Field(default=150.0, description="Radiasi (W/m²)")
    wind_ms: float = Field(default=1.0, description="Kecepatan angin (m/s)")

class DroughtForecastRequest(BaseModel):
    current_water_table_depth_cm: float = Field(..., description="TMAT sensor mutakhir (cm, misal -159.37)")
    land_type: str = Field(default="gambut", description="'gambut' atau 'lahan_biasa'")
    scenarios: Optional[List[DroughtForecastScenarioDay]] = Field(default=None, description="Opsional 7 hari skenario iklim")


class PeatlandCalculationRequest(BaseModel):
    tmat_cm: float = Field(..., description="Tinggi muka air tanah sensor (cm, negatif = di bawah permukaan, misal -159.37)")
    temp_c: float = Field(..., description="Suhu udara (°C)")
    rh_pct: float = Field(..., ge=0, le=100, description="Kelembapan udara (%)")
    solar_wm2: float = Field(..., ge=0, description="Intensitas sinar matahari (W/m²)")
    wind_ms: Optional[float] = Field(default=None, ge=0, description="Opsional: kecepatan angin (m/s); default asumsi 1.0")
    rain_7d_mm: Optional[float] = Field(default=None, ge=0, description="Opsional: akumulasi hujan 7 hari (mm); bila kosong komponen RD dibuang")
    tmat_7d_ago_cm: Optional[float] = Field(default=None, description="Opsional: TMAT 7 hari lalu (cm) untuk tren")
    dry_days: Optional[int] = Field(default=None, ge=0, description="Opsional: hari kering beruntun")
    tanggal: Optional[str] = Field(default=None, description="Opsional: YYYY-MM-DD")


class PeatlandForecastRequest(BaseModel):
    current_tmat_cm: float = Field(..., description="TMAT sensor mutakhir (cm, negatif)")
    current_weather: Optional[DroughtForecastScenarioDay] = Field(default=None, description="Opsional: cuaca hari ini")
    mode: str = Field(default="persistence", description="'persistence' | 'kering' | 'basah' (dipakai bila scenarios kosong)")
    scenarios: Optional[List[DroughtForecastScenarioDay]] = Field(default=None, description="Opsional skenario cuaca D+1..D+N")
    horizon: int = Field(default=7, ge=1, le=14, description="Jumlah hari prakiraan bila scenarios kosong")


@app.post("/api/v1/peatland/drought/calculate", tags=["Peatland Drought (PDWI-Gambut)"])
def peatland_drought_calculate(
    req: PeatlandCalculationRequest,
    _user: UserInfo = Depends(require_peatland),
):
    """PDWI-Gambut dari satu pembacaan sensor (hanya akun is_peatland=True / superadmin)."""
    from datetime import date as _date
    on_date = None
    if req.tanggal:
        try:
            on_date = _date.fromisoformat(req.tanggal)
        except ValueError:
            raise HTTPException(status_code=422, detail="Format tanggal harus YYYY-MM-DD")
    _, cal = pdwi.load_site_dataset()
    return pdwi.calculate_from_sensor(
        req.tmat_cm, req.temp_c, req.rh_pct, req.solar_wm2, cal,
        wind_ms=req.wind_ms, rain_7d_mm=req.rain_7d_mm,
        tmat_7d_ago_cm=req.tmat_7d_ago_cm, dry_days=req.dry_days, on_date=on_date,
    )


@app.post("/api/v1/peatland/drought/forecast", tags=["Peatland Drought (PDWI-Gambut)"])
def peatland_drought_forecast(
    req: PeatlandForecastRequest,
    _user: UserInfo = Depends(require_peatland),
):
    """Prakiraan PDWI-Gambut N hari dari TMAT saat ini + skenario cuaca."""
    if req.mode not in ("persistence", "kering", "basah"):
        raise HTTPException(status_code=422, detail="mode harus persistence | kering | basah")
    _, cal = pdwi.load_site_dataset()
    wtd, _inund = pdwi.wtd_from_tmat(req.current_tmat_cm)
    cur = req.current_weather.model_dump() if req.current_weather else None
    if req.scenarios:
        scen = [s.model_dump() for s in req.scenarios]
    else:
        base = pdwi._weather(cur, pdwi.DEFAULT_ASSUMPTIONS)
        scen = pdwi.build_scenarios(req.mode, base, req.horizon, cal)
    fc = pdwi.forecast_pdwi(wtd, cur, scen, cal)
    w0 = pdwi._weather(cur, pdwi.DEFAULT_ASSUMPTIONS)
    cur_state = pdwi.compute_pdwi_state(
        wtd, pdwi.calculate_vpd(w0["temp_c"], w0["rh_pct"]), w0["solar_wm2"], cal)
    return {
        "tipe_lahan": "gambut",
        "wtd_saat_ini_cm": round(wtd, 2),
        "mode": "skenario_pengguna" if req.scenarios else req.mode,
        "forecast": fc,
        "peringatan_dini": pdwi.early_warning_summary(fc, cur_state["status_operasional"]["class"]),
    }


@app.get("/api/v1/peatland/drought/latest", tags=["Peatland Drought (PDWI-Gambut)"])
def peatland_drought_latest(_user: UserInfo = Depends(require_peatland)):
    """Keadaan terakhir dataset lokasi + riwayat + forecast 7 hari + skenario kering/basah."""
    daily, cal = pdwi.load_site_dataset()
    return pdwi.analyze_latest(daily, cal)


@app.get("/api/v1/peatland/drought/assumptions", tags=["Peatland Drought (PDWI-Gambut)"])
def peatland_drought_assumptions(_user: UserInfo = Depends(require_peatland)):
    """Katalog asumsi parameter + ringkasan kalibrasi lokasi."""
    _, cal = pdwi.load_site_dataset()
    return {"asumsi": pdwi.assumptions_catalog(), "kalibrasi_lokasi": cal.summary()}


# ---------------------------------------------------------------------------
# PFWI-Gambut — Peatland Fire Weather Index (khusus is_peatland=True)
# ---------------------------------------------------------------------------
class PeatlandFWICalculationRequest(BaseModel):
    tmat_cm: float = Field(..., description="Tinggi muka air tanah sensor (cm, negatif = di bawah permukaan)")
    temp_c: float = Field(..., description="Suhu udara siang (idealnya pukul 12:00 WIB), °C")
    rh_pct: float = Field(..., ge=0, le=100, description="Kelembapan udara siang (%)")
    wind_ms: Optional[float] = Field(default=None, ge=0, description="Opsional: angin sensor 2 m (m/s); default asumsi 1.0")
    rain_24h_mm: float = Field(default=0.0, ge=0, description="Hujan 24 jam terakhir s/d pukul 12:00 (mm)")
    prev_ffmc: Optional[float] = Field(default=None, ge=0, le=101, description="Opsional: FFMC kemarin")
    prev_dmc: Optional[float] = Field(default=None, ge=0, description="Opsional: DMC kemarin")
    prev_dc: Optional[float] = Field(default=None, ge=0, description="Opsional: DC kemarin")
    tmat_7d_ago_cm: Optional[float] = Field(default=None, description="Opsional: TMAT 7 hari lalu (cm) untuk tren")
    dry_days: Optional[int] = Field(default=None, ge=0, description="Opsional: hari kering beruntun (hujan < 1 mm)")
    tanggal: Optional[str] = Field(default=None, description="Opsional: YYYY-MM-DD")

    model_config = {"json_schema_extra": {"example": {
        "tmat_cm": -95.0, "temp_c": 32.5, "rh_pct": 55.0, "wind_ms": 2.5, "rain_24h_mm": 0.0,
        "prev_ffmc": 86.0, "prev_dmc": 25.0, "prev_dc": 280.0, "tmat_7d_ago_cm": -70.0, "dry_days": 9,
    }}}


class PeatlandFWIForecastRequest(PeatlandForecastRequest):
    prev_ffmc: Optional[float] = Field(default=None, ge=0, le=101, description="Opsional: FFMC hari ini")
    prev_dmc: Optional[float] = Field(default=None, ge=0, description="Opsional: DMC hari ini")
    prev_dc: Optional[float] = Field(default=None, ge=0, description="Opsional: DC hari ini")


def _parse_date(s: Optional[str]):
    from datetime import date as _date
    if not s:
        return None
    try:
        return _date.fromisoformat(s)
    except ValueError:
        raise HTTPException(status_code=422, detail="Format tanggal harus YYYY-MM-DD")


def _prev_codes(req, wtd: float):
    vals = (req.prev_ffmc, req.prev_dmc, req.prev_dc)
    if all(v is None for v in vals):
        return None
    start = pfwi.startup_codes(wtd)
    return pfwi.FWICodes(
        start.ffmc if req.prev_ffmc is None else req.prev_ffmc,
        start.dmc if req.prev_dmc is None else req.prev_dmc,
        start.dc if req.prev_dc is None else req.prev_dc,
    )


@app.post("/api/v1/peatland/fwi/calculate", tags=["Peatland Fire Weather (PFWI-Gambut)"])
def peatland_fwi_calculate(
    req: PeatlandFWICalculationRequest,
    _user: UserInfo = Depends(require_peatland),
):
    """PFWI-Gambut dari satu pembacaan sensor siang hari (hanya akun is_peatland=True / superadmin)."""
    on_date = _parse_date(req.tanggal)
    _, _, cal = pfwi.load_site_fwi()
    wtd, _inund = pdwi.wtd_from_tmat(req.tmat_cm)
    return pfwi.calculate_from_sensor(
        req.tmat_cm, req.temp_c, req.rh_pct, cal,
        wind_ms=req.wind_ms, rain_24h_mm=req.rain_24h_mm, prev_codes=_prev_codes(req, wtd),
        tmat_7d_ago_cm=req.tmat_7d_ago_cm, dry_days=req.dry_days, on_date=on_date,
    )


@app.post("/api/v1/peatland/fwi/forecast", tags=["Peatland Fire Weather (PFWI-Gambut)"])
def peatland_fwi_forecast(
    req: PeatlandFWIForecastRequest,
    _user: UserInfo = Depends(require_peatland),
):
    """Prakiraan PFWI-Gambut N hari dari TMAT + kode FWI saat ini + skenario cuaca harian."""
    if req.mode not in ("persistence", "kering", "basah"):
        raise HTTPException(status_code=422, detail="mode harus persistence | kering | basah")
    _, _, cal = pfwi.load_site_fwi()
    wtd, _inund = pdwi.wtd_from_tmat(req.current_tmat_cm)
    cur = req.current_weather.model_dump() if req.current_weather else None
    if req.scenarios:
        scen = [s.model_dump() for s in req.scenarios]
    else:
        base = pdwi._weather(cur, pdwi.DEFAULT_ASSUMPTIONS)
        scen = pdwi.build_scenarios(req.mode, base, req.horizon, cal.pdwi_cal)
    codes = _prev_codes(req, wtd)
    notes = []
    if codes is None:
        codes = pfwi.startup_codes(wtd)
        notes.append("Kode FWI saat ini tidak diberikan -> nilai start-up (DC disetarakan dari WTD).")
    fc = pfwi.forecast_pfwi(wtd, codes, cur, scen, cal)
    return {
        "engine": pfwi.ENGINE_NAME,
        "tipe_lahan": "gambut",
        "wtd_saat_ini_cm": round(wtd, 2),
        "kode_fwi_awal": codes.to_dict(),
        "mode": "skenario_pengguna" if req.scenarios else req.mode,
        "forecast": fc,
        "peringatan_dini": pfwi.early_warning_summary(fc, current_class=-1),
        "asumsi_diterapkan": notes,
    }


@app.get("/api/v1/peatland/fwi/latest", tags=["Peatland Fire Weather (PFWI-Gambut)"])
def peatland_fwi_latest(_user: UserInfo = Depends(require_peatland)):
    """Keadaan PFWI terakhir dataset lokasi + riwayat + forecast 7 hari + skenario kering/basah."""
    noon, series, cal = pfwi.load_site_fwi()
    return pfwi.analyze_latest(noon, series, cal)


@app.get("/api/v1/peatland/fwi/assumptions", tags=["Peatland Fire Weather (PFWI-Gambut)"])
def peatland_fwi_assumptions(_user: UserInfo = Depends(require_peatland)):
    """Katalog asumsi PFWI-Gambut + kalibrasi lokasi."""
    _, _, cal = pfwi.load_site_fwi()
    return {"engine": pfwi.ENGINE_NAME, "asumsi": pfwi.assumptions_catalog(),
            "kelas_fdrs": pfwi.FDRS_BOUNDS, "kalibrasi_lokasi": cal.summary()}


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root_info():
    """
    API Welcome and documentation summary.
    """
    job = scheduler.get_job("daily_retrain_job")
    next_retrain = str(job.next_run_time) if job and job.next_run_time else "Not scheduled"
    
    return {
        "service": "Nimbus Fire & Environmental Intelligence API",
        "version": "1.2.0",
        "status": "online",
        "data_moat_strategy": "Proprietary Environmental Intelligence from continuous localized IoT observations & Full-History Training",
        "auto_retrain_schedule": "Setiap hari pukul 01:00 WIB (Daily at 01:00 AM)",
        "next_auto_retrain": next_retrain,
        "docs_url": "/docs",
        "endpoints": {
            "predict_single": "POST /api/v1/predict",
            "predict_batch": "POST /api/v1/predict/batch",
            "telemetry_ingest": "POST /api/v1/telemetry",
            "forecast_7_days": "POST /api/v1/forecast/7-days",
            "forecast_10_days": "POST /api/v1/forecast/10-days",
            "forecast_7_days_get": "GET /api/v1/forecast/7-days",
            "forecast_10_days_get": "GET /api/v1/forecast/10-days",
            "dataset_analysis_forecast": "GET /api/v1/dataset/analysis-and-forecast",
            "dataset_7_days_alias": "GET /api/v1/dataset/7-days",
            "fwi_calculator": "POST /api/v1/fwi/calculate",
            "retrain_manual": "POST /api/v1/retrain",
            "scheduler_status": "GET /api/v1/scheduler/status",
            "model_metrics": "GET /api/v1/metrics",
            "health_check": "GET /health"
        }
    }

@app.get("/health", tags=["General"])
def health_check():
    """
    Verifies that the server, ML inference models, and auto-scheduler are loaded and healthy.
    """
    models_ready = predictor is not None
    job = scheduler.get_job("daily_retrain_job")
    
    return {
        "status": "healthy" if models_ready else "degraded",
        "models_loaded": models_ready,
        "scheduler_running": scheduler.running,
        "next_auto_retrain": str(job.next_run_time) if job and job.next_run_time else None,
        "models_available": {
            "rain_model": RAIN_MODEL_PATH.exists(),
            "fire_risk_model": FIRE_MODEL_PATH.exists(),
            "drought_risk_model": DROUGHT_MODEL_PATH.exists(),
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/scheduler/status", tags=["MLOps Automation"])
def get_scheduler_status():
    """
    Checks the status and next execution time of the automated retraining scheduler.
    """
    job = scheduler.get_job("daily_retrain_job")
    return {
        "scheduler_active": scheduler.running,
        "job_id": job.id if job else None,
        "job_name": job.name if job else None,
        "schedule_cron": "01:00 AM every day",
        "next_execution": str(job.next_run_time) if job and job.next_run_time else None,
    }

@app.post("/api/v1/predict", tags=["Inference"])
def predict_disaster(
    payload: TelemetryPayload,
    _user: UserInfo = Depends(require_non_peatland)
):
    """
    Real-Time Prediction Endpoint untuk satu data telemetri IoT AWS.
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    Mengembalikan probabilitas bencana, level risiko, EWS alerts, dan forecast multi-horizon.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Models are not ready or currently reloading."
        )
    
    # Convert Pydantic model to dictionary
    data_dict = payload.model_dump(by_alias=False)
    
    # If dew_point is missing, approximate it: Td ~ T - ((100 - RH) / 5)
    if data_dict.get("dew_point_c") is None:
        rh = data_dict["relative_humidity_pct"]
        temp = data_dict["air_temperature_c"]
        data_dict["dew_point_c"] = temp - ((100.0 - rh) / 5.0)

    try:
        prediction_result = predictor.predict(data_dict)
        return {
            "success": True,
            "data": prediction_result
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference computation error: {str(e)}")

@app.post("/api/v1/telemetry", tags=["IoT Ingestion"])
def ingest_iot_telemetry(payload: TelemetryPayload):
    """
    IoT Ingestion endpoint: Receives telemetry directly from AWS hardware,
    computes disaster risks, and returns EWS flags for storage or notifications.
    """
    return predict_disaster(payload)

@app.post("/api/v1/predict/batch", tags=["Inference"])
def predict_disaster_batch(
    payloads: List[TelemetryPayload],
    _user: UserInfo = Depends(require_non_peatland)
):
    """
    Batch Inference Endpoint: Memproses beberapa rekaman sensor secara bersamaan.
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Models are not loaded."
        )
    
    results = []
    for item in payloads:
        data_dict = item.model_dump(by_alias=False)
        if data_dict.get("dew_point_c") is None:
            rh = data_dict["relative_humidity_pct"]
            temp = data_dict["air_temperature_c"]
            data_dict["dew_point_c"] = temp - ((100.0 - rh) / 5.0)
        results.append(predictor.predict(data_dict, include_forecast=False))
        
    return {
        "success": True,
        "total_records": len(results),
        "data": results
    }

@app.post("/api/v1/fwi/calculate", tags=["Fire Weather Index (Dokumen 1)"])
def compute_direct_fwi(
    payload: FWIRequest,
    _user: UserInfo = Depends(require_non_peatland)
):
    """
    Kalkulator Langsung Pseudo Fire Weather Index (FWI) berdasarkan Dokumen 1.
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    
    Formula:
      Pseudo FWI = (0.35 × Suhu) + (0.35 × RH) + (0.20 × Radiasi) + (0.10 × Hujan)
    
    Ambang Batas Index:
      - 0.00 – 0.25 : Rendah (Low)
      - 0.26 – 0.50 : Sedang (Moderate)
      - 0.51 – 0.75 : Tinggi (High)
      - 0.76 – 1.00 : Ekstrim / Bahaya (Extreme / Danger)
    """
    result = calculate_pseudo_fwi(
        temperature=payload.air_temperature_c,
        relative_humidity=payload.relative_humidity_pct,
        solar_radiation=payload.solar_radiation_wm2,
        rainfall=payload.rainfall_intensity_mmh,
    )
    return {
        "success": True,
        "data": result
    }

@app.post("/api/v1/drought/calculate", tags=["Drought Warning Index (Google Slides)"])
def compute_drought_index(
    payload: DroughtCalculationRequest,
    _user: UserInfo = Depends(require_non_peatland)
):
    """
    Kalkulator Indeks Kekeringan Ilmiah.
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    Mendukung Lahan Gambut (PDWI) dan Lahan Biasa (MDWI).
    
    Level 1: Sensor-Based Index (H_t, A_VPD, A_Solar)
    Level 2: Climate-Hydrology Index (+ Rainfall Deficit 7D)
    """
    try:
        # Load daily dataset for site-specific empirical baseline
        df_raw = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
        daily_df = aggregate_5min_to_daily(df_raw)
    except Exception:
        daily_df = None

    land = LandType.PEATLAND if payload.land_type.lower() == "gambut" else LandType.MINERAL
    wtd_val = convert_tmat_to_wtd(payload.water_table_depth_cm)
    vpd_val = calculate_vpd(payload.air_temperature_c, payload.relative_humidity_pct)
    pet_val = calculate_pet_proxy(payload.solar_radiation_wm2, vpd_val)

    # 1. Level 1
    l1_result = calculate_drought_level_1(
        wtd_cm=wtd_val,
        vpd_kpa=vpd_val,
        solar_radiation_wm2=payload.solar_radiation_wm2,
        baseline_df=daily_df,
        soil_moisture_pct=payload.soil_moisture_pct,
        land_type=land
    )

    # 2. Level 2 (jika rain_7d_mm tersedia atau hitung default dari rainfall_intensity_mmh * 7)
    rain_7d = payload.rain_7d_mm if payload.rain_7d_mm is not None else float(payload.rainfall_intensity_mmh * 24.0 * 7.0 / 12.0)
    rain_7d_ref = (
        float(daily_df["rain_mm"].rolling(7, min_periods=1).sum().median())
        if daily_df is not None and "rain_mm" in daily_df.columns
        else 15.0
    )

    l2_result = calculate_drought_level_2(
        wtd_cm=wtd_val,
        vpd_kpa=vpd_val,
        solar_radiation_wm2=payload.solar_radiation_wm2,
        rain_7d_mm=rain_7d,
        rain_7d_ref_mm=rain_7d_ref,
        baseline_df=daily_df,
        soil_moisture_pct=payload.soil_moisture_pct,
        land_type=land
    )

    return {
        "success": True,
        "tipe_lahan": land.value,
        "tmat_input_cm": payload.water_table_depth_cm,
        "wtd_konversi_cm": wtd_val,
        "vpd_kpa": vpd_val,
        "pet_proxy_mm": pet_val,
        "level_1_sensor_based": l1_result,
        "level_2_climate_hydrology": l2_result,
        "interpretasi_ilmiah": {
            "prinsip_arsitektur": "Sensor -> PDWI -> Hydrological Forecast -> Fire Vulnerability -> AI Fire Probability",
            "catatan": "PDWI != Fire Probability. Drought adalah driver hidrologis; Fire risk adalah outcome kebakaran."
        }
    }

@app.post("/api/v1/drought/forecast-7days", tags=["Drought Warning Index (Google Slides)"])
def forecast_drought_7days(
    payload: DroughtForecastRequest,
    _user: UserInfo = Depends(require_non_peatland)
):
    """
    7-Day Climate-Hydrology Drought Forecast.
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    """
    try:
        df_raw = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
        daily_df = aggregate_5min_to_daily(df_raw)
    except Exception:
        daily_df = None

    land = LandType.PEATLAND if payload.land_type.lower() == "gambut" else LandType.MINERAL
    current_wtd = convert_tmat_to_wtd(payload.current_water_table_depth_cm)

    # Jika skenario tidak disertakan, pakai default skenario simulasi dari Slide 17
    if payload.scenarios and len(payload.scenarios) > 0:
        scenarios = [s.model_dump() for s in payload.scenarios]
    else:
        scenarios = [
            {"rain_mm": 0.0, "temp_c": 25.5, "rh_pct": 82.0, "solar_wm2": 150.0, "wind_ms": 0.8},
            {"rain_mm": 0.0, "temp_c": 25.8, "rh_pct": 80.0, "solar_wm2": 165.0, "wind_ms": 0.9},
            {"rain_mm": 5.0, "temp_c": 26.0, "rh_pct": 78.0, "solar_wm2": 180.0, "wind_ms": 1.0},
            {"rain_mm": 12.0, "temp_c": 25.4, "rh_pct": 88.0, "solar_wm2": 110.0, "wind_ms": 0.7},
            {"rain_mm": 0.0, "temp_c": 25.8, "rh_pct": 84.0, "solar_wm2": 155.0, "wind_ms": 0.8},
            {"rain_mm": 0.0, "temp_c": 26.1, "rh_pct": 80.0, "solar_wm2": 175.0, "wind_ms": 0.9},
            {"rain_mm": 8.0, "temp_c": 25.6, "rh_pct": 86.0, "solar_wm2": 125.0, "wind_ms": 0.7},
        ]

    forecast_results = run_7day_climate_hydrology_forecast(
        current_wtd_cm=current_wtd,
        climate_scenarios=scenarios,
        baseline_df=daily_df,
        land_type=land
    )

    return {
        "success": True,
        "tipe_lahan": land.value,
        "current_tmat_cm": payload.current_water_table_depth_cm,
        "current_wtd_cm": current_wtd,
        "proyeksi_7_hari": forecast_results
    }

@app.get("/api/v1/drought/latest-comparison", tags=["Drought Warning Index (Google Slides)"])
def get_drought_latest_comparison(_user: UserInfo = Depends(require_non_peatland)):
    """
    Evaluasi Titik Data Sensor Mutakhir pada Dataset.
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    Membandingkan indeks kekeringan antara Lahan Gambut (PDWI) vs Lahan Biasa/Mineral (MDWI).
    """
    df_raw = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
    daily_df = aggregate_5min_to_daily(df_raw)
    last_row = daily_df.iloc[-1]

    rain_7d = daily_df["rain_mm"].tail(7).sum()
    rain_7d_ref = daily_df["rain_mm"].rolling(7, min_periods=1).sum().median()

    gambut_l1 = calculate_drought_level_1(
        wtd_cm=last_row["wtd_mean"],
        vpd_kpa=last_row["vpd_mean"],
        solar_radiation_wm2=last_row["solar_mean"],
        baseline_df=daily_df,
        land_type=LandType.PEATLAND
    )
    gambut_l2 = calculate_drought_level_2(
        wtd_cm=last_row["wtd_mean"],
        vpd_kpa=last_row["vpd_mean"],
        solar_radiation_wm2=last_row["solar_mean"],
        rain_7d_mm=rain_7d,
        rain_7d_ref_mm=rain_7d_ref,
        baseline_df=daily_df,
        land_type=LandType.PEATLAND
    )

    mineral_l1 = calculate_drought_level_1(
        wtd_cm=last_row["wtd_mean"],
        vpd_kpa=last_row["vpd_mean"],
        solar_radiation_wm2=last_row["solar_mean"],
        soil_moisture_pct=last_row["soil_moisture_mean"],
        baseline_df=daily_df,
        land_type=LandType.MINERAL
    )
    mineral_l2 = calculate_drought_level_2(
        wtd_cm=last_row["wtd_mean"],
        vpd_kpa=last_row["vpd_mean"],
        solar_radiation_wm2=last_row["solar_mean"],
        rain_7d_mm=rain_7d,
        rain_7d_ref_mm=rain_7d_ref,
        soil_moisture_pct=last_row["soil_moisture_mean"],
        baseline_df=daily_df,
        land_type=LandType.MINERAL
    )

    return {
        "success": True,
        "tanggal_observasi": str(last_row["date"]),
        "sensor_aktual": {
            "tmat_mean_cm": round(-last_row["wtd_mean"], 2),
            "wtd_mean_cm": round(last_row["wtd_mean"], 2),
            "suhu_mean_c": round(last_row["temp_mean"], 2),
            "rh_mean_pct": round(last_row["rh_mean"], 2),
            "vpd_kpa": round(last_row["vpd_mean"], 3),
            "solar_wm2": round(last_row["solar_mean"], 1),
            "soil_moisture_pct": round(last_row["soil_moisture_mean"], 1),
            "rain_7d_mm": round(rain_7d, 2),
        },
        "lahan_gambut_pdwi": {
            "level_1_sensor": gambut_l1,
            "level_2_climate_hydrology": gambut_l2,
        },
        "lahan_biasa_mineral_mdwi": {
            "level_1_sensor": mineral_l1,
            "level_2_climate_hydrology": mineral_l2,
        }
    }

@app.post("/api/v1/forecast/7-days", tags=["Inference & Forecasting"])
def predict_7_day_forecast(
    payload: TelemetryPayload,
    _user: UserInfo = Depends(require_non_peatland)
):
    """
    Prediksi Proyeksi 7 Hari ke Depan (7-Day Multi-Disaster Forecast).
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Models are not ready."
        )
    data_dict = payload.model_dump(by_alias=False)
    if data_dict.get("dew_point_c") is None:
        rh = data_dict["relative_humidity_pct"]
        temp = data_dict["air_temperature_c"]
        data_dict["dew_point_c"] = temp - ((100.0 - rh) / 5.0)
    
    full_result = predictor.predict(data_dict, include_forecast=True)
    return {
        "success": True,
        "training_scope": "Full Historical Dataset (All observations used for model training)",
        "ringkasan_hari_ini": full_result["ringkasan"],
        "prediksi_7_hari": full_result["prediksi_7_hari"]
    }

@app.get("/api/v1/forecast/7-days", tags=["Inference & Forecasting"])
def get_7_day_forecast_latest():
    """
    GET Proyeksi 7 Hari ke Depan dari Titik Sensor Mutakhir Dataset:
    Menggunakan model AI yang dilatih dari SELURUH data historis.
    """
    df = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
    last_row = df.iloc[-1].to_dict()
    forecast_7 = generate_7_day_forecast(last_row)
    return {
        "success": True,
        "training_scope": "Full Historical Dataset (All 17,567 observations and 61 full days used for model training)",
        "titik_sensor_mutakhir": {
            "waktu": str(last_row["timestamp"]),
            "suhu_c": float(last_row["air_temperature_c"]),
            "rh_persen": float(last_row["relative_humidity_pct"]),
            "tmat_cm": float(last_row["water_table_depth_cm"]),
        },
        "prediksi_7_hari": forecast_7
    }

@app.post("/api/v1/forecast/10-days", tags=["Inference & Forecasting"])
def predict_10_day_forecast(
    payload: TelemetryPayload,
    _user: UserInfo = Depends(require_non_peatland)
):
    """
    Prediksi Proyeksi 10 Hari ke Depan (10-Day Multi-Disaster Forecast).
    Hanya tersedia untuk lahan non-gambut (is_peatland=False).
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Models are not ready."
        )
    data_dict = payload.model_dump(by_alias=False)
    if data_dict.get("dew_point_c") is None:
        rh = data_dict["relative_humidity_pct"]
        temp = data_dict["air_temperature_c"]
        data_dict["dew_point_c"] = temp - ((100.0 - rh) / 5.0)
    
    full_result = predictor.predict(data_dict, include_forecast=True)
    return {
        "success": True,
        "training_scope": "Full Historical Dataset (All observations used for model training)",
        "ringkasan_hari_ini": full_result["ringkasan"],
        "prediksi_10_hari": full_result["prediksi_10_hari"]
    }

@app.get("/api/v1/forecast/10-days", tags=["Inference & Forecasting"])
def get_10_day_forecast_latest():
    """
    GET Proyeksi 10 Hari ke Depan dari Titik Sensor Mutakhir Dataset:
    Menggunakan model AI yang dilatih dari SELURUH data historis.
    """
    df = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
    last_row = df.iloc[-1].to_dict()
    forecast_10 = generate_10_day_forecast(last_row)
    return {
        "success": True,
        "training_scope": "Full Historical Dataset (All 17,567 observations and 61 full days used for model training)",
        "titik_sensor_mutakhir": {
            "waktu": str(last_row["timestamp"]),
            "suhu_c": float(last_row["air_temperature_c"]),
            "rh_persen": float(last_row["relative_humidity_pct"]),
            "tmat_cm": float(last_row["water_table_depth_cm"]),
        },
        "prediksi_10_hari": forecast_10
    }

@app.get("/api/v1/dataset/analysis-and-forecast", tags=["Dataset & Historical Analytics"])
def get_dataset_historical_and_forecast(days: Optional[int] = None, land_type: str = "lahan_biasa"):
    """
    Menganalisis data riil langsung dari dataset Excel (Data Dummy AWS dan TMAT.xlsx):
    - Model dilatih menggunakan SELURUH data historis (17.567 data telemetri / 61 hari penuh)
    - Menyajikan riwayat observasi harian terkalibrasi (default: seluruh hari atau parameter 'days')
    - Titik status mutakhir sensor pada dataset
    - Menghasilkan proyeksi prediksi 7 hari dan 10 hari ke depan serta milestone Layer 4 (1d, 3d, 7d, 10d)
    - Mendukung tipe lahan: 'lahan_biasa' (tanah mineral) atau 'gambut'
    """
    df = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
    result = analyze_and_forecast_from_dataset(df, n_history_days=days, n_forecast_days=10, land_type=land_type)
    return {
        "success": True,
        "data": result
    }

@app.get("/api/v1/dataset/7-days", tags=["Dataset & Historical Analytics"])
def get_dataset_7_days_alias():
    """
    Endpoint kompatibilitas untuk peramalan 7 hari berbasis seluruh data historis dataset.
    Melatih AI pada seluruh rekaman data (bukan sekadar 7 hari) dan menyajikan analisis serta peramalan.
    """
    return get_dataset_historical_and_forecast(days=7)

@app.get("/api/v1/dataset/all-days", tags=["Dataset & Historical Analytics"])
def get_dataset_all_days_ai_detection(days: Optional[int] = None):
    """
    🔥 Endpoint Evaluasi & Deteksi Model AI Seluruh Hari Dataset:
    - Menjalankan deteksi AI (Model Kebakaran, Kekeringan, Hujan, Status TMAT, EWS) pada SETIAP HARI di seluruh dataset (61 hari penuh).
    - Didukung model yang dilatih pada 17.567 data telemetri historis.
    - Menyajikan ringkasan statistik komprehensif, distribusi risiko, kejadian puncak (peak events), dan proyeksi 10 hari ke depan.
    - Parameter 'days' opsional: jika dikosongkan akan memproses SELURUH 61 HARI yang ada di data.
    """
    df = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
    result = analyze_full_dataset_with_ai_models(df, predictor=predictor, n_days=days)
    return result

@app.get("/api/v1/dataset/full-history", tags=["Dataset & Historical Analytics"])
def get_dataset_full_history_alias(days: Optional[int] = None):
    """
    Alias untuk /api/v1/dataset/all-days (Deteksi Model AI untuk keseluruhan hari pada data).
    """
    return get_dataset_all_days_ai_detection(days=days)

@app.get("/api/v1/dataset/export", tags=["Dataset & Historical Analytics"])
def export_dataset_telemetry(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None
):
    """
    Export data telemetri historis dalam format CSV berdasarkan rentang waktu tertentu.
    Format tanggal: YYYY-MM-DD (contoh: start_date=2026-06-01&end_date=2026-06-15).
    """
    import io
    import pandas as pd
    from fastapi.responses import StreamingResponse

    df = load_aws_dataset(DEFAULT_RAW_DATA_PATH)
    df["ts_temp"] = pd.to_datetime(df["timestamp"])
    
    if start_date:
        try:
            df = df[df["ts_temp"] >= pd.to_datetime(start_date)]
        except Exception:
            pass
    if end_date:
        try:
            end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            df = df[df["ts_temp"] < end_ts]
        except Exception:
            pass
    elif days:
        try:
            latest_date = df["ts_temp"].max()
            cutoff_date = latest_date - pd.Timedelta(days=days)
            df = df[df["ts_temp"] >= cutoff_date]
        except Exception:
            pass
        
    df = df.drop(columns=["ts_temp"], errors="ignore")
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    filename = f"nimbus_telemetry_{start_date or 'start'}_to_{end_date or 'latest'}.csv"
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )
    return response


@app.post("/api/v1/retrain", response_model=RetrainResponse, tags=["MLOps Automation"])
def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Manual trigger for model retraining asynchronously in the background.
    Will not block the API or cause HTTP timeouts.
    The updated model bundle is hot-reloaded automatically once training completes.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    background_tasks.add_task(execute_background_retraining)
    
    return RetrainResponse(
        status="retraining_initiated",
        message="Model retraining job has been triggered manually in the background. Inference remains active.",
        started_at=now_str
    )

@app.get("/api/v1/metrics", tags=["Model Governance"])
def get_model_metrics():
    """
    Returns the latest training accuracy, precision, recall, F1-scores, and top features.
    """
    if not METRICS_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics report not found. Please train models first."
        )
    
    with open(METRICS_REPORT_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    
    return {
        "status": "success",
        "models": metrics_data
    }

# ---------------------------------------------------------------------------
# RBAC Protected Routes Example
# ---------------------------------------------------------------------------
@app.get("/api/v1/user/dashboard", tags=["RBAC Protected Routes"])
def user_dashboard(user: UserInfo = Depends(require_user)):
    """
    Dashboard yang dapat diakses oleh user biasa maupun superadmin.
    """
    return {
        "status": "success",
        "access": "granted",
        "message": f"Selamat datang di Dashboard User, {user.full_name or user.email}!",
        "current_user": {
            "id": user.id,
            "email": user.email,
            "role": user.role
        }
    }

@app.get("/api/v1/superadmin/dashboard", tags=["RBAC Protected Routes"])
def superadmin_dashboard(user: UserInfo = Depends(require_superadmin)):
    """
    Dashboard khusus Superadmin. Role 'user' biasa akan menerima 403 Forbidden.
    """
    return {
        "status": "success",
        "access": "granted",
        "message": f"Selamat datang di Panel Kontrol Superadmin, {user.full_name or user.email}!",
        "current_user": {
            "id": user.id,
            "email": user.email,
            "role": user.role
        },
        "system_status": {
            "database": "Supabase Connected",
            "scheduler_active": scheduler.running
        }
    }

@app.get("/api/v1/superadmin/users", tags=["RBAC Protected Routes"])
def superadmin_get_users(user: UserInfo = Depends(require_superadmin)):
    """
    Endpoint khusus Superadmin untuk melihat seluruh data user di database.
    Role 'user' biasa akan menerima 403 Forbidden.
    """
    users = get_all_users()
    return {
        "status": "success",
        "access": "granted",
        "message": f"Daftar seluruh pengguna database ({len(users)} akun).",
        "authorized_by": {
            "email": user.email,
            "role": user.role
        },
        "total_users": len(users),
        "users": users
    }

@app.put("/api/v1/superadmin/users/{user_id}", tags=["RBAC Protected Routes"])
def superadmin_update_user(
    user_id: str,
    payload: UserUpdateRequest,
    admin: UserInfo = Depends(require_superadmin)
):
    """
    Endpoint khusus Superadmin untuk mengubah data pengguna (nama, email, role, atau reset password).
    Role 'user' biasa akan menerima 403 Forbidden.
    """
    update_data = payload.model_dump(exclude_unset=True)
    updated = update_user_by_id(user_id, update_data)
    return {
        "status": "success",
        "message": f"Data pengguna '{updated.get('email')}' (ID: {user_id}) berhasil diperbarui.",
        "updated_by": admin.email,
        "user": updated
    }

@app.delete("/api/v1/superadmin/users/{user_id}", tags=["RBAC Protected Routes"])
def superadmin_delete_user(
    user_id: str,
    admin: UserInfo = Depends(require_superadmin)
):
    """
    Endpoint khusus Superadmin untuk menghapus pengguna dari database Supabase.
    Role 'user' biasa akan menerima 403 Forbidden.
    """
    if str(admin.id) == str(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tindakan Ditolak: Anda tidak dapat menghapus akun Superadmin Anda sendiri."
        )

    delete_user_by_id(user_id)
    return {
        "status": "success",
        "message": f"Pengguna dengan ID '{user_id}' berhasil dihapus dari database Supabase.",
        "deleted_by": admin.email
    }

if __name__ == "__main__":
    import uvicorn
    # Run dev server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
