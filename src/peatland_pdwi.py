"""
PDWI-Gambut — Peatland Drought Warning Index Engine (Nimbus)
============================================================
Algoritma kekeringan KHUSUS lahan gambut (akun ``is_peatland = True``).

Sumber data sensor IoT : suhu udara, kelembapan udara, radiasi matahari, tinggi muka air (TMAT).
Referensi              : Google Slides "Algoritma Prediksi kebakaran lahan gambut"
                         (Step 0-9, Level 1 / Level 2, trend, regresi dWTD, forecast 7 hari)
                         + rancangan PDWI Nimbus di chat ChatGPT (persentil site-specific +
                         ambang ekologis absolut + faktor meteorologi) + FAO-56 Penman-Monteith.

Urutan perhitungan (sequence)
-----------------------------
  S0  Kelas drought                      NORMAL / WATCH / MODERATE / HIGH / SEVERE
  S1  Konversi sensor                    WTD = -TMAT  (cm di bawah permukaan gambut)
  S2  Agregasi 5-menit -> harian         (``aggregate_5min_to_daily``)
  S3  VPD (Tetens)                       VPD = e_s(T) * (1 - RH/100)
  S4  PET gambut                         PET = Kc * ET0  (FAO-56 Penman-Monteith harian)
  S5  Hydrological Stress  H_t           campuran skor ekologis absolut & persentil lokasi
  S6  Atmospheric Drying   A_t           0.70 * A_VPD + 0.30 * A_Solar
  S7  Rainfall Deficit     RD_t          100 * max(0, 1 - P7 / P7_ref)
  S8  LEVEL 1  (sensor)                  0.60 H + 0.25 A_VPD + 0.15 A_Solar
  S9  LEVEL 2  (iklim-hidrologi)         0.50 H + 0.20 A_VPD + 0.10 A_Solar + 0.20 RD
  S10 Tren & persistensi                 dWTD_7 = WTD_t - WTD_(t-7);  dry-spell
  S11 Model neraca air dWTD              dWTD = b0 + b1 (P - f*PET) + b3 WTD     (b1<0, b3<=0)
  S12 Forecast N hari                    cuaca -> dWTD -> WTD(t+1) -> H_t -> PDWI harian

Setiap parameter yang belum ada di data memakai NILAI ASUMSI yang dideklarasikan eksplisit di
``PeatlandAssumptions`` dan dilaporkan di keluaran API.

Catatan keselarasan waktu (penting): data simulasi menunjukkan hujan hari ``t`` baru tampak pada
rata-rata WTD hari ``t+1`` (mis. hujan 70.65 mm pada 19-Jun -> WTD turun pada 20-Jun). Karena itu
transisi WTD memakai cuaca hari sebelumnya:  WTD(t+1) = WTD(t) + dWTD( cuaca(t), WTD(t) ).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import lsq_linear

from src.config import DEFAULT_LATITUDE, DEFAULT_RAW_DATA_PATH
from src.drought_index import (
    aggregate_5min_to_daily,
    calculate_vpd,
    classify_drought_score,
    convert_tmat_to_wtd,
)

ENGINE_NAME = "PDWI-Gambut Nimbus v1"


# ---------------------------------------------------------------------------
# 0. PARAMETER ASUMSI (semua parameter yang tidak tersedia dari sensor)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeatlandAssumptions:
    """Seluruh nilai asumsi algoritma. Ubah di sini untuk kalibrasi lokasi lain."""

    # --- Stasiun / sensor ---------------------------------------------------
    latitude_deg: float = DEFAULT_LATITUDE        # koordinat AWS default sistem
    elevation_m: float = 5.0                      # dataran rendah gambut Kalimantan Barat
    wind_sensor_height_m: float = 2.0             # tinggi anemometer (FAO-56 butuh u2)
    default_wind_ms: float = 1.0                  # jika sensor angin tidak mengirim data

    # --- Evapotranspirasi ---------------------------------------------------
    albedo: float = 0.23                          # albedo permukaan referensi FAO-56
    kc_peat: float = 1.0                          # koefisien tanaman gambut: PET = Kc * ET0

    # --- Hidrologi gambut ---------------------------------------------------
    gw_et_fraction: float = 0.5                   # porsi ET yang diambil dari air tanah (f)
    sy_bounds: Tuple[float, float] = (0.08, 0.60)  # rentang specific yield gambut (fisik)
    prior_sy: float = 0.30                        # Sy awal bila belum ada data kalibrasi
    prior_base_drawdown_cm_day: float = 1.5       # laju surut dasar bila belum ada kalibrasi
    prior_wtd_feedback: float = -0.005            # umpan balik kedalaman (drainase melambat)
    max_wtd_cm: float = 300.0                     # batas fisik ketebalan gambut untuk forecast

    # --- Hydrological stress H_abs : WTD(cm) -> skor 0..100 ------------------
    # 40 cm = ambang kerusakan gambut (PP 71/2014 jo. PP 57/2016); 80/120/160 = tingkat lanjut.
    wtd_anchors: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0), (10.0, 5.0), (40.0, 40.0), (80.0, 70.0), (120.0, 90.0), (160.0, 100.0)
    )

    # --- Atmospheric drying (nilai harian rata-rata iklim tropis basah) -------
    vpd_low_kpa: float = 0.2                      # VPD harian yang dianggap "lembap" -> skor 0
    vpd_high_kpa: float = 1.2                     # VPD harian yang dianggap "sangat kering" -> 100
    solar_full_wm2: float = 250.0                 # radiasi harian rata-rata langit cerah ekuator
    w_vpd: float = 0.70                           # bobot A_VPD pada A_t (Slide 8)
    w_solar: float = 0.30                         # bobot A_Solar pada A_t (Slide 8)

    # --- Rainfall deficit ---------------------------------------------------
    p7_clim_mm: float = 40.0                      # normal hujan 7-hari musim kemarau (asumsi)

    # --- Kematangan data (data moat): bobot baseline lokasi vs ambang asumsi -----
    lambda_max_h: float = 0.5                     # H_t: ambang regulasi tetap dipertahankan 50%
    lambda_max_atm: float = 0.9                   # A_t & RD: akhirnya didominasi data lokasi
    n_full_days: int = 365                        # 1 siklus musim penuh
    min_calibration_days: int = 14                # minimal hari agar kalibrasi dianggap sah

    # --- Tren & dry-spell (indikator pendukung) --------------------------------
    trend_stable_cm: float = 2.0                  # |dWTD7| <= 2 cm dianggap stabil (Slide 15)
    trend_rapid_cm: float = 20.0                  # dWTD7 >= 20 cm/minggu = pengeringan cepat
    dry_day_mm: float = 1.0                       # hujan < 1 mm/hari dihitung hari kering

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_ASSUMPTIONS = PeatlandAssumptions()

# Bobot Level 1 & Level 2 (persis Slide 9 & Slide 14)
L1_WEIGHTS = {"H": 0.60, "A_VPD": 0.25, "A_SOLAR": 0.15}
L2_WEIGHTS = {"H": 0.50, "A_VPD": 0.20, "A_SOLAR": 0.10, "RD": 0.20}

RECOMMENDATIONS = {
    0: "Kondisi hidrologis gambut aman. Lanjutkan pemantauan rutin.",
    1: "Waspada: muka air mulai menurun. Periksa sekat kanal dan tingkatkan frekuensi pemantauan.",
    2: "Moderat: kekeringan gambut berkembang. Siapkan pembasahan (rewetting) dan patroli area rawan.",
    3: "Tinggi: gambut kering signifikan. Aktifkan siaga karhutla, pembasahan kanal, dan koordinasi pemadaman.",
    4: "Kritis: kekeringan gambut ekstrem. Siaga darurat; prioritaskan pembasahan dan patroli titik panas.",
}


def assumptions_catalog(a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> List[Dict[str, Any]]:
    """Katalog asumsi + dasar penetapan, untuk transparansi di API & dokumentasi."""
    return [
        {"parameter": "elevation_m", "nilai": a.elevation_m, "satuan": "m",
         "dasar": "Dataran rendah gambut Kalimantan Barat; dipakai untuk tekanan udara & Rso."},
        {"parameter": "wind_sensor_height_m", "nilai": a.wind_sensor_height_m, "satuan": "m",
         "dasar": "Anemometer standar WMO 2 m (FAO-56 memakai u2)."},
        {"parameter": "default_wind_ms", "nilai": a.default_wind_ms, "satuan": "m/s",
         "dasar": "Rata-rata angin harian lahan gambut bervegetasi; dipakai bila sensor tidak mengirim angin."},
        {"parameter": "albedo", "nilai": a.albedo, "satuan": "-",
         "dasar": "Albedo permukaan referensi FAO-56."},
        {"parameter": "kc_peat", "nilai": a.kc_peat, "satuan": "-",
         "dasar": "ET gambut tropis ~ ET0 (0.9-1.1). Belum dikalibrasi dengan eddy-covariance."},
        {"parameter": "gw_et_fraction", "nilai": a.gw_et_fraction, "satuan": "-",
         "dasar": "Porsi ET yang dipasok dari air tanah. Tidak teridentifikasi dari data simulasi."},
        {"parameter": "sy_bounds", "nilai": list(a.sy_bounds), "satuan": "-",
         "dasar": "Rentang specific yield gambut fibrik-hemik (0.1-0.5); membatasi koefisien hujan."},
        {"parameter": "wtd_anchors", "nilai": [list(p) for p in a.wtd_anchors], "satuan": "cm -> skor",
         "dasar": "40 cm = ambang kerusakan gambut (PP 57/2016); titik 80/120/160 cm adalah asumsi tingkat lanjut."},
        {"parameter": "vpd_low_kpa / vpd_high_kpa", "nilai": [a.vpd_low_kpa, a.vpd_high_kpa], "satuan": "kPa",
         "dasar": "VPD harian rata-rata iklim tropis basah (0.2 lembap, 1.2 sangat kering)."},
        {"parameter": "solar_full_wm2", "nilai": a.solar_full_wm2, "satuan": "W/m2",
         "dasar": "Radiasi harian rata-rata langit cerah di ekuator."},
        {"parameter": "p7_clim_mm", "nilai": a.p7_clim_mm, "satuan": "mm/7 hari",
         "dasar": "Normal hujan mingguan musim kemarau Kalimantan Barat (asumsi)."},
        {"parameter": "lambda_max_h / lambda_max_atm", "nilai": [a.lambda_max_h, a.lambda_max_atm], "satuan": "-",
         "dasar": "Bobot maksimum baseline lokasi; tumbuh linear sampai n_full_days data."},
        {"parameter": "n_full_days", "nilai": a.n_full_days, "satuan": "hari",
         "dasar": "Satu siklus musim penuh dibutuhkan agar persentil lokasi representatif."},
        {"parameter": "trend_rapid_cm", "nilai": a.trend_rapid_cm, "satuan": "cm/7 hari",
         "dasar": "Batas pengeringan cepat (indikator pendukung, tidak masuk skor)."},
    ]


# ---------------------------------------------------------------------------
# S0 + helper umum
# ---------------------------------------------------------------------------
def _r(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def _level_dict(score: float, missing: List[str]) -> Dict[str, Any]:
    cls, label = classify_drought_score(score)
    return {
        "score": round(float(score), 2),
        "class": cls,
        "label": label,
        "komponen_tidak_tersedia": missing,
    }


def _weighted(components: Dict[str, Optional[float]], weights: Dict[str, float]) -> Tuple[float, List[str]]:
    """Rata-rata berbobot; komponen tak tersedia dibuang & bobot dinormalisasi ulang."""
    avail = {k: v for k, v in components.items() if k in weights and v is not None}
    missing = [k for k in weights if k not in avail]
    wsum = sum(weights[k] for k in avail)
    if wsum <= 0:
        raise ValueError("Tidak ada komponen indeks yang tersedia.")
    return _clip01(sum(weights[k] * v for k, v in avail.items()) / wsum), missing


# ---------------------------------------------------------------------------
# S1. Konversi sensor
# ---------------------------------------------------------------------------
def wtd_from_tmat(tmat_cm: float) -> Tuple[float, bool]:
    """WTD = -TMAT (cm di bawah permukaan). TMAT >= 0 berarti tergenang -> WTD = 0."""
    wtd = float(convert_tmat_to_wtd(float(tmat_cm)))
    return max(0.0, wtd), wtd < 0.0


# ---------------------------------------------------------------------------
# S4. PET gambut — FAO-56 Penman-Monteith harian
# ---------------------------------------------------------------------------
def extraterrestrial_radiation(latitude_deg: float, doy: int) -> float:
    """Ra (MJ m-2 hari-1), FAO-56 pers. 21."""
    phi = math.radians(latitude_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    decl = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(decl))))
    return (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.sin(ws)
    )


def penman_monteith_et0(
    temp_c: float,
    rh_pct: float,
    solar_wm2: float,
    wind_ms: float,
    doy: int,
    a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS,
    vpd_kpa: Optional[float] = None,
) -> float:
    """
    ET0 harian (mm/hari) FAO-56 Penman-Monteith dari data sensor rata-rata harian.
    ``vpd_kpa`` (rata-rata VPD 5-menit) dipakai bila ada, karena lebih akurat daripada VPD dari T & RH rata-rata.
    """
    t = float(temp_c)
    rh = float(np.clip(rh_pct, 0.0, 100.0))
    es = 0.6108 * math.exp(17.27 * t / (t + 237.3))
    ea = es * rh / 100.0
    deficit = max(0.0, es - ea) if vpd_kpa is None else max(0.0, float(vpd_kpa))
    delta = 4098.0 * es / (t + 237.3) ** 2
    pressure = 101.3 * ((293.0 - 0.0065 * a.elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * pressure
    u2 = float(wind_ms) * 4.87 / math.log(67.8 * a.wind_sensor_height_m - 5.42)

    rs = max(0.0, float(solar_wm2)) * 0.0864                       # W/m2 -> MJ/m2/hari
    ra = extraterrestrial_radiation(a.latitude_deg, doy)
    rso = (0.75 + 2e-5 * a.elevation_m) * ra
    rel = float(np.clip(rs / rso, 0.3, 1.0)) if rso > 0 else 0.5
    rns = (1.0 - a.albedo) * rs
    rnl = 4.903e-9 * (t + 273.16) ** 4 * (0.34 - 0.14 * math.sqrt(max(ea, 0.0))) * (1.35 * rel - 0.35)
    rn = rns - rnl

    et0 = (0.408 * delta * rn + gamma * 900.0 / (t + 273.0) * u2 * deficit) / (delta + gamma * (1.0 + 0.34 * u2))
    return float(max(0.0, et0))


def pet_peat(temp_c: float, rh_pct: float, solar_wm2: float, wind_ms: float, doy: int,
             a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS, vpd_kpa: Optional[float] = None) -> float:
    """PET gambut = Kc * ET0."""
    return a.kc_peat * penman_monteith_et0(temp_c, rh_pct, solar_wm2, wind_ms, doy, a, vpd_kpa)


# ---------------------------------------------------------------------------
# Kalibrasi lokasi (site calibration)
# ---------------------------------------------------------------------------
@dataclass
class SiteCalibration:
    """Hasil belajar dari riwayat sensor lokasi: baseline persentil + parameter hidrologi."""

    source: str                                   # "dataset_lokasi" | "prior_asumsi"
    n_days: int = 0
    lambda_h: float = 0.0                         # bobot persentil lokasi pada H_t
    lambda_atm: float = 0.0                       # bobot persentil lokasi pada A_t & RD
    wtd_baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    vpd_baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    solar_baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    p7_ref_mm: float = DEFAULT_ASSUMPTIONS.p7_clim_mm
    wtd_range: Tuple[float, float] = (0.0, 0.0)   # rentang WTD yang teramati saat kalibrasi (cm)
    beta0: float = DEFAULT_ASSUMPTIONS.prior_base_drawdown_cm_day
    beta1: float = -1.0 / (10.0 * DEFAULT_ASSUMPTIONS.prior_sy)
    beta3: float = DEFAULT_ASSUMPTIONS.prior_wtd_feedback
    hydro_source: str = "prior_asumsi"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def implied_sy(self) -> Optional[float]:
        return None if self.beta1 >= 0 else -1.0 / (10.0 * self.beta1)

    def summary(self) -> Dict[str, Any]:
        return {
            "sumber": self.source,
            "jumlah_hari_data": self.n_days,
            "bobot_persentil_lokasi_H": _r(self.lambda_h, 3),
            "bobot_persentil_lokasi_atmosfer": _r(self.lambda_atm, 3),
            "p7_referensi_mm": _r(self.p7_ref_mm),
            "rentang_wtd_kalibrasi_cm": [_r(self.wtd_range[0]), _r(self.wtd_range[1])],
            "model_dwtd": {
                "sumber_parameter": self.hydro_source,
                "b0_cm_hari": _r(self.beta0, 3),
                "b1_cm_per_mm": _r(self.beta1, 4),
                "b3_per_cm": _r(self.beta3, 4),
                "specific_yield_tersirat": _r(self.implied_sy, 3),
            },
            "diagnostik": self.diagnostics,
        }


def prior_calibration(a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> SiteCalibration:
    """Kalibrasi tanpa data lokasi: murni ambang asumsi + parameter hidrologi prior."""
    return SiteCalibration(
        source="prior_asumsi",
        p7_ref_mm=a.p7_clim_mm,
        beta0=a.prior_base_drawdown_cm_day,
        beta1=-1.0 / (10.0 * a.prior_sy),
        beta3=a.prior_wtd_feedback,
    )


def data_maturity_lambda(n_days: int, lam_max: float, a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    """lambda = lam_max * min(1, n/n_full); 0 bila data < min_calibration_days."""
    if n_days < a.min_calibration_days:
        return 0.0
    return float(lam_max * min(1.0, n_days / float(a.n_full_days)))


def _fit_hydro(rain: np.ndarray, pet: np.ndarray, wtd: np.ndarray, a: PeatlandAssumptions
               ) -> Optional[Tuple[float, float, float]]:
    """
    Regresi berbatas fisik:  dWTD(t) = b0 + b1 * (P - f*PET) + b3 * WTD
      b0 in [0, 30] ; b1 = -1/(10*Sy), Sy in sy_bounds ; b3 in [-0.2, 0]
    (b1 < 0 : hujan menaikkan muka air; tanda PET ikut b1 sehingga PET menurunkan muka air).
    """
    y = wtd[1:] - wtd[:-1]
    nwi = rain[:-1] - a.gw_et_fraction * pet[:-1]
    if len(y) < a.min_calibration_days or float(np.std(nwi)) < 0.5:
        return None
    x = np.column_stack([np.ones(len(y)), nwi, wtd[:-1]])
    b1_lo, b1_hi = -1.0 / (10.0 * a.sy_bounds[0]), -1.0 / (10.0 * a.sy_bounds[1])
    res = lsq_linear(x, y, bounds=([0.0, b1_lo, -0.2], [30.0, b1_hi, 0.0]))
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def _dwtd(b0: float, b1: float, b3: float, rain: float, pet: float, wtd: float, f: float) -> float:
    return b0 + b1 * (rain - f * pet) + b3 * wtd


def _walk_forward_diagnostics(daily: pd.DataFrame, a: PeatlandAssumptions, min_train: int = 30) -> Dict[str, Any]:
    """Validasi walk-forward: galat 1 hari & galat level WTD 7 hari (cuaca aktual) vs persistensi."""
    n = len(daily)
    if n < min_train + 9:
        return {"catatan": f"data < {min_train + 9} hari, validasi walk-forward dilewati"}
    rain = daily["rain_mm"].to_numpy(float)
    pet = daily["pet_peat"].to_numpy(float)
    wtd = daily["wtd_mean"].to_numpy(float)
    e1, e7, ep = [], [], []
    for t in range(min_train, n - 8):
        fit = _fit_hydro(rain[: t + 1], pet[: t + 1], wtd[: t + 1], a)
        if fit is None:
            continue
        b0, b1, b3 = fit
        e1.append(abs((wtd[t + 1] - wtd[t]) - _dwtd(b0, b1, b3, rain[t], pet[t], wtd[t], a.gw_et_fraction)))
        w = wtd[t]
        for h in range(7):
            w = float(np.clip(w + _dwtd(b0, b1, b3, rain[t + h], pet[t + h], w, a.gw_et_fraction), 0.0, a.max_wtd_cm))
        e7.append(abs(w - wtd[t + 7]))
        ep.append(abs(wtd[t] - wtd[t + 7]))
    if not e1:
        return {"catatan": "validasi walk-forward tidak dapat dihitung"}
    return {
        "mae_1_hari_cm": _r(np.mean(e1)),
        "mae_level_wtd_7_hari_cm": _r(np.mean(e7)),
        "mae_persistensi_7_hari_cm": _r(np.mean(ep)),
        "n_uji": len(e1),
        "catatan": "MAE 7 hari memakai cuaca aktual (batas atas akurasi); forecast sungguhan lebih lebar.",
    }


def calibrate_site(daily: pd.DataFrame, a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> SiteCalibration:
    """
    Belajar dari riwayat harian lokasi (keluaran ``aggregate_5min_to_daily``):
      1. baseline persentil WTD / VPD / radiasi,  2. referensi hujan 7-hari,
      3. parameter b0,b1,b3 model neraca air dWTD,  4. diagnostik walk-forward.
    """
    df = daily.copy()
    df["_ts"] = pd.to_datetime(df["date"])
    df = df.sort_values("_ts").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return prior_calibration(a)

    df["doy_of_year"] = df["_ts"].dt.dayofyear
    df["pet_peat"] = [
        pet_peat(r.temp_mean, r.rh_mean, r.solar_mean, r.wind_mean, int(r.doy_of_year), a, r.vpd_mean)
        for r in df.itertuples()
    ]

    lam_h = data_maturity_lambda(n, a.lambda_max_h, a)
    lam_atm = data_maturity_lambda(n, a.lambda_max_atm, a)

    p7 = df["rain_mm"].rolling(7, min_periods=7).sum().dropna()
    p7_med = float(p7.median()) if len(p7) else None
    p7_ref = a.p7_clim_mm if p7_med is None else (1.0 - lam_atm) * a.p7_clim_mm + lam_atm * p7_med

    cal = SiteCalibration(
        source="dataset_lokasi",
        n_days=n,
        lambda_h=lam_h,
        lambda_atm=lam_atm,
        wtd_baseline=df["wtd_mean"].dropna().to_numpy(float),
        vpd_baseline=df["vpd_mean"].dropna().to_numpy(float),
        solar_baseline=df["solar_mean"].dropna().to_numpy(float),
        p7_ref_mm=float(p7_ref),
        wtd_range=(float(df["wtd_mean"].min()), float(df["wtd_mean"].max())),
    )

    fit = _fit_hydro(df["rain_mm"].to_numpy(float), df["pet_peat"].to_numpy(float),
                     df["wtd_mean"].to_numpy(float), a)
    if fit is not None:
        cal.beta0, cal.beta1, cal.beta3 = fit
        cal.hydro_source = "regresi_dataset_lokasi"
    else:
        prior = prior_calibration(a)
        cal.beta0, cal.beta1, cal.beta3 = prior.beta0, prior.beta1, prior.beta3
        cal.hydro_source = "prior_asumsi (data lokasi belum cukup variatif)"
    cal.diagnostics = _walk_forward_diagnostics(df, a)
    return cal


# ---------------------------------------------------------------------------
# S5-S7. Subskor
# ---------------------------------------------------------------------------
def empirical_percentile(value: float, baseline: np.ndarray) -> Optional[float]:
    """Persentil empiris 0-100 terhadap baseline lokasi (Slide 7: 100 * CDF)."""
    arr = np.asarray(baseline, float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return None
    return float(stats.percentileofscore(arr, value, kind="weak"))


def _blend(abs_score: float, emp_score: Optional[float], lam: float) -> float:
    if emp_score is None or lam <= 0.0:
        return _clip01(abs_score)
    return _clip01((1.0 - lam) * abs_score + lam * emp_score)


def score_wtd_absolute(wtd_cm: float, a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    """H_abs: skor ekologis absolut WTD (interpolasi linear antar titik ambang)."""
    xs, ys = zip(*a.wtd_anchors)
    return float(np.interp(wtd_cm, xs, ys))


def hydrological_stress(wtd_cm: float, cal: SiteCalibration,
                        a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> Dict[str, Optional[float]]:
    """H_t = (1-lambda_h) * H_abs + lambda_h * H_emp."""
    h_abs = score_wtd_absolute(wtd_cm, a)
    h_emp = empirical_percentile(wtd_cm, cal.wtd_baseline)
    return {"H": _blend(h_abs, h_emp, cal.lambda_h), "H_abs": h_abs, "H_emp": h_emp}


def atmospheric_drying(vpd_kpa: float, solar_wm2: float, cal: SiteCalibration,
                       a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> Dict[str, Optional[float]]:
    """A_t = 0.70 * A_VPD + 0.30 * A_Solar  (Slide 8), tiap komponen = campuran absolut & persentil."""
    vpd_abs = _clip01((vpd_kpa - a.vpd_low_kpa) / (a.vpd_high_kpa - a.vpd_low_kpa) * 100.0)
    sol_abs = _clip01(solar_wm2 / a.solar_full_wm2 * 100.0)
    a_vpd = _blend(vpd_abs, empirical_percentile(vpd_kpa, cal.vpd_baseline), cal.lambda_atm)
    a_sol = _blend(sol_abs, empirical_percentile(solar_wm2, cal.solar_baseline), cal.lambda_atm)
    return {"A_VPD": a_vpd, "A_SOLAR": a_sol, "A": a.w_vpd * a_vpd + a.w_solar * a_sol}


def rainfall_deficit(rain_7d_mm: float, p7_ref_mm: float) -> float:
    """RD = 100 * max(0, 1 - P7/P7_ref)  (Slide 12)."""
    return _clip01(100.0 * max(0.0, 1.0 - max(0.0, rain_7d_mm) / max(0.1, p7_ref_mm)))


def wtd_trend(wtd_now_cm: float, wtd_7d_ago_cm: float, a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> Dict[str, Any]:
    """dWTD_7 = WTD_t - WTD_(t-7)  (Slide 15). Positif = muka air makin dalam = mengering."""
    d = float(wtd_now_cm) - float(wtd_7d_ago_cm)
    if d >= a.trend_rapid_cm:
        label = "Pengeringan Cepat"
    elif d > a.trend_stable_cm:
        label = "Pengeringan"
    elif d < -a.trend_stable_cm:
        label = "Pembasahan"
    else:
        label = "Stabil"
    return {"delta_wtd_7_hari_cm": round(d, 2), "tren": label, "mengering": d > a.trend_stable_cm}


def count_dry_days(rain_series_mm: Sequence[float], a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS) -> int:
    """Jumlah hari kering berturut-turut (hujan < dry_day_mm) menghitung mundur dari hari terakhir."""
    n = 0
    for r in reversed(list(rain_series_mm)):
        if r < a.dry_day_mm:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# S8-S9. Level 1 & Level 2 dari satu keadaan
# ---------------------------------------------------------------------------
def compute_pdwi_state(
    wtd_cm: float,
    vpd_kpa: float,
    solar_wm2: float,
    cal: SiteCalibration,
    a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS,
    rain_7d_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """PDWI-L1 & PDWI-L2 untuk satu keadaan harian."""
    h = hydrological_stress(wtd_cm, cal, a)
    at = atmospheric_drying(vpd_kpa, solar_wm2, cal, a)
    rd = None if rain_7d_mm is None else rainfall_deficit(rain_7d_mm, cal.p7_ref_mm)

    comps = {"H": h["H"], "A_VPD": at["A_VPD"], "A_SOLAR": at["A_SOLAR"], "RD": rd}
    s1, miss1 = _weighted(comps, L1_WEIGHTS)
    s2, miss2 = _weighted(comps, L2_WEIGHTS)
    level_1 = _level_dict(s1, miss1)
    level_2 = _level_dict(s2, miss2)

    # Status operasional: Level 2 (iklim-hidrologi) bila RD tersedia, jika tidak Level 1.
    if "RD" in miss2:
        status = {**level_1, "level_dipakai": "L1"}
    else:
        status = {**level_2, "level_dipakai": "L2"}
    status["rekomendasi"] = RECOMMENDATIONS[status["class"]]

    return {
        "subskor": {
            "H_t": _r(h["H"]), "H_ekologis_absolut": _r(h["H_abs"]), "H_persentil_lokasi": _r(h["H_emp"]),
            "A_VPD": _r(at["A_VPD"]), "A_Solar": _r(at["A_SOLAR"]), "A_t": _r(at["A"]),
            "RD_t": _r(rd),
        },
        "level_1": level_1,
        "level_2": level_2,
        "status_operasional": status,
    }


# ---------------------------------------------------------------------------
# S11-S12. Model neraca air & forecast
# ---------------------------------------------------------------------------
DEFAULT_WEATHER = {"rain_mm": 0.0, "temp_c": 25.0, "rh_pct": 82.0, "solar_wm2": 120.0, "wind_ms": 1.0}


def _weather(d: Optional[Dict[str, Any]], a: PeatlandAssumptions) -> Dict[str, float]:
    base = dict(DEFAULT_WEATHER)
    base["wind_ms"] = a.default_wind_ms
    if d:
        base.update({k: float(v) for k, v in d.items() if v is not None and k in base})
    return base


def build_scenarios(mode: str, current: Dict[str, float], horizon: int, cal: SiteCalibration) -> List[Dict[str, float]]:
    """
    Skenario iklim sintetis (asumsi):
      persistence : cuaca hari ini diulang
      kering      : tanpa hujan, T +0.5 C, RH -5 poin, radiasi x1.15
      basah       : hujan 15 mm/hari, T -0.5 C, RH 92 %, radiasi x0.5
    """
    out = []
    for _ in range(horizon):
        w = dict(current)
        if mode == "kering":
            w.update(rain_mm=0.0, temp_c=current["temp_c"] + 0.5,
                     rh_pct=max(40.0, current["rh_pct"] - 5.0), solar_wm2=min(300.0, current["solar_wm2"] * 1.15))
        elif mode == "basah":
            w.update(rain_mm=15.0, temp_c=current["temp_c"] - 0.5, rh_pct=92.0, solar_wm2=current["solar_wm2"] * 0.5)
        elif mode != "persistence":
            raise ValueError(f"Mode skenario tidak dikenal: {mode}")
        out.append(w)
    return out


def forecast_pdwi(
    current_wtd_cm: float,
    current_weather: Optional[Dict[str, Any]],
    scenarios: Sequence[Dict[str, Any]],
    cal: SiteCalibration,
    a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS,
    start_date: Optional[date] = None,
    recent_rain_mm: Optional[Sequence[float]] = None,
    wtd_history_cm: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Forecast PDWI N hari:   cuaca(t) -> dWTD -> WTD(t+1) -> H_t -> PDWI(t+1)
      current_wtd_cm : WTD hari ini (cm di bawah permukaan, positif)
      scenarios      : cuaca harian hari D+1 .. D+N
      recent_rain_mm : hujan harian sampai hari ini (elemen terakhir = hari ini); None -> diasumsikan netral
      wtd_history_cm : WTD harian sampai hari ini (elemen terakhir = hari ini); dipakai untuk tren
    """
    start = start_date or date.today()
    cur = _weather(current_weather, a)

    have_rain_hist = recent_rain_mm is not None and len(recent_rain_mm) > 0
    if have_rain_hist:
        rain_series = [float(x) for x in recent_rain_mm]
    else:  # asumsi netral: hujan harian = referensi iklim, kecuali hari ini
        rain_series = [cal.p7_ref_mm / 7.0] * 6 + [cur["rain_mm"]]

    path: List[float] = [float(x) for x in wtd_history_cm[:-1]] if wtd_history_cm else []
    path.append(float(current_wtd_cm))

    wtd = float(current_wtd_cm)
    prev = cur
    results: List[Dict[str, Any]] = []
    for h, sc in enumerate(scenarios, start=1):
        day = start + timedelta(days=h)
        prev_day = start + timedelta(days=h - 1)

        # 1. transisi WTD digerakkan cuaca hari sebelumnya
        pet_prev = pet_peat(prev["temp_c"], prev["rh_pct"], prev["solar_wm2"], prev["wind_ms"],
                            prev_day.timetuple().tm_yday, a)
        delta = _dwtd(cal.beta0, cal.beta1, cal.beta3, prev["rain_mm"], pet_prev, wtd, a.gw_et_fraction)
        wtd = float(np.clip(wtd + delta, 0.0, a.max_wtd_cm))
        path.append(wtd)

        # 2. cuaca hari ini (skenario) -> subskor atmosfer & jendela hujan 7 hari
        w = _weather(sc, a)
        rain_series.append(w["rain_mm"])
        vpd = calculate_vpd(w["temp_c"], w["rh_pct"])
        rain_7d = float(sum(rain_series[-7:])) if len(rain_series) >= 7 else None

        state = compute_pdwi_state(wtd, vpd, w["solar_wm2"], cal, a, rain_7d)
        trend = wtd_trend(path[-1], path[-8], a) if len(path) >= 8 else None

        results.append({
            "hari": f"D+{h}",
            "hari_ke": h,
            "tanggal": day.isoformat(),
            "wtd_cm": round(wtd, 2),
            "tmat_cm": round(-wtd, 2),
            "delta_wtd_cm": round(delta, 2),
            "pet_gambut_mm": _r(pet_prev, 2),
            "skenario_iklim": {k: round(v, 2) for k, v in w.items()} | {"vpd_kpa": round(vpd, 3)},
            "hujan_7_hari_mm": _r(rain_7d),
            "subskor": state["subskor"],
            "level_1": {k: state["level_1"][k] for k in ("score", "class", "label")},
            "level_2": {k: state["level_2"][k] for k in ("score", "class", "label")},
            "status_kekeringan": state["status_operasional"]["label"],
            "status_skor": state["status_operasional"]["score"],
            "status_kelas": state["status_operasional"]["class"],
            "tren_7_hari": trend,
            "hari_kering_beruntun": count_dry_days(rain_series, a) if have_rain_hist else None,
            "di_luar_rentang_kalibrasi": bool(cal.wtd_range[1] > 0 and wtd > cal.wtd_range[1]),
        })
        prev = w
    return results


def early_warning_summary(forecast: List[Dict[str, Any]], current_class: int, threshold_class: int = 3) -> Dict[str, Any]:
    """Kapan kelas >= HIGH tercapai (0 = sudah tercapai hari ini), puncak skor, dan titik akhir WTD."""
    already = current_class >= threshold_class
    first = next((f for f in forecast if f["status_kelas"] >= threshold_class), None)
    peak = max(forecast, key=lambda f: f["status_skor"]) if forecast else None
    if already:
        days_to = 0
    else:
        days_to = None if first is None else first["hari_ke"]
    return {
        "kelas_saat_ini": current_class,
        "sudah_kelas_tinggi": already,
        "hari_menuju_kelas_tinggi": days_to,
        "tanggal_kelas_tinggi_pertama": (None if already or first is None else first["tanggal"]),
        "puncak_skor": None if peak is None else peak["status_skor"],
        "puncak_pada": None if peak is None else peak["hari"],
        "wtd_akhir_horizon_cm": None if not forecast else forecast[-1]["wtd_cm"],
        "ada_ekstrapolasi_di_luar_kalibrasi": any(f["di_luar_rentang_kalibrasi"] for f in forecast),
    }


# ---------------------------------------------------------------------------
# Analisis dari dataset lokasi
# ---------------------------------------------------------------------------
def _daily_with_et(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["_ts"] = pd.to_datetime(df["date"])
    return df.sort_values("_ts").reset_index(drop=True)


def analyze_history(daily: pd.DataFrame, cal: SiteCalibration, a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS,
                    last_n: int = 7) -> List[Dict[str, Any]]:
    """PDWI L1/L2 untuk ``last_n`` hari terakhir dataset (RD & tren dihitung dari jendela riil)."""
    df = _daily_with_et(daily)
    p7 = df["rain_mm"].rolling(7, min_periods=7).sum()
    out = []
    for i in range(max(0, len(df) - last_n), len(df)):
        r = df.iloc[i]
        wtd = float(r["wtd_mean"])
        state = compute_pdwi_state(wtd, float(r["vpd_mean"]), float(r["solar_mean"]), cal, a,
                                   None if np.isnan(p7.iloc[i]) else float(p7.iloc[i]))
        trend = wtd_trend(wtd, float(df["wtd_mean"].iloc[i - 7]), a) if i >= 7 else None
        out.append({
            "tanggal": str(r["date"]),
            "wtd_cm": round(wtd, 2),
            "tmat_cm": round(-wtd, 2),
            "hujan_mm": round(float(r["rain_mm"]), 2),
            "level_1": {k: state["level_1"][k] for k in ("score", "label")},
            "level_2": {k: state["level_2"][k] for k in ("score", "label")},
            "status_kekeringan": state["status_operasional"]["label"],
            "tren_7_hari": trend,
        })
    return out


def analyze_latest(daily: pd.DataFrame, cal: SiteCalibration, a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS,
                   horizon: int = 7, history_days: int = 7) -> Dict[str, Any]:
    """Keadaan terakhir dataset + forecast (persistence/kering/basah) + riwayat."""
    df = _daily_with_et(daily)
    last = df.iloc[-1]
    wtd_now = float(last["wtd_mean"])
    rain_hist = df["rain_mm"].tail(7).astype(float).tolist()
    wtd_hist = df["wtd_mean"].tail(8).astype(float).tolist()
    last_date = pd.Timestamp(last["date"]).date()

    # Cuaca "hari ini" = observasi terakhir; skenario persistence = rata-rata 7 hari terakhir
    tail = df.tail(7)
    today = {"rain_mm": float(last["rain_mm"]), "temp_c": float(last["temp_mean"]),
             "rh_pct": float(last["rh_mean"]), "solar_wm2": float(last["solar_mean"]),
             "wind_ms": float(last["wind_mean"])}
    mean7 = {"rain_mm": float(tail["rain_mm"].mean()), "temp_c": float(tail["temp_mean"].mean()),
             "rh_pct": float(tail["rh_mean"].mean()), "solar_wm2": float(tail["solar_mean"].mean()),
             "wind_ms": float(tail["wind_mean"].mean())}

    p7_now = float(sum(rain_hist))
    state = compute_pdwi_state(wtd_now, float(last["vpd_mean"]), float(last["solar_mean"]), cal, a, p7_now)
    trend = wtd_trend(wtd_now, float(df["wtd_mean"].iloc[-8]), a) if len(df) >= 8 else None

    def run(mode: str) -> List[Dict[str, Any]]:
        return forecast_pdwi(wtd_now, today, build_scenarios(mode, mean7, horizon, cal), cal, a,
                             start_date=last_date, recent_rain_mm=rain_hist, wtd_history_cm=wtd_hist)

    main_fc = run("persistence")
    return {
        "tanggal_data_terakhir": last_date.isoformat(),
        "keadaan_saat_ini": {
            "wtd_cm": round(wtd_now, 2),
            "tmat_cm": round(-wtd_now, 2),
            "vpd_kpa": round(float(last["vpd_mean"]), 3),
            "hujan_7_hari_mm": round(p7_now, 2),
            "hari_kering_beruntun": count_dry_days(rain_hist, a),
            "tren_7_hari": trend,
            **state,
        },
        "riwayat": analyze_history(df, cal, a, history_days),
        "forecast": main_fc,
        "skenario_pembanding": {
            "kering": [_compact(f) for f in run("kering")],
            "basah": [_compact(f) for f in run("basah")],
        },
        "peringatan_dini": early_warning_summary(main_fc, state["status_operasional"]["class"]),
    }


def _compact(f: Dict[str, Any]) -> Dict[str, Any]:
    return {"hari": f["hari"], "tanggal": f["tanggal"], "wtd_cm": f["wtd_cm"],
            "skor": f["status_skor"], "status": f["status_kekeringan"]}


# ---------------------------------------------------------------------------
# Titik masuk untuk satu pembacaan sensor
# ---------------------------------------------------------------------------
def calculate_from_sensor(
    tmat_cm: float,
    temp_c: float,
    rh_pct: float,
    solar_wm2: float,
    cal: SiteCalibration,
    a: PeatlandAssumptions = DEFAULT_ASSUMPTIONS,
    wind_ms: Optional[float] = None,
    rain_7d_mm: Optional[float] = None,
    tmat_7d_ago_cm: Optional[float] = None,
    dry_days: Optional[int] = None,
    on_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Hitung PDWI-Gambut dari 4 sensor wajib (+ opsional). Nilai tidak ada -> asumsi / komponen dibuang."""
    notes: List[str] = []
    wtd, inundated = wtd_from_tmat(tmat_cm)
    if inundated:
        notes.append("TMAT >= 0 (tergenang): WTD dibatasi 0 cm.")

    vpd = calculate_vpd(temp_c, rh_pct)
    if wind_ms is None:
        wind_ms = a.default_wind_ms
        notes.append(f"wind_speed_ms tidak diberikan -> asumsi {a.default_wind_ms} m/s.")
    day = on_date or date.today()
    if on_date is None:
        notes.append("tanggal tidak diberikan -> memakai tanggal hari ini untuk radiasi ekstraterestrial.")
    et0 = penman_monteith_et0(temp_c, rh_pct, solar_wm2, wind_ms, day.timetuple().tm_yday, a)

    state = compute_pdwi_state(wtd, vpd, solar_wm2, cal, a, rain_7d_mm)
    if rain_7d_mm is None:
        notes.append("rain_7d_mm tidak diberikan -> Rainfall Deficit dibuang; status memakai Level 1.")

    trend = None
    if tmat_7d_ago_cm is not None:
        trend = wtd_trend(wtd, wtd_from_tmat(tmat_7d_ago_cm)[0], a)
    else:
        notes.append("tmat_7d_ago_cm tidak diberikan -> tren 7 hari tidak dihitung.")

    if cal.source == "prior_asumsi":
        notes.append("Tanpa baseline lokasi: H_t, A_t & RD memakai ambang asumsi murni (lambda = 0).")
    else:
        notes.append(f"Baseline lokasi dari {cal.n_days} hari data (bobot H={cal.lambda_h:.2f}, atmosfer={cal.lambda_atm:.2f}).")

    available = 4 + sum(x is not None for x in (rain_7d_mm, tmat_7d_ago_cm))
    return {
        "tipe_lahan": "gambut",
        "input": {
            "tmat_cm": _r(tmat_cm), "suhu_c": _r(temp_c), "rh_persen": _r(rh_pct),
            "radiasi_wm2": _r(solar_wm2), "angin_ms": _r(wind_ms),
            "hujan_7_hari_mm": _r(rain_7d_mm), "tmat_7_hari_lalu_cm": _r(tmat_7d_ago_cm),
        },
        "turunan": {
            "wtd_cm": _r(wtd), "tergenang": inundated, "vpd_kpa": _r(vpd, 3),
            "et0_mm_hari": _r(et0), "pet_gambut_mm_hari": _r(a.kc_peat * et0),
        },
        **state,
        "tren_7_hari": trend,
        "hari_kering_beruntun": dry_days,
        "kelengkapan_data": {"sensor_wajib": "4/4", "input_opsional_terisi": f"{available - 4}/2"},
        "asumsi_diterapkan": notes,
    }


# ---------------------------------------------------------------------------
# Cache kalibrasi dataset (dibuat ulang bila berkas dataset berubah)
# ---------------------------------------------------------------------------
_CAL_CACHE: Dict[str, Any] = {"key": None, "cal": None, "daily": None}


def load_site_dataset(path: Optional[os.PathLike] = None) -> Tuple[pd.DataFrame, SiteCalibration]:
    """Muat dataset lokasi -> (daily_df, kalibrasi). Di-cache berdasarkan mtime berkas."""
    from src.data_loader import load_aws_dataset  # import lokal: hindari beban impor saat unit test murni

    p = os.fspath(path or DEFAULT_RAW_DATA_PATH)
    key = (p, os.path.getmtime(p))
    if _CAL_CACHE["key"] != key:
        daily = aggregate_5min_to_daily(load_aws_dataset(p))
        _CAL_CACHE.update(key=key, daily=daily, cal=calibrate_site(daily))
    return _CAL_CACHE["daily"], _CAL_CACHE["cal"]
