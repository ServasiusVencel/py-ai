"""
PFWI-Gambut — Peatland Fire Weather Index Engine (Nimbus)
=========================================================
Algoritma bahaya kebakaran KHUSUS lahan gambut (akun ``is_peatland = True``).

Mengapa tidak memakai Pseudo-FWI Dokumen 1?
  Pseudo-FWI (``src/fwi.py``) hanya membaca cuaca sesaat (suhu, RH, radiasi, hujan) sehingga cocok
  untuk lahan mineral. Di lahan gambut, kebakaran ditentukan oleh DUA lapis bahan bakar:
    1. Bahan bakar permukaan (serasah, semak, gambut fibrik)  -> mudah tersulut, digerakkan cuaca harian.
    2. Bahan bakar bawah permukaan (tubuh gambut)             -> terbakar membara (smouldering), sulit
       dipadamkan, ketersediaannya ditentukan kedalaman muka air tanah (WTD/TMAT) & kekeringan berhari-hari.
  Karena itu PFWI-Gambut memakai sistem berstatus (memiliki memori harian) yang menggabungkan:
    * Canadian Forest Fire Weather Index System (Van Wagner 1987) — FFMC, DMC, DC, ISI, BUI, FWI —
      dengan penyesuaian panjang hari ekuator (Lawson & Armitage 2008) dan kelas bahaya
      Indonesia/Malaysia Fire Danger Rating System (FDRS; de Groot et al. 2006, dipakai BMKG).
    * Hydrological Stress H_t hasil ukur sensor TMAT dari engine PDWI-Gambut (``src/peatland_pdwi.py``).

Urutan perhitungan (sequence)
-----------------------------
  F0  Kelas bahaya                       RENDAH / SEDANG / TINGGI / EKSTRIM (0-25-50-75-100)
  F1  Cuaca siang (noon)                 T, RH, angin jam 12:00 WIB; hujan 24 jam berakhir 12:00
  F2  Angin 10 m                         u10 = u2 * ln(67.8*10 - 5.42) / 4.87   (profil log FAO-56) -> km/jam
  F3  FFMC  (bahan bakar halus)          Van Wagner (1987) pers. 1-10
  F4  DMC   (lapisan organik dangkal)    pers. 11-16, Le ekuator = 9 jam
  F5  DC    (lapisan organik dalam)      pers. 18-21, Lf ekuator = 1.4
  F6  ISI, BUI, FWI, DSR                 pers. 24-30
  F7  Subskor 0-100                      S_ign = f(FFMC), S_spread = f(ISI), S_DC = f(DC)
                                         (titik belok = batas kelas FDRS Indonesia)
  F8  Peat Fuel Availability             S_peat = 0.60 * H_t(WTD) + 0.40 * S_DC
  F9  Persistensi kemarau                S_dry  = f(hari kering beruntun)
  F10 PFWI                               0.45 S_peat + 0.30 S_ign + 0.15 S_spread + 0.10 S_dry
  F11 Aturan eskalasi gambut             R1: WTD >= 40 cm & FFMC >= 83            -> min. TINGGI
                                         R2: WTD >= 80 cm & DC >= 350 & FFMC >= 78 -> EKSTRIM
                                         R3: pengeringan cepat (dWTD7 >= 20 cm) & WTD >= 40 cm & FFMC >= 73
                                             -> min. TINGGI
  F12 Forecast N hari                    WTD dari model neraca air PDWI + kode FWI dilanjutkan harian

Semua parameter yang tidak tersedia dari sensor memakai NILAI ASUMSI di ``PeatFWIAssumptions`` dan
dilaporkan di keluaran API.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_LATITUDE, DEFAULT_RAW_DATA_PATH, FIRE_RISK_LABELS
from src import peatland_pdwi as pdwi

ENGINE_NAME = "PFWI-Gambut Nimbus v1"


# ---------------------------------------------------------------------------
# 0. PARAMETER ASUMSI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeatFWIAssumptions:
    """Seluruh nilai asumsi PFWI-Gambut. Ubah di sini untuk kalibrasi lokasi lain."""

    # --- Stasiun ---------------------------------------------------------------
    latitude_deg: float = DEFAULT_LATITUDE
    wind_sensor_height_m: float = 2.0             # tinggi anemometer AWS
    default_wind_ms: float = 1.0                  # bila sensor angin tidak mengirim data
    noon_hour: int = 12                           # jam pengamatan standar FWI (waktu lokal WIB)
    noon_window_h: float = 1.0                    # rata-rata 11:00-13:00 untuk meredam noise 5-menit

    # --- Konversi cuaca harian-rata-rata -> siang (untuk skenario forecast) ------
    prior_t_noon_offset_c: float = 3.0            # T_siang - T_rata2 (dikalibrasi dari data lokasi)
    prior_rh_noon_offset_pct: float = 12.0        # RH_rata2 - RH_siang
    prior_wind_noon_factor: float = 1.5           # angin_siang / angin_rata2

    # --- Nilai awal (start-up) kode FWI — Van Wagner standar ---------------------
    startup_ffmc: float = 85.0
    startup_dmc: float = 6.0
    startup_dc: float = 15.0
    dc_init_from_wtd: bool = True                 # DC awal disetarakan dengan kelas WTD terukur

    # --- Pemetaan kode FWI -> subskor 0..100 (batas kelas FDRS Indonesia -> 25/50/75) ---
    ffmc_anchors: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0), (50.0, 5.0), (73.0, 25.0), (78.0, 50.0), (83.0, 75.0), (92.0, 100.0))
    isi_anchors: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0), (2.0, 25.0), (4.0, 50.0), (6.0, 75.0), (10.0, 100.0))
    dc_anchors: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0), (140.0, 25.0), (260.0, 50.0), (350.0, 75.0), (500.0, 100.0))
    dry_day_anchors: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0), (3.0, 25.0), (7.0, 50.0), (14.0, 75.0), (21.0, 100.0))

    # --- Bobot komponen --------------------------------------------------------
    w_peat_h: float = 0.60                        # porsi H_t (WTD terukur) dalam S_peat
    w_peat_dc: float = 0.40                       # porsi S_DC (kekeringan dalam berbasis cuaca)
    weights: Tuple[Tuple[str, float], ...] = (
        ("S_peat", 0.45), ("S_ign", 0.30), ("S_spread", 0.15), ("S_dry", 0.10))

    # --- Aturan eskalasi gambut -------------------------------------------------
    wtd_damage_cm: float = 40.0                   # ambang kerusakan gambut (PP 57/2016)
    wtd_deep_cm: float = 80.0                     # gambut kering dalam
    ffmc_extreme: float = 83.0                    # FDRS: penyulutan ekstrem
    ffmc_high: float = 78.0                       # FDRS: penyulutan tinggi
    ffmc_moderate: float = 73.0                   # FDRS: bahan bakar permukaan mulai dapat tersulut
    dc_extreme: float = 350.0                     # FDRS: DC ekstrem (asap & bara gambut)
    rapid_drying_cm: float = 20.0                 # dWTD 7 hari

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_FWI_ASSUMPTIONS = PeatFWIAssumptions()

# Kelas FDRS Indonesia (de Groot et al. 2006 / BMKG): batas bawah kelas Sedang, Tinggi, Ekstrim
FDRS_BOUNDS = {
    "FFMC": (73.0, 78.0, 83.0),
    "DMC": (15.0, 30.0, 40.0),
    "DC": (140.0, 260.0, 350.0),
    "ISI": (2.0, 4.0, 6.0),
    "BUI": (20.0, 40.0, 60.0),
    "FWI": (2.0, 7.0, 13.0),
}

RECOMMENDATIONS = {
    0: "Bahaya kebakaran gambut rendah. Lanjutkan pemantauan TMAT & cuaca rutin.",
    1: "Waspada: bahan bakar permukaan mulai mudah tersulut. Larang pembakaran lahan, cek sekat kanal & sumber air.",
    2: "Tinggi: api mudah menyala dan gambut dapat ikut terbakar. Patroli harian, siagakan regu & pompa, lakukan pembasahan.",
    3: "Ekstrim: kebakaran gambut membara sulit dipadamkan. Siaga darurat karhutla, pembasahan masif, koordinasi BPBD/Manggala Agni.",
}


def assumptions_catalog(a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS) -> List[Dict[str, Any]]:
    """Katalog asumsi + dasar penetapan, untuk transparansi di API & dokumentasi."""
    return [
        {"parameter": "noon_hour / noon_window_h", "nilai": [a.noon_hour, a.noon_window_h], "satuan": "jam",
         "dasar": "FWI dihitung dari cuaca jam 12:00 waktu lokal standar; rata-rata ±1 jam meredam noise 5-menit."},
        {"parameter": "wind_sensor_height_m", "nilai": a.wind_sensor_height_m, "satuan": "m",
         "dasar": "FWI butuh angin 10 m; dikonversi dari 2 m dengan profil logaritmik FAO-56."},
        {"parameter": "default_wind_ms", "nilai": a.default_wind_ms, "satuan": "m/s",
         "dasar": "Dipakai bila sensor angin tidak mengirim data."},
        {"parameter": "day_length_factors", "nilai": {"Le_DMC": dmc_day_length(a.latitude_deg, 1),
                                                      "Lf_DC": dc_day_length(a.latitude_deg, 1)},
         "satuan": "-", "dasar": "Penyesuaian lintang ekuator (Lawson & Armitage 2008 / cffdrs)."},
        {"parameter": "startup_ffmc / dmc / dc", "nilai": [a.startup_ffmc, a.startup_dmc, a.startup_dc],
         "satuan": "-", "dasar": "Nilai awal standar Van Wagner bila kode hari sebelumnya tidak tersedia."},
        {"parameter": "dc_init_from_wtd", "nilai": a.dc_init_from_wtd, "satuan": "-",
         "dasar": "Tanpa riwayat, DC awal disetarakan dengan skor WTD terukur (sensor TMAT lebih akurat daripada start-up 15)."},
        {"parameter": "ffmc/isi/dc_anchors", "nilai": {"FFMC": [list(p) for p in a.ffmc_anchors],
                                                      "ISI": [list(p) for p in a.isi_anchors],
                                                      "DC": [list(p) for p in a.dc_anchors]},
         "satuan": "kode -> skor", "dasar": "Batas kelas FDRS Indonesia (de Groot et al. 2006) dipetakan ke skor 25/50/75."},
        {"parameter": "dry_day_anchors", "nilai": [list(p) for p in a.dry_day_anchors], "satuan": "hari -> skor",
         "dasar": "Asumsi: 7 hari tanpa hujan = sedang, 14 hari = tinggi, >=21 hari = ekstrim."},
        {"parameter": "w_peat_h / w_peat_dc", "nilai": [a.w_peat_h, a.w_peat_dc], "satuan": "-",
         "dasar": "WTD terukur diberi bobot lebih besar daripada DC (proksi WTD berbasis cuaca)."},
        {"parameter": "weights", "nilai": dict(a.weights), "satuan": "-",
         "dasar": "Gambut: ketersediaan bahan bakar bawah permukaan paling dominan, lalu penyulutan & penjalaran."},
        {"parameter": "prior_t_noon_offset_c / prior_rh_noon_offset_pct / prior_wind_noon_factor",
         "nilai": [a.prior_t_noon_offset_c, a.prior_rh_noon_offset_pct, a.prior_wind_noon_factor], "satuan": "-",
         "dasar": "Konversi cuaca rata-rata harian -> siang untuk skenario forecast; diganti hasil kalibrasi data lokasi."},
        {"parameter": "aturan_eskalasi", "nilai": {"wtd_damage_cm": a.wtd_damage_cm, "wtd_deep_cm": a.wtd_deep_cm,
                                                  "ffmc_extreme": a.ffmc_extreme, "ffmc_high": a.ffmc_high,
                                                  "ffmc_moderate": a.ffmc_moderate,
                                                  "dc_extreme": a.dc_extreme, "rapid_drying_cm": a.rapid_drying_cm},
         "satuan": "-", "dasar": "40 cm ambang kerusakan gambut (PP 57/2016); FFMC 83 & DC 350 kelas Ekstrim FDRS."},
    ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _r(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def _interp(x: float, anchors: Sequence[Tuple[float, float]]) -> float:
    xs, ys = zip(*anchors)
    return float(np.interp(float(x), xs, ys))


def _inverse_interp(score: float, anchors: Sequence[Tuple[float, float]]) -> float:
    xs, ys = zip(*anchors)
    return float(np.interp(float(score), ys, xs))


def classify_pfwi(score: float) -> Tuple[int, str]:
    """0-25 Rendah, 25-50 Sedang, 50-75 Tinggi, >=75 Ekstrim (label sama dengan Pseudo-FWI)."""
    s = float(np.clip(score, 0.0, 100.0))
    cls = 0 if s < 25.0 else 1 if s < 50.0 else 2 if s < 75.0 else 3
    return cls, FIRE_RISK_LABELS[cls]


def fdrs_class(code: str, value: float) -> Tuple[int, str]:
    """Kelas FDRS Indonesia untuk satu komponen sistem FWI."""
    lo, mid, hi = FDRS_BOUNDS[code]
    cls = 0 if value < lo else 1 if value < mid else 2 if value < hi else 3
    return cls, FIRE_RISK_LABELS[cls]


# ---------------------------------------------------------------------------
# F2. Angin 10 m
# ---------------------------------------------------------------------------
def wind_10m_kmh(wind_ms: float, sensor_height_m: float = 2.0) -> float:
    """Konversi angin sensor ke ketinggian 10 m (profil log FAO-56) dalam km/jam."""
    w = max(0.0, float(wind_ms))
    if abs(sensor_height_m - 10.0) > 1e-6:
        u2 = w * 4.87 / math.log(67.8 * sensor_height_m - 5.42)       # -> 2 m
        w = u2 * math.log(67.8 * 10.0 - 5.42) / 4.87                    # 2 m -> 10 m
    return w * 3.6


# ---------------------------------------------------------------------------
# F3-F6. Canadian FWI System (Van Wagner 1987), sesuai implementasi cffdrs
# ---------------------------------------------------------------------------
_ELL_46N = (6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0)
_ELL_20N = (7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8)
_ELL_20S = (10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2)
_ELL_40S = (11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8)
_FL_N = (-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6)
_FL_S = (6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8)


def dmc_day_length(lat: float, month: int) -> float:
    """Faktor panjang hari efektif Le untuk DMC."""
    if lat > 30:
        return _ELL_46N[month - 1]
    if lat > 10:
        return _ELL_20N[month - 1]
    if lat > -10:
        return 9.0                                 # pita ekuator
    if lat > -30:
        return _ELL_20S[month - 1]
    return _ELL_40S[month - 1]


def dc_day_length(lat: float, month: int) -> float:
    """Faktor panjang hari Lf untuk DC."""
    if lat > 20:
        return _FL_N[month - 1]
    if lat > -20:
        return 1.4                                 # pita ekuator
    return _FL_S[month - 1]


def ffmc_next(ffmc_yda: float, temp: float, rh: float, wind_kmh: float, rain: float) -> float:
    """Fine Fuel Moisture Code."""
    rh = float(np.clip(rh, 0.0, 100.0))
    mo = 147.2 * (101.0 - ffmc_yda) / (59.5 + ffmc_yda)
    if rain > 0.5:
        rf = rain - 0.5
        add = 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        if mo > 150.0:
            add += 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        mo = min(mo + add, 250.0)
    ed = 0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind_kmh) * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = 0.618 * rh ** 0.753 + 10.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind_kmh) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            kw = k1 * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    return float(np.clip(59.5 * (250.0 - m) / (147.2 + m), 0.0, 101.0))


def dmc_next(dmc_yda: float, temp: float, rh: float, rain: float, month: int, lat: float) -> float:
    """Duff Moisture Code."""
    rh = float(np.clip(rh, 0.0, 100.0))
    t = max(temp, -1.1)
    rk = 1.894 * (t + 1.1) * (100.0 - rh) * dmc_day_length(lat, month) * 1e-4
    pr = dmc_yda
    if rain > 1.5:
        rw = 0.92 * rain - 1.27
        wmi = 20.0 + 280.0 / math.exp(0.023 * dmc_yda)
        if dmc_yda <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_yda)
        elif dmc_yda <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_yda)
        else:
            b = 6.2 * math.log(dmc_yda) - 17.2
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr = max(0.0, 43.43 * (5.6348 - math.log(wmr - 20.0)))
    return max(0.0, pr + rk)


def dc_next(dc_yda: float, temp: float, rain: float, month: int, lat: float) -> float:
    """Drought Code."""
    t = max(temp, -2.8)
    pe = max(0.0, (0.36 * (t + 2.8) + dc_day_length(lat, month)) / 2.0)
    dr = dc_yda
    if rain > 2.8:
        rw = 0.83 * rain - 1.27
        smi = 800.0 * math.exp(-dc_yda / 400.0)
        dr = max(0.0, dc_yda - 400.0 * math.log(1.0 + 3.937 * rw / smi))
    return max(0.0, dr + pe)


def isi_calc(ffmc: float, wind_kmh: float) -> float:
    """Initial Spread Index."""
    fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * math.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)
    return 0.208 * math.exp(0.05039 * wind_kmh) * ff


def bui_calc(dmc: float, dc: float) -> float:
    """Buildup Index."""
    if dmc == 0 and dc == 0:
        return 0.0
    bui1 = 0.8 * dc * dmc / (dmc + 0.4 * dc)
    if bui1 < dmc:
        p = (dmc - bui1) / dmc
        cc = 0.92 + (0.0114 * dmc) ** 1.7
        return max(0.0, dmc - cc * p)
    return bui1


def fwi_calc(isi: float, bui: float) -> float:
    """Fire Weather Index (kanonik)."""
    if bui > 80.0:
        bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 / math.exp(0.023 * bui)))
    else:
        bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
    return bb if bb <= 1.0 else math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)


@dataclass
class FWICodes:
    """Kode berstatus (dibawa dari hari ke hari)."""
    ffmc: float
    dmc: float
    dc: float

    def to_dict(self) -> Dict[str, float]:
        return {"ffmc": round(self.ffmc, 2), "dmc": round(self.dmc, 2), "dc": round(self.dc, 2)}


def startup_codes(wtd_cm: Optional[float] = None,
                  a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS) -> FWICodes:
    """Kode awal. Bila ``dc_init_from_wtd``: DC awal = DC dengan skor setara skor ekologis WTD terukur."""
    dc0 = a.startup_dc
    if a.dc_init_from_wtd and wtd_cm is not None:
        dc0 = max(a.startup_dc, _inverse_interp(pdwi.score_wtd_absolute(wtd_cm), a.dc_anchors))
    return FWICodes(a.startup_ffmc, a.startup_dmc, dc0)


def step_fwi_system(prev: FWICodes, temp: float, rh: float, wind_kmh: float, rain: float,
                    month: int, a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS) -> Dict[str, Any]:
    """Satu langkah harian sistem FWI -> kode baru + indeks turunan."""
    ffmc = ffmc_next(prev.ffmc, temp, rh, wind_kmh, rain)
    dmc = dmc_next(prev.dmc, temp, rh, rain, month, a.latitude_deg)
    dc = dc_next(prev.dc, temp, rain, month, a.latitude_deg)
    isi = isi_calc(ffmc, wind_kmh)
    bui = bui_calc(dmc, dc)
    fwi = fwi_calc(isi, bui)
    return {
        "codes": FWICodes(ffmc, dmc, dc),
        "FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi, "BUI": bui, "FWI": fwi,
        "DSR": 0.0272 * fwi ** 1.77,
    }


# ---------------------------------------------------------------------------
# F7-F11. Indeks PFWI-Gambut
# ---------------------------------------------------------------------------
def compute_pfwi_state(
    fwi_sys: Dict[str, Any],
    wtd_cm: float,
    cal: pdwi.SiteCalibration,
    a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS,
    dry_days: Optional[int] = None,
    delta_wtd_7d_cm: Optional[float] = None,
) -> Dict[str, Any]:
    """Gabungkan sistem FWI + hidrologi gambut terukur -> skor PFWI 0-100, kelas & eskalasi."""
    h = pdwi.hydrological_stress(wtd_cm, cal)
    s_ign = _interp(fwi_sys["FFMC"], a.ffmc_anchors)
    s_spread = _interp(fwi_sys["ISI"], a.isi_anchors)
    s_dc = _interp(fwi_sys["DC"], a.dc_anchors)
    s_peat = (a.w_peat_h * h["H"] + a.w_peat_dc * s_dc) / (a.w_peat_h + a.w_peat_dc)
    s_dry = None if dry_days is None else _interp(dry_days, a.dry_day_anchors)

    comps = {"S_peat": s_peat, "S_ign": s_ign, "S_spread": s_spread, "S_dry": s_dry}
    score, missing = pdwi._weighted(comps, dict(a.weights))
    base_cls, _ = classify_pfwi(score)

    cls = base_cls
    rules: List[str] = []
    ffmc, dc = fwi_sys["FFMC"], fwi_sys["DC"]
    if wtd_cm >= a.wtd_damage_cm and ffmc >= a.ffmc_extreme and cls < 2:
        cls = 2
        rules.append(f"R1: WTD >= {a.wtd_damage_cm:g} cm & FFMC >= {a.ffmc_extreme:g} -> minimal TINGGI")
    if wtd_cm >= a.wtd_deep_cm and dc >= a.dc_extreme and ffmc >= a.ffmc_high and cls < 3:
        cls = 3
        rules.append(f"R2: WTD >= {a.wtd_deep_cm:g} cm & DC >= {a.dc_extreme:g} & FFMC >= {a.ffmc_high:g} -> EKSTRIM")
    if (delta_wtd_7d_cm is not None and delta_wtd_7d_cm >= a.rapid_drying_cm
            and wtd_cm >= a.wtd_damage_cm and ffmc >= a.ffmc_moderate and cls < 2):
        cls = 2
        rules.append(f"R3: pengeringan cepat (dWTD7 >= {a.rapid_drying_cm:g} cm) & WTD >= {a.wtd_damage_cm:g} cm "
                     f"& FFMC >= {a.ffmc_moderate:g} -> minimal TINGGI")

    fdrs = {k: {"nilai": _r(fwi_sys[k]), "kelas": fdrs_class(k, fwi_sys[k])[0],
                "label": fdrs_class(k, fwi_sys[k])[1]} for k in FDRS_BOUNDS}
    fdrs["DSR"] = {"nilai": _r(fwi_sys["DSR"], 3)}

    return {
        "pfwi": {
            "score": round(score, 2),
            "class": cls,
            "label": FIRE_RISK_LABELS[cls],
            "kelas_sebelum_eskalasi": base_cls,
            "eskalasi_gambut": rules,
            "komponen_tidak_tersedia": missing,
            "rekomendasi": RECOMMENDATIONS[cls],
        },
        "subskor": {
            "S_peat": _r(s_peat), "H_t_wtd": _r(h["H"]), "S_DC": _r(s_dc),
            "S_ign_ffmc": _r(s_ign), "S_spread_isi": _r(s_spread), "S_dry": _r(s_dry),
        },
        "sistem_fwi_fdrs": fdrs,
    }


# ---------------------------------------------------------------------------
# F1. Agregasi data 5-menit -> cuaca siang harian
# ---------------------------------------------------------------------------
def aggregate_noon_daily(df_sensor: pd.DataFrame, a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS) -> pd.DataFrame:
    """
    Data 5-menit -> satu baris per tanggal:
      t_noon, rh_noon, wind_noon : rata-rata jendela noon_hour ± noon_window_h
      rain_24h_noon              : hujan (D-1 12:00, D 12:00]  — standar FWI
      wtd_mean, temp_mean, rh_mean, wind_mean, rain_mm (hari kalender) dari agregasi PDWI
    """
    df = df_sensor.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "wind_speed_ms" not in df.columns:
        df["wind_speed_ms"] = a.default_wind_ms
    rain_col = "rainfall_intensity_mmh" if "rainfall_intensity_mmh" in df.columns else "rain_mm"
    df["_rain5"] = df[rain_col].astype(float) * (5.0 / 60.0)

    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    win = df[(hour >= a.noon_hour - a.noon_window_h) & (hour <= a.noon_hour + a.noon_window_h)].copy()
    win["date"] = win["timestamp"].dt.date
    noon = win.groupby("date").agg(t_noon=("air_temperature_c", "mean"),
                                   rh_noon=("relative_humidity_pct", "mean"),
                                   wind_noon=("wind_speed_ms", "mean"))

    shift = pd.Timedelta(hours=24 - a.noon_hour) - pd.Timedelta(seconds=1)
    df["_fwi_day"] = (df["timestamp"] + shift).dt.date
    rain24 = df.groupby("_fwi_day")["_rain5"].sum().rename("rain_24h_noon")

    daily = pdwi.aggregate_5min_to_daily(df_sensor).set_index("date")
    out = daily.join(noon, how="left").join(rain24, how="left")
    out["rain_24h_noon"] = out["rain_24h_noon"].fillna(out["rain_mm"])
    # Hari tanpa data siang -> perkiraan dari rata-rata harian + offset asumsi
    out["t_noon"] = out["t_noon"].fillna(out["temp_mean"] + a.prior_t_noon_offset_c)
    out["rh_noon"] = out["rh_noon"].fillna((out["rh_mean"] - a.prior_rh_noon_offset_pct).clip(0, 100))
    out["wind_noon"] = out["wind_noon"].fillna(out["wind_mean"] * a.prior_wind_noon_factor)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Kalibrasi lokasi
# ---------------------------------------------------------------------------
@dataclass
class PeatFWICalibration:
    """Kalibrasi PDWI (H_t & model WTD) + konversi cuaca rata-rata -> siang hasil belajar data lokasi."""
    pdwi_cal: pdwi.SiteCalibration
    t_noon_offset_c: float = DEFAULT_FWI_ASSUMPTIONS.prior_t_noon_offset_c
    rh_noon_offset_pct: float = DEFAULT_FWI_ASSUMPTIONS.prior_rh_noon_offset_pct
    wind_noon_factor: float = DEFAULT_FWI_ASSUMPTIONS.prior_wind_noon_factor
    source: str = "prior_asumsi"
    n_days: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "sumber_konversi_siang": self.source,
            "jumlah_hari_data": self.n_days,
            "offset_suhu_siang_c": _r(self.t_noon_offset_c),
            "offset_rh_siang_persen": _r(self.rh_noon_offset_pct),
            "faktor_angin_siang": _r(self.wind_noon_factor, 3),
            "kalibrasi_pdwi": self.pdwi_cal.summary(),
            **self.extra,
        }


def calibrate_fwi_site(noon_daily: pd.DataFrame, pdwi_cal: pdwi.SiteCalibration,
                       a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS) -> PeatFWICalibration:
    """Belajar selisih cuaca siang vs rata-rata harian (median) untuk dipakai di skenario forecast."""
    n = len(noon_daily)
    cal = PeatFWICalibration(pdwi_cal=pdwi_cal, n_days=n)
    if n >= pdwi.DEFAULT_ASSUMPTIONS.min_calibration_days:
        cal.t_noon_offset_c = float((noon_daily["t_noon"] - noon_daily["temp_mean"]).median())
        cal.rh_noon_offset_pct = float((noon_daily["rh_mean"] - noon_daily["rh_noon"]).median())
        wm = noon_daily["wind_mean"].replace(0, np.nan)
        ratio = (noon_daily["wind_noon"] / wm).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            cal.wind_noon_factor = float(np.clip(ratio.median(), 1.0, 3.0))
        cal.source = "dataset_lokasi"
    return cal


def prior_fwi_calibration(pdwi_cal: Optional[pdwi.SiteCalibration] = None) -> PeatFWICalibration:
    return PeatFWICalibration(pdwi_cal=pdwi_cal or pdwi.prior_calibration())


# ---------------------------------------------------------------------------
# Deret harian (spin-up kode FWI di sepanjang riwayat)
# ---------------------------------------------------------------------------
def compute_pfwi_series(noon_daily: pd.DataFrame, cal: PeatFWICalibration,
                        a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS,
                        init: Optional[FWICodes] = None) -> pd.DataFrame:
    """Jalankan PFWI hari demi hari di seluruh riwayat (kode FWI dibawa ke hari berikutnya)."""
    df = noon_daily.copy()
    df["_ts"] = pd.to_datetime(df["date"])
    df = df.sort_values("_ts").reset_index(drop=True)
    codes = init or startup_codes(float(df["wtd_mean"].iloc[0]), a)
    rows = []
    for i, r in df.iterrows():
        wind = wind_10m_kmh(r["wind_noon"], a.wind_sensor_height_m)
        sys_ = step_fwi_system(codes, float(r["t_noon"]), float(r["rh_noon"]), wind,
                               float(r["rain_24h_noon"]), int(r["_ts"].month), a)
        codes = sys_["codes"]
        dry = pdwi.count_dry_days(df["rain_mm"].iloc[: i + 1].tolist())
        d7 = float(r["wtd_mean"] - df["wtd_mean"].iloc[i - 7]) if i >= 7 else None
        st = compute_pfwi_state(sys_, float(r["wtd_mean"]), cal.pdwi_cal, a, dry, d7)
        rows.append({
            "date": r["date"], "wtd_cm": float(r["wtd_mean"]), "t_noon": float(r["t_noon"]),
            "rh_noon": float(r["rh_noon"]), "wind_10m_kmh": wind, "rain_24h_mm": float(r["rain_24h_noon"]),
            "FFMC": sys_["FFMC"], "DMC": sys_["DMC"], "DC": sys_["DC"], "ISI": sys_["ISI"],
            "BUI": sys_["BUI"], "FWI": sys_["FWI"], "dry_days": dry, "dwtd7": d7,
            "pfwi_score": st["pfwi"]["score"], "pfwi_class": st["pfwi"]["class"],
            "pfwi_label": st["pfwi"]["label"], "eskalasi": st["pfwi"]["eskalasi_gambut"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# F12. Forecast
# ---------------------------------------------------------------------------
def noon_from_daily_weather(w: Dict[str, float], cal: PeatFWICalibration,
                            a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS) -> Dict[str, float]:
    """Cuaca harian (rata-rata) -> cuaca siang untuk input FWI."""
    return {
        "t_noon": w["temp_c"] + cal.t_noon_offset_c,
        "rh_noon": float(np.clip(w["rh_pct"] - cal.rh_noon_offset_pct, 5.0, 100.0)),
        "wind_kmh": wind_10m_kmh(w["wind_ms"] * cal.wind_noon_factor, a.wind_sensor_height_m),
        "rain_mm": w["rain_mm"],
    }


def forecast_pfwi(
    current_wtd_cm: float,
    current_codes: FWICodes,
    current_weather: Optional[Dict[str, Any]],
    scenarios: Sequence[Dict[str, Any]],
    cal: PeatFWICalibration,
    a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS,
    start_date: Optional[date] = None,
    recent_rain_mm: Optional[Sequence[float]] = None,
    wtd_history_cm: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Forecast PFWI N hari:
      WTD(t+1)  <- model neraca air PDWI (cuaca hari t)
      kode FWI  <- dilanjutkan dari ``current_codes`` dengan cuaca siang skenario
      PFWI      <- gabungan keduanya + aturan eskalasi
    """
    start = start_date or date.today()
    hydro = pdwi.forecast_pdwi(current_wtd_cm, current_weather, scenarios, cal.pdwi_cal,
                               start_date=start, recent_rain_mm=recent_rain_mm, wtd_history_cm=wtd_history_cm)
    codes = current_codes
    out = []
    for h_day, sc in zip(hydro, scenarios):
        w = pdwi._weather(sc, pdwi.DEFAULT_ASSUMPTIONS)
        nw = noon_from_daily_weather(w, cal, a)
        month = date.fromisoformat(h_day["tanggal"]).month
        sys_ = step_fwi_system(codes, nw["t_noon"], nw["rh_noon"], nw["wind_kmh"], nw["rain_mm"], month, a)
        codes = sys_["codes"]
        trend = h_day["tren_7_hari"]
        st = compute_pfwi_state(sys_, h_day["wtd_cm"], cal.pdwi_cal, a, h_day["hari_kering_beruntun"],
                                None if trend is None else trend["delta_wtd_7_hari_cm"])
        out.append({
            "hari": h_day["hari"], "hari_ke": h_day["hari_ke"], "tanggal": h_day["tanggal"],
            "wtd_cm": h_day["wtd_cm"], "tmat_cm": h_day["tmat_cm"],
            "cuaca_siang": {"suhu_c": _r(nw["t_noon"]), "rh_persen": _r(nw["rh_noon"]),
                            "angin_10m_kmh": _r(nw["wind_kmh"]), "hujan_24j_mm": _r(nw["rain_mm"])},
            "kode_fwi": codes.to_dict(),
            "pfwi_skor": st["pfwi"]["score"], "pfwi_kelas": st["pfwi"]["class"], "pfwi_label": st["pfwi"]["label"],
            "eskalasi_gambut": st["pfwi"]["eskalasi_gambut"],
            "subskor": st["subskor"],
            "fwi_kanonik": st["sistem_fwi_fdrs"]["FWI"]["nilai"],
            "tren_7_hari": trend,
            "hari_kering_beruntun": h_day["hari_kering_beruntun"],
            "status_kekeringan_pdwi": h_day["status_kekeringan"],
            "di_luar_rentang_kalibrasi": h_day["di_luar_rentang_kalibrasi"],
        })
    return out


def early_warning_summary(forecast: List[Dict[str, Any]], current_class: int, threshold_class: int = 2) -> Dict[str, Any]:
    """Kapan kelas >= TINGGI tercapai (0 = sudah hari ini), puncak skor PFWI."""
    already = current_class >= threshold_class
    first = next((f for f in forecast if f["pfwi_kelas"] >= threshold_class), None)
    peak = max(forecast, key=lambda f: f["pfwi_skor"]) if forecast else None
    return {
        "kelas_saat_ini": current_class,
        "sudah_bahaya_tinggi": already,
        "hari_menuju_bahaya_tinggi": 0 if already else (None if first is None else first["hari_ke"]),
        "tanggal_bahaya_tinggi_pertama": None if already or first is None else first["tanggal"],
        "puncak_skor_pfwi": None if peak is None else peak["pfwi_skor"],
        "puncak_pada": None if peak is None else peak["hari"],
        "ada_ekstrapolasi_di_luar_kalibrasi": any(f["di_luar_rentang_kalibrasi"] for f in forecast),
    }


# ---------------------------------------------------------------------------
# Titik masuk: satu pembacaan sensor
# ---------------------------------------------------------------------------
def calculate_from_sensor(
    tmat_cm: float,
    temp_c: float,
    rh_pct: float,
    cal: PeatFWICalibration,
    a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS,
    wind_ms: Optional[float] = None,
    rain_24h_mm: float = 0.0,
    prev_codes: Optional[FWICodes] = None,
    tmat_7d_ago_cm: Optional[float] = None,
    dry_days: Optional[int] = None,
    on_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    PFWI-Gambut dari satu pembacaan (idealnya pukul 12:00 WIB).
    ``prev_codes`` = FFMC/DMC/DC kemarin (dari respons hari sebelumnya). Bila kosong -> start-up.
    """
    notes: List[str] = []
    wtd, inundated = pdwi.wtd_from_tmat(tmat_cm)
    if inundated:
        notes.append("TMAT >= 0 (tergenang): WTD dibatasi 0 cm; gambut tidak tersedia sebagai bahan bakar.")
    if wind_ms is None:
        wind_ms = a.default_wind_ms
        notes.append(f"wind_ms tidak diberikan -> asumsi {a.default_wind_ms} m/s.")
    day = on_date or date.today()
    if prev_codes is None:
        prev_codes = startup_codes(wtd, a)
        notes.append("Kode FWI kemarin tidak diberikan -> nilai start-up "
                     f"(FFMC {a.startup_ffmc:g}, DMC {a.startup_dmc:g}, DC {prev_codes.dc:.1f}"
                     f"{' disetarakan dari WTD' if a.dc_init_from_wtd else ''}). "
                     "Kirim 'kode_fwi_hari_ini' dari respons ini sebagai prev_* esok hari agar memori kekeringan akurat.")

    wind_kmh = wind_10m_kmh(wind_ms, a.wind_sensor_height_m)
    sys_ = step_fwi_system(prev_codes, float(temp_c), float(rh_pct), wind_kmh, float(rain_24h_mm), day.month, a)

    d7 = None
    trend = None
    if tmat_7d_ago_cm is not None:
        trend = pdwi.wtd_trend(wtd, pdwi.wtd_from_tmat(tmat_7d_ago_cm)[0])
        d7 = trend["delta_wtd_7_hari_cm"]
    if dry_days is None:
        notes.append("dry_days tidak diberikan -> komponen S_dry dibuang & bobot dinormalisasi ulang.")

    st = compute_pfwi_state(sys_, wtd, cal.pdwi_cal, a, dry_days, d7)
    return {
        "engine": ENGINE_NAME,
        "tipe_lahan": "gambut",
        "tanggal": day.isoformat(),
        "input": {"tmat_cm": _r(tmat_cm), "suhu_siang_c": _r(temp_c), "rh_siang_persen": _r(rh_pct),
                  "angin_ms": _r(wind_ms), "hujan_24j_mm": _r(rain_24h_mm),
                  "kode_kemarin": prev_codes.to_dict(), "tmat_7_hari_lalu_cm": _r(tmat_7d_ago_cm),
                  "hari_kering_beruntun": dry_days},
        "turunan": {"wtd_cm": _r(wtd), "tergenang": inundated, "angin_10m_kmh": _r(wind_kmh)},
        **st,
        "kode_fwi_hari_ini": sys_["codes"].to_dict(),
        "tren_7_hari": trend,
        "asumsi_diterapkan": notes,
    }


# ---------------------------------------------------------------------------
# Analisis dataset lokasi
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Any] = {"key": None}


def load_site_fwi(path: Optional[os.PathLike] = None) -> Tuple[pd.DataFrame, pd.DataFrame, PeatFWICalibration]:
    """Muat dataset -> (noon_daily, deret PFWI, kalibrasi). Di-cache berdasarkan mtime berkas."""
    from src.data_loader import load_aws_dataset

    p = os.fspath(path or DEFAULT_RAW_DATA_PATH)
    key = (p, os.path.getmtime(p))
    if _CACHE["key"] != key:
        _daily, pdwi_cal = pdwi.load_site_dataset(p)
        noon = aggregate_noon_daily(load_aws_dataset(p))
        cal = calibrate_fwi_site(noon, pdwi_cal)
        series = compute_pfwi_series(noon, cal)
        _CACHE.update(key=key, noon=noon, series=series, cal=cal)
    return _CACHE["noon"], _CACHE["series"], _CACHE["cal"]


def _series_row(r: pd.Series) -> Dict[str, Any]:
    return {
        "tanggal": str(r["date"]), "wtd_cm": _r(r["wtd_cm"]), "hujan_24j_mm": _r(r["rain_24h_mm"]),
        "kode_fwi": {"ffmc": _r(r["FFMC"]), "dmc": _r(r["DMC"]), "dc": _r(r["DC"])},
        "isi": _r(r["ISI"]), "bui": _r(r["BUI"]), "fwi_kanonik": _r(r["FWI"]),
        "pfwi_skor": _r(r["pfwi_score"]), "pfwi_label": r["pfwi_label"], "eskalasi_gambut": r["eskalasi"],
    }


def analyze_latest(noon: pd.DataFrame, series: pd.DataFrame, cal: PeatFWICalibration,
                   a: PeatFWIAssumptions = DEFAULT_FWI_ASSUMPTIONS, horizon: int = 7,
                   history_days: int = 7) -> Dict[str, Any]:
    """Keadaan terakhir + riwayat + forecast (persistence/kering/basah) + peringatan dini."""
    nd = noon.copy()
    nd["_ts"] = pd.to_datetime(nd["date"])
    nd = nd.sort_values("_ts").reset_index(drop=True)
    last, last_s = nd.iloc[-1], series.iloc[-1]
    last_date = pd.Timestamp(last["date"]).date()
    codes = FWICodes(float(last_s["FFMC"]), float(last_s["DMC"]), float(last_s["DC"]))

    # Keadaan saat ini dihitung ulang lengkap (subskor & kelas FDRS) dari kode kemarin
    prev = series.iloc[-2] if len(series) > 1 else None
    prev_codes = startup_codes(float(last["wtd_mean"]), a) if prev is None else FWICodes(prev["FFMC"], prev["DMC"], prev["DC"])
    sys_now = step_fwi_system(prev_codes, float(last["t_noon"]), float(last["rh_noon"]),
                              wind_10m_kmh(float(last["wind_noon"]), a.wind_sensor_height_m),
                              float(last["rain_24h_noon"]), last_date.month, a)
    d7 = None if pd.isna(last_s["dwtd7"]) else float(last_s["dwtd7"])
    state = compute_pfwi_state(sys_now, float(last["wtd_mean"]), cal.pdwi_cal, a, int(last_s["dry_days"]), d7)

    tail = nd.tail(7)
    today = {"rain_mm": float(last["rain_mm"]), "temp_c": float(last["temp_mean"]),
             "rh_pct": float(last["rh_mean"]), "solar_wm2": float(last["solar_mean"]), "wind_ms": float(last["wind_mean"])}
    mean7 = {"rain_mm": float(tail["rain_mm"].mean()), "temp_c": float(tail["temp_mean"].mean()),
             "rh_pct": float(tail["rh_mean"].mean()), "solar_wm2": float(tail["solar_mean"].mean()),
             "wind_ms": float(tail["wind_mean"].mean())}
    rain_hist = nd["rain_mm"].tail(7).astype(float).tolist()
    wtd_hist = nd["wtd_mean"].tail(8).astype(float).tolist()

    def run(mode: str) -> List[Dict[str, Any]]:
        scen = pdwi.build_scenarios(mode, mean7, horizon, cal.pdwi_cal)
        return forecast_pfwi(float(last["wtd_mean"]), codes, today, scen, cal, a, last_date, rain_hist, wtd_hist)

    main_fc = run("persistence")
    compact = lambda f: {"hari": f["hari"], "tanggal": f["tanggal"], "wtd_cm": f["wtd_cm"],  # noqa: E731
                         "skor": f["pfwi_skor"], "label": f["pfwi_label"]}
    return {
        "engine": ENGINE_NAME,
        "tanggal_data_terakhir": last_date.isoformat(),
        "keadaan_saat_ini": {
            "wtd_cm": _r(last["wtd_mean"]), "tmat_cm": _r(-last["wtd_mean"]),
            "cuaca_siang": {"suhu_c": _r(last["t_noon"]), "rh_persen": _r(last["rh_noon"]),
                            "angin_ms": _r(last["wind_noon"]), "hujan_24j_mm": _r(last["rain_24h_noon"])},
            "hari_kering_beruntun": int(last_s["dry_days"]),
            "delta_wtd_7_hari_cm": _r(d7),
            **state,
            "kode_fwi_hari_ini": codes.to_dict(),
        },
        "riwayat": [_series_row(r) for _, r in series.tail(history_days).iterrows()],
        "forecast": main_fc,
        "skenario_pembanding": {"kering": [compact(f) for f in run("kering")],
                                "basah": [compact(f) for f in run("basah")]},
        "peringatan_dini": early_warning_summary(main_fc, state["pfwi"]["class"]),
        "statistik_dataset": {
            "jumlah_hari": len(series),
            "distribusi_kelas_pfwi": {FIRE_RISK_LABELS[k]: int(v) for k, v in
                                      series["pfwi_class"].value_counts().sort_index().items()},
        },
    }
