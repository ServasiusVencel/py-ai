from fastapi.testclient import TestClient

from main import app
from src import peatland_fwi as pf
from src.auth.supabase_service import create_access_token

client = TestClient(app)

SENSOR = {"tmat_cm": -95.0, "temp_c": 32.5, "rh_pct": 55.0, "wind_ms": 2.5, "rain_24h_mm": 0.0,
          "prev_ffmc": 86.0, "prev_dmc": 25.0, "prev_dc": 380.0, "tmat_7d_ago_cm": -70.0, "dry_days": 9}


def _h(is_peatland, role="user"):
    tok = create_access_token("u_test", "t@test.com", role, is_peatland=is_peatland)
    return {"Authorization": f"Bearer {tok}"}


def test_van_wagner_reference():
    """Kasus uji standar Van Wagner & Pickett (1985), lintang 46°N, April."""
    a = pf.PeatFWIAssumptions(latitude_deg=46.0)
    s = pf.step_fwi_system(pf.FWICodes(85, 6, 15), 17, 42, 25, 0, 4, a)
    expected = {"FFMC": 87.7, "DMC": 8.5, "DC": 19.0, "ISI": 10.9, "BUI": 8.5, "FWI": 10.1}
    for k, v in expected.items():
        assert round(s[k], 1) == v, (k, s[k])


def test_engine_wet_vs_dry():
    _, _, cal = pf.load_site_fwi()
    wet = pf.calculate_from_sensor(5.0, 24.0, 95.0, cal, wind_ms=0.5, rain_24h_mm=30.0, dry_days=0)
    dry = pf.calculate_from_sensor(-95.0, 32.5, 55.0, cal, wind_ms=2.5, rain_24h_mm=0.0,
                                   prev_codes=pf.FWICodes(86, 25, 380), tmat_7d_ago_cm=-70.0, dry_days=9)
    assert wet["pfwi"]["class"] == 0
    assert dry["pfwi"]["class"] == 3
    assert dry["pfwi"]["score"] > wet["pfwi"]["score"]
    assert set(dry["kode_fwi_hari_ini"]) == {"ffmc", "dmc", "dc"}


def test_peat_escalation_rules():
    cal = pf.prior_fwi_calibration()
    sys_ = {"FFMC": 84.0, "DMC": 5.0, "DC": 20.0, "ISI": 1.0, "BUI": 5.0, "FWI": 1.0, "DSR": 0.0}
    st = pf.compute_pfwi_state(sys_, 45.0, cal.pdwi_cal)                 # R1
    assert st["pfwi"]["kelas_sebelum_eskalasi"] < 2 and st["pfwi"]["class"] == 2
    sys_.update(DC=360.0, FFMC=79.0)
    st = pf.compute_pfwi_state(sys_, 90.0, cal.pdwi_cal)                 # R2
    assert st["pfwi"]["class"] == 3 and any(r.startswith("R2") for r in st["pfwi"]["eskalasi_gambut"])


def test_series_and_latest():
    noon, series, cal = pf.load_site_fwi()
    assert len(series) == len(noon) > 0
    assert series["pfwi_score"].between(0, 100).all()
    r = pf.analyze_latest(noon, series, cal)
    assert len(r["forecast"]) == 7
    dry = r["skenario_pembanding"]["kering"][-1]["skor"]
    wet = r["skenario_pembanding"]["basah"][-1]["skor"]
    assert dry > wet


def test_api_access_control():
    url = "/api/v1/peatland/fwi/calculate"
    assert client.post(url, json=SENSOR).status_code in (401, 403)
    assert client.post(url, json=SENSOR, headers=_h(False)).status_code == 403
    r = client.post(url, json=SENSOR, headers=_h(True))
    assert r.status_code == 200, r.text
    assert r.json()["pfwi"]["label"].startswith("Ekstrim")
    assert client.post(url, json=SENSOR, headers=_h(False, "superadmin")).status_code == 200


def test_api_other_routes():
    h = _h(True)
    assert client.get("/api/v1/peatland/fwi/latest", headers=h).status_code == 200
    assert client.get("/api/v1/peatland/fwi/assumptions", headers=h).status_code == 200
    for mode in ("persistence", "kering", "basah"):
        r = client.post("/api/v1/peatland/fwi/forecast", headers=h,
                        json={"current_tmat_cm": -120, "mode": mode, "prev_dc": 300})
        assert r.status_code == 200, r.text
        assert len(r.json()["forecast"]) == 7
    assert client.post("/api/v1/peatland/fwi/forecast", headers=h,
                       json={"current_tmat_cm": -100, "mode": "x"}).status_code == 422
    assert client.get("/api/v1/peatland/fwi/latest", headers=_h(False)).status_code == 403
