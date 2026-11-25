from fastapi.testclient import TestClient

from main import app
from src import peatland_pdwi as pdwi
from src.auth.supabase_service import create_access_token

client = TestClient(app)

SENSOR = {"tmat_cm": -159.37, "temp_c": 27.0, "rh_pct": 80.0, "solar_wm2": 150.0}


def _h(is_peatland, role="user"):
    tok = create_access_token("u_test", "t@test.com", role, is_peatland=is_peatland)
    return {"Authorization": f"Bearer {tok}"}


def test_engine_unit():
    _, cal = pdwi.load_site_dataset()
    assert cal.beta1 < 0 and cal.beta3 <= 0
    r = pdwi.calculate_from_sensor(-159.37, 27, 80, 150, cal)
    assert r["tipe_lahan"] == "gambut"
    assert r["level_1"]["label"] in ("HIGH", "SEVERE")
    assert "RD" in r["level_1"]["komponen_tidak_tersedia"] or "RD" in str(r["level_2"])
    r2 = pdwi.calculate_from_sensor(-159.37, 27, 80, 150, cal, rain_7d_mm=5.0)
    assert r2["level_2"]["komponen_tidak_tersedia"] == []


def test_api_access_control():
    url = "/api/v1/peatland/drought/calculate"
    assert client.post(url, json=SENSOR).status_code in (401, 403)
    assert client.post(url, json=SENSOR, headers=_h(False)).status_code == 403
    r = client.post(url, json=SENSOR, headers=_h(True))
    assert r.status_code == 200, r.text
    assert r.json()["level_1"]["score"] > 0
    assert client.post(url, json=SENSOR, headers=_h(False, "superadmin")).status_code == 200


def test_api_other_routes():
    h = _h(True)
    assert client.get("/api/v1/peatland/drought/latest", headers=h).status_code == 200
    assert client.get("/api/v1/peatland/drought/assumptions", headers=h).status_code == 200
    for mode in ("persistence", "kering", "basah"):
        r = client.post("/api/v1/peatland/drought/forecast", headers=h,
                        json={"current_tmat_cm": -159.37, "mode": mode})
        assert r.status_code == 200, r.text
        assert len(r.json()["forecast"]) == 7
    assert client.post("/api/v1/peatland/drought/forecast", headers=h,
                       json={"current_tmat_cm": -100, "mode": "x"}).status_code == 422
    assert client.post("/api/v1/fwi/calculate", headers=h, json={
        "air_temperature_c": 31.0, "relative_humidity_pct": 60.0,
        "solar_radiation_wm2": 500.0, "rainfall_intensity_mmh": 0.0}).status_code == 403
