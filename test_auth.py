import sys
from fastapi.testclient import TestClient
from main import app
from src.auth.dependencies import get_current_user
from src.auth.schemas import UserInfo

client = TestClient(app)

def test_unauthenticated_access():
    print("\n--- 1. Testing Unauthenticated Access ---")
    resp_user = client.get("/api/v1/auth/user-access")
    print(f"GET /api/v1/auth/user-access -> Status: {resp_user.status_code}")
    assert resp_user.status_code == 401, "Should be 401 Unauthorized"

    resp_admin = client.get("/api/v1/auth/superadmin-access")
    print(f"GET /api/v1/auth/superadmin-access -> Status: {resp_admin.status_code}")
    assert resp_admin.status_code == 401, "Should be 401 Unauthorized"

    resp_admin_dash = client.get("/api/v1/superadmin/dashboard")
    print(f"GET /api/v1/superadmin/dashboard -> Status: {resp_admin_dash.status_code}")
    assert resp_admin_dash.status_code == 401, "Should be 401 Unauthorized"
    print("✅ Unauthenticated protection works as expected!")


def test_end_to_end_register_and_login_flow():
    print("\n--- 2. Testing Direct Register & Login (TANPA CONFIRM EMAIL) ---")
    
    # 1. Register Regular User
    user_email = "andi_operator@perusahaan.com"
    reg_user_resp = client.post("/api/v1/auth/register", json={
        "email": user_email,
        "password": "PasswordUser123!",
        "role": "user",
        "full_name": "Andi Operator"
    })
    print(f"POST /api/v1/auth/register (User) -> Status: {reg_user_resp.status_code}")
    print("Response:", reg_user_resp.json())
    assert reg_user_resp.status_code in (200, 201, 400) # 400 if already created in rerun

    # 2. Login as Regular User (Langsung bisa login seketika!)
    login_user_resp = client.post("/api/v1/auth/login", json={
        "email": user_email,
        "password": "PasswordUser123!"
    })
    print(f"POST /api/v1/auth/login (User) -> Status: {login_user_resp.status_code}")
    assert login_user_resp.status_code == 200, f"Login failed: {login_user_resp.text}"
    user_token_data = login_user_resp.json()
    user_jwt = user_token_data["access_token"]
    user_role = user_token_data["user"]["role"]
    print(f"✅ Login Berhasil! Role didapat: '{user_role}', Token: {user_jwt[:25]}...")
    assert user_role == "user"

    # 3. Register Superadmin
    admin_email = "bos_superadmin@perusahaan.com"
    reg_admin_resp = client.post("/api/v1/auth/register", json={
        "email": admin_email,
        "password": "PasswordAdmin999!",
        "role": "superadmin",
        "full_name": "Budi Superadmin"
    })
    print(f"POST /api/v1/auth/register (Superadmin) -> Status: {reg_admin_resp.status_code}")
    assert reg_admin_resp.status_code in (200, 201, 400)

    # 4. Login as Superadmin
    login_admin_resp = client.post("/api/v1/auth/login", json={
        "email": admin_email,
        "password": "PasswordAdmin999!"
    })
    print(f"POST /api/v1/auth/login (Superadmin) -> Status: {login_admin_resp.status_code}")
    assert login_admin_resp.status_code == 200
    admin_token_data = login_admin_resp.json()
    admin_jwt = admin_token_data["access_token"]
    admin_role = admin_token_data["user"]["role"]
    print(f"✅ Superadmin Login Berhasil! Role: '{admin_role}'")
    assert admin_role == "superadmin"

    print("\n--- 3. Testing Real JWT Token Bearer RBAC Enforcement ---")
    headers_user = {"Authorization": f"Bearer {user_jwt}"}
    headers_admin = {"Authorization": f"Bearer {admin_jwt}"}

    # User access user route -> 200 OK
    res1 = client.get("/api/v1/auth/user-access", headers=headers_user)
    print(f"User token -> /api/v1/auth/user-access: {res1.status_code}")
    assert res1.status_code == 200

    # User access user dashboard -> 200 OK
    res_dash = client.get("/api/v1/user/dashboard", headers=headers_user)
    print(f"User token -> /api/v1/user/dashboard: {res_dash.status_code}")
    assert res_dash.status_code == 200

    # User access superadmin route -> 403 FORBIDDEN!
    res2 = client.get("/api/v1/auth/superadmin-access", headers=headers_user)
    print(f"User token -> /api/v1/auth/superadmin-access: {res2.status_code}")
    assert res2.status_code == 403, "User must NOT access superadmin endpoint!"
    print("Detail penolakan:", res2.json()["detail"])

    # User access superadmin dashboard -> 403 FORBIDDEN!
    res_admin_dash = client.get("/api/v1/superadmin/dashboard", headers=headers_user)
    print(f"User token -> /api/v1/superadmin/dashboard: {res_admin_dash.status_code}")
    assert res_admin_dash.status_code == 403

    # Superadmin access user route -> 200 OK
    res3 = client.get("/api/v1/auth/user-access", headers=headers_admin)
    print(f"Superadmin token -> /api/v1/auth/user-access: {res3.status_code}")
    assert res3.status_code == 200

    # Superadmin access superadmin route -> 200 OK
    res4 = client.get("/api/v1/auth/superadmin-access", headers=headers_admin)
    print(f"Superadmin token -> /api/v1/auth/superadmin-access: {res4.status_code}")
    assert res4.status_code == 200

    # Superadmin access superadmin dashboard -> 200 OK
    res5 = client.get("/api/v1/superadmin/dashboard", headers=headers_admin)
    print(f"Superadmin token -> /api/v1/superadmin/dashboard: {res5.status_code}")
    assert res5.status_code == 200

    # 5. Check GET /api/v1/auth/me
    res_me = client.get("/api/v1/auth/me", headers=headers_admin)
    print(f"Superadmin token -> /api/v1/auth/me: {res_me.status_code}")
    assert res_me.status_code == 200
    print("User profile /me:", res_me.json())

    # 6. Logout
    res_logout = client.post("/api/v1/auth/logout", headers=headers_admin)
    print(f"POST /api/v1/auth/logout: {res_logout.status_code}")
    assert res_logout.status_code == 200
    print("Logout response:", res_logout.json())

    print("✅ Direct Database User Login & Role Authorization works 100% perfectly without any email confirmation!")


def test_invalid_login():
    print("\n--- 4. Testing Invalid Password Rejection ---")
    resp = client.post("/api/v1/auth/login", json={
        "email": "andi_operator@perusahaan.com",
        "password": "WrongPassword999!"
    })
    print(f"POST /api/v1/auth/login (password salah) -> Status: {resp.status_code}")
    assert resp.status_code == 401
    print("Response detail:", resp.json())
    print("✅ Invalid password correctly rejected with 401!")


def test_superadmin_exclusive_user_management():
    print("\n--- 5. Testing Exclusive Superadmin Access to View All Users ---")
    from src.auth.supabase_service import create_access_token
    user_tok = create_access_token("u1", "user@test.com", "user")
    admin_tok = create_access_token("a1", "admin@test.com", "superadmin")

    # 1. User biasa mencoba melihat daftar users -> HARUS DITOLAK 403!
    r_user = client.get("/api/v1/superadmin/users", headers={"Authorization": f"Bearer {user_tok}"})
    print(f"User mencoba /api/v1/superadmin/users -> Status: {r_user.status_code}")
    assert r_user.status_code == 403, "Regular user MUST NOT view all users!"

    # 2. Superadmin melihat daftar users -> HARUS SUKSES 200!
    r_admin = client.get("/api/v1/superadmin/users", headers={"Authorization": f"Bearer {admin_tok}"})
    print(f"Superadmin membuka /api/v1/superadmin/users -> Status: {r_admin.status_code}")
    assert r_admin.status_code == 200
    user_list = r_admin.json()["users"]
    print(f"✅ Superadmin berhasil melihat {len(user_list)} data user!")
    assert len(user_list) > 0
    assert "password_hash" not in user_list[0], "password_hash must be excluded!"


def test_peatland_restriction():
    print("\n--- 6. Testing is_peatland Endpoint Restrictions ---")
    from src.auth.supabase_service import create_access_token
    # User non-gambut (is_peatland=False)
    tok_non_peatland = create_access_token("u_mineral", "mineral@test.com", "user", is_peatland=False)
    # User lahan gambut (is_peatland=True)
    tok_peatland = create_access_token("u_peat", "peat@test.com", "user", is_peatland=True)

    fwi_payload = {
        "air_temperature_c": 31.0,
        "relative_humidity_pct": 60.0,
        "solar_radiation_wm2": 500.0,
        "rainfall_intensity_mmh": 0.0
    }

    # 1. User non-gambut mengakses /api/v1/fwi/calculate -> HARUS 200 OK
    resp_ok = client.post(
        "/api/v1/fwi/calculate",
        json=fwi_payload,
        headers={"Authorization": f"Bearer {tok_non_peatland}"}
    )
    print(f"User is_peatland=False -> POST /api/v1/fwi/calculate: Status {resp_ok.status_code}")
    assert resp_ok.status_code == 200, f"Expected 200, got {resp_ok.text}"

    # 2. User lahan gambut mengakses /api/v1/fwi/calculate -> HARUS DITOLAK 403 Forbidden!
    resp_blocked = client.post(
        "/api/v1/fwi/calculate",
        json=fwi_payload,
        headers={"Authorization": f"Bearer {tok_peatland}"}
    )
    print(f"User is_peatland=True -> POST /api/v1/fwi/calculate: Status {resp_blocked.status_code}")
    assert resp_blocked.status_code == 403, f"Expected 403 Forbidden, got {resp_blocked.status_code}"
    print("Pesan penolakan:", resp_blocked.json()["detail"])
    assert "is_peatland=True" in resp_blocked.json()["detail"]

    # 3. User lahan gambut mengakses /api/v1/predict -> HARUS DITOLAK 403 Forbidden!
    resp_pred_blocked = client.post(
        "/api/v1/predict",
        json={
            "air_temperature_c": 31.0,
            "relative_humidity_pct": 60.0,
            "solar_radiation_wm2": 500.0,
            "rainfall_intensity_mmh": 0.0,
            "water_table_depth_cm": -45.0,
            "soil_moisture_pct": 55.0
        },
        headers={"Authorization": f"Bearer {tok_peatland}"}
    )
    print(f"User is_peatland=True -> POST /api/v1/predict: Status {resp_pred_blocked.status_code}")
    assert resp_pred_blocked.status_code == 403
    print("✅ is_peatland=False/True guard bekerja 100% sempurna!")


if __name__ == "__main__":
    try:
        test_unauthenticated_access()
        test_end_to_end_register_and_login_flow()
        test_invalid_login()
        test_superadmin_exclusive_user_management()
        test_peatland_restriction()
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! SIAP DIGUNAKAN!")
    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected test error: {e}")
        sys.exit(1)
