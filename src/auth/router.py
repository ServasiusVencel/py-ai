from typing import Dict, Any
from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.auth.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    UserInfo,
    LogoutResponse,
    GenericResponse,
    UserUpdateRequest,
)
from src.auth.supabase_service import (
    register_user,
    login_user,
    logout_user,
    get_all_users,
    update_user_by_id,
    delete_user_by_id,
)
from src.auth.dependencies import (
    oauth2_scheme,
    get_current_user,
    require_user,
    require_superadmin,
)

auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & RBAC"])


@auth_router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Registrasi Akun Baru",
    description="Mendaftarkan akun baru ke Supabase Auth dengan pilihan role: 'user' atau 'superadmin'."
)
def register(payload: UserRegisterRequest) -> Dict[str, Any]:
    """
    Endpoint registrasi user ke Supabase Auth.
    Menyimpan role ('user' atau 'superadmin') dan status tipe lahan (is_peatland).
    """
    return register_user(
        email=str(payload.email),
        password=payload.password,
        role=payload.role,
        full_name=payload.full_name,
        is_peatland=payload.is_peatland
    )


@auth_router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login Pengguna",
    description="Login ke Supabase menggunakan email dan password. Mengembalikan Bearer Token dan Role pengguna."
)
def login(payload: UserLoginRequest):
    """
    Endpoint login pengguna.
    Memverifikasi kredensial ke Supabase dan menerbitkan access token JWT.
    """
    return login_user(email=str(payload.email), password=payload.password)


@auth_router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout Pengguna",
    description="Mencabut sesi autentikasi pengguna di Supabase."
)
async def logout(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)):
    """
    Endpoint logout untuk mengakhiri sesi token di Supabase.
    """
    token = credentials.credentials
    success = await logout_user(token)
    return LogoutResponse(
        success=success,
        message="Logout berhasil. Sesi telah diakhiri di Supabase."
    )


@auth_router.get(
    "/me",
    response_model=UserInfo,
    summary="Profil Pengguna Aktif",
    description="Mengambil data akun dan role pengguna yang sedang login berdasarkan Bearer token."
)
def get_profile(current_user: UserInfo = Depends(get_current_user)):
    """
    Mendapatkan identitas dan role pengguna saat ini.
    """
    return current_user


# ---------------------------------------------------------------------------
# Contoh Endpoint Terproteksi Berdasarkan Role (RBAC)
# ---------------------------------------------------------------------------

@auth_router.get(
    "/user-access",
    response_model=GenericResponse,
    summary="Endpoint Khusus Role 'user' & 'superadmin'",
    description="Rute ini dapat diakses oleh akun dengan role 'user' maupun 'superadmin'."
)
def protected_user_route(current_user: UserInfo = Depends(require_user)):
    return GenericResponse(
        success=True,
        message=f"Halo {current_user.full_name or current_user.email}! Anda berhasil mengakses area Pengguna.",
        data={
            "user_id": current_user.id,
            "email": current_user.email,
            "role": current_user.role,
            "allowed_features": [
                "read_aws_telemetry",
                "request_weather_forecast",
                "view_fire_risk_alerts"
            ]
        }
    )


@auth_router.get(
    "/superadmin-access",
    response_model=GenericResponse,
    summary="Endpoint Eksklusif Role 'superadmin'",
    description="Rute ini HANYA bisa diakses oleh akun dengan role 'superadmin'. Akun dengan role 'user' akan ditolak (403 Forbidden)."
)
def protected_superadmin_route(current_user: UserInfo = Depends(require_superadmin)):
    return GenericResponse(
        success=True,
        message=f"Selamat datang Superadmin {current_user.full_name or current_user.email}! Anda memiliki hak akses penuh.",
        data={
            "user_id": current_user.id,
            "email": current_user.email,
            "role": current_user.role,
            "superadmin_features": [
                "trigger_model_retraining",
                "manage_station_coordinates",
                "manage_user_roles",
                "view_system_audit_logs"
            ]
        }
    )


@auth_router.get(
    "/users",
    summary="Lihat Seluruh Data Pengguna (Hanya Superadmin)",
    description="Endpoint khusus untuk Superadmin guna melihat seluruh data user terdaftar di Supabase database. Role 'user' biasa akan ditolak (403 Forbidden)."
)
def get_all_users_endpoint(current_user: UserInfo = Depends(require_superadmin)):
    """
    Mengambil semua data pengguna yang tersimpan di tabel database Supabase.
    Hanya dapat diakses oleh akun dengan role 'superadmin'.
    """
    users = get_all_users()
    return {
        "success": True,
        "message": f"Berhasil mengambil seluruh data pengguna ({len(users)} pengguna ditemukan).",
        "authorized_admin": {
            "id": current_user.id,
            "email": current_user.email,
            "role": current_user.role
        },
        "total_users": len(users),
        "data": users
    }


@auth_router.put(
    "/users/{user_id}",
    summary="Edit / Update Data Pengguna (Hanya Superadmin)",
    description="Superadmin dapat memperbarui nama lengkap, role ('user'/'superadmin'), email, atau mereset password pengguna."
)
def update_user_endpoint(
    user_id: str,
    payload: UserUpdateRequest,
    current_user: UserInfo = Depends(require_superadmin)
):
    """
    Mengubah data akun pengguna di Supabase.
    Eksklusif untuk role 'superadmin'.
    """
    update_dict = payload.model_dump(exclude_unset=True)
    updated = update_user_by_id(user_id, update_dict)
    return {
        "success": True,
        "message": f"Data pengguna '{updated.get('email')}' (ID: {user_id}) berhasil diperbarui.",
        "updated_by": current_user.email,
        "data": updated
    }


@auth_router.delete(
    "/users/{user_id}",
    summary="Hapus Pengguna (Hanya Superadmin)",
    description="Superadmin dapat menghapus akun pengguna secara permanen dari database Supabase."
)
def delete_user_endpoint(
    user_id: str,
    current_user: UserInfo = Depends(require_superadmin)
):
    """
    Menghapus akun pengguna dari database Supabase.
    Eksklusif untuk role 'superadmin'.
    """
    if str(current_user.id) == str(user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tindakan Ditolak: Anda tidak dapat menghapus akun Superadmin Anda sendiri demi keamanan sistem."
        )

    delete_user_by_id(user_id)
    return {
        "success": True,
        "message": f"Pengguna dengan ID '{user_id}' berhasil dihapus dari database Supabase.",
        "deleted_by": current_user.email
    }

