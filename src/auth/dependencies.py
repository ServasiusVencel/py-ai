from typing import List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.auth.schemas import UserInfo
from src.auth.supabase_service import get_user_from_token

# HTTP Bearer scheme with OpenAPI integration
oauth2_scheme = HTTPBearer(
    auto_error=True,
    description="Masukkan access token Supabase JWT (format: Bearer <token>)"
)

# Optional scheme for endpoints where login is not strictly required
oauth2_optional_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)
) -> UserInfo:
    """
    Dependency untuk memvalidasi token dan mengambil data user aktif dari Supabase.
    Jika token tidak valid, akan mengembalikan HTTP 401 Unauthorized.
    """
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token autentikasi tidak ditemukan.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return get_user_from_token(token)


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_optional_scheme)
) -> Optional[UserInfo]:
    """
    Dependency opsional: jika token ada dan valid, kembalikan UserInfo, jika tidak ada kembalikan None.
    """
    if not credentials or not credentials.credentials:
        return None
    try:
        return get_user_from_token(credentials.credentials)
    except HTTPException:
        return None


def require_roles(allowed_roles: List[str]):
    """
    Role-Based Access Control (RBAC) dependency factory.
    Memeriksa apakah role pengguna saat ini terdaftar di dalam allowed_roles.
    Contoh:
        @app.get('/admin', dependencies=[Depends(require_roles(['superadmin']))])
    """
    allowed_set = {r.lower().strip() for r in allowed_roles}

    def role_checker(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
        user_role = current_user.role.lower().strip()
        if user_role not in allowed_set:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Akses Ditolak: Anda tidak memiliki wewenang untuk mengakses endpoint ini. "
                    f"Dibutuhkan salah satu dari role: {list(allowed_set)}, sedangkan role akun Anda: '{current_user.role}'."
                )
            )
        return current_user

    return role_checker


# Shorthand dependencies siap pakai
require_user = require_roles(["user", "superadmin"])
require_superadmin = require_roles(["superadmin"])


def require_peatland(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """
    Dependency guard untuk endpoint algoritma PDWI-Gambut (/api/v1/peatland/...).
    Hanya akun bertipe lahan gambut (is_peatland=True) yang diizinkan.
    Role 'superadmin' juga diizinkan untuk keperluan operasional dan pengujian.
    """
    if current_user.is_peatland or current_user.role.lower().strip() == "superadmin":
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Akses Ditolak: Algoritma PDWI-Gambut hanya tersedia untuk akun dengan tipe lahan gambut "
            "(is_peatland=True). Akun non-gambut (is_peatland=False) menggunakan algoritma FWI/Drought/Forecast reguler."
        )
    )


def require_non_peatland(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """
    Dependency guard untuk endpoint algoritma (FWI, Drought, Forecast, Predict).
    Memblokir akses jika user memiliki status lahan gambut (is_peatland=True).
    Algoritma hanya tersedia untuk lahan non-gambut (is_peatland=False).
    """
    if current_user.is_peatland:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Akses Ditolak: Algoritma FWI, Drought, dan Forecast tidak tersedia untuk akun dengan "
                "tipe lahan gambut (is_peatland=True). Fitur ini hanya diperuntukkan bagi lahan non-gambut."
            )
        )
    return current_user
