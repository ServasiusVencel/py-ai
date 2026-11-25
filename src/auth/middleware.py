import logging
from typing import Dict, List, Optional, Set
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from fastapi import HTTPException

from src.auth.supabase_service import get_user_from_token
from src.auth.schemas import UserInfo

logger = logging.getLogger("Auth-Middleware")


class RoleAuthMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware komprehensif untuk melindungi seluruh akses API dengan Bearer JWT
    dan menegakkan Role-Based Access Control (RBAC).

    Fitur:
    1. Melindungi seluruh rute di bawah `api_prefix` (default: `/api/v1`).
    2. Melewati (whitelist) rute publik seperti login, register, docs, dan health check.
    3. Jika request tidak menyertakan Bearer token -> Mengembalikan HTTP 401 Unauthorized.
    4. Jika rute memiliki batasan role tertentu (misal: hanya 'superadmin') -> Memvalidasi role pengguna,
       dan jika tidak sesuai mengembalikan HTTP 403 Forbidden.
    5. Menyimpan profil pengguna yang terverifikasi ke dalam `request.state.user`.
    """

    def __init__(
        self,
        app,
        protect_all_api: bool = True,
        api_prefix: str = "/api/v1",
        public_routes: Optional[List[str]] = None,
        role_protected_routes: Optional[Dict[str, List[str]]] = None
    ):
        super().__init__(app)
        self.protect_all_api = protect_all_api
        self.api_prefix = api_prefix

        # Rute publik yang bebas diakses tanpa Bearer token
        default_public = [
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
            "/openapi.json",
            "/health",
            "/",
            "/favicon.ico"
        ]
        self.public_routes: Set[str] = set(public_routes or default_public)

        # Rute dengan batasan role spesifik
        self.role_protected_routes: Dict[str, List[str]] = role_protected_routes or {
            "/api/v1/superadmin": ["superadmin"],
            "/api/v1/retrain": ["superadmin"],
        }

    def _is_public_path(self, path: str) -> bool:
        """Mengecek apakah path termasuk rute publik."""
        if path in self.public_routes:
            return True
        for pub in self.public_routes:
            if pub != "/" and path.startswith(pub):
                return True
        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        current_path = request.url.path
        request.state.user = None

        # 0. Izinkan preflight request CORS (OPTIONS) lewat tanpa autentikasi
        if request.method == "OPTIONS":
            return await call_next(request)

        # 1. Jika rute publik, izinkan lewat tanpa token
        if self._is_public_path(current_path):
            return await call_next(request)

        # 2. Cek apakah rute ini termasuk dalam API yang diproteksi
        is_api_protected = self.protect_all_api and current_path.startswith(self.api_prefix)

        auth_header = request.headers.get("Authorization")
        user_info: Optional[UserInfo] = None
        token: Optional[str] = None

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()

        # 3. Jika rute diproteksi dan token tidak ada
        if is_api_protected and not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        f"Akses Ditolak: Rute '{current_path}' dilindungi dan memerlukan autentikasi Bearer JWT. "
                        f"Silakan login di /api/v1/auth/login untuk memperoleh access token."
                    )
                },
                headers={"WWW-Authenticate": "Bearer"}
            )

        # 4. Validasi token jika ada
        if token:
            try:
                user_info = get_user_from_token(token)
                request.state.user = user_info
            except HTTPException as exc:
                if is_api_protected:
                    return JSONResponse(
                        status_code=exc.status_code,
                        content={"detail": exc.detail},
                        headers={"WWW-Authenticate": "Bearer"}
                    )
            except Exception as e:
                logger.warning(f"Gagal verifikasi token pada '{current_path}': {e}")
                if is_api_protected:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": f"Token autentikasi tidak valid atau sudah kedaluwarsa: {str(e)}"},
                        headers={"WWW-Authenticate": "Bearer"}
                    )

        # 5. Cek pembatasan Role (RBAC) pada rute bertarget khusus
        for prefix, allowed_roles in self.role_protected_routes.items():
            if current_path.startswith(prefix):
                if not user_info:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": f"Endpoint '{current_path}' memerlukan login sebagai: {allowed_roles}."},
                        headers={"WWW-Authenticate": "Bearer"}
                    )

                allowed_set = {r.lower().strip() for r in allowed_roles}
                if user_info.role.lower().strip() not in allowed_set:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": (
                                f"Akses Ditolak (403 Forbidden): Endpoint '{current_path}' hanya dapat diakses oleh "
                                f"role: {allowed_roles}. Role Anda saat ini adalah '{user_info.role}'."
                            )
                        }
                    )
                break

        response = await call_next(request)
        return response
