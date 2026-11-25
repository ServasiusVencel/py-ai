from src.auth.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserInfo,
    TokenResponse,
    LogoutResponse,
    GenericResponse,
    UserUpdateRequest,
)
from src.auth.supabase_service import (
    get_supabase_client,
    register_user,
    login_user,
    logout_user,
    get_user_from_token,
    get_all_users,
    update_user_by_id,
    delete_user_by_id,
)
from src.auth.dependencies import (
    oauth2_scheme,
    get_current_user,
    get_optional_user,
    require_roles,
    require_user,
    require_superadmin,
    require_non_peatland,
    require_peatland,
)
from src.auth.middleware import RoleAuthMiddleware
from src.auth.router import auth_router

__all__ = [
    "UserRegisterRequest",
    "UserLoginRequest",
    "UserInfo",
    "TokenResponse",
    "LogoutResponse",
    "GenericResponse",
    "get_supabase_client",
    "register_user",
    "login_user",
    "logout_user",
    "get_user_from_token",
    "get_all_users",
    "oauth2_scheme",
    "get_current_user",
    "get_optional_user",
    "require_roles",
    "require_user",
    "require_superadmin",
    "require_non_peatland",
    "require_peatland",
    "RoleAuthMiddleware",
    "auth_router",
]
