from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Email pengguna")
    password: str = Field(..., min_length=6, description="Password minimal 6 karakter")
    role: Literal["user", "superadmin"] = Field(
        default="user",
        description="Role pengguna: 'user' atau 'superadmin'"
    )
    full_name: Optional[str] = Field(default=None, description="Nama lengkap pengguna (opsional)")
    is_peatland: bool = Field(default=False, description="Status lahan gambut: True = lahan gambut, False = bukan lahan gambut (non-gambut)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "user@example.com",
                "password": "SecretPassword123!",
                "role": "user",
                "full_name": "Budi Santoso",
                "is_peatland": False
            }
        }
    }


class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="Email terdaftar")
    password: str = Field(..., description="Password akun")

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "user@example.com",
                "password": "SecretPassword123!"
            }
        }
    }


class UserInfo(BaseModel):
    id: str = Field(..., description="Supabase Auth UUID")
    email: str = Field(..., description="Email pengguna")
    role: str = Field(default="user", description="Role yang dimiliki (user / superadmin)")
    full_name: Optional[str] = Field(default=None, description="Nama lengkap pengguna")
    is_peatland: bool = Field(default=False, description="Status tipe lahan gambut: True = lahan gambut, False = bukan lahan gambut")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="User metadata dari Supabase")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT Bearer access token")
    token_type: str = Field(default="bearer", description="Tipe token")
    expires_in: Optional[int] = Field(default=3600, description="Durasi masa berlaku token dalam detik")
    refresh_token: Optional[str] = Field(default=None, description="Refresh token untuk memperbarui sesi")
    user: UserInfo = Field(..., description="Informasi profil dan role pengguna")


class LogoutResponse(BaseModel):
    success: bool = Field(default=True)
    message: str = Field(..., description="Pesan status logout")


class GenericResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, description="Nama lengkap baru (opsional)")
    role: Optional[Literal["user", "superadmin"]] = Field(default=None, description="Role baru: 'user' atau 'superadmin'")
    email: Optional[EmailStr] = Field(default=None, description="Email baru (opsional)")
    password: Optional[str] = Field(default=None, min_length=6, description="Password baru jika ingin mereset password (opsional)")
    is_peatland: Optional[bool] = Field(default=None, description="Status tipe lahan gambut: True = lahan gambut, False = bukan lahan gambut (opsional)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "full_name": "Nama Baru Pengguna",
                "role": "superadmin",
                "email": "emailbaru@perusahaan.com",
                "password": "PasswordBaru123!",
                "is_peatland": False
            }
        }
    }

