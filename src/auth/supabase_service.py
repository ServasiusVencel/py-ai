import os
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone

import jwt
import bcrypt
from fastapi import HTTPException, status
from supabase import create_client, Client

from src.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    JWT_SECRET,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    BASE_DIR
)
from src.auth.schemas import UserInfo, TokenResponse

logger = logging.getLogger("Auth-Supabase-DB")

# Singleton Supabase Client
_supabase_client: Optional[Client] = None

# Local SQLite fallback path (digunakan jika tabel Supabase belum dibuat di dashboard)
LOCAL_DB_PATH = BASE_DIR / "data" / "local_users.db"


def get_supabase_client() -> Client:
    """
    Mengambil atau menginisialisasi instance singleton Supabase Client.
    """
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL dan SUPABASE_KEY harus disetel di file .env")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ---------------------------------------------------------------------------
# Helper Enkripsi Password (Bcrypt) & JWT
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Mengenkripsi plain password menggunakan salt bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Memverifikasi plain password terhadap hash bcrypt."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    full_name: Optional[str] = None,
    is_peatland: bool = False,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Membuat token JWT dengan role dan status lahan gambut pengguna."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_id,
        "email": email,
        "role": role.lower().strip(),
        "full_name": full_name or "",
        "is_peatland": is_peatland,
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


# ---------------------------------------------------------------------------
# Local SQLite Fallback Helper (agar API langsung bisa dites tanpa nunggu DDL)
# ---------------------------------------------------------------------------

def _get_local_db_conn():
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                full_name TEXT,
                is_peatland INTEGER NOT NULL DEFAULT 0, -- 0 = false, 1 = true
                created_at TEXT
            )
        """)
        # Migrasi: tambahkan kolom is_peatland jika belum ada
        try:
            conn.execute("ALTER TABLE users ADD COLUMN is_peatland INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Kolom sudah ada, abaikan error
    return conn


# ---------------------------------------------------------------------------
# Operasi Database Tabel Users (Supabase Database + Local Fallback)
# ---------------------------------------------------------------------------

def _is_table_not_found_error(err_obj: Any) -> bool:
    err_str = str(err_obj)
    return "PGRST205" in err_str or "Could not find the table" in err_str


def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    client = get_supabase_client()
    clean_email = email.lower().strip()
    try:
        res = client.table("users").select("*").eq("email", clean_email).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as e:
        if _is_table_not_found_error(e):
            logger.warning("Tabel 'users' di Supabase belum dibuat. Menggunakan penyimpanan fallback lokal.")
            conn = _get_local_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email = ? LIMIT 1", (clean_email,))
            row = cur.fetchone()
            conn.close()
            if row:
                return dict(row)
            return None
        logger.error(f"Error querying user by email: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


def insert_new_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    client = get_supabase_client()
    try:
        res = client.table("users").insert(user_data).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
        return user_data
    except Exception as e:
        if _is_table_not_found_error(e):
            logger.warning("Tabel 'users' di Supabase belum dibuat. Menyimpan user ke fallback lokal.")
            conn = _get_local_db_conn()
            with conn:
                conn.execute(
                    "INSERT INTO users (id, email, password_hash, role, full_name, is_peatland, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_data["id"],
                        user_data["email"],
                        user_data["password_hash"],
                        user_data["role"],
                        user_data["full_name"],
                        1 if user_data.get("is_peatland", False) else 0,
                        user_data.get("created_at", datetime.now().isoformat())
                    )
                )
            conn.close()
            return user_data
        logger.error(f"Error inserting user into Supabase: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan ke database Supabase: {str(e)}")


def get_all_users() -> List[Dict[str, Any]]:
    """
    Mengambil seluruh daftar pengguna dari tabel 'users' di database Supabase (atau fallback).
    Kolom password_hash otomatis disaring (dihapus) demi keamanan.
    """
    client = get_supabase_client()
    try:
        res = client.table("users").select("id, email, role, full_name, is_peatland, created_at").order("created_at", desc=True).execute()
        users_list = res.data or []
        for u in users_list:
            u.pop("password_hash", None)
        return users_list
    except Exception as e:
        if _is_table_not_found_error(e):
            conn = _get_local_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT id, email, role, full_name, is_peatland, created_at FROM users ORDER BY created_at DESC")
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows
        logger.error(f"Error fetching all users from Supabase: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mengambil data user dari Supabase: {str(e)}")


def update_user_by_id(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mengupdate data pengguna berdasarkan user_id di tabel 'users' Supabase (atau fallback).
    Jika password diubah, otomatis di-hash dengan bcrypt.
    """
    client = get_supabase_client()

    clean_updates = {}
    if "full_name" in updates and updates["full_name"] is not None:
        clean_updates["full_name"] = str(updates["full_name"]).strip()
    if "role" in updates and updates["role"] is not None:
        r = str(updates["role"]).lower().strip()
        if r not in ("user", "superadmin"):
            raise HTTPException(status_code=400, detail="Role harus 'user' atau 'superadmin'.")
        clean_updates["role"] = r
    if "email" in updates and updates["email"] is not None:
        clean_updates["email"] = str(updates["email"]).lower().strip()
    if "password" in updates and updates["password"]:
        clean_updates["password_hash"] = hash_password(str(updates["password"]))
    if "is_peatland" in updates and updates["is_peatland"] is not None:
        clean_updates["is_peatland"] = bool(updates["is_peatland"])

    if not clean_updates:
        raise HTTPException(status_code=400, detail="Tidak ada data yang diberikan untuk diperbarui.")

    clean_updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        res = client.table("users").update(clean_updates).eq("id", user_id).execute()
        if res.data and len(res.data) > 0:
            updated_user = res.data[0]
            updated_user.pop("password_hash", None)
            return updated_user
        raise HTTPException(status_code=404, detail=f"Pengguna dengan ID '{user_id}' tidak ditemukan.")
    except HTTPException:
        raise
    except Exception as e:
        if _is_table_not_found_error(e):
            conn = _get_local_db_conn()
            set_clauses = [f"{k} = ?" for k in clean_updates.keys()]
            values = list(clean_updates.values()) + [user_id]
            with conn:
                cur = conn.execute(f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?", values)
                if cur.rowcount == 0:
                    conn.close()
                    raise HTTPException(status_code=404, detail=f"Pengguna dengan ID '{user_id}' tidak ditemukan.")
            cur = conn.cursor()
            cur.execute("SELECT id, email, role, full_name, is_peatland, created_at FROM users WHERE id = ?", (user_id,))
            row = cur.fetchone()
            conn.close()
            return dict(row)
        logger.error(f"Error updating user in Supabase: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memperbarui pengguna: {str(e)}")


def delete_user_by_id(user_id: str) -> bool:
    """
    Menghapus data pengguna berdasarkan user_id dari tabel 'users' Supabase (atau fallback).
    """
    client = get_supabase_client()
    try:
        # Cek apakah user ada sebelum delete
        check = client.table("users").select("id").eq("id", user_id).limit(1).execute()
        if not check.data or len(check.data) == 0:
            raise HTTPException(status_code=404, detail=f"Pengguna dengan ID '{user_id}' tidak ditemukan.")

        client.table("users").delete().eq("id", user_id).execute()
        return True
    except HTTPException:
        raise
    except Exception as e:
        if _is_table_not_found_error(e):
            conn = _get_local_db_conn()
            with conn:
                cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.close()
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail=f"Pengguna dengan ID '{user_id}' tidak ditemukan.")
            return True
        logger.error(f"Error deleting user in Supabase: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal menghapus pengguna: {str(e)}")


# ---------------------------------------------------------------------------
# Layanan Auth: Register, Login, Logout, Verify Token
# ---------------------------------------------------------------------------

def register_user(
    email: str,
    password: str,
    role: str = "user",
    full_name: Optional[str] = None,
    is_peatland: bool = False
) -> Dict[str, Any]:
    """
    Mendaftarkan pengguna baru langsung ke tabel 'users' di database.
    TANPA konfirmasi email - akun langsung aktif seketika!
    """
    clean_email = email.lower().strip()
    clean_role = role.lower().strip()

    if clean_role not in ("user", "superadmin"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role harus bernilai 'user' atau 'superadmin'."
        )

    # 1. Cek apakah email sudah terdaftar
    existing_user = find_user_by_email(clean_email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email '{clean_email}' sudah terdaftar. Silakan login."
        )

    # 2. Enkripsi password menggunakan bcrypt
    pwd_hash = hash_password(password)

    # 3. Buat ID dan simpan ke database Supabase
    import uuid
    user_id = str(uuid.uuid4())
    user_payload = {
        "id": user_id,
        "email": clean_email,
        "password_hash": pwd_hash,
        "role": clean_role,
        "full_name": full_name or clean_email.split("@")[0],
        "is_peatland": bool(is_peatland),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    saved = insert_new_user(user_payload)

    user_info = UserInfo(
        id=str(saved.get("id", user_id)),
        email=saved.get("email", clean_email),
        role=saved.get("role", clean_role),
        full_name=saved.get("full_name"),
        is_peatland=bool(saved.get("is_peatland", is_peatland)),
        metadata={"created_at": saved.get("created_at")}
    )

    return {
        "success": True,
        "message": f"Akun berhasil dibuat dengan role '{clean_role}'. Anda dapat langsung login tanpa konfirmasi email!",
        "user": user_info
    }


def login_user(email: str, password: str) -> TokenResponse:
    """
    Login langsung memverifikasi kredensial ke tabel 'users' di database Supabase.
    Langsung menerbitkan JWT Token dengan role pengguna (user / superadmin).
    """
    clean_email = email.lower().strip()

    # 1. Ambil data user dari tabel database
    user_record = find_user_by_email(clean_email)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah."
        )

    # 2. Verifikasi kecocokan password hash
    stored_hash = user_record.get("password_hash")
    if not stored_hash or not verify_password(password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah."
        )

    user_id = str(user_record.get("id"))
    user_role = str(user_record.get("role", "user")).lower().strip()
    full_name = user_record.get("full_name")
    user_is_peatland = bool(user_record.get("is_peatland", False))

    # 3. Terbitkan JWT Token
    access_token = create_access_token(
        user_id=user_id,
        email=clean_email,
        role=user_role,
        full_name=full_name,
        is_peatland=user_is_peatland
    )

    user_info = UserInfo(
        id=user_id,
        email=clean_email,
        role=user_role,
        full_name=full_name,
        is_peatland=user_is_peatland,
        metadata={"created_at": user_record.get("created_at")}
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=None,
        user=user_info
    )


async def logout_user(token: str) -> bool:
    """
    Logout token pengguna.
    """
    # Pada JWT stateless, logout ditandai sukses sehingga client menghapus token dari local storage / cookies
    return True


def get_user_from_token(token: str) -> UserInfo:
    """
    Memverifikasi dan mendecode JWT token.
    Mengekstrak ID, email, dan role ('user' atau 'superadmin').
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        email = payload.get("email")
        role = payload.get("role", "user")
        full_name = payload.get("full_name")
        is_peatland = bool(payload.get("is_peatland", False))

        if not user_id or not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Klaim token tidak lengkap.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        return UserInfo(
            id=str(user_id),
            email=str(email),
            role=str(role),
            full_name=full_name,
            is_peatland=is_peatland,
            metadata={"iat": payload.get("iat"), "exp": payload.get("exp")}
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token autentikasi sudah kedaluwarsa. Silakan login kembali.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token autentikasi tidak valid: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
