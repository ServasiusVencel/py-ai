-- ========================================================================
-- SCHEMA TABEL USERS UNTUK SUPABASE
-- Jalankan query ini di SQL Editor pada Dashboard Supabase Anda:
-- Dashboard Supabase -> Project -> SQL Editor -> New Query -> Run
-- ========================================================================

-- 1. Buat tabel users jika belum ada
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user', -- Nilai: 'user' atau 'superadmin'
    full_name TEXT,
    is_peatland BOOLEAN NOT NULL DEFAULT FALSE, -- Tipe lahan gambut: true = lahan gambut, false = bukan lahan gambut
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tambahkan kolom is_peatland jika tabel sudah ada sebelumnya (migrasi)
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_peatland BOOLEAN NOT NULL DEFAULT FALSE;

-- 2. Nonaktifkan Row Level Security (RLS) agar API backend dapat melakukan INSERT & SELECT langsung
ALTER TABLE public.users DISABLE ROW LEVEL SECURITY;

-- 3. (Opsional) Buat Superadmin default jika belum ada (Password default: Admin123!)
-- Password hash di bawah menggunakan bcrypt untuk: 'Admin123!'
INSERT INTO public.users (email, password_hash, role, full_name)
VALUES (
    'admin@nimbus.local',
    '$2b$12$e6mZ85d0i/fG8NqU9kXl4u2U82e7t1B/zP8v6O5U.C0gGgUeYc62u',
    'superadmin',
    'Superadmin Default'
)
ON CONFLICT (email) DO NOTHING;
