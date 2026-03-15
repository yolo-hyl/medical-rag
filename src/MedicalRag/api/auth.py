"""
SQLite-backed auth, session management, and message persistence.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

DB_PATH = str(Path(__file__).resolve().parents[4] / "medical_rag.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                phone TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('doctor','patient','admin')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                service_type TEXT NOT NULL CHECK(service_type IN ('chat','agent')),
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                extra_data TEXT,
                timestamp TEXT NOT NULL
            );
        """)
        # Migrate existing databases
        try:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN extra_data TEXT")
        except Exception:
            pass  # Column already exists


def register_user(phone: str, password: str, role: str) -> str:
    """Returns user_id. Raises ValueError on duplicate phone."""
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO users (id, phone, password_hash, role, created_at) VALUES (?,?,?,?,?)",
                (user_id, phone, _hash(password), role, now),
            )
    except sqlite3.IntegrityError:
        raise ValueError("手机号已注册")
    return user_id


def login_user(phone: str, password: str) -> tuple[str, str, str]:
    """Returns (user_id, token, role). Raises ValueError on bad credentials."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash, role FROM users WHERE phone = ?", (phone,)
        ).fetchone()
    if not row or row["password_hash"] != _hash(password):
        raise ValueError("手机号或密码错误")

    token = str(uuid.uuid4())
    expires_at = (datetime.utcnow() + timedelta(days=30)).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?,?,?)",
            (token, row["id"], expires_at),
        )
    return row["id"], token, row["role"]


def get_user_info(user_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, phone, role FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    return {"user_id": row["id"], "phone": row["phone"], "role": row["role"]}


def verify_token(token: str) -> Optional[str]:
    """Returns user_id if token is valid and not expired, else None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at FROM auth_tokens WHERE token = ?", (token,)
        ).fetchone()
    if not row:
        return None
    if datetime.utcnow().isoformat() > row["expires_at"]:
        return None
    return row["user_id"]


def upsert_session(
    session_id: str, user_id: str, service_type: str, title: Optional[str] = None
) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        else:
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, service_type, title, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, user_id, service_type, title, now, now),
            )


def save_message(session_id: str, role: str, content: str, extra_data: Optional[str] = None) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, extra_data, timestamp) VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), session_id, role, content, extra_data, now),
        )


def list_sessions(user_id: str, service_type: str) -> List[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions "
            "WHERE user_id = ? AND service_type = ? ORDER BY updated_at DESC",
            (user_id, service_type),
        ).fetchall()
    return [dict(r) for r in rows]


def get_messages(session_id: str, user_id: str) -> List[dict]:
    """Returns messages for a session, ownership-checked."""
    with _conn() as conn:
        session = conn.execute(
            "SELECT user_id FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session or session["user_id"] != user_id:
            raise PermissionError("无权访问该会话")
        rows = conn.execute(
            "SELECT role, content, extra_data, timestamp FROM chat_messages "
            "WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str, user_id: str) -> bool:
    with _conn() as conn:
        session = conn.execute(
            "SELECT user_id FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session or session["user_id"] != user_id:
            return False
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    return True
