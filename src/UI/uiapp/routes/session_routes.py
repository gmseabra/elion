# =============================================================================
# routes/session_routes.py
# Session management for Elion chat.
#
# Features:
#   - SQLite DB at visualizer/elion_sessions.db
#   - Keyed by (ip_address, chat_type)  — chat_type: "vina" | "attn"
#   - Persists conversation history across page refreshes
#   - Model registry: Qwen2.5-7B on Lysine, Qwen3-235B on cluster
#   - GET  /session/history?chat=vina|attn   — load history for this IP+chat
#   - POST /session/append                   — save one turn
#   - POST /session/clear                    — clear history for this IP+chat
#   - GET  /session/sessions                 — all sessions for this IP
#   - GET  /session/models                   — available models + active model
#   - POST /session/set_model                — switch active model
#
# Registration: loaded by uiapp/routes/__init__.py before its consumers.
#   _load("uiapp.routes.session_routes", "session_routes")
# =============================================================================

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import g, jsonify, request
from uiapp import app
from uiapp import config as _scfg
from uiapp.routes.shared import logger, _VIZ

# =============================================================================
# ── Database ──────────────────────────────────────────────────────────────────
# =============================================================================
# Config-driven so the DB lands under data/ (gitignored) rather than at the
# repo root, and so UI_SESSIONS_DB can move it. Its parent is created on
# first use because _init_db() runs at import time.
_DB_PATH = _scfg.SESSIONS_DB
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)

def _get_db() -> sqlite3.Connection:
    """Return a per-request SQLite connection (stored on Flask g)."""
    if "db" not in g:
        g.db = sqlite3.connect(_DB_PATH, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def _close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def _init_db():
    """Create tables if they don't exist. Called at module load."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ip          TEXT    NOT NULL,
            chat_type   TEXT    NOT NULL,   -- 'vina' | 'attn'
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL,
            model       TEXT    NOT NULL DEFAULT 'qwen2.5-7b',
            turn_count  INTEGER NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_ip_chat
            ON sessions (ip, chat_type);

        CREATE TABLE IF NOT EXISTS turns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES sessions(id),
            role        TEXT    NOT NULL,   -- 'user' | 'assistant'
            content     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_turns_session
            ON turns (session_id);
    """)
    conn.commit()
    conn.close()
    logger.info("[Session] DB initialised at %s", _DB_PATH)

_init_db()


# =============================================================================
# ── Model registry ─────────────────────────────────────────────────────────────
# Qwen2.5-7B-Instruct-GGUF (both parts) on Lysine
# Qwen3-235B-A22B-Instruct on cluster
# =============================================================================
_MODEL_REGISTRY = {
    "qwen2.5-7b": {
        "key":         "qwen2.5-7b",
        "label":       "Qwen2.5-7B",
        "description": "Instruct · GGUF q4_k_m (both parts) · Lysine",
        "badge":       "Qwen2.5-7B",
        "env":         "lysine",
        "active":      True,            # always available
        "alias":       "qwen2.5-14b",   # --model_alias in serve_qwen.sh
        "context":     8192,
    },
    "qwen3-235b": {
        "key":         "qwen3-235b",
        "label":       "Qwen3-235B-A22B-Instruct-2507",
        "description": "MoE flagship · fp8 · HPC /blue partition",
        "badge":       "Qwen3-235B",
        "env":         "cluster",
        "active":      False,           # greyed out — cluster only
        "alias":       "qwen3-235b-a22b-instruct-2507",
        "context":     16384,
    },
}

# Active model per-process (shared across all users on same server instance)
# In production this would be per-session; for now it's a server-level default.
_active_model_key = "qwen2.5-7b"


def get_active_model() -> dict:
    """Return the currently active model config dict."""
    return _MODEL_REGISTRY.get(_active_model_key, _MODEL_REGISTRY["qwen2.5-7b"])


# =============================================================================
# ── Helpers ────────────────────────────────────────────────────────────────────
# =============================================================================

def _client_ip() -> str:
    """Extract real client IP, respecting X-Forwarded-For from a proxy."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_or_create_session(ip: str, chat_type: str, db: sqlite3.Connection) -> int:
    """Return session id, creating the row if it doesn't exist."""
    row = db.execute(
        "SELECT id FROM sessions WHERE ip=? AND chat_type=?", (ip, chat_type)
    ).fetchone()
    if row:
        return row["id"]
    now = _now()
    cur = db.execute(
        "INSERT INTO sessions (ip, chat_type, created_at, updated_at, model, turn_count) "
        "VALUES (?,?,?,?,?,0)",
        (ip, chat_type, now, now, _active_model_key),
    )
    db.commit()
    return cur.lastrowid


def load_history(ip: str, chat_type: str, last_n: int = 20) -> list[tuple[str, str]]:
    """
    Load the last N turns for this IP+chat_type from the DB.
    Returns list of (role, content) tuples — same format as _vina_chat_history.
    Called by vina_chat_routes and attn_routes at request time.
    """
    try:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        sess = conn.execute(
            "SELECT id FROM sessions WHERE ip=? AND chat_type=?", (ip, chat_type)
        ).fetchone()
        if not sess:
            conn.close()
            return []
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (sess["id"], last_n),
        ).fetchall()
        conn.close()
        return [(r["role"], r["content"]) for r in reversed(rows)]
    except Exception as exc:
        logger.warning("[Session] load_history error: %s", exc)
        return []


def save_turn(ip: str, chat_type: str, role: str, content: str):
    """
    Persist one turn to the DB. Called after each Qwen response.
    Thread-safe via a dedicated connection.
    """
    try:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        now = _now()
        sess_id = _get_or_create_session(ip, chat_type, conn)
        conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (sess_id, role, content, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at=?, turn_count=turn_count+1 WHERE id=?",
            (now, sess_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[Session] save_turn error: %s", exc)


def clear_session(ip: str, chat_type: str):
    """Delete all turns for this IP+chat_type."""
    try:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        sess = conn.execute(
            "SELECT id FROM sessions WHERE ip=? AND chat_type=?", (ip, chat_type)
        ).fetchone()
        if sess:
            conn.execute("DELETE FROM turns WHERE session_id=?", (sess["id"],))
            conn.execute(
                "UPDATE sessions SET turn_count=0, updated_at=? WHERE id=?",
                (_now(), sess["id"]),
            )
            conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[Session] clear_session error: %s", exc)


# =============================================================================
# ── Routes ────────────────────────────────────────────────────────────────────
# =============================================================================

@app.route("/session/history", methods=["GET"])
def session_history():
    """
    GET /session/history?chat=vina|attn
    Returns the conversation history for this IP + chat type.
    """
    chat_type = request.args.get("chat", "vina")
    ip        = _client_ip()
    history   = load_history(ip, chat_type, last_n=40)
    return jsonify({
        "ip":        ip,
        "chat_type": chat_type,
        "history":   [{"role": r, "content": c} for r, c in history],
        "count":     len(history),
    })


@app.route("/session/append", methods=["POST"])
def session_append():
    """
    POST /session/append
    Body: {"chat": "vina"|"attn", "role": "user"|"assistant", "content": "..."}
    Saves one turn. Called by the frontend after each exchange.
    """
    data      = request.get_json(force=True) or {}
    chat_type = data.get("chat", "vina")
    role      = data.get("role", "user")
    content   = (data.get("content") or "").strip()
    ip        = _client_ip()
    if content:
        save_turn(ip, chat_type, role, content)
    return jsonify({"ok": True})


@app.route("/session/clear", methods=["POST"])
def session_clear():
    """
    POST /session/clear
    Body: {"chat": "vina"|"attn"}
    Clears history for this IP + chat type.
    """
    data      = request.get_json(force=True) or {}
    chat_type = data.get("chat", "vina")
    ip        = _client_ip()
    clear_session(ip, chat_type)
    return jsonify({"ok": True})


@app.route("/session/sessions", methods=["GET"])
def session_list():
    """
    GET /session/sessions
    Returns all sessions for this IP — used by the session viewer panel.
    """
    ip = _client_ip()
    try:
        db   = _get_db()
        rows = db.execute(
            "SELECT chat_type, created_at, updated_at, model, turn_count "
            "FROM sessions WHERE ip=? ORDER BY updated_at DESC",
            (ip,),
        ).fetchall()
        sessions = []
        for r in rows:
            # Get last 3 turns preview
            sess_row = db.execute(
                "SELECT id FROM sessions WHERE ip=? AND chat_type=?",
                (ip, r["chat_type"]),
            ).fetchone()
            preview = []
            if sess_row:
                turns = db.execute(
                    "SELECT role, content FROM turns WHERE session_id=? "
                    "ORDER BY id DESC LIMIT 3",
                    (sess_row["id"],),
                ).fetchall()
                preview = [{"role": t["role"], "content": t["content"][:120]}
                           for t in reversed(turns)]
            sessions.append({
                "chat_type":   r["chat_type"],
                "label":       {"vina": "Vina Docking",
                                "attn": "ChemBERT Attn",
                                "deepatom": "DeepAtom CNN"}.get(r["chat_type"], r["chat_type"]),
                "icon":        {"vina": "⚗️", "attn": "🧠", "deepatom": "⚛️"}.get(r["chat_type"], "💬"),
                "created_at":  r["created_at"],
                "updated_at":  r["updated_at"],
                "model":       r["model"],
                "turn_count":  r["turn_count"],
                "preview":     preview,
            })
        return jsonify({"ip": ip, "sessions": sessions})
    except Exception as exc:
        logger.error("[Session] session_list error: %s", exc)
        return jsonify({"ip": ip, "sessions": [], "error": str(exc)})


@app.route("/session/models", methods=["GET"])
def session_models():
    """
    GET /session/models
    Returns all models and which one is active.
    Used by the model picker UI.
    """
    return jsonify({
        "models":       list(_MODEL_REGISTRY.values()),
        "active_model": get_active_model(),
    })


@app.route("/session/set_model", methods=["POST"])
def session_set_model():
    """
    POST /session/set_model
    Body: {"model": "qwen2.5-7b" | "qwen3-235b"}
    Sets the active model. Returns error if model unavailable.
    """
    global _active_model_key
    data  = request.get_json(force=True) or {}
    key   = data.get("model", "qwen2.5-7b")
    if key not in _MODEL_REGISTRY:
        return jsonify({"ok": False, "error": f"Unknown model: {key}"}), 400
    model = _MODEL_REGISTRY[key]
    if not model["active"]:
        return jsonify({
            "ok":    False,
            "error": f"{model['label']} is only available on the {model['env']} environment.",
        }), 400
    _active_model_key = key
    logger.info("[Session] model switched to %s", key)
    return jsonify({"ok": True, "active_model": model})