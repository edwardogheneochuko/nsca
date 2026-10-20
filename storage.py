import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

DB_PATH = os.getenv("NSCA_DB_PATH", "nsca_local.sqlite3")

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cases (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, outcome TEXT NOT NULL, payload TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL
    )""")
    return conn

def save_case(result: dict):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO cases (id, created_at, outcome, payload) VALUES (?, ?, ?, ?)",
            (result["case_id"], datetime.now(timezone.utc).isoformat(), result["outcome"], json.dumps(result))
        )
        conn.commit()
    finally:
        conn.close()

def save_feedback(payload: dict) -> str:
    feedback_id = str(uuid4())
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO feedback (id, created_at, payload) VALUES (?, ?, ?)",
            (feedback_id, datetime.now(timezone.utc).isoformat(), json.dumps(payload))
        )
        conn.commit()
    finally:
        conn.close()
    return feedback_id

def load_feedback_count() -> int:
    conn = _connect()
    try:
        return int(conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
    finally:
        conn.close()
