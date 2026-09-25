"""SQLite storage."""
import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS loans (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lender TEXT DEFAULT '',
    principal REAL NOT NULL,
    rate REAL NOT NULL,
    term_months INTEGER NOT NULL,
    first_due TEXT NOT NULL,
    payment REAL,
    extra REAL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('expense', 'income')),
    amount REAL NOT NULL,
    category TEXT NOT NULL,
    day TEXT NOT NULL,
    note TEXT DEFAULT '',
    merchant TEXT DEFAULT '',
    receipt TEXT,
    source TEXT DEFAULT 'manual',
    loan_id INTEGER REFERENCES loans(id) ON DELETE SET NULL,
    loan_period INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_tx_day ON transactions(day);
CREATE UNIQUE INDEX IF NOT EXISTS ix_tx_loan_period
    ON transactions(loan_id, loan_period) WHERE loan_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(path: str):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def get_setting(key: str, default: str = "") -> str:
    row = get_db().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row and row["value"] is not None else default


def set_setting(key: str, value: str):
    db = get_db()
    db.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
               "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
    db.commit()
