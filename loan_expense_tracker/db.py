"""Storage for transactions, loans and settings.

Uses Postgres (e.g. Supabase) when DATABASE_URL is set, otherwise a local
SQLite file. Queries are written once with `?` / `:name` placeholders and
translated for Postgres.
"""
import re
import sqlite3

from flask import current_app, g

SQLITE_SCHEMA = """
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

# Same tables in Postgres types. Dates stay ISO text so queries are shared.
PG_SCHEMA = (SQLITE_SCHEMA
             .replace("id INTEGER PRIMARY KEY", "id SERIAL PRIMARY KEY")
             .replace("REAL", "DOUBLE PRECISION")
             .replace("created_at TEXT DEFAULT CURRENT_TIMESTAMP",
                      "created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP"))

_NAMED = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def to_pg(sql: str) -> str:
    """Translate sqlite-style placeholders to psycopg's."""
    return _NAMED.sub(r"%(\1)s", sql.replace("?", "%s"))


class PGConnection:
    """Just enough of the sqlite3.Connection API for this app."""

    def __init__(self, url: str):
        import psycopg
        from psycopg.rows import dict_row
        # prepare_threshold=None: Supabase's pooler (transaction mode) can't
        # keep prepared statements across connections.
        self._con = psycopg.connect(url, row_factory=dict_row, prepare_threshold=None)

    def execute(self, sql: str, params=()):
        return self._con.execute(to_pg(sql), params or None)

    def commit(self):
        self._con.commit()

    def close(self):
        self._con.close()


def is_pg(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://"))


def connect(url: str):
    if is_pg(url):
        return PGConnection(url)
    con = sqlite3.connect(url)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def get_db():
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE"])
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(url: str):
    con = connect(url)
    if is_pg(url):
        con.execute(PG_SCHEMA)
    else:
        con.executescript(SQLITE_SCHEMA)
    con.commit()
    con.close()


def insert(sql: str, params) -> int:
    """Run an INSERT and return the new row's id on either database."""
    con = get_db()
    if isinstance(con, PGConnection):
        return con.execute(sql + " RETURNING id", params).fetchone()["id"]
    return con.execute(sql, params).lastrowid


def get_setting(key: str, default: str = "") -> str:
    row = get_db().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row and row["value"] is not None else default


def set_setting(key: str, value: str):
    db = get_db()
    db.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
               "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
    db.commit()
