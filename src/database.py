"""
SQLite database layer (read, write, modify, delete).

Planned contents:
- init_db(): create tables `clips` and `transcripts`
- insert_clips() / insert_transcripts(): write rows from pandas DataFrames
- read_clips() / read_transcripts() / read_joined(): read into DataFrames
- update_notes() / update_wer(): modify existing rows
- delete_transcripts() / delete_clip(): remove rows
- a small demo (python -m src.database) that shows read -> modify -> read
"""
import sqlite3

import numpy as np
import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id        TEXT PRIMARY KEY,
    source_clip_id TEXT,
    filename       TEXT,
    filepath       TEXT,
    condition      TEXT,
    snr_db         REAL,
    duration_s     REAL,
    sample_rate    INTEGER,
    true_text      TEXT
);
CREATE TABLE IF NOT EXISTS transcripts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id        TEXT NOT NULL,
    asr_model      TEXT NOT NULL,
    predicted_text TEXT,
    confidence     REAL,
    notes          TEXT,
    wer            REAL,
    UNIQUE (clip_id, asr_model),
    FOREIGN KEY (clip_id) REFERENCES clips (clip_id)
);
"""

CLIP_COLS = ["clip_id", "source_clip_id", "filename", "filepath", "condition",
             "snr_db", "duration_s", "sample_rate", "true_text"]


def connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)


def _rows(df, cols):
    sub = df[cols].astype(object)
    sub = sub.where(sub.notna(), None)
    return [tuple(r) for r in sub.itertuples(index=False, name=None)]


# ---------------------------------------------------------------- write
def insert_clips(df: pd.DataFrame):
    init_db()
    sql = f"INSERT OR REPLACE INTO clips ({','.join(CLIP_COLS)}) VALUES ({','.join('?' * len(CLIP_COLS))})"
    with connect() as conn:
        conn.executemany(sql, _rows(df, CLIP_COLS))


def insert_transcripts(df: pd.DataFrame):
    """Insert or overwrite transcripts. Columns: clip_id, asr_model, predicted_text, confidence, notes."""
    init_db()
    cols = ["clip_id", "asr_model", "predicted_text", "confidence", "notes"]
    sql = """INSERT INTO transcripts (clip_id, asr_model, predicted_text, confidence, notes)
             VALUES (?,?,?,?,?)
             ON CONFLICT(clip_id, asr_model) DO UPDATE SET
               predicted_text = excluded.predicted_text,
               confidence = excluded.confidence,
               notes = excluded.notes,
               wer = NULL"""
    with connect() as conn:
        conn.executemany(sql, _rows(df, cols))


# ----------------------------------------------------------------- read
def read_clips(condition=None) -> pd.DataFrame:
    init_db()
    q, params = "SELECT * FROM clips", ()
    if condition:
        q, params = q + " WHERE condition = ?", (condition,)
    with connect() as conn:
        return pd.read_sql_query(q, conn, params=params)


def read_transcripts(model=None) -> pd.DataFrame:
    init_db()
    q, params = "SELECT * FROM transcripts", ()
    if model:
        q, params = q + " WHERE asr_model = ?", (model,)
    with connect() as conn:
        return pd.read_sql_query(q, conn, params=params)


def read_joined(model=None) -> pd.DataFrame:
    """Clips joined with their transcripts (one row per clip x ASR model)."""
    init_db()
    q = """SELECT c.clip_id, c.source_clip_id, c.filename, c.condition, c.snr_db,
                  c.duration_s, c.true_text, t.asr_model, t.predicted_text,
                  t.confidence, t.notes, t.wer
           FROM transcripts t JOIN clips c ON c.clip_id = t.clip_id"""
    params = ()
    if model:
        q, params = q + " WHERE t.asr_model = ?", (model,)
    with connect() as conn:
        return pd.read_sql_query(q, conn, params=params)


def transcribed_ids(model) -> set:
    return set(read_transcripts(model)["clip_id"])


# --------------------------------------------------------------- modify
def update_notes(clip_id, model, notes):
    with connect() as conn:
        conn.execute("UPDATE transcripts SET notes = ? WHERE clip_id = ? AND asr_model = ?",
                     (notes, clip_id, model))


def update_wer(rows):
    """rows: iterable of (wer, clip_id, asr_model)."""
    rows = [(None if (w is None or (isinstance(w, float) and np.isnan(w))) else float(w), c, m)
            for w, c, m in rows]
    with connect() as conn:
        conn.executemany("UPDATE transcripts SET wer = ? WHERE clip_id = ? AND asr_model = ?", rows)


# --------------------------------------------------------------- delete
def delete_transcripts(model):
    with connect() as conn:
        conn.execute("DELETE FROM transcripts WHERE asr_model = ?", (model,))


def delete_clip(clip_id):
    with connect() as conn:
        conn.execute("DELETE FROM transcripts WHERE clip_id = ?", (clip_id,))
        conn.execute("DELETE FROM clips WHERE clip_id = ?", (clip_id,))


# ----------------------------------------------------------------- demo
def demo():
    clips, trans = read_clips(), read_transcripts()
    print(f"clips: {len(clips)} rows | transcripts: {len(trans)} rows")
    if trans.empty:
        print("No transcripts yet - run python -m src.transcribe first.")
        return
    row = trans.iloc[0]
    print("READ    :", row["clip_id"], row["asr_model"], "notes =", row["notes"])
    update_notes(row["clip_id"], row["asr_model"], "checked manually (demo)")
    after = read_transcripts(row["asr_model"]).query("clip_id == @row.clip_id").iloc[0]
    print("MODIFY  : notes ->", after["notes"])
    update_notes(row["clip_id"], row["asr_model"], row["notes"])
    print("RESTORE : notes ->", row["notes"])


if __name__ == "__main__":
    demo()
