import sqlite3
import os
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "flare_catalog.db")

# Ensure the data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    """Initializes the SQLite database and creates the flare_catalog table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flare_catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            peak_time TEXT,
            end_time TEXT,
            peak_flux REAL,
            class TEXT,
            source TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_flare_event(start_time: str, peak_time: str, end_time: str, peak_flux: float, flare_class: str, source: str = "GOES"):
    """Inserts a new detected flare event into the catalog database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO flare_catalog (start_time, peak_time, end_time, peak_flux, class, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (start_time, peak_time, end_time, peak_flux, flare_class, source))
    conn.commit()
    conn.close()

def get_latest_flares(limit: int = 30) -> List[Dict[str, Any]]:
    """Retrieves the latest N recorded flare events from the database."""
    init_db()  # Safety check
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT start_time, peak_time, end_time, peak_flux, class, source 
        FROM flare_catalog 
        ORDER BY start_time DESC 
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
