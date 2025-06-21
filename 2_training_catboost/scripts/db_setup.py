# db_setup.py

import sqlite3
import os

def db_setup(DB_PATH):
    if not os.path.exists(DB_PATH):
        print(f"Rank 0: Initializing SQLite at {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")  # Improve write performance
        conn.execute("PRAGMA temp_store = MEMORY;")  # Store temp tables in RAM
        conn.close()