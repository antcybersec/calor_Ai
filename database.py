import os
import json
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", "calor_ai.db")

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database schema if tables do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Meals Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                raw_input TEXT,
                meal_type TEXT DEFAULT 'unknown',
                items_json TEXT NOT NULL,
                total_calories REAL NOT NULL DEFAULT 0.0,
                total_protein_g REAL NOT NULL DEFAULT 0.0,
                total_carbs_g REAL NOT NULL DEFAULT 0.0,
                total_fat_g REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'active',
                replaced_by_meal_id INTEGER NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User Memories Table (Structured Selective Memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, memory_key)
            )
        """)

        # Chat History Table (Session-based multi-turn logs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                image_path TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

# --- MEALS OPERATIONAL HELPERS ---

def log_meal(
    user_id: str,
    raw_input: str,
    items: List[Dict[str, Any]],
    total_calories: float,
    total_protein_g: float = 0.0,
    total_carbs_g: float = 0.0,
    total_fat_g: float = 0.0,
    meal_type: str = "unknown",
    log_date: Optional[str] = None,
    db_path: str = DB_PATH
) -> Dict[str, Any]:
    """Insert a new meal record."""
    init_db(db_path)
    current_date = log_date or date.today().isoformat()
    now_iso = datetime.now().isoformat()
    items_json_str = json.dumps(items)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meals (
                user_id, date, timestamp, raw_input, meal_type,
                items_json, total_calories, total_protein_g, total_carbs_g, total_fat_g, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
        """, (
            user_id, current_date, now_iso, raw_input, meal_type,
            items_json_str, float(total_calories), float(total_protein_g),
            float(total_carbs_g), float(total_fat_g)
        ))
        meal_id = cursor.lastrowid
        conn.commit()

    return {
        "meal_id": meal_id,
        "user_id": user_id,
        "date": current_date,
        "meal_type": meal_type,
        "items": items,
        "total_calories": total_calories,
        "total_protein_g": total_protein_g,
        "total_carbs_g": total_carbs_g,
        "total_fat_g": total_fat_g,
        "status": "active"
    }

def correct_last_meal(
    user_id: str,
    raw_input: str,
    items: List[Dict[str, Any]],
    total_calories: float,
    total_protein_g: float = 0.0,
    total_carbs_g: float = 0.0,
    total_fat_g: float = 0.0,
    meal_type: Optional[str] = None,
    log_date: Optional[str] = None,
    db_path: str = DB_PATH
) -> Dict[str, Any]:
    """
    Corrects the user's most recent active meal on the given date (or today).
    Marks previous meal as 'superseded' and inserts the corrected meal to prevent double-counting.
    """
    init_db(db_path)
    current_date = log_date or date.today().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Find the latest active meal for this user on this date (or overall)
        cursor.execute("""
            SELECT id, meal_type FROM meals
            WHERE user_id = ? AND date = ? AND status = 'active'
            ORDER BY id DESC LIMIT 1
        """, (user_id, current_date))
        row = cursor.fetchone()

        previous_meal_id = None
        inherited_meal_type = meal_type or "unknown"
        if row:
            previous_meal_id = row["id"]
            if not meal_type:
                inherited_meal_type = row["meal_type"]
            
            # Mark previous meal as superseded
            cursor.execute("""
                UPDATE meals SET status = 'superseded', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (previous_meal_id,))

        # Insert new updated meal
        now_iso = datetime.now().isoformat()
        items_json_str = json.dumps(items)

        cursor.execute("""
            INSERT INTO meals (
                user_id, date, timestamp, raw_input, meal_type,
                items_json, total_calories, total_protein_g, total_carbs_g, total_fat_g, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
        """, (
            user_id, current_date, now_iso, raw_input, inherited_meal_type,
            items_json_str, float(total_calories), float(total_protein_g),
            float(total_carbs_g), float(total_fat_g)
        ))
        new_meal_id = cursor.lastrowid

        if previous_meal_id:
            cursor.execute("""
                UPDATE meals SET replaced_by_meal_id = ? WHERE id = ?
            """, (new_meal_id, previous_meal_id))

        conn.commit()

    return {
        "meal_id": new_meal_id,
        "previous_meal_id": previous_meal_id,
        "user_id": user_id,
        "date": current_date,
        "meal_type": inherited_meal_type,
        "items": items,
        "total_calories": total_calories,
        "total_protein_g": total_protein_g,
        "total_carbs_g": total_carbs_g,
        "total_fat_g": total_fat_g,
        "status": "active",
        "action": "corrected"
    }

def get_daily_totals(user_id: str, query_date: Optional[str] = None, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Calculate running daily totals for active meals on a given date (default today)."""
    init_db(db_path)
    target_date = query_date or date.today().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(id) as meal_count,
                COALESCE(SUM(total_calories), 0.0) as total_calories,
                COALESCE(SUM(total_protein_g), 0.0) as total_protein_g,
                COALESCE(SUM(total_carbs_g), 0.0) as total_carbs_g,
                COALESCE(SUM(total_fat_g), 0.0) as total_fat_g
            FROM meals
            WHERE user_id = ? AND date = ? AND status = 'active'
        """, (user_id, target_date))
        row = cursor.fetchone()

        cursor.execute("""
            SELECT id, meal_type, items_json, total_calories, total_protein_g, timestamp
            FROM meals
            WHERE user_id = ? AND date = ? AND status = 'active'
            ORDER BY id ASC
        """, (user_id, target_date))
        meal_rows = cursor.fetchall()

    logged_meals = []
    for m in meal_rows:
        logged_meals.append({
            "id": m["id"],
            "meal_type": m["meal_type"],
            "items": json.loads(m["items_json"]),
            "total_calories": round(m["total_calories"], 1),
            "total_protein_g": round(m["total_protein_g"], 1),
            "timestamp": m["timestamp"]
        })

    return {
        "user_id": user_id,
        "date": target_date,
        "meal_count": row["meal_count"],
        "total_calories": round(row["total_calories"], 1),
        "total_protein_g": round(row["total_protein_g"], 1),
        "total_carbs_g": round(row["total_carbs_g"], 1),
        "total_fat_g": round(row["total_fat_g"], 1),
        "meals": logged_meals
    }

def get_meal_history(user_id: str, limit: int = 5, query_date: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve history of active meals for user."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if query_date:
            cursor.execute("""
                SELECT * FROM meals
                WHERE user_id = ? AND date = ? AND status = 'active'
                ORDER BY id DESC LIMIT ?
            """, (user_id, query_date, limit))
        else:
            cursor.execute("""
                SELECT * FROM meals
                WHERE user_id = ? AND status = 'active'
                ORDER BY id DESC LIMIT ?
            """, (user_id, limit))
        
        rows = cursor.fetchall()

    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "date": r["date"],
            "timestamp": r["timestamp"],
            "raw_input": r["raw_input"],
            "meal_type": r["meal_type"],
            "items": json.loads(r["items_json"]),
            "total_calories": r["total_calories"],
            "total_protein_g": r["total_protein_g"],
            "total_carbs_g": r["total_carbs_g"],
            "total_fat_g": r["total_fat_g"]
        })
    return history

# --- MEMORY OPERATIONAL HELPERS ---

def save_memory(
    user_id: str,
    category: str,
    memory_key: str,
    memory_value: str,
    confidence: float = 1.0,
    db_path: str = DB_PATH
) -> Dict[str, Any]:
    """Store or update a selective memory item."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_memories (user_id, category, memory_key, memory_value, confidence, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, memory_key) DO UPDATE SET
                category = excluded.category,
                memory_value = excluded.memory_value,
                confidence = excluded.confidence,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, category, memory_key.lower().strip(), memory_value, confidence))
        conn.commit()

    return {
        "user_id": user_id,
        "category": category,
        "memory_key": memory_key.lower().strip(),
        "memory_value": memory_value,
        "confidence": confidence
    }

def get_memories(user_id: str, category: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve stored selective memories for a user."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if category:
            cursor.execute("""
                SELECT category, memory_key, memory_value, confidence, updated_at
                FROM user_memories
                WHERE user_id = ? AND category = ?
                ORDER BY updated_at DESC
            """, (user_id, category))
        else:
            cursor.execute("""
                SELECT category, memory_key, memory_value, confidence, updated_at
                FROM user_memories
                WHERE user_id = ?
                ORDER BY updated_at DESC
            """, (user_id,))
        
        rows = cursor.fetchall()

    return [dict(r) for r in rows]

def delete_memory(user_id: str, memory_key: str, db_path: str = DB_PATH) -> bool:
    """Delete a memory item."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM user_memories WHERE user_id = ? AND memory_key = ?
        """, (user_id, memory_key.lower().strip()))
        conn.commit()
        return cursor.rowcount > 0

# --- CHAT HISTORY HELPERS ---

def save_chat_message(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    image_path: Optional[str] = None,
    db_path: str = DB_PATH
) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (user_id, session_id, role, content, image_path)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, session_id, role, content, image_path))
        conn.commit()

def get_chat_history(user_id: str, session_id: str, limit: int = 20, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, image_path, created_at FROM chat_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY id ASC LIMIT ?
        """, (user_id, session_id, limit))
        return [dict(r) for r in cursor.fetchall()]

def clear_user_data(user_id: str, db_path: str = DB_PATH) -> None:
    """Reset user database data for evaluation testing."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM meals WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
