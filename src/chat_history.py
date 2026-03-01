"""
chat_history.py
---------------
Lightweight SQLite persistence for chat messages.
Each message row: (id, session_id, role, content, created_at).
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# Store the DB next to the project root so it survives runs
_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"


def _get_conn() -> sqlite3.Connection:
    """
    Establish a connection to the SQLite database.

    Returns:
        sqlite3.Connection: A connection object with Row factory enabled.
    """
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the SQLite database schema.
    Creates the 'messages' table if it does not already exist.
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
        """)
        conn.commit()


def load_messages(session_id: str) -> list[dict]:
    """
    Retrieve all chat messages for a specific session.

    Args:
        session_id (str): The unique identifier for the chat session.

    Returns:
        list[dict]: A list of message dictionaries with 'role' and 'content'.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Persist a new message to the database.

    Args:
        session_id (str): The unique identifier for the chat session.
        role (str): The role of the sender ('user' or 'assistant').
        content (str): The text content of the message.
    """
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()


def clear_messages(session_id: str) -> None:
    """
    Delete all messages associated with a specific session ID.

    Args:
        session_id (str): The unique identifier for the chat session.
    """
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()


def list_sessions() -> list[str]:
    """
    Retrieve a list of all unique session IDs, ordered by most recent activity.

    Returns:
        list[str]: A list of session IDs.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT session_id FROM messages GROUP BY session_id ORDER BY MAX(id) DESC"
        ).fetchall()
    return [r["session_id"] for r in rows]
