import sqlite3
from datetime import datetime

DB_NAME = "enterprise_chatbot.db"


def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        role TEXT DEFAULT 'user',
        preferred_language TEXT DEFAULT 'English',
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        user_message TEXT,
        bot_response TEXT,
        language TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS escalation_tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question TEXT,
        bot_response TEXT,
        status TEXT DEFAULT 'Open',
        assigned_to TEXT,
        admin_note TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        uploaded_by TEXT,
        source_type TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        expected_answer TEXT,
        bot_answer TEXT,
        score REAL,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def add_user(username, password_hash, role="user", preferred_language="English"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO users (username, password_hash, role, preferred_language, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (username, password_hash, role, preferred_language, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_user(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, username, password_hash, role, preferred_language
    FROM users
    WHERE username = ?
    """, (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_all_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, username, role, preferred_language, created_at
    FROM users
    ORDER BY created_at DESC
    """)
    users = cursor.fetchall()
    conn.close()
    return users


def update_user_role(username, role):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
    conn.commit()
    conn.close()


def save_chat(username, user_message, bot_response, language):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO chat_history (username, user_message, bot_response, language, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (username, user_message, bot_response, language, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_chat_history(username=None):
    conn = get_connection()
    cursor = conn.cursor()
    if username:
        cursor.execute("""
        SELECT username, user_message, bot_response, language, created_at
        FROM chat_history
        WHERE username = ?
        ORDER BY created_at DESC
        """, (username,))
    else:
        cursor.execute("""
        SELECT username, user_message, bot_response, language, created_at
        FROM chat_history
        ORDER BY created_at DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def create_escalation_ticket(username, question, bot_response):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO escalation_tickets
    (username, question, bot_response, status, assigned_to, admin_note, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, question, bot_response, "Open", "", "", now, now))
    conn.commit()
    conn.close()


def get_escalation_tickets():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, username, question, bot_response, status, assigned_to, admin_note, created_at, updated_at
    FROM escalation_tickets
    ORDER BY created_at DESC
    """)
    tickets = cursor.fetchall()
    conn.close()
    return tickets


def update_ticket(ticket_id, status, assigned_to, admin_note):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE escalation_tickets
    SET status = ?, assigned_to = ?, admin_note = ?, updated_at = ?
    WHERE id = ?
    """, (status, assigned_to, admin_note, datetime.now().isoformat(), ticket_id))
    conn.commit()
    conn.close()


def save_knowledge_file(filename, uploaded_by, source_type):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO knowledge_files (filename, uploaded_by, source_type, created_at)
    VALUES (?, ?, ?, ?)
    """, (filename, uploaded_by, source_type, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_knowledge_files():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, filename, uploaded_by, source_type, created_at
    FROM knowledge_files
    ORDER BY created_at DESC
    """)
    files = cursor.fetchall()
    conn.close()
    return files


def save_evaluation_result(question, expected_answer, bot_answer, score):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO evaluation_results
    (question, expected_answer, bot_answer, score, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (question, expected_answer, bot_answer, score, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_evaluation_results():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT question, expected_answer, bot_answer, score, created_at
    FROM evaluation_results
    ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows
