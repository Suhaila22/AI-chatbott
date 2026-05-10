import bcrypt
from database import add_user, get_user


def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def register_user(username, password, role="user", preferred_language="English"):
    if not username or not password:
        return False, "Username and password are required."

    existing_user = get_user(username)
    if existing_user:
        return False, "Username already exists."

    password_hash = hash_password(password)
    add_user(username, password_hash, role, preferred_language)
    return True, "User registered successfully."


def login_user(username, password):
    user = get_user(username)
    if not user:
        return False, None, "User not found."

    user_id, username, password_hash, role, preferred_language = user
    if verify_password(password, password_hash):
        return True, {
            "id": user_id,
            "username": username,
            "role": role,
            "preferred_language": preferred_language
        }, "Login successful."

    return False, None, "Incorrect password."
