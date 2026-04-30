"""
Authentication module for user login, signup, and password reset functionality.
"""
import sqlite3
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Database path for users
AUTH_DB_PATH = "users.db"

def get_auth_db():
    """Get connection to authentication database."""
    conn = sqlite3.connect(AUTH_DB_PATH)
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )''')
    
    # Create password reset tokens table
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT UNIQUE NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        used INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    conn.commit()
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash a password with a salt. Returns (hash, salt)."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use PBKDF2 for password hashing
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return password_hash.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verify a password against a hash and salt."""
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == password_hash


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password strength.
    Returns (is_valid, error_message)
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
    
    return True, ""


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate username.
    Returns (is_valid, error_message)
    Requirements:
    - 3-20 characters
    - Alphanumeric and underscores only
    """
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(username) > 20:
        return False, "Username must be no more than 20 characters long"
    
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    
    return True, ""


def create_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Create a new user account.
    Returns (success, message)
    """
    # Validate inputs
    username_valid, username_error = validate_username(username)
    if not username_valid:
        return False, username_error
    
    if not validate_email(email):
        return False, "Invalid email format"
    
    password_valid, password_error = validate_password(password)
    if not password_valid:
        return False, password_error
    
    # Check if username or email already exists
    conn = get_auth_db()
    c = conn.cursor()
    
    c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if c.fetchone():
        conn.close()
        return False, "Username or email already exists"
    
    # Create user
    try:
        password_hash, salt = hash_password(password)
        c.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, salt)
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username or email already exists"
    except Exception as e:
        conn.close()
        return False, f"Error creating account: {str(e)}"


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[dict], str]:
    """
    Authenticate a user.
    Returns (success, user_data, message)
    """
    if not username or not password:
        return False, None, "Username and password are required"
    
    conn = get_auth_db()
    c = conn.cursor()
    
    c.execute(
        "SELECT id, username, email, password_hash, salt, is_active FROM users WHERE username = ? OR email = ?",
        (username, username)
    )
    user = c.fetchone()
    
    if not user:
        conn.close()
        return False, None, "Invalid username or password"
    
    user_id, db_username, email, password_hash, salt, is_active = user
    
    if not is_active:
        conn.close()
        return False, None, "Account is deactivated. Please contact support."
    
    if not verify_password(password, password_hash, salt):
        conn.close()
        return False, None, "Invalid username or password"
    
    # Update last login
    c.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_id))
    conn.commit()
    conn.close()
    
    user_data = {
        "id": user_id,
        "username": db_username,
        "email": email
    }
    
    return True, user_data, "Login successful!"


def generate_reset_token(email: str) -> Tuple[bool, Optional[str], str]:
    """
    Generate a password reset token for a user.
    Returns (success, token, message)
    """
    if not validate_email(email):
        return False, None, "Invalid email format"
    
    conn = get_auth_db()
    c = conn.cursor()
    
    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return False, None, "No account found with this email address"
    
    user_id = user[0]
    
    # Generate token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=1)  # Token expires in 1 hour
    
    # Invalidate old tokens for this user
    c.execute("UPDATE password_reset_tokens SET used = 1 WHERE user_id = ?", (user_id,))
    
    # Create new token
    c.execute(
        "INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user_id, token, expires_at)
    )
    conn.commit()
    conn.close()
    
    return True, token, "Password reset token generated"


def verify_reset_token(token: str) -> Tuple[bool, Optional[int], str]:
    """
    Verify a password reset token.
    Returns (is_valid, user_id, message)
    """
    if not token:
        return False, None, "Token is required"
    
    conn = get_auth_db()
    c = conn.cursor()
    
    c.execute(
        "SELECT user_id, expires_at, used FROM password_reset_tokens WHERE token = ?",
        (token,)
    )
    token_data = c.fetchone()
    
    if not token_data:
        conn.close()
        return False, None, "Invalid or expired reset token"
    
    user_id, expires_at_str, used = token_data
    
    if used:
        conn.close()
        return False, None, "This reset token has already been used"
    
    expires_at = datetime.fromisoformat(expires_at_str)
    if datetime.now() > expires_at:
        conn.close()
        return False, None, "This reset token has expired"
    
    conn.close()
    return True, user_id, "Token is valid"


def reset_password(token: str, new_password: str) -> Tuple[bool, str]:
    """
    Reset a user's password using a reset token.
    Returns (success, message)
    """
    # Validate password
    password_valid, password_error = validate_password(new_password)
    if not password_valid:
        return False, password_error
    
    # Verify token
    token_valid, user_id, message = verify_reset_token(token)
    if not token_valid:
        return False, message
    
    # Update password
    conn = get_auth_db()
    c = conn.cursor()
    
    password_hash, salt = hash_password(new_password)
    c.execute(
        "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
        (password_hash, salt, user_id)
    )
    
    # Mark token as used
    c.execute("UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (token,))
    
    conn.commit()
    conn.close()
    
    return True, "Password reset successfully!"


def change_password(user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
    """
    Change a user's password (requires old password).
    Returns (success, message)
    """
    # Validate new password
    password_valid, password_error = validate_password(new_password)
    if not password_valid:
        return False, password_error
    
    conn = get_auth_db()
    c = conn.cursor()
    
    # Get current password hash
    c.execute("SELECT password_hash, salt FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return False, "User not found"
    
    password_hash, salt = user
    
    # Verify old password
    if not verify_password(old_password, password_hash, salt):
        conn.close()
        return False, "Current password is incorrect"
    
    # Update password
    new_password_hash, new_salt = hash_password(new_password)
    c.execute(
        "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
        (new_password_hash, new_salt, user_id)
    )
    conn.commit()
    conn.close()
    
    return True, "Password changed successfully!"


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Get user data by ID."""
    conn = get_auth_db()
    c = conn.cursor()
    
    c.execute("SELECT id, username, email, created_at, last_login FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        return None
    
    return {
        "id": user[0],
        "username": user[1],
        "email": user[2],
        "created_at": user[3],
        "last_login": user[4]
    }

