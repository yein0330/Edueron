# src/utils/auth.py

import bcrypt

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Parameters
    ----------
    password : str
        Plain text password to hash
        
    Returns
    -------
    str
        Hashed password
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against a hash.
    
    Parameters
    ----------
    password : str
        Plain text password to verify
    hashed : str
        Hashed password to compare against
        
    Returns
    -------
    bool
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def migrate_plain_passwords(user_db: dict) -> dict:
    """
    Migrate plain text passwords to hashed passwords.
    
    Parameters
    ----------
    user_db : dict
        User database with plain passwords
        
    Returns
    -------
    dict
        Updated database with hashed passwords
    """
    for username, user_info in user_db["users"].items():
        password = user_info.get("password", "")
        # Check if password is already hashed (bcrypt hashes start with $2b$)
        if not password.startswith("$2b$"):
            user_info["password"] = hash_password(password)
    return user_db