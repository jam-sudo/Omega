"""OAuth2/JWT authentication helpers for the Omega PBPK API.

Auth is **disabled by default** (OMEGA_AUTH_ENABLED env var must be set to
enable it). When disabled all existing endpoints behave exactly as before.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    from jose import JWTError
    from jose import jwt as _jwt

    HAS_JOSE = True
except ImportError:  # pragma: no cover
    HAS_JOSE = False

try:
    from passlib.context import CryptContext as _CryptContext

    _pwd_context = _CryptContext(schemes=["bcrypt"], deprecated="auto")
    HAS_PASSLIB = True
except ImportError:  # pragma: no cover
    HAS_PASSLIB = False
    _pwd_context = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
AUTH_ENABLED: bool = os.environ.get("OMEGA_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")

SECRET_KEY: str = os.environ.get("OMEGA_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# ---------------------------------------------------------------------------
# Data models (frozen dataclasses — use tuples for sequence fields)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenData:
    """Decoded JWT payload."""

    username: str
    scopes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class User:
    """Stored user record."""

    username: str
    hashed_password: str
    scopes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# In-memory user store
# ---------------------------------------------------------------------------
_users: dict[str, User] = {}


def create_user(
    username: str,
    password: str,
    scopes: tuple[str, ...] = (),
) -> User:
    """Hash *password* and store a new user; return the User record."""
    if not HAS_PASSLIB:
        raise RuntimeError("passlib is required for auth. Install: pip install omega-pbpk[auth]")
    hashed = _pwd_context.hash(password)  # type: ignore[union-attr]
    user = User(username=username, hashed_password=hashed, scopes=scopes)
    _users[username] = user
    return user


def get_user(username: str) -> User | None:
    """Return the User record for *username*, or None if not found."""
    return _users.get(username)


def clear_users() -> None:
    """Remove all users from the in-memory store (useful for testing)."""
    _users.clear()


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Return bcrypt hash of *password*."""
    if not HAS_PASSLIB:  # pragma: no cover
        raise RuntimeError("passlib is required for auth.")
    return _pwd_context.hash(password)  # type: ignore[union-attr]


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    if not HAS_PASSLIB:  # pragma: no cover
        return False
    return bool(_pwd_context.verify(plain, hashed))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Encode *data* as a signed JWT and return the token string."""
    if not HAS_JOSE:  # pragma: no cover
        raise RuntimeError(
            "python-jose is required for auth. Install: pip install omega-pbpk[auth]"
        )
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return str(_jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM))


def verify_token(token: str) -> TokenData | None:
    """Decode and verify *token*; return TokenData or None on failure."""
    if not HAS_JOSE:  # pragma: no cover
        return None
    try:
        payload: dict[str, Any] = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            return None
        raw_scopes = payload.get("scopes", [])
        scopes: tuple[str, ...] = tuple(raw_scopes) if isinstance(raw_scopes, list) else ()
        return TokenData(username=username, scopes=scopes)
    except JWTError:
        return None
