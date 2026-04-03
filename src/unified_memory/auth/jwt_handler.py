"""
JWT creation and verification.

Requires ``python-jose[cryptography]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional


@dataclass
class AuthenticatedUser:
    """Decoded identity carried through the request lifecycle."""

    user_id: str
    tenant_id: str
    email: str
    roles: List[str] = field(default_factory=list)


def create_access_token(
    user_id: str,
    tenant_id: str,
    email: str,
    roles: List[str],
    secret: str,
    algorithm: str = "HS256",
    expire_minutes: int = 60,
) -> str:
    from jose import jwt

    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "roles": json.dumps(roles),
        "iat": now,
        "exp": now + timedelta(minutes=expire_minutes),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
) -> Optional[AuthenticatedUser]:
    """Return an ``AuthenticatedUser`` or ``None`` on any failure."""
    try:
        from jose import jwt, JWTError

        payload = jwt.decode(token, secret, algorithms=[algorithm])
        return AuthenticatedUser(
            user_id=payload["sub"],
            tenant_id=payload["tenant_id"],
            email=payload.get("email", ""),
            roles=json.loads(payload.get("roles", "[]")),
        )
    except Exception:
        return None
