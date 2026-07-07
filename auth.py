import os
from typing import Optional

from fastapi import Header, HTTPException, status


def configured_api_key() -> Optional[str]:
    value = os.getenv("APP_API_KEY", "").strip()
    if not value or value == "change-me":
        return None
    return value


def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    expected = configured_api_key()
    if expected is None:
        return

    token = None
    if authorization:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() == "bearer" and credentials:
            token = credentials.strip()
    if token is None and x_api_key:
        token = x_api_key.strip()

    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
