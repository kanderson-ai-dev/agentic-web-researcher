"""JWT issuance/verification and the auth dependency used by routers.

Auth is *disabled* when ``jwt_secret_key`` is not configured (local/dev mode) —
``require_auth`` becomes a no-op and ``/auth/login`` returns 503. When it is
enabled, every token error produces the same generic 401 so nothing about the
key material or account validity is leaked.
"""

import time

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


class TokenError(Exception):
    """Raised for any token problem — callers must not distinguish causes."""


def create_access_token(subject: str, settings: Settings) -> str:
    """Issue a signed JWT for ``subject``."""
    if settings.jwt_secret_key is None:
        raise TokenError("jwt not configured")
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + settings.jwt_ttl_minutes * 60,
        "iss": settings.jwt_issuer,
    }
    return jwt.encode(
        payload, settings.jwt_secret_key.get_secret_value(), algorithm="HS256"
    )


def verify_access_token(token: str, settings: Settings) -> str:
    """Verify a JWT and return its subject. All failures raise ``TokenError``."""
    if settings.jwt_secret_key is None:
        raise TokenError("jwt not configured")
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
        )
    except jwt.PyJWTError as exc:
        raise TokenError("invalid token") from exc
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise TokenError("invalid token")
    return subject


def auth_enabled(settings: Settings) -> bool:
    """True when JWT auth is configured (production mode)."""
    return settings.jwt_secret_key is not None


def credentials_configured(settings: Settings) -> bool:
    """True when the login endpoint can actually verify credentials."""
    return settings.jwt_secret_key is not None and settings.admin_password is not None


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency: enforce Bearer auth when configured.

    Returns the authenticated subject (or ``"anonymous"`` when auth is off).
    """
    settings: Settings = request.app.state.settings
    if not auth_enabled(settings):
        return "anonymous"
    if credentials is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        return verify_access_token(credentials.credentials, settings)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail="unauthorized") from exc


def get_settings_dep() -> Settings:
    """Settings dependency (kept for routers that don't need the request)."""
    return get_settings()
