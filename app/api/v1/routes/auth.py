"""Auth endpoints: login issues a short-lived JWT.

Rate-limited per client IP. Wrong credentials, missing config, and locked-out
clients all produce generic responses — nothing distinguishes "user not found"
from "wrong password".
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.rate_limit import rate_limiter
from app.core.security import (
    auth_enabled,
    create_access_token,
    credentials_configured,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Credentials payload for ``POST /auth/login``."""

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    """Issued token payload."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, request: Request) -> TokenResponse:
    settings = request.app.state.settings
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(
        "login",
        client_ip,
        limit=settings.rate_limit_login_per_minute,
        window_seconds=60.0,
    ):
        raise HTTPException(status_code=429, detail="too many requests")
    if not auth_enabled(settings):
        raise HTTPException(status_code=503, detail="auth is not configured")
    if not credentials_configured(settings):
        raise HTTPException(status_code=503, detail="auth is not configured")

    assert settings.admin_password is not None
    valid = (
        payload.username == settings.admin_username
        and payload.password == settings.admin_password.get_secret_value()
    )
    if not valid:
        raise HTTPException(status_code=401, detail="invalid credentials")

    token = create_access_token(payload.username, settings)
    return TokenResponse(access_token=token, expires_in=settings.jwt_ttl_minutes * 60)
