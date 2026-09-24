"""Tests for application settings and secret handling."""

import pytest
from pydantic import SecretStr

from app.core.config import Settings


def test_empty_string_secrets_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SEARCH_API_KEY", "   ")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    monkeypatch.setenv("JWT_SECRET_KEY", "")

    settings = Settings()

    assert settings.openai_api_key is None
    assert settings.search_api_key is None
    assert settings.langchain_api_key is None
    assert settings.jwt_secret_key is None
    assert not settings.has_llm_credentials()
    assert not settings.has_search_credentials()


def test_real_secrets_are_parsed_as_secretstr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("SEARCH_API_KEY", "tvly-test-456")

    settings = Settings()

    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.openai_api_key.get_secret_value() == "sk-test-123"
    assert settings.has_llm_credentials()
    assert settings.has_search_credentials()


def test_operational_defaults() -> None:
    settings = Settings()

    assert settings.max_sub_questions > 0
    assert settings.max_critic_rounds >= 1
    assert settings.scrape_delay_seconds >= 1.0
    assert settings.scrape_max_bytes > 0
    assert settings.database_url.startswith("sqlite:///")
