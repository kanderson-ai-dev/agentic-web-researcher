"""Application settings loaded from environment / .env.

Every secret is ``SecretStr | None``: an empty string (e.g. an unconfigured CI
secret) is treated as ``None`` so it can never be mistaken for a real key.
"""

from functools import lru_cache
from typing import Any

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SECRET_FIELDS = (
    "openai_api_key",
    "search_api_key",
    "langchain_api_key",
    "jwt_secret_key",
)


class Settings(BaseSettings):
    """Runtime configuration for the research microservice."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider (planner / critic / writer agents)
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"

    # Web search provider (Tavily-compatible API)
    search_api_key: SecretStr | None = None
    search_api_base_url: str = "https://api.tavily.com"

    # LangSmith tracing (optional)
    langchain_api_key: SecretStr | None = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "agentic-web-researcher"

    # Auth
    jwt_secret_key: SecretStr | None = None

    # Agent / pipeline limits
    max_sub_questions: int = 6
    max_critic_rounds: int = 2
    max_search_results_per_question: int = 3

    # Scraper ethics / limits
    scrape_delay_seconds: float = 1.0
    scrape_timeout_seconds: float = 15.0
    scrape_max_bytes: int = 2_000_000
    scrape_user_agent: str = (
        "agentic-web-researcher/0.1 (+https://github.com/kanderson-ai-dev; research bot)"
    )

    # Persistence
    database_url: str = "sqlite:///./data/jobs.sqlite"

    @field_validator(*_SECRET_FIELDS, mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def has_llm_credentials(self) -> bool:
        """True when a real LLM provider key is configured."""
        return self.openai_api_key is not None

    def has_search_credentials(self) -> bool:
        """True when a real web search provider key is configured."""
        return self.search_api_key is not None


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
