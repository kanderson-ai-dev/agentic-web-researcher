# Contributing

## Setup

```bash
uv sync --all-extras
cp .env.example .env   # fill in keys as needed — the app degrades cleanly without them
uv run uvicorn app.main:app --reload
# http://localhost:8000/         → landing
# http://localhost:8000/console/ → operator console
```

## Conventions

- Python ≥ 3.11, package manager `uv`. `requirements.txt` is generated
  (`uv export`), never hand-edited.
- `ruff check .` must pass (line-length 100, rules `E,F,I,UP,B,SIM`, ignore
  `B008`).
- `mypy --strict app/` must pass with 0 errors.
- Tests must be deterministic and fully offline — no network, no real
  credentials. Tests that genuinely need live credentials go behind explicit
  `requires_*` skip markers (`requires_search_api`, `requires_openai_key`,
  `requires_langsmith_key`) defined in `tests/conftest.py`, so the default
  suite passes with zero secrets.
- Quality is gated like tests: `uv run python -m evaluation.run_eval` must
  pass its thresholds (citation support ≥ 0.90, source precision ≥ 0.80,
  topic coverage ≥ 0.85). A quality regression fails CI — never adjust a
  reported number to look better than the measured one.
- Frontend is plain HTML/CSS/JS, no build step: Tailwind via CDN on `/`,
  token-based `styles.css` on `/console`. Don't introduce a bundler.
- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`,
  `refactor:`).

## Security rules that are not optional

- Scraping always goes through `app/services/scraper_client.py`
  (rate-limited, `robots.txt`-respecting) — never fetch a URL directly from
  a graph node.
- Scraped content is **untrusted data, never instructions**: it must pass the
  sanitization guardrail (`app/graph/guardrails.py`) before reaching any
  prompt, and is never interpolated raw into a system/instruction prompt.
- The research topic passes the input guardrail before any paid LLM or
  search call — keep the guardrail-first order when modifying the graph.
- Secrets are `pydantic.SecretStr | None` in `app/core/config.py`, treating
  `""` as `None`. Never accept keys via query string; return generic error
  messages.
- Log redaction covers `*_key`, `*_token`, `*_password`, `authorization` and
  `*_secret` — name sensitive fields so the redactor catches them.
- Never commit secrets. `.env`, `data/*.sqlite`, and internal planning docs
  are gitignored; `gitleaks` and `pip-audit` run in CI.

## Before you push

```bash
ruff check .
mypy --strict app/
pytest -v --cov=app
uv run python -m evaluation.run_eval
```

CI (`.github/workflows/ci.yml`) runs exactly this plus `pip-audit` and
`gitleaks`, on `pull_request` with `contents: read`.
