# Security Posture

This document maps the controls implemented in this service to the OWASP
Top 10 (web) and OWASP LLM Top 10. It is written to be auditable: every claim
points at the module that enforces it.

## Threat model

The service accepts a research topic, autonomously searches and scrapes the
public web, feeds parsed content to an LLM, and publishes a report behind a
human review gate. The two trust boundaries are:

1. **User input** — the topic string (API consumers, authenticated).
2. **Web content** — pages/documents fetched from arbitrary URLs found by
   search. This is the distinguishing risk of the project: *indirect prompt
   injection* embedded in third-party content.

## OWASP LLM Top 10 mapping

| Risk | Control | Implementation |
|---|---|---|
| LLM01 Prompt Injection (direct) | Input guardrail screens every topic before any LLM call or paid request; blocked topics get a generic 422 | `app/graph/guardrails.py:screen_topic`, `input_guardrail` node, `POST /research` |
| LLM01 Prompt Injection (indirect) | All scraped content is treated as hostile *data*: invisible-character normalization, instruction-phrase neutralization, length caps — applied before anything reaches the LLM | `sanitize_scraped_content` in `app/graph/guardrails.py`, invoked in `document_worker` |
| LLM02 Sensitive Information Disclosure | Secrets are `SecretStr \| None` (empty string → unset); generic error messages; structlog processor redacts `*_key`, `*_token`, `*_password`, `authorization`, `*_secret` fields; no secrets in query strings (Bearer header only) | `app/core/config.py`, `app/core/logging.py`, `app/core/security.py` |
| LLM03 Supply Chain | Locked dependencies (`uv.lock`), generated `requirements.txt`, `pip-audit` + `gitleaks` in CI | `pyproject.toml`, `.github/workflows/ci.yml` |
| LLM04 Data & Model Poisoning | Output is grounded in fetched sources: every citation's quote must be verifiably present in its source's extracted text (verbatim, punctuation-insensitive, or ≥0.85 sliding-window similarity) or it is dropped; adversarial fixture proves the guardrail | `verify_citations`, `tests/fixtures/adversarial_page.html` |
| LLM05 Improper Output Handling | Console renders the report escaped (no raw HTML injection); output guardrail strips unsupported citations before the human gate | `frontend/console/app.js:renderMarkdown`, `output_guardrail` node |
| LLM06 Excessive Agency | Bounded re-plan loop (`max_critic_rounds`), read-only tools, per-job cost ceiling visible in metrics, mandatory human review before a report is considered published | `app/graph/assembly.py`, `human_review` node |
| LLM07 System Prompt Leakage | Prompts are server-side only; refusal messages are generic and carry no prompt material | `rejection_output` node, generic API errors |
| LLM08 Vector & Embedding Weaknesses | Not applicable — the service does not use embedding stores | — |
| LLM09 Misinformation | Citation support rate is measured and gated in CI (≥ 0.90); unresolved sub-questions are surfaced honestly instead of being written around | `evaluation/`, `output_guardrail`, `report_assembler` |
| LLM10 Unbounded Consumption | Rate limiting on `/auth/login` and `POST /research`; bounded critic loop; scraper byte/time limits; per-job token accounting with USD estimate | `app/core/rate_limit.py`, `app/services/usage.py`, `ScraperClient` |

## OWASP Top 10 (web) mapping

| Risk | Control |
|---|---|
| Broken Access Control | JWT Bearer auth on all `/api/v1/research*` endpoints (HS256, issuer + expiry checked); uniform 401 responses |
| Cryptographic Failures | `SecretStr` everywhere; secrets only via headers/env — never query strings; no secrets in logs (redaction processor) |
| Injection | No string-interpolated queries (SQLite via parameterized `aiosqlite`); topic guardrail before LLM |
| Insecure Design | Guardrail-first pipeline: malicious input is rejected before any LLM spend |
| Security Misconfiguration | Security headers middleware (`CSP`, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`); CI `permissions: contents: read`, `pull_request` only |
| Vulnerable Components | `pip-audit` in CI; locked dependency file |
| Auth Failures | Rate-limited login; generic "invalid credentials"; expired/invalid tokens share one 401 |
| SSRF | Scraper fetches only `http(s)` URLs, respects `robots.txt`, caps response size and enforces timeouts |
| Logging Failures | structlog JSON pipeline, `job_id` correlation, sensitive-field redaction |
| Integrity of Data | LangGraph SQLite checkpoints per job; HITL gate records the human decision in state |

## Responsible scraping

`ScraperClient` enforces: `robots.txt` compliance (cached per origin), a
configurable delay between requests, a descriptive User-Agent, response size
and content-type limits, and an opt-in Playwright fallback (lazy import —
never loaded unless configured).

## Secret hygiene

- `.env`, `data/*.sqlite`, and all planning docs are gitignored and
  dockerignored.
- `.env.example` documents every variable with empty values.
- `gitleaks detect` runs in CI; no secret may enter code, tests, fixtures,
  commit messages or this file.

## Known limitations

- Auth is a single static credential (`ADMIN_*`) — appropriate for a
  portfolio/demo deployment, not multi-tenant production.
- The rate limiter is in-process: it does not coordinate across replicas.
- The SSRF guard allowlists schemes but does not yet block private IP ranges
  (documented for a follow-up hardening pass).
