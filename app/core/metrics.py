"""Prometheus metrics for the research service.

HTTP-level metrics come from ``prometheus-fastapi-instrumentator`` (mounted in
``main.create_app``); these are the domain metrics the console/alerts care
about: job outcomes, latency, token usage and estimated cost.
"""

from prometheus_client import Counter, Histogram

RESEARCH_JOBS = Counter(
    "research_jobs_total", "Research jobs by final status", ["status"]
)
RESEARCH_DURATION = Histogram(
    "research_job_duration_seconds",
    "End-to-end research job latency",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)
LLM_TOKENS = Counter(
    "llm_tokens_total", "LLM tokens consumed, by agent role and direction", ["role", "kind"]
)
JOB_COST = Histogram(
    "research_job_cost_usd",
    "Estimated LLM cost per research job",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
)
