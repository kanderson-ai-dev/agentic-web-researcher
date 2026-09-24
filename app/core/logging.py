"""Structured logging via structlog, correlated by ``job_id``.

Any field whose name looks like a secret (``*_key``, ``*_token``,
``*_password``, ``authorization``, ``*_secret``) is redacted before the event
reaches the renderer — logs must never carry credential material.
"""

import logging
import re

import structlog
from structlog.typing import EventDict

_SENSITIVE = re.compile(r"(?:_key|_token|_password|authorization|_secret)$", re.I)


def redact_secrets(_logger: object, _method: str, event_dict: EventDict) -> EventDict:
    """structlog processor: mask values of sensitive-looking keys."""
    for key in list(event_dict):
        if _SENSITIVE.search(key):
            event_dict[key] = "***"
    return event_dict


def configure_logging(*, json_output: bool = False) -> None:
    """Configure stdlib logging + structlog pipeline once per process."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            redact_secrets,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
