"""CLI logging: call configure_logging() before constructing RAGPipeline."""
from __future__ import annotations

import logging
import sys
from typing import Final

_LOGGER_NAME: Final = "simple_rag"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log = logging.getLogger(_LOGGER_NAME)
    log.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False
    for noisy in ("httpx", "httpcore", "chromadb", "kubernetes", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger() -> logging.Logger:
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        configure_logging(False)
    return log
