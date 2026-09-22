"""Shared application logging configured once by the entry point.

Call ``setup_logging(log_file=...)`` with values from configuration before
creating the pipeline. Modules can then call ``get_logger(__name__)``.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "anpr"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(
    log_file: str | Path,
    *,
    level: str | int = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    console: bool = True,
) -> logging.Logger:
    """Configure UTF-8 rotating file logs and optional console output.

    The caller supplies the configured log path. Repeated calls replace this
    application's handlers without changing the root logger. Configure logging
    at startup, before worker threads start. File-system errors propagate to
    the caller so an unusable log destination is visible.
    """
    if isinstance(level, str):
        resolved_level = logging.getLevelName(level.upper())
        if not isinstance(resolved_level, int):
            raise ValueError(f"Unknown logging level: {level!r}")
    elif isinstance(level, int):
        resolved_level = level
    else:
        raise TypeError("level must be a logging level name or integer")

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if backup_count < 1:
        raise ValueError("backup_count must be at least 1")

    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handlers: list[logging.Handler] = [file_handler]
    if console:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(formatter)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for handler in handlers:
        logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the application logger or a child sharing its configured handlers."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
