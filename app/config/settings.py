"""Load and validate explicit YAML settings without import-time side effects."""

"""It checks required fields, data types, thresholds, and supported options, then returns typed settings. Invalid configuration raises ConfigurationException.
tests/test_settings.py tests that this loading and validation work correctly."""

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from app.core.exceptions import ConfigurationException


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class DetectionSettings:
    """Detector weights, confidence thresholds, and inference device."""

    vehicle_model: Path
    plate_model: Path
    vehicle_confidence: float
    plate_confidence: float
    iou_threshold: float
    device: str


@dataclass(frozen=True)
class TrackingSettings:
    """Ultralytics tracker configuration name."""

    tracker: str


@dataclass(frozen=True)
class MatchingSettings:
    """Plate association strategy and minimum IoU."""

    method: str
    iou_threshold: float


@dataclass(frozen=True)
class VideoSettings:
    """Video input and output options; sources may contain credentials."""

    source: str | int | Path = field(repr=False)
    output_dir: Path
    save_output: bool


@dataclass(frozen=True)
class OCRSettings:
    """OCR language, minimum recognition confidence, and device."""

    language: str
    confidence: float
    device: str


@dataclass(frozen=True)
class StorageSettings:
    """Destination for cropped plates."""

    plate_dir: Path


@dataclass(frozen=True)
class DatabaseSettings:
    """SQLite database file."""

    path: Path


@dataclass(frozen=True)
class LoggingSettings:
    """Options accepted by app.core.logger.setup_logging."""

    log_file: Path
    level: str
    max_bytes: int
    backup_count: int
    console: bool


@dataclass(frozen=True)
class Settings:
    """Validated application configuration."""

    detection: DetectionSettings
    tracking: TrackingSettings
    matching: MatchingSettings
    video: VideoSettings
    ocr: OCRSettings
    storage: StorageSettings
    database: DatabaseSettings
    logging: LoggingSettings


def _resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve file paths independently of the working directory."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _expand(value: Any, environment: dict[str, str | None]) -> Any:
    """Expand ${NAME} in parsed values without interpreting secrets as YAML."""
    if isinstance(value, dict):
        return {key: _expand(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        result = environment.get(name)
        if result is None:
            raise ConfigurationException(f"Environment variable {name} is not set")
        return result

    return _ENV_VARIABLE.sub(replace, value)


def _section(cls: type, data: Any, name: str, root: Path) -> Any:
    """Validate a section's exact keys and normalize its typed fields."""
    expected = {item.name for item in fields(cls)}
    if not isinstance(data, dict) or set(data) != expected:
        raise ConfigurationException(
            f"{name} must contain exactly: {', '.join(sorted(expected))}"
        )
    values = {}
    for item in fields(cls):
        value = data[item.name]
        label = f"{name}.{item.name}"
        if item.name == "source":
            if type(value) is int and value >= 0:
                pass
            elif isinstance(value, str) and value.strip():
                if value.isdecimal():
                    value = int(value)
                elif not value.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
                    value = _resolve_path(value, root)
            else:
                raise ConfigurationException(f"{label} must be a path, URL, or nonnegative camera index")
        elif item.type is Path:
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationException(f"{label} must be a nonempty path")
            value = _resolve_path(value, root)
        elif item.type is float:
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ConfigurationException(f"{label} must be a number between 0 and 1")
            value = float(value)
        elif item.type is int:
            if type(value) is not int or value <= 0:
                raise ConfigurationException(f"{label} must be a positive integer")
        elif item.type is bool:
            if type(value) is not bool:
                raise ConfigurationException(f"{label} must be a boolean")
        elif not isinstance(value, str) or not value.strip():
            raise ConfigurationException(f"{label} must be a nonempty string")
        values[item.name] = value

    choices = {
        "tracking": ("tracker", {"bytetrack.yaml", "botsort.yaml"}),
        "matching": ("method", {"center", "iou"}),
        "logging": ("level", {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}),
    }
    if name in choices:
        key, allowed = choices[name]
        if values[key] not in allowed:
            raise ConfigurationException(f"{name}.{key} must be one of: {', '.join(sorted(allowed))}")
    return cls(**values)


def load_settings(
    config_path: str | Path = "app/config/config.yaml",
    *,
    project_root: str | Path = PROJECT_ROOT,
    env_file: str | Path = ".env",
) -> Settings:
    """Read YAML and optional .env, validate, and return immutable settings.

    Relative paths use project_root, including custom config and .env paths.
    ${NAME} placeholders use process environment values before .env values.
    Environment values remain strings; numeric and boolean settings belong in
    YAML. This function does not change os.environ, create directories, or
    require model/video files to exist. Consumers validate input availability.
    """
    root = Path(project_root).expanduser().resolve()
    try:
        path = _resolve_path(config_path, root)
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationException("Cannot read configuration: file is missing, unreadable, or invalid YAML") from exc

    try:
        env_path = _resolve_path(env_file, root)
        environment = dict(dotenv_values(env_path, interpolate=False)) if env_path.exists() else {}
    except (OSError, UnicodeError) as exc:
        raise ConfigurationException("Cannot read environment file") from exc
    environment.update(os.environ)
    raw = _expand(raw, environment)
    expected = {item.name for item in fields(Settings)}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ConfigurationException(
            f"Configuration must contain exactly: {', '.join(sorted(expected))}"
        )
    return Settings(**{
        item.name: _section(item.type, raw[item.name], item.name, root)
        for item in fields(Settings)
    })
