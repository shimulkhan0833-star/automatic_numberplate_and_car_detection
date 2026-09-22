# Automatic Number Plate Recognition (ANPR)

A Python project for vehicle detection, tracking, license plate recognition,
and database storage from video or CCTV streams.

## Current status

Implemented: project scaffolding, rotating file and console logging, shared
exceptions, validated YAML configuration, and tests. Detection, tracking, matching, OCR,
database access, and pipeline files are placeholders. `app/main.py` is empty;
end-to-end video processing is not available yet.

## Planned pipeline

```text
Video / CCTV
  -> YOLO11 vehicle detection
  -> ByteTrack tracking and vehicle IDs
  -> Custom YOLO11 plate detection
  -> Vehicle-to-plate matching (center point / IoU)
  -> Plate cropping and PaddleOCR
  -> SQLite storage and logs
```

BoT-SORT is an optional tracker. Planned records include vehicle ID, plate text,
confidence, timestamp, image path, and video source.

## Installation (Anaconda on Windows)

Open Anaconda Prompt or Anaconda PowerShell Prompt and navigate to this
repository. Create a dedicated environment with Python 3.11:

```powershell
conda create -n anpr python=3.11 pip -y
conda activate anpr
python -m pip install -r requirements.txt
python -m pip check
```

If you already have a project environment, activate it instead of creating
`anpr`. Run `conda activate anpr` in each new terminal session, and select that
environment's Python interpreter in your IDE. `requirements.txt` is installed
with pip inside the active Conda environment; see the
[Conda environment guide](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

Requirements include the CPU PaddlePaddle runtime. GPU installation depends on
hardware and platform; follow the official
[PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)
and [Ultralytics installation guide](https://docs.ultralytics.com/quickstart).
PaddleOCR 3.x requires PaddlePaddle 3.0 or newer; see the
[PaddleOCR quick start](https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html).

Version ranges are initial development constraints, not a tested lock file.
The full AI dependency installation and inference have not been verified in
this repository. SQLite, logging, and unittest are included with Python.

## Current project structure

```text
app/
  main.py
  config/          # Settings and config.yaml
  core/            # Logging, exceptions, constants
  detection/       # Vehicle and plate detectors
  tracking/        # Vehicle tracking
  matching/        # Vehicle-to-plate association
  recognition/     # OCR
  database/        # Connection and repository
  pipeline/        # Pipeline orchestration
models/            # Model weights
data/              # See paths below
database/          # SQLite files
logs/              # Application logs
tests/
requirements.txt
setup_project.py
INSTRUCTIONS.txt
```

Current data paths are `data/videos/input/`, `data/videos/output/`, and
`data/plates/`. The structure reflects the current scaffold; `INSTRUCTIONS.txt`
describes the broader target architecture.

Create missing scaffold files with:

```powershell
python setup_project.py
```

Files are created alongside the setup script in the repository root, without
an extra `ANPR_Project` directory. Existing files are preserved.

## Models and configuration

The vehicle detector will use pretrained YOLO11 weights. The plate detector
uses the custom-trained weights at `models/best.pt`.
Installing dependencies does not supply the custom plate model.

Configuration lives in `app/config/config.yaml`, loaded through
`app/config/settings.py`. Keep model paths, confidence thresholds, video paths,
and database paths in YAML, and sensitive values in `.env`.

```python
from dataclasses import asdict
from app.config.settings import load_settings
from app.core.logger import setup_logging

settings = load_settings()
setup_logging(**asdict(settings.logging))
```

Paths resolve against the repository root, independently of the working
directory. The loader validates required sections, keys, types, thresholds,
tracker names, and log settings, raising `ConfigurationException` on errors.
It does not create output directories or require model/video files to exist.
`sample.mp4` is a placeholder: set `video.source` to your video, an integer
camera index such as `0`, or an RTSP/HTTP URL. OCR defaults to English;
choose a language supported by your OCR model for your plates.

For a private camera URL, set `video.source: "${ANPR_VIDEO_SOURCE}"` in YAML
and `ANPR_VIDEO_SOURCE=rtsp://user:password@camera/live` in the root `.env`.
Process environment values take precedence over `.env`; missing referenced
variables raise an error. Loading does not modify the process environment.
Substitutions remain strings; put numeric and boolean options directly in YAML.
The `.env` file is optional and excluded from Git.

## Logging and exceptions

Configure logging once at startup, passing the log path from configuration.
Standalone example from the repository root:

```python
from app.core.logger import get_logger, setup_logging

setup_logging(log_file="logs/anpr.log", level="INFO")
logger = get_logger(__name__)
logger.info("Application initialized")
```

Logging supports UTF-8, console output, and rotating files (5 MiB and three
backups by default). Repeated setup replaces handlers to avoid duplicate logs.
Use `logger.exception(...)` inside an exception handler to record tracebacks.

Domain errors inherit from `ANPRException` in `app/core/exceptions.py`, including
configuration, model, video, camera, detection, tracking, matching, OCR, database,
and storage errors. Preserve causes with `raise OCRException(message) from exc`.

## Tests

Tests use unittest. Configuration tests require only PyYAML and python-dotenv,
without AI dependencies:

```powershell
python -m pip install "PyYAML>=6.0,<7" "python-dotenv>=1.0,<2"
python -m unittest discover -s tests -v
```

They cover Unicode logging, handler reconfiguration, rotation, tracebacks,
exception chaining, configuration validation, path resolution, and environment
overrides. Model inference is not tested yet.

## Guidelines and roadmap

Follow `INSTRUCTIONS.txt`: use focused modules, type hints, docstrings, logging,
custom exceptions, and meaningful tests. Keep database queries in the repository
layer and AI modules independent.

- Phase 1: configuration, detection, tracking, matching, OCR, and SQLite.
- Phase 2: FastAPI, PostgreSQL, REST API, and authentication.
- Phase 3: multiple CCTV streams, dashboard, cloud deployment, monitoring,
  and user management.
