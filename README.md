# Automatic Number Plate Recognition (ANPR)

A Python project for vehicle detection, tracking, license plate recognition,
and database storage from video or CCTV streams.

## Current status

Implemented: YAML settings, logging, vehicle tracking, plate detection,
vehicle-to-plate matching, OCR with per-vehicle caching, and ANPRPipeline.
The pipeline can preview results and save annotated video. SQLite recognition storage is implemented;
best-reading plate crops are saved as PNG images. Pipeline tests mock inference and
video devices; they do not verify real model accuracy or installed codecs.

## Run the pipeline

From the repository root, configure `app/config/config.yaml`, then run:

```powershell
python -m app.main
```

Press Q or close the preview window to stop. When `video.save_output` is true,
a uniquely named MP4 is written to `video.output_dir`. For preview only, use
`python test_video.py`.

For processing without a preview window:

```python
from app.pipeline.anpr_pipeline import ANPRPipeline

pipeline = ANPRPipeline(display=False)
frame_count = pipeline.run()
print(frame_count, pipeline.output_path)
```

Each run creates fresh tracking and OCR cache state. Sources can be video paths,
camera indices, or RTSP/HTTP URLs. A failed read after processing frames ends the
run; automatic stream reconnection is not implemented. Output uses the reported
source FPS, or 30 FPS when unavailable.

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

Configuration lives in `app/config/config.yaml`. `app/config/settings.py`
reads it once on import and exposes uppercase constants such as
`VEHICLE_MODEL_PATH`, `PLATE_CONFIDENCE`, and `DATABASE_PATH`.

```python
from app.config import settings
from app.core.logger import setup_logging

setup_logging(settings.LOG_FILE, level=settings.LOG_LEVEL,
              max_bytes=settings.LOG_MAX_BYTES,
              backup_count=settings.LOG_BACKUP_COUNT,
              console=settings.LOG_CONSOLE)
```

Run commands from the repository root: configuration and relative file paths
use the current working directory. Restart the application after editing YAML.
Set `video.source` to a local video path, camera index such as `0`, or RTSP/HTTP
URL. Put numeric and boolean options directly in YAML. This simple configuration
module does not load `.env`, expand environment placeholders, or perform schema
validation. OCR model names determine the recognition language.

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

Tests use unittest. Configuration tests require only PyYAML,
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

## Recognition database

The pipeline creates the SQLite file at `database.path` automatically and writes
`plate_recognitions` after OCR. Each run has a new `pipeline.session_id`.
All tracked cars, motorcycles, buses, and trucks are stored, even without a
readable plate. People are excluded. Plate text, OCR confidence, and the best
reading timestamp stay NULL until recognition succeeds. Vehicle class is stored
as `class_id` and `class_name`. Existing databases are migrated automatically
on connection, preserving their records (old records have unknown vehicle class).

The primary key `(session_id, vehicle_id)` prevents duplicate rows per frame
and separates tracking IDs from different runs. Unmatched and unreadable OCR results do not overwrite vehicle records. A higher-confidence reading replaces the text, confidence, and
video timestamp; equal confidence retains the previous reading.

`first_seen` and `last_seen` are UTC processing times when the tracked vehicle
is observed, including frames without a readable plate. Previously stored records
retain their original observation times; earlier missing vehicles cannot be recovered
without processing the video again.
`video_timestamp` is the best reading's position in seconds: frame index / FPS
for files (30 FPS fallback), and elapsed processing time for live sources.
It is approximate for variable-frame-rate files and buffered live streams.
Each frame's writes commit together; a database failure stops processing and is
logged. The preview runner also saves recognition records, though it disables
annotated video output. Fresh best-reading plate crops are saved as PNG images.

### Plate images

`storage.plate_dir` (normally `data/plates`) contains PNG crops from fresh OCR
results that improve the saved confidence. Filenames include session ID, vehicle
ID, and a unique suffix. SQLite `image_path` points to the crop that produced the
saved text. Cached, lower-confidence, and equal-confidence readings do not write
another image. Unreadable/unmatched plates and vehicles without plates have no
image path. Existing databases gain the nullable column automatically on startup.

Superseded images remain on disk; the record points to the newest best image.
New files are removed if the database transaction fails. A process crash between
file creation and database commit can leave an unreferenced image. Crop saving
also runs in the preview script, independently of annotated video saving.

## Phase 2: read-only FastAPI backend

Run from the repository root in your Python environment:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.api.app:app --reload
```

Open http://127.0.0.1:8000/docs to try the API. All routes and explanatory
comments are in `app/api/app.py`. Starting the server prepares/migrates the
SQLite schema but does not load models or run video processing.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Database health check |
| GET | `/vehicles` | Paginated vehicle search |
| GET | `/vehicles/{session_id}/{vehicle_id}` | Vehicle details |
| GET | `/vehicles/{session_id}/{vehicle_id}/plate-image` | Saved PNG crop |
| GET | `/sessions` | Paginated session summaries |
| GET | `/sessions/{session_id}` | Session counts and observation times |

Vehicle filters: `session_id`, `plate_text` (literal substring), `vehicle_type`,
`has_plate`, `date_from`, and `date_to` (inclusive first-seen UTC dates).
List routes accept `limit` (1?200) and `offset` (0 or greater).
Missing records/images return 404; invalid parameters return 422; database
query failures return 503. Images are restricted to the configured plate folder.

Sessions are derived from vehicle records: empty runs, processing status, and
output video links are not represented yet. Background jobs and authentication
are not implemented; the default command serves locally on 127.0.0.1.

Install `requirements-dev.txt` as well to run the API tests, then use
`python -m unittest discover -s tests`. Tests use temporary databases and images.
