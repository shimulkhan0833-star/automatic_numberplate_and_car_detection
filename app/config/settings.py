"""Configuration values loaded from config.yaml."""

from pathlib import Path

import yaml


CONFIG_PATH = Path("app/config/config.yaml")

with open(CONFIG_PATH, encoding="utf-8-sig") as file:
    config = yaml.safe_load(file)


VEHICLE_MODEL_PATH = config["detection"]["vehicle_model"]
PLATE_MODEL_PATH = config["detection"]["plate_model"]
VEHICLE_CONFIDENCE = config["detection"]["vehicle_confidence"]
PLATE_CONFIDENCE = config["detection"]["plate_confidence"]
IOU_THRESHOLD = config["detection"]["iou_threshold"]
DEVICE = config["detection"]["device"]

TRACKER = config["tracking"]["tracker"]

MATCHING_METHOD = config["matching"]["method"]
MATCHING_IOU_THRESHOLD = config["matching"]["iou_threshold"]

VIDEO_SOURCE = config["video"]["source"]
OUTPUT_DIR = config["video"]["output_dir"]
SAVE_OUTPUT = config["video"]["save_output"]

OCR_LANGUAGE = config["ocr"]["language"]
OCR_CONFIDENCE = config["ocr"]["confidence"]
OCR_DEVICE = config["ocr"]["device"]
OCR_DETECTION_MODEL = config["ocr"]["detection_model"]
OCR_RECOGNITION_MODEL = config["ocr"]["recognition_model"]
OCR_RETRY_FRAMES = config["ocr"]["retry_frames"]
OCR_REFRESH_FRAMES = config["ocr"]["refresh_frames"]
OCR_CACHE_TTL_FRAMES = config["ocr"]["cache_ttl_frames"]

PLATE_DIR = config["storage"]["plate_dir"]
DATABASE_PATH = config["database"]["path"]

LOG_FILE = config["logging"]["log_file"]
LOG_LEVEL = config["logging"]["level"]
LOG_MAX_BYTES = config["logging"]["max_bytes"]
LOG_BACKUP_COUNT = config["logging"]["backup_count"]
LOG_CONSOLE = config["logging"]["console"]
