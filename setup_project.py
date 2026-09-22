from pathlib import Path


# Create the project structure alongside this setup script.
PROJECT_ROOT = Path(__file__).resolve().parent


# Folder structure
folders = [
    "app",
    "app/config",
    "app/core",
    "app/detection",
    "app/tracking",
    "app/matching",
    "app/recognition",
    "app/database",
    "app/pipeline",

    "models",

    "data/videos/input",
    "data/videos/output",
    "data/plates",

    "database",

    "logs",

    "tests",
]


# Files to create
files = [
    "app/main.py",

    # Config
    "app/config/__init__.py",
    "app/config/settings.py",
    "app/config/config.yaml",

    # Core
    "app/core/__init__.py",
    "app/core/logger.py",
    "app/core/exceptions.py",
    "app/core/constants.py",

    # Detection
    "app/detection/__init__.py",
    "app/detection/vehicle_detector.py",
    "app/detection/plate_detector.py",

    # Tracking
    "app/tracking/__init__.py",
    "app/tracking/tracker.py",

    # Matching
    "app/matching/__init__.py",
    "app/matching/matcher.py",

    # OCR
    "app/recognition/__init__.py",
    "app/recognition/ocr.py",

    # Database
    "app/database/__init__.py",
    "app/database/database.py",
    "app/database/repository.py",

    # Pipeline
    "app/pipeline/__init__.py",
    "app/pipeline/anpr_pipeline.py",

    # Root files
    "models/.gitkeep",

    "database/.gitkeep",

    "logs/.gitkeep",

    "data/plates/.gitkeep",

    "requirements.txt",
    ".env",
    ".gitignore",
    "README.md",
]


def create_project():

    root = PROJECT_ROOT

    # Create folders
    for folder in folders:
        path = root / folder
        path.mkdir(
            parents=True,
            exist_ok=True
        )


    # Create files
    for file in files:
        path = root / file

        if not path.exists():
            path.touch()


    print("✅ ANPR Project structure created successfully!")
    print(f"Location: {root.absolute()}")


if __name__ == "__main__":
    create_project()
