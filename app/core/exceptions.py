"""Domain exceptions shared across the ANPR application.

Wrap underlying library errors using ``raise OCRException(message) from exc``
to retain their cause. Log failures at the handling boundary, rather than in
exception constructors, to avoid duplicate log entries.
"""


class ANPRException(Exception):
    """Base class for application failures that callers may handle together."""


class ConfigurationException(ANPRException):
    """Application configuration is missing or invalid."""


class ModelException(ANPRException):
    """A model cannot be loaded or initialized."""


class ModelNotFoundException(ModelException):
    """A required model file does not exist."""


class VideoInputException(ANPRException):
    """A video source cannot be opened or decoded."""


class CameraException(VideoInputException):
    """A camera connection or frame capture fails."""


class DetectionException(ANPRException):
    """Vehicle or license plate inference fails."""


class TrackingException(ANPRException):
    """Vehicle tracking fails."""


class MatchingException(ANPRException):
    """Vehicle and plate association cannot be performed."""


class OCRException(ANPRException):
    """OCR initialization or text recognition fails."""


class DatabaseException(ANPRException):
    """A database connection, query, or transaction fails."""


class StorageException(ANPRException):
    """An image or video cannot be saved or retrieved."""
