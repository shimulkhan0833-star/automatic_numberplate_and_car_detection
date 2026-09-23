"""PaddleOCR 3.x adapter for cropped license plates."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
import numpy as np
from app.core.exceptions import OCRException
from app.core.logger import get_logger
from app.config import settings
logger = get_logger(__name__)

@dataclass(frozen=True)
class OCRResult:
    """Recognized text and mean confidence of accepted text segments."""
    text: str
    confidence: float

class OCREngine:
    """Initialize OCR once; return None for unreadable plate crops."""
    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                # Explicit model names determine the recognition language.
                device=settings.OCR_DEVICE,
                text_detection_model_name=settings.OCR_DETECTION_MODEL,
                text_recognition_model_name=settings.OCR_RECOGNITION_MODEL,
                # Avoid the PaddlePaddle 3.3.1 oneDNN/PIR CPU inference crash.
                enable_mkldnn=False,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as exc:
            raise OCRException('Could not initialize PaddleOCR') from exc
        logger.info('OCR initialized (language=%s)', settings.OCR_LANGUAGE)

    def recognize(self, crop: np.ndarray) -> OCRResult | None:
        """Read BGR uint8 pixels, filtering segments by configured confidence."""
        if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0 or crop.dtype != np.uint8:
            return None
        try:
            segments = []
            for result in self._engine.predict(crop):
                for text, score in zip(result['rec_texts'], result['rec_scores'], strict=True):
                    text, score = str(text).strip(), float(score)
                    if text and isfinite(score) and settings.OCR_CONFIDENCE <= score <= 1:
                        segments.append((text, score))
            if not segments:
                return None
            return OCRResult(' '.join(text for text, _ in segments),
                             sum(score for _, score in segments) / len(segments))
        except Exception as exc:
            raise OCRException('Plate text recognition failed') from exc
