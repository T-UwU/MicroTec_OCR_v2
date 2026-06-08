"""Singletons perezosos de los motores OCR."""
from .ocr_engine import OCREngine
from .field_extractor import FieldExtractor
from .validator import Validator

_ocr_engine = None
_field_extractor = None
_validator = None


def get_ocr_engine() -> OCREngine:
    """Obtiene instancia singleton del OCR Engine."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


def get_field_extractor() -> FieldExtractor:
    """Obtiene instancia singleton del Field Extractor."""
    global _field_extractor
    if _field_extractor is None:
        _field_extractor = FieldExtractor()
    return _field_extractor


def get_validator() -> Validator:
    """Obtiene instancia singleton del Validator."""
    global _validator
    if _validator is None:
        _validator = Validator()
    return _validator

