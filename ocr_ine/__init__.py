"""
OCR Avanzado para INE Mexicana — versión modular.

Arquitectura:
- config           : configuración de entorno (CUDA off, shims, Tesseract). Se importa primero.
- models           : dataclasses de datos
- paddle_loader    : carga perezosa de PaddleOCR
- preprocessor     : preprocesamiento adaptativo de imagen
- zone_extractor   : extracción de zonas de la INE
- ocr_engine       : motor OCR (PaddleOCR + Tesseract)
- field_extractor  : extracción de campos (nombre, CURP, domicilio, MRZ)
- validator        : validación cruzada frente/reverso
- factory          : singletons de los motores
- api              : funciones públicas de extracción
- legacy           : funciones de compatibilidad

Uso:
    from ocr_ine import extraer_datos_ine_frente, extraer_datos_ine_reverso
    datos = extraer_datos_ine_frente(img)
"""

# CRÍTICO: config debe ejecutarse antes de cualquier otra importación
from . import config  # noqa: F401

# API pública principal
from .api import (
    extraer_datos_ine,
    extraer_datos_ine_frente,
    extraer_datos_ine_reverso,
    validar_cruzado_ine,
    validar_ocr_mejorado,
    ocr_combinado,
)

# Factory / singletons
from .factory import get_ocr_engine, get_field_extractor, get_validator

# Modelos de datos
from .models import (
    Detection, OCRResult, NameData, AddressData, FrontData,
    MRZData, BackData, MatchResult, CorrectedName, INEData,
)

# Clases principales (para uso avanzado)
from .preprocessor import Preprocessor
from .zone_extractor import INEZoneExtractor
from .ocr_engine import OCREngine
from .field_extractor import FieldExtractor
from .validator import Validator
from .paddle_loader import is_paddleocr_available, get_paddleocr_reader

__all__ = [
    "extraer_datos_ine", "extraer_datos_ine_frente", "extraer_datos_ine_reverso",
    "validar_cruzado_ine", "validar_ocr_mejorado", "ocr_combinado",
    "get_ocr_engine", "get_field_extractor", "get_validator",
    "Detection", "OCRResult", "NameData", "AddressData", "FrontData",
    "MRZData", "BackData", "MatchResult", "CorrectedName", "INEData",
    "Preprocessor", "INEZoneExtractor", "OCREngine", "FieldExtractor", "Validator",
    "is_paddleocr_available", "get_paddleocr_reader",
]
