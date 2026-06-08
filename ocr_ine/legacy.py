"""Funciones legacy para compatibilidad con scripts existentes."""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from .models import OCRResult, Detection
from .factory import get_field_extractor, get_ocr_engine
from .preprocessor import Preprocessor

def extraer_curp(texto: str) -> Optional[str]:
    """Función legacy para extraer CURP."""
    extractor = get_field_extractor()
    return extractor.extract_curp(texto)


def extraer_clave_elector(texto: str) -> Optional[str]:
    """Función legacy para extraer clave de elector."""
    extractor = get_field_extractor()
    return extractor.extract_clave_elector(texto)


def extraer_fecha_nacimiento(texto: str) -> Optional[str]:
    """Función legacy para extraer fecha de nacimiento."""
    extractor = get_field_extractor()
    return extractor.extract_fecha_nacimiento(texto)


def extraer_sexo(texto: str) -> Optional[str]:
    """Función legacy para extraer sexo."""
    extractor = get_field_extractor()
    return extractor.extract_sexo(texto)


def extraer_mrz(texto: str) -> Dict:
    """Función legacy para extraer MRZ."""
    extractor = get_field_extractor()
    mrz = extractor.extract_mrz(texto)
    return {
        "lineas_raw": mrz.lineas_raw,  # Líneas raw del OCR (pueden estar contaminadas)
        "lineas_clean": mrz.lineas_clean,  # MRZ limpio validado (solo desde IDMEX...)
        "documento_tipo": mrz.documento_tipo,
        "pais": mrz.pais,
        "numero_documento": mrz.numero_documento,
        "nombre_completo": mrz.nombre_completo,
        "fecha_nacimiento": mrz.fecha_nacimiento,
        "sexo": mrz.sexo,
        "fecha_expiracion": mrz.fecha_expiracion
    }


def extraer_nombre_completo(texto: str, detections: List[dict] = None) -> Dict:
    """Función legacy para extraer nombre."""
    extractor = get_field_extractor()
    
    # Convertir detections a formato Detection si es necesario
    det_list = detections or []
    
    # Crear OCRResult mock
    ocr_result = OCRResult(
        combined_text=texto,
        detections=[Detection(text=d.get("text", ""), 
                             confidence=d.get("confidence", 0),
                             bbox=d.get("bbox", [])) for d in det_list],
        confidence=0.0,
        engine="legacy"
    )
    
    nombre = extractor.extract_name(texto, det_list)
    return {
        "apellido_paterno": nombre.apellido_paterno,
        "apellido_materno": nombre.apellido_materno,
        "nombre": nombre.nombre,
        "nombre_completo": nombre.nombre_completo
    }


def extraer_domicilio(texto: str) -> Dict:
    """Función legacy para extraer domicilio."""
    extractor = get_field_extractor()
    address = extractor.parse_address(texto)
    return {
        "calle": address.calle,
        "numero_exterior": address.numero_exterior,
        "colonia": address.colonia,
        "codigo_postal": address.codigo_postal,
        "municipio": address.municipio,
        "estado": address.estado,
        "domicilio_completo": address.domicilio_completo
    }

def corregir_curp_ocr(curp: str) -> str:
    """Función legacy para corregir errores OCR en CURP."""
    extractor = get_field_extractor()
    return extractor._correct_curp_ocr(curp)


def validar_digito_verificador_curp(curp: str) -> bool:
    """Función legacy para validar dígito verificador de CURP."""
    extractor = get_field_extractor()
    return extractor._validate_curp_checksum(curp)


def ocr_rapido_paddleocr(img: np.ndarray) -> str:
    """
    Función legacy para OCR rápido con PaddleOCR.
    Retorna solo el texto combinado.
    """
    engine = get_ocr_engine()
    result = engine.run_ocr(img)
    return result.combined_text


def calcular_confianza_ocr(texto: str, detections: List[dict]) -> float:
    """
    Calcula una métrica de confianza del OCR.
    Función legacy para compatibilidad.
    """
    score = 0.0
    
    # Longitud del texto (máx 30 puntos)
    score += min(len(texto) / 100, 0.3) * 100
    
    # Confianza promedio de detecciones (máx 40 puntos)
    if detections:
        avg_conf = np.mean([d.get("confidence", 0) for d in detections])
        score += avg_conf * 40
    
    # Presencia de campos clave (máx 30 puntos)
    campos_clave = ["NOMBRE", "CURP", "ELECTOR", "DOMICILIO", "VIGENCIA", 
                    "NACIMIENTO", "INSTITUTO", "NACIONAL", "ELECTORAL", "IDMEX"]
    campos_encontrados = sum(1 for campo in campos_clave if campo in texto.upper())
    score += (campos_encontrados / len(campos_clave)) * 30
    
    return min(round(score, 2), 100.0)


# Funciones de preprocesamiento legacy
def corregir_orientacion(img: np.ndarray) -> np.ndarray:
    """Función legacy para corregir orientación."""
    preprocessor = Preprocessor()
    return preprocessor.correct_orientation(img)


def mejorar_contraste_clahe(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Función legacy para mejorar contraste con CLAHE."""
    preprocessor = Preprocessor()
    return preprocessor.enhance_contrast(img, clip_limit)


def aumentar_resolucion(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Función legacy para aumentar resolución."""
    h, w = img.shape[:2]
    new_size = (int(w * factor), int(h * factor))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)


def binarizar_adaptativo(gray: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
    """Función legacy para binarización adaptativa."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )


def binarizar_otsu(gray: np.ndarray) -> np.ndarray:
    """Función legacy para binarización Otsu."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
