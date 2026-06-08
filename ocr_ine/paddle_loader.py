"""Carga perezosa y verificación de PaddleOCR."""
import os
import logging

_paddleocr_reader = None
_paddleocr_available = None

# Configuración del modelo (ajustable para A/B). lang='es'→latin v3 rec.
OCR_LANG = 'es'
OCR_VERSION = None  # None=default; "PP-OCRv4" para forzar v4


def reset_reader():
    """Fuerza recarga del reader (para cambiar de modelo en runtime)."""
    global _paddleocr_reader
    _paddleocr_reader = None


def is_paddleocr_available() -> bool:
    """Verifica si PaddleOCR está instalado SIN IMPORTARLO."""
    global _paddleocr_available
    
    if _paddleocr_available is not None:
        return _paddleocr_available
    
    _paddleocr_available = False
    
    try:
        import importlib.util
        spec_paddle = importlib.util.find_spec("paddle")
        spec_paddleocr = importlib.util.find_spec("paddleocr")
        _paddleocr_available = spec_paddle is not None and spec_paddleocr is not None
    except Exception:
        _paddleocr_available = False
    
    return _paddleocr_available


def get_paddleocr_reader():
    """Carga PaddleOCR de forma lazy."""
    global _paddleocr_reader
    
    if _paddleocr_reader == "unavailable":
        return None
    
    if _paddleocr_reader is not None:
        return _paddleocr_reader
    
    if not is_paddleocr_available():
        _paddleocr_reader = "unavailable"
        return None
    
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['FLAGS_use_cuda'] = '0'
        os.environ['FLAGS_use_gpu'] = '0'
        
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        logging.getLogger('paddle').setLevel(logging.ERROR)
        
        from paddleocr import PaddleOCR

        print("Cargando PaddleOCR (PP-OCRv5)...")
        # API 3.x: desactivar módulos de documento (orientación/unwarping) que
        # no aplican a recortes de INE y sólo añaden latencia.
        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=OCR_LANG,
        )
        if OCR_VERSION:
            kwargs['ocr_version'] = OCR_VERSION
        _paddleocr_reader = PaddleOCR(**kwargs)
        print("PaddleOCR cargado exitosamente")
        return _paddleocr_reader
        
    except Exception as e:
        print(f"Error cargando PaddleOCR: {e}")
        _paddleocr_reader = "unavailable"
        return None


