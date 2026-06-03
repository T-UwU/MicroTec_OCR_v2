"""
OCR Avanzado para INE Mexicana - v2.0
=====================================
Módulo refactorizado con arquitectura modular para máxima extracción de datos 
de credenciales INE usando PaddleOCR, con enfoque en validación cruzada frente-reverso.

Configurado para CPU (sin GPU) - compatible con Debian 12 x86_64 y macOS.

Arquitectura:
- Preprocessor: Preprocesamiento adaptativo de imagen
- OCREngine: Motor OCR con múltiples pasadas
- FieldExtractor: Extracción de campos específicos
- Validator: Validación cruzada y corrección de datos
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

# ============================================================================
# CONFIGURACIÓN CRÍTICA: Suprimir CUDA/GPU ANTES de cualquier importación
# ============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FLAGS_use_cuda'] = '0'
os.environ['FLAGS_use_gpu'] = '0'
os.environ['PADDLEOCR_USE_GPU'] = '0'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Configurar ruta de Tesseract en Windows si no está en PATH
import platform
if platform.system() == 'Windows':
    import shutil
    if not shutil.which('tesseract'):
        _tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        import os as _os
        if _os.path.isfile(_tess_path):
            pytesseract.pytesseract.tesseract_cmd = _tess_path
import re
from difflib import SequenceMatcher

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Shim de compatibilidad para NumPy 2.x: np.sctypes fue eliminado en NumPy 2.0
# pero PaddlePaddle 2.x lo sigue usando internamente.
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Detection:
    """Representa una detección de texto OCR."""
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)


@dataclass
class OCRResult:
    """Resultado de OCR combinado."""
    combined_text: str
    detections: List[Detection]
    confidence: float
    engine: str  # "paddleocr" | "tesseract"


@dataclass
class NameData:
    """Datos de nombre estructurados."""
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None
    nombre: Optional[str] = None
    nombre_completo: Optional[str] = None


@dataclass
class AddressData:
    """Datos de domicilio estructurados."""
    calle: Optional[str] = None
    numero_exterior: Optional[str] = None
    numero_interior: Optional[str] = None
    colonia: Optional[str] = None
    codigo_postal: Optional[str] = None
    municipio: Optional[str] = None
    estado: Optional[str] = None
    domicilio_completo: Optional[str] = None


@dataclass
class FrontData:
    """Datos extraídos del frente de INE."""
    nombre: NameData = field(default_factory=NameData)
    sexo: Optional[str] = None
    curp: Optional[str] = None
    clave_elector: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    domicilio: AddressData = field(default_factory=AddressData)
    seccion: Optional[str] = None
    vigencia: Optional[str] = None
    anio_registro: Optional[str] = None
    anio_emision: Optional[str] = None
    tipo_ine: Optional[str] = None  # "IFE" (C/D) o "INE" (E/F/G/H)
    modelo_ine: Optional[str] = None  # Modelo específico: "C", "D", "E", "F", "G", "H"
    confianza_ocr: float = 0.0


@dataclass
class MRZData:
    """Datos extraídos del MRZ."""
    lineas_raw: List[str] = field(default_factory=list)  # Líneas raw del OCR (pueden estar contaminadas)
    lineas_clean: Optional[str] = None  # MRZ limpio validado (solo desde IDMEX...)
    documento_tipo: Optional[str] = None
    pais: Optional[str] = None
    numero_documento: Optional[str] = None
    nombre_completo: Optional[str] = None
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None
    nombres: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    sexo: Optional[str] = None
    fecha_expiracion: Optional[str] = None


@dataclass
class BackData:
    """Datos extraídos del reverso de INE."""
    mrz: MRZData = field(default_factory=MRZData)
    curp: Optional[str] = None
    cic: Optional[str] = None  # Código de Identificación de Credencial (9 dígitos)
    ocr_vertical: Optional[str] = None  # Identificador ciudadano (13 dígitos)
    confianza_ocr: float = 0.0


@dataclass
class MatchResult:
    """Resultado de validación cruzada frente/reverso."""
    porcentaje_match: int  # 0-100
    resultado: str  # "MATCH_CONFIABLE" | "MATCH_PARCIAL" | "NO_MATCH"
    mensaje: str
    validaciones_detalle: List[dict] = field(default_factory=list)
    puede_aprobar_automatico: bool = False


@dataclass
class CorrectedName:
    """Nombre corregido usando MRZ como ground truth."""
    nombre_completo: str
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None
    nombres: Optional[str] = None
    fuente: str = "Frontal"  # "MRZ" | "Frontal" | "MRZ+Frontal"
    confianza: float = 0.0
    correccion_aplicada: bool = False


@dataclass
class INEData:
    """Datos completos de INE."""
    tipo: str  # "INE_FRENTE" | "INE_REVERSO"
    campos_frente: Optional[FrontData] = None
    campos_reverso: Optional[BackData] = None
    validacion_match: Optional[MatchResult] = None
    texto_crudo: str = ""
    confianza_ocr: float = 0.0


# ============================================================================
# PADDLEOCR - VERIFICACIÓN SEGURA
# ============================================================================
_paddleocr_reader = None
_paddleocr_available = None


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
        
        print("Cargando PaddleOCR...")
        _paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='es')
        print("PaddleOCR cargado exitosamente")
        return _paddleocr_reader
        
    except Exception as e:
        print(f"Error cargando PaddleOCR: {e}")
        _paddleocr_reader = "unavailable"
        return None


# ============================================================================
# PREPROCESSOR - Preprocesamiento Adaptativo de Imagen
# ============================================================================

class Preprocessor:
    """
    Preprocesador de imágenes para OCR óptimo.
    OPTIMIZADO: Genera solo variantes esenciales para mejor performance.
    """
    
    MIN_SIZE = 900  # 900px: balance óptimo velocidad/calidad (vs 1200 orig); 1.5-2.5x más rápido en reconocimiento
    INE_ASPECT_RATIO = 1.58  # Ratio de aspecto de INE
    INE_ASPECT_TOLERANCE = 0.2  # Tolerancia para ratio de aspecto
    
    def preprocess(self, img: np.ndarray, fast_mode: bool = False) -> List[Tuple[str, np.ndarray]]:
        """
        Genera variantes preprocesadas de la imagen.
        OPTIMIZADO: En fast_mode solo genera 2 variantes esenciales.
        
        Args:
            img: Imagen a preprocesar
            fast_mode: Si True, genera solo variantes mínimas (default: False)
        
        Returns:
            Lista de tuplas (nombre_variante, imagen_procesada)
        """
        results = []
        
        # 0. Extraer región de tarjeta (SKIP en fast_mode - muy costoso)
        if fast_mode:
            img_card = img
            card_was_cropped = False
        else:
            img_card, card_was_cropped = self.extract_card_region(img)
        
        # 1. Corrección de orientación SIMPLIFICADA (sin OCR)
        img_oriented = self._correct_orientation_fast(img_card)
        
        # 2. Corrección de perspectiva (SKIP en fast_mode)
        if card_was_cropped or fast_mode:
            img_perspective = img_oriented
        else:
            img_perspective = self._correct_perspective_fast(img_oriented)
        
        # 3. Escalar si es necesario
        img_scaled = self.scale_image(img_perspective)
        
        # 4. Convertir a escala de grises
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY) if len(img_scaled.shape) == 3 else img_scaled
        
        # Variante 1: Original escalado (siempre)
        results.append(("original", gray))
        
        # Variante 2: CLAHE (siempre - muy efectivo)
        img_clahe = self.enhance_contrast(img_scaled)
        gray_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY) if len(img_clahe.shape) == 3 else img_clahe
        results.append(("clahe", gray_clahe))
        
        # En fast_mode, solo 2 variantes
        if fast_mode:
            return results
        
        # Variante 3: Binarizado adaptativo (solo si no es fast_mode)
        binary_adaptive = self._binarize_adaptive(gray_clahe)
        results.append(("binary_adaptive", binary_adaptive))
        
        return results
    
    def _correct_orientation_fast(self, img: np.ndarray) -> np.ndarray:
        """
        Corrección de orientación RÁPIDA sin usar OCR.
        Solo detecta rotaciones de 90° usando análisis de bordes.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            h, w = gray.shape[:2]
            
            # INE es horizontal (w > h). Si h > w, rotar 90°
            if h > w * 1.2:  # Claramente vertical
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            # Detectar si está al revés usando gradientes (texto va de izq a der)
            # Esto es heurístico pero muy rápido
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            left_half = np.abs(edges[:, :w//2]).mean()
            right_half = np.abs(edges[:, w//2:]).mean()
            
            # Si hay más actividad en la derecha, podría estar al revés
            # Pero esto es muy poco confiable, mejor no rotar
            return img
        except:
            return img
    
    def _correct_perspective_fast(self, img: np.ndarray) -> np.ndarray:
        """
        Corrección de perspectiva RÁPIDA sin validación OCR.
        Solo aplica si detecta claramente una tarjeta inclinada.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 75, 200)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            
            h, w = img.shape[:2]
            img_area = h * w
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Solo procesar si el contorno es grande (>30% de la imagen)
                if area < img_area * 0.3:
                    continue
                
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    rect = self._order_points(pts.astype("float32"))
                    
                    width = max(
                        np.linalg.norm(rect[0] - rect[1]),
                        np.linalg.norm(rect[2] - rect[3])
                    )
                    height = max(
                        np.linalg.norm(rect[0] - rect[3]),
                        np.linalg.norm(rect[1] - rect[2])
                    )
                    
                    aspect = width / height if height > 0 else 0
                    if abs(aspect - self.INE_ASPECT_RATIO) < self.INE_ASPECT_TOLERANCE:
                        dst = np.array([
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1]
                        ], dtype="float32")
                        
                        M = cv2.getPerspectiveTransform(rect, dst)
                        return cv2.warpPerspective(img, M, (int(width), int(height)))
        except:
            pass
        return img
    
    def extract_card_region(self, img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detecta y extrae solo la región de la tarjeta INE, eliminando el fondo.
        Usa múltiples métodos para robustez.
        
        Args:
            img: Imagen original que puede contener fondo
            
        Returns:
            Tupla (imagen recortada, fue_recortada) - fue_recortada indica si se aplicó warp
        """
        h, w = img.shape[:2]
        min_card_area = h * w * 0.15  # La tarjeta debe ser al menos 15% de la imagen
        max_card_area = h * w * 0.98  # No más del 98% (debe haber algo de fondo)
        
        best_card = None
        best_score = 0
        
        # Método 1: Detección por contornos (bordes de la tarjeta)
        card_contour = self._detect_card_by_contours(img, min_card_area, max_card_area)
        if card_contour is not None:
            score = self._score_card_candidate(card_contour, h, w)
            if score > best_score:
                best_score = score
                best_card = card_contour
        
        # Método 2: Detección por color (la INE es mayormente blanca/beige)
        card_color = self._detect_card_by_color(img, min_card_area, max_card_area)
        if card_color is not None:
            score = self._score_card_candidate(card_color, h, w)
            if score > best_score:
                best_score = score
                best_card = card_color
        
        # Si encontramos una tarjeta válida, recortar
        if best_card is not None and best_score > 0.5:
            cropped, warp_ok = self._crop_card(img, best_card)
            if warp_ok:
                return cropped, True
        
        # Fallback: devolver imagen original
        return img, False
    
    def _detect_card_by_contours(self, img: np.ndarray, min_area: float, max_area: float) -> Optional[np.ndarray]:
        """Detecta la tarjeta por contornos (bordes)."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Blur para reducir ruido
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detección de bordes
            edges = cv2.Canny(blur, 50, 150)
            
            # Dilatar para conectar bordes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Buscar contorno rectangular con ratio de INE
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                
                # Aproximar a polígono
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Debe ser cuadrilátero
                if len(approx) == 4:
                    # Verificar ratio de aspecto
                    rect = cv2.minAreaRect(contour)
                    w_rect, h_rect = rect[1]
                    if w_rect > 0 and h_rect > 0:
                        aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
                        if 1.3 < aspect < 1.9:  # Rango de INE
                            return approx.reshape(4, 2)
            
            return None
        except:
            return None
    
    def _detect_card_by_color(self, img: np.ndarray, min_area: float, max_area: float) -> Optional[np.ndarray]:
        """Detecta la tarjeta por color (INE es mayormente blanca/beige)."""
        try:
            # Convertir a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Rango para colores claros (blanco, beige, gris claro)
            # H: cualquier, S: bajo (poco saturado), V: alto (brillante)
            lower = np.array([0, 0, 180])
            upper = np.array([180, 60, 255])
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # Operaciones morfológicas para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                
                # Verificar ratio de aspecto
                rect = cv2.minAreaRect(contour)
                w_rect, h_rect = rect[1]
                if w_rect > 0 and h_rect > 0:
                    aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
                    if 1.3 < aspect < 1.9:
                        # Obtener los 4 puntos del rectángulo
                        box = cv2.boxPoints(rect)
                        return box.astype(np.int32)
            
            return None
        except:
            return None
    
    def _score_card_candidate(self, points: np.ndarray, img_h: int, img_w: int) -> float:
        """
        Calcula un score para un candidato a tarjeta.
        Mayor score = mejor candidato.
        """
        try:
            # Calcular área del candidato
            rect = cv2.minAreaRect(points)
            w_rect, h_rect = rect[1]
            area = w_rect * h_rect
            img_area = img_h * img_w
            
            # Score basado en:
            # 1. Proporción del área (ideal: 30-80% de la imagen)
            area_ratio = area / img_area
            if area_ratio < 0.15 or area_ratio > 0.95:
                return 0.0
            
            area_score = 1.0 - abs(area_ratio - 0.5)  # Mejor si está cerca del 50%
            
            # 2. Ratio de aspecto (ideal: ~1.58 para INE)
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect) if min(w_rect, h_rect) > 0 else 0
            aspect_score = 1.0 - min(abs(aspect - self.INE_ASPECT_RATIO) / 0.5, 1.0)
            
            # 3. Centrado (mejor si está cerca del centro)
            center = rect[0]
            center_dist = np.sqrt((center[0] - img_w/2)**2 + (center[1] - img_h/2)**2)
            max_dist = np.sqrt((img_w/2)**2 + (img_h/2)**2)
            center_score = 1.0 - (center_dist / max_dist)
            
            # Score final ponderado
            return 0.3 * area_score + 0.5 * aspect_score + 0.2 * center_score
        except:
            return 0.0
    
    def _crop_card(self, img: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Recorta la imagen usando los 4 puntos de la tarjeta.
        
        Returns:
            Tupla (imagen_resultado, warp_exitoso)
            - Si warp_exitoso=True, imagen_resultado es la tarjeta recortada
            - Si warp_exitoso=False, imagen_resultado es la imagen original
        """
        MIN_DIMENSION = 300  # Tamaño mínimo para considerar warp válido
        
        try:
            # Ordenar puntos
            rect = self._order_points(points.astype("float32"))
            
            # Calcular dimensiones del rectángulo destino
            width = max(
                np.linalg.norm(rect[0] - rect[1]),
                np.linalg.norm(rect[2] - rect[3])
            )
            height = max(
                np.linalg.norm(rect[0] - rect[3]),
                np.linalg.norm(rect[1] - rect[2])
            )
            
            # Validar dimensiones antes de warp
            if width < MIN_DIMENSION or height < MIN_DIMENSION:
                return img, False
            
            # Asegurar que width > height (INE es horizontal)
            # Si height > width, necesitamos reordenar los puntos para rotar
            if height > width:
                width, height = height, width
                # Rotar puntos 90° para que el warp salga horizontal
                rect = np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")
            
            # Validar ratio de aspecto (INE ~1.58)
            aspect = width / height if height > 0 else 0
            if aspect < 1.3 or aspect > 1.9:
                return img, False
            
            # Puntos destino
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            # Transformación de perspectiva
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (int(width), int(height)))
            
            # Validar resultado final
            if warped.size == 0:
                return img, False
            
            h_out, w_out = warped.shape[:2]
            if h_out < MIN_DIMENSION or w_out < MIN_DIMENSION:
                return img, False
            
            # Validar que el área del resultado sea razonable vs original
            h_orig, w_orig = img.shape[:2]
            area_ratio = (h_out * w_out) / (h_orig * w_orig)
            if area_ratio < 0.1 or area_ratio > 1.5:
                return img, False
            
            return warped, True
            
        except Exception:
            return img, False
    
    def _safe_correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica corrección de perspectiva con sanity check ligero.
        Si la corrección empeora el score de texto, conserva la original.
        Optimizado para bajo costo computacional.
        """
        # Tamaño mínimo para considerar warp válido (evita warps degenerados)
        MIN_DIMENSION = 300
        
        # Intentar corrección de perspectiva primero
        img_corrected = self.correct_perspective(img)
        
        # Verificar si realmente cambió (warpPerspective siempre devuelve array nuevo)
        # Comparar por tamaño primero (rápido)
        h_orig, w_orig = img.shape[:2]
        h_corr, w_corr = img_corrected.shape[:2]
        
        # Rechazar warp degenerado (imagen muy pequeña)
        if h_corr < MIN_DIMENSION or w_corr < MIN_DIMENSION:
            return img
        
        # Si el tamaño es exactamente igual, verificar si el contenido cambió
        if (h_orig, w_orig) == (h_corr, w_corr):
            # Comparación rápida: downscale a gray y diferencia media
            # Usar INTER_AREA para shrink (más estable)
            try:
                scale = 0.1
                small_orig = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                small_corr = cv2.resize(img_corrected, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                # Convertir ambos a gray para comparación consistente (evita diff BGR vs gray)
                if len(small_orig.shape) == 3:
                    small_orig = cv2.cvtColor(small_orig, cv2.COLOR_BGR2GRAY)
                if len(small_corr.shape) == 3:
                    small_corr = cv2.cvtColor(small_corr, cv2.COLOR_BGR2GRAY)
                
                diff = np.mean(np.abs(small_orig.astype(float) - small_corr.astype(float)))
                if diff < 1.0:  # Prácticamente idénticas
                    return img
            except:
                pass
        
        # Sanity check: Si la corrección cambió drásticamente el tamaño, probablemente falló
        area_ratio = (h_corr * w_corr) / (h_orig * w_orig)
        if area_ratio < 0.5 or area_ratio > 2.0:
            return img
        
        # Verificar ratio de aspecto (INE es ~1.58)
        aspect_corr = w_corr / h_corr if h_corr > 0 else 0
        if aspect_corr < 1.2 or aspect_corr > 2.0:  # Fuera de rango razonable para INE
            return img
        
        # Score rápido: solo verificar si hay texto legible (1 intento, imagen pequeña)
        score_original = self._calculate_text_score_fast(img)
        
        # Si original ya tiene buen score, comparar con corrección
        if score_original >= 2:
            score_corrected = self._calculate_text_score_fast(img_corrected)
            if score_corrected < score_original:
                return img
        
        return img_corrected
    
    def _calculate_text_score_fast(self, img: np.ndarray) -> float:
        """
        Versión rápida de _calculate_text_score.
        Usa imagen más pequeña y menos keywords.
        """
        try:
            h, w = img.shape[:2]
            # Escalar a máximo 400px para OCR rápido
            scale = min(400 / max(h, w), 1.0)
            small = cv2.resize(img, None, fx=scale, fy=scale)
            
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
            
            # OCR rápido con config mínima
            text = pytesseract.image_to_string(gray, lang='spa', config='--psm 6 --oem 1')
            text_upper = text.upper()
            
            # Keywords esenciales de INE (menos que la versión completa)
            keywords = ['NOMBRE', 'CURP', 'ELECTOR', 'DOMICILIO', 'INE']
            return sum(1 for kw in keywords if kw in text_upper)
        except:
            return 0
    
    def correct_orientation(self, img: np.ndarray) -> np.ndarray:
        """
        Corrige rotación de 0°, 90°, 180°, 270° y ángulos menores.
        Usa OCR rápido para detectar orientación del texto.
        """
        # Primero intentar detectar rotaciones de 90°
        img_corrected = self._detect_and_correct_90_rotation(img)
        
        # Luego corregir ángulos menores con Hough
        img_corrected = self._correct_small_angle(img_corrected)
        
        return img_corrected
    
    def _detect_and_correct_90_rotation(self, img: np.ndarray) -> np.ndarray:
        """Detecta y corrige rotaciones de 90°, 180°, 270°."""
        best_img = img
        best_score = self._calculate_text_score(img)
        
        # Probar rotaciones de 90°, 180°, 270°
        for angle in [90, 180, 270]:
            rotated = self._rotate_90(img, angle)
            score = self._calculate_text_score(rotated)
            if score > best_score:
                best_score = score
                best_img = rotated
        
        return best_img
    
    def _rotate_90(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rota imagen en múltiplos de 90°."""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img
    
    def _calculate_text_score(self, img: np.ndarray) -> float:
        """
        Calcula un score de orientación basado en detección de texto.
        Mayor score = mejor orientación.
        """
        try:
            # Reducir imagen para OCR rápido
            h, w = img.shape[:2]
            scale = min(500 / max(h, w), 1.0)
            small = cv2.resize(img, None, fx=scale, fy=scale)
            
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
            
            # OCR rápido con Tesseract
            try:
                text = pytesseract.image_to_string(gray, lang='spa', config='--psm 6 --oem 3')
                # Score basado en palabras clave de INE
                keywords = ['NOMBRE', 'CURP', 'ELECTOR', 'DOMICILIO', 'VIGENCIA', 
                           'INSTITUTO', 'NACIONAL', 'ELECTORAL', 'IDMEX', 'MEX']
                score = sum(1 for kw in keywords if kw in text.upper())
                return score
            except:
                return 0
        except:
            return 0
    
    def _correct_small_angle(self, img: np.ndarray) -> np.ndarray:
        """Corrige ángulos menores a 45° usando detección de líneas Hough."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                     minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 15:  # Solo ángulos pequeños
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.5:  # Solo si hay desviación significativa
                        h, w = img.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), 
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass
        return img
    
    def correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """
        Corrige perspectiva detectando los 4 bordes de la INE.
        Valida ratio de aspecto (~1.58).
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 75, 200)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    rect = self._order_points(pts)
                    
                    width = max(
                        np.linalg.norm(rect[0] - rect[1]),
                        np.linalg.norm(rect[2] - rect[3])
                    )
                    height = max(
                        np.linalg.norm(rect[0] - rect[3]),
                        np.linalg.norm(rect[1] - rect[2])
                    )
                    
                    aspect = width / height if height > 0 else 0
                    if abs(aspect - self.INE_ASPECT_RATIO) < self.INE_ASPECT_TOLERANCE:
                        dst = np.array([
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1]
                        ], dtype="float32")
                        
                        M = cv2.getPerspectiveTransform(rect, dst)
                        return cv2.warpPerspective(img, M, (int(width), int(height)))
        except Exception:
            pass
        return img
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Ordena 4 puntos: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def enhance_contrast(self, img: np.ndarray, clip_limit: float = None) -> np.ndarray:
        """
        Aplica CLAHE con parámetros adaptativos según histograma.
        """
        if clip_limit is None:
            clip_limit = self._calculate_optimal_clip_limit(img)
        
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(img)
    
    def _calculate_optimal_clip_limit(self, img: np.ndarray) -> float:
        """Calcula clip_limit óptimo basado en histograma."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Calcular contraste actual
        std = np.std(gray)
        
        # Ajustar clip_limit según contraste
        if std < 30:  # Bajo contraste
            return 4.0
        elif std < 50:  # Contraste medio
            return 3.0
        else:  # Alto contraste
            return 2.0
    
    def scale_image(self, img: np.ndarray, max_size: int = None) -> np.ndarray:
        """
        Escala imagen al tamaño óptimo para OCR.
        - Upscale si es menor a MIN_SIZE (usa INTER_CUBIC)
        - Downscale si es mayor a max_size (usa INTER_AREA)
        """
        h, w = img.shape[:2]
        max_dim = max(h, w)
        
        # Upscale si es muy pequeña
        if max_dim < self.MIN_SIZE:
            scale = self.MIN_SIZE / max_dim
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Downscale si es muy grande (opcional, para performance)
        if max_size and max_dim > max_size:
            scale = max_size / max_dim
            new_size = (int(w * scale), int(h * scale))
            # INTER_AREA es mejor para shrink (evita aliasing)
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        return img
    
    def _binarize_adaptive(self, gray: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
        """Binarización adaptativa."""
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, C
        )
    
    def _binarize_otsu(self, gray: np.ndarray) -> np.ndarray:
        """Binarización con método Otsu."""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def _high_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Genera versión de alto contraste."""
        # Normalizar a rango completo
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # Aplicar gamma correction para aumentar contraste
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(normalized, table)
    
    def _anti_reflejo(self, img: np.ndarray) -> np.ndarray:
        """
        Reduce reflejos usando recorte de V (valor) en HSV.
        Aplica blur suave solo para reducir reflejos sin perder detalles finos.
        
        Args:
            img: Imagen en BGR o escala de grises
            
        Returns:
            Imagen con reflejos reducidos
        """
        try:
            # Convertir a HSV si es color
            if len(img.shape) == 3:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
            else:
                # Si es escala de grises, usar como V
                v = img.copy()
            
            # Recortar valores altos (reflejos) en el canal V
            # Valores > 240 se consideran reflejos
            v_clipped = np.clip(v, 0, 240)
            
            # Aplicar blur muy suave (medianBlur 3x3) solo para suavizar transiciones
            v_blurred = cv2.medianBlur(v_clipped, 3)
            
            # Re-escalar para usar todo el rango dinámico
            v_normalized = cv2.normalize(v_blurred, None, 0, 255, cv2.NORM_MINMAX)
            
            # Reconstruir imagen
            if len(img.shape) == 3:
                hsv_processed = cv2.merge([h, s, v_normalized])
                return cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2BGR)
            else:
                return v_normalized
        except Exception:
            # Si falla, retornar original
            return img


# ============================================================================
# INE ZONE EXTRACTOR - Extracción por zonas específicas
# ============================================================================

class INEZoneExtractor:
    """
    Extrae zonas específicas de la INE para OCR más preciso.
    Las zonas están basadas en la estructura estándar de INE mexicana.
    """
    
    # Zonas del frente de INE (coordenadas relativas 0-1)
    # Formato: (x_start, y_start, x_end, y_end)
    # IMPORTANTE: nombre y domicilio NO deben solaparse
    # NOTA: Las coordenadas están calibradas para INE modelo E/F/G/H
    # AJUSTE: zona nombre expandida verticalmente para capturar nombres de 3 líneas
    # NOTA: y_start=0.20 para capturar desde justo debajo del header
    # AJUSTE v4: y_end=0.50 para capturar nombres de 3+ líneas (apellido paterno, materno, nombre)
    # La INE tiene el nombre en ~18-48% del alto, DOMICILIO empieza ~50%
    FRONT_ZONES = {
        'foto': (0.03, 0.18, 0.33, 0.72),      # Foto del titular (izquierda)
        'nombre': (0.32, 0.20, 0.78, 0.50),    # Zona de nombre (expandida: y_start=0.20, y_end=0.50)
        'domicilio': (0.32, 0.50, 0.95, 0.60), # Zona de domicilio (empieza donde termina nombre)
        'datos': (0.32, 0.60, 0.70, 0.85),     # CURP, Clave elector, etc. (expandida hacia abajo)
        'fechas': (0.68, 0.60, 0.95, 0.82),    # Fechas y sección
        'inferior': (0.03, 0.82, 0.95, 0.95),  # Vigencia y año registro
    }
    
    # Márgenes por zona (en píxeles) - ajustados para mejor captura
    ZONE_MARGINS = {
        'nombre': 10,     # Margen amplio para capturar texto en bordes (nombres de 3 líneas)
        'domicilio': 5,   # Margen pequeño
        'datos': 5,
        'fechas': 5,
        'inferior': 5,
        'foto': 5,
        'mrz': 10,        # MRZ necesita más margen
        'codigo_barras': 5,
        'datos_extra': 5,
    }
    
    # Zonas del reverso de INE
    # NOTA: MRZ expandido para capturar mejor las 3 líneas
    BACK_ZONES = {
        'mrz': (0.02, 0.55, 0.98, 0.98),       # Zona MRZ (inferior) - expandida hacia arriba
        'codigo_barras': (0.02, 0.02, 0.40, 0.35),  # Código de barras
        'datos_extra': (0.40, 0.02, 0.98, 0.55),    # Datos adicionales
    }
    
    def __init__(self):
        self.preprocessor = Preprocessor()
    
    def extract_zones(self, img: np.ndarray, tipo: str = "frente") -> Dict[str, np.ndarray]:
        """
        Extrae las zonas de interés de una imagen de INE.
        Usa márgenes diferenciados por zona para evitar contaminación.
        
        Args:
            img: Imagen de INE
            tipo: "frente" o "reverso"
            
        Returns:
            Diccionario con nombre de zona -> imagen recortada
        """
        h, w = img.shape[:2]
        zones = self.FRONT_ZONES if tipo == "frente" else self.BACK_ZONES
        
        extracted = {}
        for zone_name, (x1, y1, x2, y2) in zones.items():
            # Convertir coordenadas relativas a absolutas
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            
            # Usar margen diferenciado por zona (0 para zonas críticas)
            margin = self.ZONE_MARGINS.get(zone_name, 5)
            px1 = max(0, px1 - margin)
            py1 = max(0, py1 - margin)
            px2 = min(w, px2 + margin)
            py2 = min(h, py2 + margin)
            
            zone_img = img[py1:py2, px1:px2]
            
            if zone_img.size > 0:
                extracted[zone_name] = zone_img
        
        return extracted
    
    def extract_mrz_zone(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae específicamente la zona MRZ del reverso.
        Usa detección de líneas para encontrar el MRZ con precisión.
        Aplica inversión automática si el texto es más claro que el fondo.
        """
        h, w = img.shape[:2]
        
        # MRZ está en la mitad inferior - expandir zona para capturar mejor
        mrz_region = img[int(h * 0.50):h, :]
        
        # Si la región es muy pequeña, retornar la región completa
        if mrz_region.shape[0] < 50 or mrz_region.shape[1] < 100:
            return mrz_region
        
        # Preprocesar para mejor detección
        gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY) if len(mrz_region.shape) == 3 else mrz_region
        
        # Aplicar CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Inversión automática: si la media > 127, el fondo es claro y texto oscuro
        # En ese caso THRESH_BINARY funciona bien. Si media <= 127, invertimos.
        mean_val = np.mean(gray)
        if mean_val <= 127:
            # Fondo oscuro, texto claro -> invertir para que texto quede oscuro
            gray_for_thresh = cv2.bitwise_not(gray)
        else:
            gray_for_thresh = gray
        
        # Binarizar con Otsu
        _, binary = cv2.threshold(gray_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Buscar líneas horizontales de texto (características del MRZ)
        # Clamp del kernel para estabilidad en imágenes de diferentes tamaños
        mrz_w = mrz_region.shape[1]
        kernel_w = max(50, min(mrz_w // 4, 250))  # Entre 50 y 250 px
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el contorno más grande (probablemente el MRZ)
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            
            # Expandir un poco el recorte
            margin = 15
            y1 = max(0, y - margin)
            y2 = min(mrz_region.shape[0], y + ch + margin)
            x1 = max(0, x - margin)
            x2 = min(mrz_region.shape[1], x + cw + margin)
            
            # Sanity check: verificar que el recorte tenga tamaño razonable
            if (y2 - y1) < 30 or (x2 - x1) < 100:
                return mrz_region
            
            return mrz_region[y1:y2, x1:x2]
        
        # Si no encontramos contornos, retornar la región completa
        return mrz_region
    
    def get_zone_ocr_config(self, zone_name: str) -> dict:
        """
        Retorna configuración OCR optimizada para cada zona.
        Incluye whitelist para Tesseract cuando aplica.
        """
        configs = {
            'nombre': {'psm': 6, 'oem': 1, 'whitelist': 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ '},
            'domicilio': {'psm': 6, 'oem': 1, 'whitelist': 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ0123456789 .,-#'},
            'datos': {'psm': 6, 'oem': 1},  # CURP, clave elector
            'fechas': {'psm': 6, 'oem': 1, 'whitelist': '0123456789/-'},
            'mrz': {'psm': 6, 'oem': 1, 'whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'},
            'inferior': {'psm': 6, 'oem': 1},
        }
        return configs.get(zone_name, {'psm': 6, 'oem': 1})


# ============================================================================
# OCR ENGINE - Motor OCR con múltiples pasadas
# ============================================================================

class OCREngine:
    """
    Motor OCR que ejecuta PaddleOCR con múltiples pasadas y combina resultados.
    Soporta OCR por zonas para mayor precisión.
    """
    
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.zone_extractor = INEZoneExtractor()
        self._paddle_reader = None
        self._low_confidence_threshold = 0.7  # Umbral para usar cls
        self._ocr_cache = {}  # Cache para evitar reprocesamiento
    
    def run_ocr(self, img: np.ndarray, fast_mode: bool = True) -> OCRResult:
        """
        Ejecuta OCR OPTIMIZADO con early-exit.
        
        Args:
            img: Imagen a procesar
            fast_mode: Si True, usa menos variantes y sale temprano si hay buenos resultados
        """
        # Generar variantes preprocesadas (menos en fast_mode)
        variants = self.preprocessor.preprocess(img, fast_mode=fast_mode)
        
        all_detections = []
        all_texts = []
        primary_engine = "tesseract"
        
        # Ejecutar PaddleOCR
        if is_paddleocr_available():
            # OPTIMIZACIÓN: Solo procesar primera variante inicialmente
            first_variant_name, first_variant_img = variants[0]
            text, detections = self._run_paddleocr(first_variant_img, use_cls=False)
            
            if text.strip():
                all_texts.append({
                    "source": f"paddleocr_{first_variant_name}",
                    "text": text.strip(),
                    "weight": self._get_variant_weight(first_variant_name)
                })
                all_detections.extend(detections)
                primary_engine = "paddleocr"
                
                # EARLY EXIT: Si primera variante tiene buena confianza, no procesar más
                if detections:
                    avg_conf = float(np.mean([d.confidence for d in detections]))
                    if avg_conf >= 0.8 and len(text.strip()) > 50:
                        # Resultado suficientemente bueno, salir temprano
                        return OCRResult(
                            combined_text=text.strip(),
                            detections=detections,
                            confidence=avg_conf,
                            engine=primary_engine
                        )
            
            # Si primera variante no fue suficiente, probar segunda (CLAHE)
            if len(variants) > 1 and (not all_texts or len(all_detections) < 10):
                second_name, second_img = variants[1]
                text2, detections2 = self._run_paddleocr(second_img, use_cls=False)
                if text2.strip():
                    all_texts.append({
                        "source": f"paddleocr_{second_name}",
                        "text": text2.strip(),
                        "weight": self._get_variant_weight(second_name)
                    })
                    all_detections.extend(detections2)
                    primary_engine = "paddleocr"
            
            # Solo usar cls si realmente es necesario (confianza muy baja)
            if all_detections:
                temp_filtered = self.filter_duplicates(all_detections)
                if temp_filtered:
                    top_k_conf = sorted([d.confidence for d in temp_filtered], reverse=True)[:10]
                    avg_conf = float(np.mean(top_k_conf))
                else:
                    avg_conf = 0.0
            else:
                avg_conf = 0.0
            
            # Solo retry con cls si confianza es MUY baja
            if avg_conf < 0.5 and not all_texts:
                # Intentar con cls en la variante CLAHE (mejor para texto difícil)
                if len(variants) > 1:
                    _, clahe_img = variants[1]
                    text_cls, detections_cls = self._run_paddleocr(clahe_img, use_cls=True)
                    if text_cls.strip():
                        all_texts.append({
                            "source": "paddleocr_cls_clahe",
                            "text": text_cls.strip(),
                            "weight": 3.3
                        })
                        all_detections.extend(detections_cls)
                        primary_engine = "paddleocr"
        
        # Fallback a Tesseract solo si PaddleOCR falló completamente
        if not all_texts:
            for name, variant_img in variants[:2]:  # Solo 2 variantes
                text = self._run_tesseract(variant_img, psm=6)
                if text.strip():
                    all_texts.append({
                        "source": f"tesseract_{name}",
                        "text": text.strip().upper(),
                        "weight": 1.0
                    })
                    break  # Salir al primer resultado
        
        # Filtrar duplicados
        filtered_detections = self.filter_duplicates(all_detections)
        
        # Combinar textos
        combined_text = self._combine_texts(all_texts)
        
        # Calcular confianza promedio final
        avg_confidence = 0.0
        if filtered_detections:
            top_k = sorted([d.confidence for d in filtered_detections], reverse=True)[:15]
            avg_confidence = float(np.mean(top_k))
        
        return OCRResult(
            combined_text=combined_text,
            detections=filtered_detections,
            confidence=avg_confidence,
            engine=primary_engine
        )
    
    def run_ocr_by_zones(self, img: np.ndarray, tipo: str = "frente") -> Dict[str, OCRResult]:
        """
        Ejecuta OCR por zonas específicas de la INE.
        OPTIMIZADO: Reduce llamadas a PaddleOCR y usa early-exit.
        
        Args:
            img: Imagen de INE preprocesada
            tipo: "frente" o "reverso"
            
        Returns:
            Diccionario con nombre de zona -> OCRResult
        """
        # Preprocesamiento SIMPLIFICADO (sin extract_card_region costoso)
        img_preprocessed = self.preprocessor._correct_orientation_fast(img)
        img_preprocessed = self.preprocessor.scale_image(img_preprocessed)
        
        # Extraer zonas
        zones = self.zone_extractor.extract_zones(img_preprocessed, tipo)
        
        results = {}
        
        # OPTIMIZACIÓN: Definir zonas prioritarias según tipo
        if tipo == "frente":
            # Para frente: nombre y datos son críticos
            priority_zones = ['nombre', 'datos', 'domicilio', 'fechas', 'inferior']
        else:
            # Para reverso: MRZ es lo único crítico
            priority_zones = ['mrz', 'datos_extra']
        
        for zone_name in priority_zones:
            if zone_name not in zones:
                continue
            
            zone_img = zones[zone_name]
            if zone_img is None or zone_img.size == 0:
                continue
            
            # OCR específico para cada zona
            if is_paddleocr_available():
                # MRZ: procesamiento especial pero OPTIMIZADO
                if zone_name == 'mrz':
                    mrz_result = self._process_mrz_zone_fast(img_preprocessed, zone_img)
                    if mrz_result:
                        results[zone_name] = mrz_result
                    continue
                
                # Otras zonas: una sola pasada sin cls (rápido)
                text, detections = self._run_paddleocr(zone_img, use_cls=False)
                avg_conf = np.mean([d.confidence for d in detections]) if detections else 0.0
                
                # Solo retry con cls para zona 'nombre' si resultado muy malo
                if zone_name == 'nombre' and (not text.strip() or avg_conf < 0.5):
                    text_cls, detections_cls = self._run_paddleocr(zone_img, use_cls=True)
                    avg_conf_cls = np.mean([d.confidence for d in detections_cls]) if detections_cls else 0.0
                    if text_cls.strip() and avg_conf_cls > avg_conf:
                        text = text_cls
                        detections = detections_cls
                        avg_conf = avg_conf_cls
                
                if text.strip():
                    results[zone_name] = OCRResult(
                        combined_text=text.strip(),
                        detections=detections,
                        confidence=avg_conf,
                        engine="paddleocr"
                    )
            else:
                # Fallback a Tesseract
                gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY) if len(zone_img.shape) == 3 else zone_img
                text = self._run_tesseract(gray, psm=6)
                if text.strip():
                    results[zone_name] = OCRResult(
                        combined_text=text.strip().upper(),
                        detections=[],
                        confidence=0.7,
                        engine="tesseract"
                    )
        
        return results
    
    def _run_tesseract_mrz(self, img: np.ndarray) -> str:
        """
        OCR rápido para MRZ usando Tesseract con whitelist A-Z0-9<.
        Tesseract tarda ~0.3-0.5s vs ~30s de PaddleOCR para líneas MRZ anchas.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            # Binarizar con Otsu para texto claro sobre fondo oscuro o viceversa
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = (
                '--psm 6 --oem 1 '
                '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
            )
            text = pytesseract.image_to_string(binary, lang='eng', config=config)
            return text.upper()
        except Exception:
            return ""

    def _process_mrz_zone_fast(self, img_full: np.ndarray, zone_img: np.ndarray) -> Optional[OCRResult]:
        """
        Procesa zona MRZ de forma OPTIMIZADA.
        Estrategia: Tesseract (~0.5s) primero → PaddleOCR (~30s) solo como fallback.
        """
        mrz_zone = self.zone_extractor.extract_mrz_zone(img_full)
        if mrz_zone is not None and mrz_zone.size > 0:
            zone_img = mrz_zone

        gray_mrz = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY) if len(zone_img.shape) == 3 else zone_img

        # Intento 1: Tesseract directo (RÁPIDO)
        text_tess = self._run_tesseract_mrz(zone_img)
        mrz_validado = self._validate_and_clean_mrz(text_tess)
        if mrz_validado:
            return OCRResult(
                combined_text=mrz_validado,
                detections=[],
                confidence=0.85,
                engine="tesseract"
            )

        # Intento 2: CLAHE + Tesseract
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_mrz)
        text_tess2 = self._run_tesseract_mrz(clahe_img)
        mrz_validado2 = self._validate_and_clean_mrz(text_tess2)
        if mrz_validado2:
            return OCRResult(
                combined_text=mrz_validado2,
                detections=[],
                confidence=0.85,
                engine="tesseract"
            )

        # Fallback: PaddleOCR sin cls (más rápido que cls=True)
        text, detections = self._run_paddleocr(zone_img, use_cls=False)
        avg_conf = float(np.mean([d.confidence for d in detections])) if detections else 0.0

        mrz_validado3 = self._validate_and_clean_mrz(text)
        if mrz_validado3:
            return OCRResult(
                combined_text=mrz_validado3,
                detections=detections,
                confidence=max(avg_conf, 0.8),
                engine="paddleocr"
            )

        # Fallback final: CLAHE + PaddleOCR
        if not text.strip() or avg_conf < 0.6:
            clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
            text2, detections2 = self._run_paddleocr(clahe_bgr, use_cls=False)
            avg_conf2 = float(np.mean([d.confidence for d in detections2])) if detections2 else 0.0
            mrz_validado4 = self._validate_and_clean_mrz(text2)
            if mrz_validado4:
                return OCRResult(
                    combined_text=mrz_validado4,
                    detections=detections2,
                    confidence=max(avg_conf2, 0.8),
                    engine="paddleocr"
                )
            if text2.strip() and ('<' in text2 or 'IDMEX' in text2.upper()) and avg_conf2 > avg_conf:
                text, detections, avg_conf = text2, detections2, avg_conf2

        if text.strip():
            return OCRResult(
                combined_text=text.strip(),
                detections=detections,
                confidence=min(avg_conf, 0.5),
                engine="paddleocr"
            )

        return None
    
    def _validate_and_clean_mrz(self, text: str) -> Optional[str]:
        """
        Valida y limpia texto MRZ descartando variantes contaminadas.
        
        Estrategia:
        1. Normalizar a charset A-Z0-9<\n (preservar saltos de línea)
        2. Localizar última ocurrencia de IDMEX
        3. Recortar desde ahí
        4. Validar estructura completa: IDMEX\\d{10}, MEX, patrón línea 3, relleno final
        
        Args:
            text: Texto crudo del OCR
            
        Returns:
            Texto MRZ limpio y validado, o None si no pasa validación
        """
        if not text:
            return None
        
        # 1. Normalizar: solo A-Z, 0-9, < y saltos de línea
        # IMPORTANTE: Preservar \n para poder reconstruir líneas después
        text_upper = text.upper()
        text_clean = re.sub(r'[^A-Z0-9<\n]', '', text_upper)  # Esto ya elimina espacios
        
        # 2. Localizar última ocurrencia de IDMEX (más robusto si hay texto antes)
        idx_idmex = text_clean.rfind('IDMEX')
        if idx_idmex == -1:
            # Intentar con variantes OCR comunes
            idx_idmex = text_clean.rfind('1DMEX')  # 1 en lugar de I
            if idx_idmex == -1:
                return None  # No hay IDMEX, descartar
        
        # 3. Recortar desde IDMEX
        mrz_candidate = text_clean[idx_idmex:]
        
        # 4. Validar estructura mínima (en orden de importancia):
        # 4.1. Validar IDMEX + número documento (debe estar cerca del inicio del bloque)
        if not re.search(r'^(IDMEX|1DMEX)\d{10}', mrz_candidate):
            return None

        
        # 4.2. Validar que exista MEX después del número de documento (línea 2 típicamente)
        # Buscar MEX después de IDMEX + 10 dígitos (posición 14) en una ventana razonable
        pos_despues_idmex = 14  # "IDMEX" (5) + 10 dígitos = 14
        if len(mrz_candidate) > pos_despues_idmex:
            # Buscar MEX en una ventana de 100 caracteres después de IDMEX+10 dígitos
            ventana = mrz_candidate[pos_despues_idmex:pos_despues_idmex + 100]
            if 'MEX' not in ventana:
                return None
        
        # 4.3. Validar que tenga suficientes < (MRZ real tiene 20+)
        # Umbral más alto para evitar aceptar basura "parecida" a MRZ
        count_lt = mrz_candidate.count('<')
        if count_lt < 10:  # Mínimo realista: INEs con nombres largos tienen ~14 '<'
            return None
        
        # 4.4. Validar patrón de línea 3 (nombres) - más estricto
        # Patrón: al menos 2 letras, <, al menos 2 letras, <<, al menos 2 letras (puede tener < intercalados)
        patron_linea3 = r'[A-Z]{2,}<[A-Z]{2,}<<[A-Z<]{2,}'
        if not re.search(patron_linea3, mrz_candidate):
            return None
        
        # 4.5. Validar que tenga bloque de relleno al final (típico de MRZ)
        # El MRZ suele terminar con varios < seguidos (relleno)
        # Buscar en los últimos 30 caracteres para ser flexible
        if len(mrz_candidate) > 30:
            tramo_final = mrz_candidate[-30:]
            if not re.search(r'<{3,}', tramo_final):
                # Si no hay relleno al final, verificar que haya al menos un tramo largo de < en algún lugar
                # (puede estar en medio si el OCR pegó líneas)
                if not re.search(r'<{5,}', mrz_candidate):
                    return None
        else:
            # Si es muy corto, verificar que tenga al menos un tramo de < seguidos
            if not re.search(r'<{3,}', mrz_candidate):
                return None
        
        # 5. Validar longitud mínima: 3 líneas × 30 chars = 90 + posibles \n = 92.
        # Tesseract puede producir exactamente 92 chars (30+\n+30+\n+30).
        if len(mrz_candidate) < 85:
            return None
        
        # 6. Limpiar saltos de línea múltiples y normalizar
        # Preservar estructura pero limpiar ruido
        mrz_candidate = re.sub(r'\n+', '\n', mrz_candidate)  # Múltiples \n -> uno
        mrz_candidate = mrz_candidate.strip()
        
        return mrz_candidate
    
    def _run_paddleocr(self, img: np.ndarray, use_cls: bool = True) -> Tuple[str, List[Detection]]:
        """
        Ejecuta PaddleOCR en una imagen.
        
        Args:
            img: Imagen a procesar
            use_cls: Si usar angle classifier (más lento pero mejor para imágenes rotadas)
        """
        reader = get_paddleocr_reader()
        if reader is None:
            return "", []
        
        try:
            # Convertir a BGR si es escala de grises
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            results = reader.ocr(img, cls=use_cls)
            
            text_parts = []
            detections = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        
                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                        else:
                            text = str(text_info)
                            confidence = 0.9
                        
                        text_parts.append(text)
                        detections.append(Detection(
                            text=text,
                            confidence=confidence,
                            bbox=bbox
                        ))
            
            return " ".join(text_parts), detections
        except Exception as e:
            return "", []
    
    def _run_tesseract(self, img: np.ndarray, psm: int = 6, whitelist: str = None) -> str:
        """
        Ejecuta Tesseract OCR con configuración optimizada.
        
        Args:
            img: Imagen a procesar
            psm: Page Segmentation Mode (6=bloque uniforme, 4=columna, 11=sparse)
            whitelist: Caracteres permitidos (opcional, mejora precisión para campos específicos)
        """
        try:
            # OEM 1 = LSTM only (mejor para texto moderno)
            config = f'--psm {psm} --oem 1'
            
            # Agregar whitelist si se especifica
            if whitelist:
                config += f' -c tessedit_char_whitelist={whitelist}'
            
            return pytesseract.image_to_string(img, lang='spa', config=config)
        except Exception:
            return ""
    
    def _get_variant_weight(self, name: str) -> float:
        """Retorna peso para cada variante de preprocesamiento."""
        weights = {
            "original": 2.5,
            "clahe": 3.0,
            "binary_adaptive": 2.0,
            "binary_otsu": 2.0,
            "high_contrast": 1.5
        }
        return weights.get(name, 1.0)
    
    def _calculate_iou(self, bbox1: List, bbox2: List) -> float:
        """
        Calcula Intersection over Union (IoU) entre dos bounding boxes.
        Cada bbox es una lista de 4 puntos [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
        """
        try:
            if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
                return 0.0
            
            # Convertir a rectángulo (min_x, min_y, max_x, max_y)
            x1_min = min(p[0] for p in bbox1)
            y1_min = min(p[1] for p in bbox1)
            x1_max = max(p[0] for p in bbox1)
            y1_max = max(p[1] for p in bbox1)
            
            x2_min = min(p[0] for p in bbox2)
            y2_min = min(p[1] for p in bbox2)
            x2_max = max(p[0] for p in bbox2)
            y2_max = max(p[1] for p in bbox2)
            
            # Calcular intersección
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # Calcular áreas
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            
            # IoU
            union_area = area1 + area2 - inter_area
            if union_area <= 0:
                return 0.0
            
            return inter_area / union_area
        except:
            return 0.0
    
    def filter_duplicates(self, detections: List[Detection]) -> List[Detection]:
        """
        Elimina detecciones duplicadas usando IoU (Intersection over Union).
        Mantiene la detección con mayor confianza cuando hay overlap significativo.
        IMPORTANTE: Protege apellidos repetidos (ej: CRUZ CRUZ) en líneas diferentes.
        """
        if not detections:
            return []
        
        IOU_THRESHOLD = 0.3  # 30% de overlap = duplicado
        SIMILARITY_THRESHOLD = 0.85  # Similitud de texto (subido para ser más conservador)
        MIN_LEN_FOR_SUBSTRING = 5  # Solo usar substring check si ambos tienen >= 5 chars
        # Umbral Y para considerar "misma línea" - ajustado para INE donde líneas están ~30-50px separadas
        Y_SAME_LINE_THRESHOLD = 20  # Reducido: solo mismo bbox si Y difiere <20px
        
        # Ordenar por confianza (mayor primero)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        kept = []
        
        for det in sorted_dets:
            text_key = det.text.upper().strip()
            if not text_key or len(text_key) < 2:
                continue
            
            is_duplicate = False
            
            for kept_det in kept:
                kept_text = kept_det.text.upper().strip()
                
                # Verificar similitud de texto con criterios más estrictos
                # 1. Exactamente igual
                if text_key == kept_text:
                    text_similar = True
                # 2. Substring solo si ambos son largos y longitudes similares (evita MEX in IDMEX)
                elif (len(text_key) >= MIN_LEN_FOR_SUBSTRING and 
                      len(kept_text) >= MIN_LEN_FOR_SUBSTRING and
                      abs(len(text_key) - len(kept_text)) <= 3):
                    text_similar = text_key in kept_text or kept_text in text_key
                # 3. SequenceMatcher para similitud general
                else:
                    text_similar = SequenceMatcher(None, text_key, kept_text).ratio() > SIMILARITY_THRESHOLD
                
                if text_similar:
                    # PROTECCIÓN DE APELLIDOS REPETIDOS (ej: CRUZ CRUZ)
                    # Si es un token corto igual (posible apellido repetido), verificar posición espacial
                    is_short_repeated = (text_key == kept_text and 
                                       text_key.isalpha() and 
                                       len(text_key) <= 8)  # Aumentado a 8 para cubrir más apellidos
                    
                    if is_short_repeated:
                        # Calcular diferencia Y entre los dos bboxes
                        try:
                            y1 = (det.bbox[0][1] + det.bbox[2][1]) / 2 if det.bbox else 0
                            y2 = (kept_det.bbox[0][1] + kept_det.bbox[2][1]) / 2 if kept_det.bbox else 0
                            y_diff = abs(y1 - y2)
                        except:
                            y_diff = 0
                        
                        # Si están en líneas diferentes (Y difiere significativamente), NO es duplicado
                        # Esto protege apellidos repetidos como CRUZ CRUZ
                        if y_diff > Y_SAME_LINE_THRESHOLD:
                            continue  # NO es duplicado, mantener ambos
                        
                        # Si están en la misma línea, verificar IoU para confirmar duplicado
                        iou = self._calculate_iou(det.bbox, kept_det.bbox)
                        if iou < 0.5:  # Si IoU es bajo, probablemente son tokens diferentes
                            continue  # NO es duplicado
                        
                        # IoU alto + misma línea = duplicado real
                        is_duplicate = True
                        break
                    
                    # Para otros casos (no apellidos cortos repetidos), usar lógica normal
                    # Verificar overlap espacial con IoU
                    iou = self._calculate_iou(det.bbox, kept_det.bbox)
                    if iou > IOU_THRESHOLD:
                        is_duplicate = True
                        break
                    
                    # También verificar proximidad Y (para textos en líneas cercanas)
                    try:
                        y1 = (det.bbox[0][1] + det.bbox[2][1]) / 2 if det.bbox else 0
                        y2 = (kept_det.bbox[0][1] + kept_det.bbox[2][1]) / 2 if kept_det.bbox else 0
                        if abs(y1 - y2) < Y_SAME_LINE_THRESHOLD:  # Usar mismo umbral
                            is_duplicate = True
                            break
                    except:
                        pass
            
            if not is_duplicate:
                kept.append(det)
        
        return kept
    
    def _combine_texts(self, texts: List[dict]) -> str:
        """Combina textos de múltiples fuentes priorizando por peso."""
        if not texts:
            return ""
        
        # Ordenar por peso
        sorted_texts = sorted(texts, key=lambda x: x.get("weight", 1.0), reverse=True)
        
        # Texto principal
        combined = sorted_texts[0]["text"]
        
        # Agregar textos únicos adicionales
        seen = {combined}
        for item in sorted_texts[1:]:
            if item["text"] not in seen:
                combined += "\n" + item["text"]
                seen.add(item["text"])
        
        return combined


# ============================================================================
# FIELD EXTRACTOR - Extracción de campos específicos
# ============================================================================

class FieldExtractor:
    """
    Extrae campos específicos del texto OCR.
    """
    
    # Palabras que NO son nombres (etiquetas de INE y basura OCR común)
    NO_NOMBRES = {
        'NOMBRE', 'SEXO', 'SEXOH', 'SEXOM', 'DOMICILIO', 'CLAVE', 'CURP', 
        # Variantes OCR de SEXOH (S→B, S→5, etc.) y combinaciones con valor
        'BEXOH', 'BEXO', 'SEXO', '5EXOH', '5EXO', 'SEKO', 'SEKOH',
        'SEXOAM', 'SEXOM', 'SEXOH', 'SEXOF',  # SEXO + valor pegado
        'BEXOAM', 'BEXOM',  # Variantes con B
        'FECHA', 'AÑO', 'ANO', 'SECCION', 'SECCIÓN', 'VIGENCIA', 'REGISTRO', 
        'NACIMIENTO', 'ELECTOR', 'EMISION', 'EMISIÓN', 'ESTADO', 'MUNICIPIO',
        'LOCALIDAD', 'INSTITUTO', 'NACIONAL', 'ELECTORAL', 'CREDENCIAL', 
        'VOTAR', 'PARA', 'MEXICO', 'MÉXICO', 'UNIDOS', 'MEXICANOS', 'ESTADOS',
        'INE', 'IFE', 'ELECTORAL', 'CALLE', 'CARR', 'CARRETERA', 'COL', 
        'COLONIA', 'AV', 'AVENIDA', 'BLVD', 'BOULEVARD', 'PRIV', 'PRIVADA',
        'NORTE', 'SUR', 'ORIENTE', 'PONIENTE', 'OTE', 'PTE', 'INT', 'EXT',
        'NUM', 'NUMERO', 'DEL', 'DE', 'LA', 'LOS', 'LAS', 'EL', 'CP',
        # Abreviaturas de estados (se cuelan al nombre cuando zona captura domicilio)
        'PUE', 'OAX', 'GRO', 'MEX', 'CDMX', 'JAL', 'VER', 'GTO', 'CHIS',
        'AGS', 'BC', 'BCS', 'CAM', 'COAH', 'DGO', 'HGO', 'MICH', 'MOR',
        'NAY', 'NL', 'QRO', 'QROO', 'SLP', 'SIN', 'SON', 'TAB', 'TAM',
        'TLAX', 'YUC', 'ZAC', 'H', 'M',
        # Estados completos (por si OCR los captura)
        'VERACRUZ', 'PUEBLA', 'OAXACA', 'GUERRERO', 'JALISCO', 'CHIAPAS',
        'GUANAJUATO', 'MICHOACAN', 'HIDALGO', 'MORELOS', 'TABASCO', 'YUCATAN',
        'TAMAULIPAS', 'SINALOA', 'SONORA', 'DURANGO', 'COAHUILA', 'NAYARIT',
        'AGUASCALIENTES', 'ZACATECAS', 'TLAXCALA', 'CAMPECHE', 'COLIMA',
        'QUERETARO', 'QUINTANAROO',
        # Ciudades/municipios comunes que se cuelan
        'TANTOYUCA', 'TANTOYUCAVER', 'YUCAVER',  # Tantoyuca, Ver.
        'XALAPA', 'VERACRUZVER', 'COATZACOALCOS', 'CORDOBA', 'ORIZABA',
        'TUXTLA', 'TAPACHULA', 'VILLAHERMOSA', 'MERIDA', 'CANCUN',
        'MONTERREY', 'GUADALAJARA', 'TIJUANA', 'JUAREZ', 'LEON',
        # Tokens de domicilio que se pegan (S/N → SN, LOMA DEL SIERVO → LOMADELSIERVO)
        'SN', 'LOMA', 'LOMADELSIERVO', 'SIERVO', 'CERRO', 'VALLE', 'LLANO',
        'CENTRO', 'BARRIO', 'FRACC', 'FRACCIONAMIENTO', 'UNIDAD', 'INFONAVIT',
        'EJIDO', 'RANCHO', 'HACIENDA', 'RESIDENCIAL', 'INDUSTRIAL', 'POPULAR',
        # Palabras de calle que se pegan al nombre
        'RAFAEL', 'OZUNA', 'OZUNASN', 'RAFAELOZUNA', 'RAFAELOZUNASN',
        # Basura OCR común que NO son nombres
        'KICO', 'K1CO', '1CO', 'ICO', 'IDMEX', 'TORRE', 'SOLEDAD',
        'MARTINEZDELATORRE',  # Lugar, no nombre
        # Texto de fondos/publicidad común
        'TU', 'CELULAR', 'TUCELULAR', 'TELCEL', 'MOVISTAR', 'ATT',
        # Errores OCR de etiquetas (primera letra cortada o mal leída)
        'OMBRE', 'OMICILIO', 'ECHA', 'ACIMIENTO', 'LECTOR', 'EXOH', 'EXOM',
        'IGENCIA', 'EGISTRO', 'ECCION', 'URSO', 'OMCILIO', 'OMICILO',
        # Variantes OCR de "NOMBRE" (N→NO, M→ME, BR→PE, etc.)
        'NOMEPE', 'NOMERE', 'NOMBPE', 'NOMBBE', 'NOMORE', 'NOMERE', 'NOMPRE',
        'NOMRE', 'NOMBE', 'NOMPE', 'NOMER', 'NOMBR', 'NOMBEE', 'NOMBFE',
        'HOMBRE', 'MOMBRE', 'ROMBRE', 'POMBRE',  # Primera letra mal leída
        'NOMDRE', 'NOMRRE', 'NOMBKE', 'NOMBRF',  # Variantes internas
        # Basura OCR severa (secuencias sin sentido)
        'JIOEAA', 'ZIIAIACA', 'IOEAA', 'IIAIACA', 'AIACA', 'OEAA',
        # Tokens que indican contaminación de domicilio
        'ELOZUNASN', 'OMADELSIERVO',  # Específicos de este caso
        # Basura OCR de texto invertido/mal leído
        'ACNINANOC', 'CONANIÑCA', 'CONANINCA',  # Texto invertido
        # Texto de firmas/sellos que se cuela
        'CLAUDIA', 'ARLETTE', 'SPINO', 'SECRETARIA', 'EJECUTIVA',
        'CLAUDIAARLETTE', 'CLAUDIAARLETTESPINO', 'SECRETARIAEJECUTIVA',
        'SECRETARIAEJECUTIVADE',
    }
    
    # Señales de que el texto de nombre está contaminado con otros campos
    SEÑALES_CONTAMINACION = {
        'DOMICILIO', 'CALLE', 'COL', 'COLONIA', 'AV', 'AVENIDA', 'PRIV',
        'CLAVE', 'ELECTOR', 'CURP', 'SECCION', 'SECCIÓN', 'VIGENCIA',
        'FECHA', 'NACIMIENTO', 'REGISTRO', 'CP', 'SN', 'S/N', 'NUM',
        'SEXO', 'SEXOH', 'SEXOM', 'BEXOH', 'BEXO',  # Variantes OCR de SEXO
    }
    
    # Patrones de primera palabra que indican contaminación (regex)
    # Si el nombre empieza con algo que parece etiqueta mal leída, es basura
    PATRONES_PRIMERA_PALABRA_BASURA = [
        r'^[BS]EXO[HMF]?$',  # SEXOH, BEXOH, SEXO, BEXO, SEXOM, SEXOF
        r'^[BS]EXO[A-Z]{1,2}$',  # SEXOAM, SEXOM, etc. (SEXO + valor pegado)
        r'^[0-9]EXO',       # 5EXOH, etc.
        r'^OMBRE$',         # NOMBRE sin N
        r'^OMICILIO$',      # DOMICILIO sin D
        r'^NOM[BEPRO][BEPRO][BEPRO]?$',  # Variantes de NOMBRE: NOMEPE, NOMBPE, NOMERE, etc.
        r'^[HNMRP]OMBRE$',  # HOMBRE, MOMBRE, ROMBRE, POMBRE (NOMBRE con primera letra mal)
        r'^NOM[A-Z]{2,4}$',  # NOM + 2-4 letras que no forman nombre válido
    ]
    
    # Abreviaturas de estados mexicanos
    ESTADOS_ABREV = {
        'AGS': 'AGUASCALIENTES', 'BC': 'BAJA CALIFORNIA', 'BCS': 'BAJA CALIFORNIA SUR',
        'CAM': 'CAMPECHE', 'CAMP': 'CAMPECHE', 'CHIS': 'CHIAPAS', 'CHIH': 'CHIHUAHUA',
        'CDMX': 'CIUDAD DE MEXICO', 'COAH': 'COAHUILA', 'COL': 'COLIMA',
        'DGO': 'DURANGO', 'GTO': 'GUANAJUATO', 'GRO': 'GUERRERO', 'HGO': 'HIDALGO',
        'JAL': 'JALISCO', 'MEX': 'ESTADO DE MEXICO', 'MICH': 'MICHOACAN',
        'MOR': 'MORELOS', 'NAY': 'NAYARIT', 'NL': 'NUEVO LEON', 'OAX': 'OAXACA',
        'PUE': 'PUEBLA', 'QRO': 'QUERETARO', 'QROO': 'QUINTANA ROO',
        'SLP': 'SAN LUIS POTOSI', 'SIN': 'SINALOA', 'SON': 'SONORA',
        'TAB': 'TABASCO', 'TAM': 'TAMAULIPAS', 'TLAX': 'TLAXCALA',
        'VER': 'VERACRUZ', 'YUC': 'YUCATAN', 'ZAC': 'ZACATECAS', 'DF': 'CIUDAD DE MEXICO'
    }
    
    # Apellidos comunes mexicanos para validación
    APELLIDOS_COMUNES = {
        'GARCIA', 'HERNANDEZ', 'MARTINEZ', 'LOPEZ', 'GONZALEZ', 'RODRIGUEZ', 'PEREZ',
        'SANCHEZ', 'RAMIREZ', 'CRUZ', 'FLORES', 'GOMEZ', 'MORALES', 'VAZQUEZ', 'REYES',
        'JIMENEZ', 'TORRES', 'DIAZ', 'RUIZ', 'MENDOZA', 'AGUILAR', 'MORENO', 'CASTILLO',
        'ROMERO', 'ALVAREZ', 'GUTIERREZ', 'ORTIZ', 'RAMOS', 'CHAVEZ', 'VARGAS', 'MEDINA',
        'CASTRO', 'GUZMAN', 'HERRERA', 'FERNANDEZ', 'RIVERA', 'SALAZAR', 'NUNEZ', 'SOTO',
        'BENITEZ', 'DELGADO', 'VEGA', 'RIOS', 'CONTRERAS', 'SANDOVAL', 'ESPINOZA', 'LEON',
        'ESTRADA', 'JUAREZ', 'DOMINGUEZ', 'ROJAS', 'SILVA', 'VELAZQUEZ', 'CAMPOS', 'LUNA',
        'SANTIAGO', 'MEJIA', 'CORTES', 'IBARRA', 'ACOSTA', 'GUERRERO', 'CABRERA', 'MENDEZ',
        'MOLINA', 'NAVARRO', 'FUENTES', 'CARRILLO', 'PENA', 'CERVANTES', 'PACHECO', 'BAUTISTA',
        'CAMACHO', 'MIRANDA', 'CARDENAS', 'LARA', 'VALENCIA', 'AVILA', 'TREJO', 'OROZCO',
        'ALVARADO', 'VILLARREAL', 'ARELLANO', 'CASTELLANOS', 'VILLANUEVA', 'VILLALOBOS',
        'MURILLO', 'PADILLA', 'BONILLA', 'SEVILLA', 'PORTILLO', 'GALLEGOS', 'CABALLERO',
        # Apellidos que también son nombres comunes
        'FRANCISCO', 'LAUREANO', 'GUADALUPE', 'ANGEL', 'JESUS', 'MARIA', 'JOSE', 'JUAN',
    }
    
    # Nombres comunes mexicanos para separar palabras pegadas
    # IMPORTANTE: Los nombres más largos tienen prioridad (sorted by len, reverse=True)
    NOMBRES_COMUNES = {
        # Nombres masculinos
        'LUIS', 'FERNANDO', 'JOSE', 'JUAN', 'CARLOS', 'MIGUEL', 'ANGEL',
        'PEDRO', 'ANTONIO', 'FRANCISCO', 'MANUEL', 'JESUS', 'ALEJANDRO', 'ROBERTO',
        'DANIEL', 'DAVID', 'RICARDO', 'EDUARDO', 'JORGE', 'ALBERTO', 'ARTURO',
        'ENRIQUE', 'RAUL', 'SERGIO', 'VICTOR', 'OSCAR', 'RAFAEL', 'MARTIN', 'PABLO',
        'EMILIANO', 'SANTIAGO', 'SEBASTIAN', 'MATEO', 'LEONARDO', 'DIEGO',
        'FACUNDO', 'RODRIGO', 'ANDRES', 'ADRIAN', 'IVAN', 'HECTOR', 'HUGO',
        'ARMANDO', 'GERARDO', 'GUSTAVO', 'JAVIER', 'MARCO', 'MARCOS', 'OMAR',
        'RAMIRO', 'RUBEN', 'SALVADOR', 'SAUL', 'TOMAS', 'ULISES', 'LAUREANO',
        'ALEX', 'AXEL', 'ERICK', 'ERIK', 'ALAN', 'ALDO', 'ABEL', 'ADAN', 'EDGAR',
        'CESAR', 'FELIX', 'ISAAC', 'JOEL', 'RAUL', 'RENE', 'SAID', 'URIEL',
        # Nombres femeninos
        'MARIA', 'ANA', 'ROSA', 'GUADALUPE', 'PATRICIA', 'ELIZABETH', 'VERONICA', 
        'ADRIANA', 'CLAUDIA', 'LETICIA', 'GABRIELA', 'SILVIA', 'MARTHA', 'CARMEN', 
        'LAURA', 'MARIANA', 'JOCELYN', 'PAOLA', 'ANDREA', 'DIANA', 'MONICA', 'SANDRA',
        'ALEJANDRA', 'FERNANDA', 'DANIELA', 'VALERIA', 'NATALIA', 'SOFIA', 'CAMILA',
        'ISABELLA', 'XIMENA', 'REGINA', 'RENATA', 'VALENTINA', 'EMILIA', 'VICTORIA',
        'NATALI', 'KARLA', 'KAREN', 'JESSICA', 'JENNIFER', 'STEPHANIE',
        'ALICIA', 'BEATRIZ', 'CECILIA', 'ELENA', 'ESTHER', 'FATIMA', 'GLORIA',
        'IRMA', 'JULIA', 'LUCIA', 'LUISA', 'NORMA', 'OLGA', 'PAULA', 'RAQUEL',
        'REBECA', 'ROCIO', 'SUSANA', 'TERESA', 'YOLANDA', 'JOSELYN', 'JOSSELYN',
        'ALMA', 'DORA', 'EDNA', 'ELSA', 'EMMA', 'GEMA', 'IRIS', 'LINA', 'NORA',
        'ANGELICA', 'ANGELI', 'ISELA', 'SOCORRO', 'DOLORES', 'CONSUELO',  # Agregados
        # Nombres cortos válidos (para validación)
        'EVA', 'LUZ', 'SOL', 'PAZ', 'FE', 'IDA', 'LEA', 'LIA', 'MIA', 'ZOE',
        'EMA', 'IAN', 'LEO', 'MAX', 'NOE', 'ROY', 'SAM', 'GIL', 'ELI',
        'LESLI', 'LESLIE',  # Nombres cortos válidos
        # Nombres compuestos comunes que NO deben separarse
        'LUISFERNANDO', 'JUANCARLOS', 'JOSEMARIA', 'MARIAELENA', 'MARIAFERNANDA',
        'JUANPABLO', 'JOSEMIGUEL', 'JOSEANTONIO', 'MARIALUISA', 'MARIADELCARMEN',
        'MARIANAJOCELYN', 'MARIAJOSE', 'JUANMANUEL', 'JOSEDELCARMEN',
    }
    
    # Correcciones OCR comunes para nombres
    OCR_CORRECTIONS = {
        # Números confundidos con letras
        '1CO': 'ICO',  # Común en OCR
        'K1CO': 'KICO',
        '1': 'I',
        '0': 'O',
        '5': 'S',
        '8': 'B',
        '6': 'G',
        '4': 'A',
        # Letras confundidas entre sí
        'CACL': 'FACL',  # C confundida con F al inicio
        'CACUND': 'FACUND',  # CACUNDO -> FACUNDO
        'CACLIN': 'FACLIN',  # Variante
        'CACI': 'FACI',  # Variante
        # Errores comunes de OCR en apellidos
        'CONZALEZ': 'GONZALEZ',
        'CARNANDEZ': 'FERNANDEZ',
        'CARCIA': 'GARCIA',
        'CERNANDEZ': 'FERNANDEZ',
        'CARCÍA': 'GARCIA',
        'CÓMEZ': 'GOMEZ',
        'COMEZ': 'GOMEZ',
        # Errores en nombres
        'CERNANDO': 'FERNANDO',
        'CRANCISCO': 'FRANCISCO',
        'CACUNDO': 'FACUNDO',
        'CACUNOO': 'FACUNDO',
        'CACUNDOI': 'FACUNDO',
        'CACLINIDO': 'FACUNDO',  # Error severo de OCR
        # Letras duplicadas por error
        'LL': 'L',  # Solo si no es parte de palabra válida
    }
    
    # Mapeo de palabras OCR mal leídas a nombres correctos
    OCR_NAME_FIXES = {
        'CACLINIDO': 'FACUNDO',
        'CACUNDO': 'FACUNDO',
        'CACUNDOI': 'FACUNDO',
        'CERNANDO': 'FERNANDO',
        'CRANCISCO': 'FRANCISCO',
        'RANCISCO': 'FRANCISCO',  # Primera letra cortada
        'CONZALEZ': 'GONZALEZ',
        'CARNANDEZ': 'FERNANDEZ',
        'CARCIA': 'GARCIA',
        'CERNANDEZ': 'FERNANDEZ',
        'COMEZ': 'GOMEZ',
        'CÓMEZ': 'GOMEZ',
        'CARCÍA': 'GARCIA',
        'CARCLA': 'GARCIA',
        'CIMINEZ': 'JIMENEZ',
        'CIMENEZ': 'JIMENEZ',
        'IMENEZ': 'JIMENEZ',  # Primera letra cortada
        'CUAN': 'JUAN',
        'COSE': 'JOSE',
        'CEDRO': 'PEDRO',
        'CARLOS': 'CARLOS',  # Este está bien, no cambiar
        'CARIOS': 'CARLOS',
        'CARIA': 'MARIA',
        'CARLA': 'MARIA',  # Podría ser Carla real, cuidado
        'CATALI': 'NATALI',
        'CATALIA': 'NATALIA',
        # Errores de primera letra cortada
        'AUREANO': 'LAUREANO',
        'AUREAN': 'LAUREAN',
        'ESLI': 'LESLI',
        'UADALUPE': 'GUADALUPE',
        'UADAL': 'GUADAL',
        'ARCIA': 'GARCIA',
        'ERNANDEZ': 'FERNANDEZ',
        'ARTINEZ': 'MARTINEZ',
        'ODRIGUEZ': 'RODRIGUEZ',
        'ANCHEZ': 'SANCHEZ',
        'AMIREZ': 'RAMIREZ',
        'ORALES': 'MORALES',
        'ASTILLO': 'CASTILLO',
        'OMERO': 'ROMERO',
        'LVAREZ': 'ALVAREZ',
        'UTIERREZ': 'GUTIERREZ',
        'RTIZ': 'ORTIZ',
        'AMOS': 'RAMOS',
        'HAVEZ': 'CHAVEZ',
        'ARGAS': 'VARGAS',
        'EDINA': 'MEDINA',
        # Errores de doble-L (OCR pierde una L)
        'ARELANO': 'ARELLANO',
        'ARELAN': 'ARELLANO',
        'CASTELANOS': 'CASTELLANOS',
        'CASTELANO': 'CASTELLANO',
        'VILAREAL': 'VILLARREAL',
        'VILARREAL': 'VILLARREAL',  # Una L faltante
        'VILALOBOS': 'VILLALOBOS',
        'VILANUEVA': 'VILLANUEVA',
        'VILALBA': 'VILLALBA',
        'VILALPANDO': 'VILLALPANDO',
        'VILASEÑOR': 'VILLASEÑOR',
        'VILASENOR': 'VILLASEÑOR',
        'CABELO': 'CABELLO',
        'MURILO': 'MURILLO',
        'MURIL': 'MURILLO',
        'CASTILO': 'CASTILLO',
        'TRUJILO': 'TRUJILLO',
        'TRUJIL': 'TRUJILLO',
        'PADILA': 'PADILLA',
        'PADIL': 'PADILLA',
        'BONILA': 'BONILLA',
        'BONIL': 'BONILLA',
        'SEVILA': 'SEVILLA',
        'SEVIL': 'SEVILLA',
        'CARILO': 'CARRILLO',
        'CARIL': 'CARRILLO',
        'PORTILO': 'PORTILLO',
        'PORTIL': 'PORTILLO',
        'SALTILO': 'SALTILLO',
        'SALTIL': 'SALTILLO',
        'CEPEDA': 'CEPEDA',  # Este está bien
        'VALEJO': 'VALLEJO',
        'VALEJ': 'VALLEJO',
        'GALEGOS': 'GALLEGOS',
        'GALEGO': 'GALLEGO',
        'GALEG': 'GALLEGO',
        'CABALERO': 'CABALLERO',
        'CABALER': 'CABALLERO',
        # Errores OCR severos (basura)
        'OMBRE': 'NOMBRE',  # Etiqueta mal leída, filtrar
        'JIOEAA': None,  # Basura OCR, eliminar
        'ZIIAIACA': None,  # Basura OCR, eliminar
    }
    
    # Nombres truncados comunes en MRZ (límite ~30 chars) -> versión completa
    # El MRZ tiene espacio limitado y trunca nombres largos
    TRUNCATED_NAMES = {
        # Nombres truncados -> completos
        'GUADA': 'GUADALUPE',
        'GUADAL': 'GUADALUPE',
        'GUADALU': 'GUADALUPE',
        'GUADALUP': 'GUADALUPE',
        'LAUREAN': 'LAUREANO',
        'LAUREN': 'LAUREANO',
        'LAURE': 'LAUREANO',
        'FERNAN': 'FERNANDO',
        'FERNAND': 'FERNANDO',
        'FRANCI': 'FRANCISCO',
        'FRANCIS': 'FRANCISCO',
        'FRANCISC': 'FRANCISCO',
        'ALEJAN': 'ALEJANDRO',
        'ALEJAND': 'ALEJANDRO',
        'ALEJANDR': 'ALEJANDRO',
        'SEBAST': 'SEBASTIAN',
        'SEBASTI': 'SEBASTIAN',
        'SEBASTIA': 'SEBASTIAN',
        'VALENT': 'VALENTINA',
        'VALENTI': 'VALENTINA',
        'VALENTIN': 'VALENTINA',  # Podría ser VALENTIN o VALENTINA
        'MONTSER': 'MONTSERRAT',
        'MONTSERR': 'MONTSERRAT',
        'MONTSERRA': 'MONTSERRAT',
        'ESPERAN': 'ESPERANZA',
        'ESPERANZ': 'ESPERANZA',
        'CONCEP': 'CONCEPCION',
        'CONCEPC': 'CONCEPCION',
        'CONCEPCI': 'CONCEPCION',
        'CONCEPCIO': 'CONCEPCION',
        'MARGARI': 'MARGARITA',
        'MARGARIT': 'MARGARITA',
        'ELIZABE': 'ELIZABETH',
        'ELIZABET': 'ELIZABETH',
        'CRISTOB': 'CRISTOBAL',
        'CRISTOBA': 'CRISTOBAL',
        'MAXIMI': 'MAXIMINO',
        'MAXIMIN': 'MAXIMINO',
        'SALVAD': 'SALVADOR',
        'SALVADO': 'SALVADOR',
        'BENJAM': 'BENJAMIN',
        'BENJAMI': 'BENJAMIN',
        'RODRIG': 'RODRIGO',
        'RODRI': 'RODRIGO',
        'GONZA': 'GONZALO',
        'GONZAL': 'GONZALO',
        'HERNAN': 'HERNANDEZ',  # Apellido truncado
        'HERNAND': 'HERNANDEZ',
        'HERNANDE': 'HERNANDEZ',
        'RODRIQU': 'RODRIGUEZ',
        'RODRIGU': 'RODRIGUEZ',
        'RODRIGUE': 'RODRIGUEZ',
        'MARTINE': 'MARTINEZ',
        'GONZALE': 'GONZALEZ',
        'FERNANDE': 'FERNANDEZ',
        'SANCH': 'SANCHEZ',
        'SANCHE': 'SANCHEZ',
        'RAMIR': 'RAMIREZ',
        'RAMIRE': 'RAMIREZ',
        'FLORE': 'FLORES',
        'MORAL': 'MORALES',
        'MORALE': 'MORALES',
        'GUTIERR': 'GUTIERREZ',
        'GUTIERRE': 'GUTIERREZ',
        'CASTIL': 'CASTILLO',
        'CASTILL': 'CASTILLO',
    }
    
    def _correct_ocr_name(self, texto: str) -> str:
        """Corrige errores OCR comunes en nombres."""
        if not texto:
            return texto
        
        texto = texto.upper()
        
        # Primero verificar si la palabra completa tiene corrección conocida
        palabras = texto.split()
        palabras_corregidas = []
        
        for palabra in palabras:
            # Verificar corrección de palabra completa
            if palabra in self.OCR_NAME_FIXES:
                correccion = self.OCR_NAME_FIXES[palabra]
                # Si la corrección es None, es basura OCR - omitir
                if correccion is not None:
                    palabras_corregidas.append(correccion)
                # Si es None, simplemente no agregamos la palabra (la eliminamos)
            # Verificar si es un nombre truncado conocido
            elif palabra in self.TRUNCATED_NAMES:
                palabras_corregidas.append(self.TRUNCATED_NAMES[palabra])
            else:
                # Aplicar correcciones parciales
                palabra_corregida = palabra
                for wrong, correct in self.OCR_CORRECTIONS.items():
                    palabra_corregida = palabra_corregida.replace(wrong, correct)
                
                # Corregir números sueltos en medio de texto
                # Ej: "RE1ES" -> "REYES", "FAC0NDO" -> "FACUNDO"
                palabra_corregida = re.sub(r'([A-Z])1([A-Z])', r'\1I\2', palabra_corregida)
                palabra_corregida = re.sub(r'([A-Z])0([A-Z])', r'\1O\2', palabra_corregida)
                palabra_corregida = re.sub(r'([A-Z])5([A-Z])', r'\1S\2', palabra_corregida)
                palabra_corregida = re.sub(r'([A-Z])8([A-Z])', r'\1B\2', palabra_corregida)
                palabra_corregida = re.sub(r'([A-Z])6([A-Z])', r'\1G\2', palabra_corregida)
                palabra_corregida = re.sub(r'([A-Z])4([A-Z])', r'\1A\2', palabra_corregida)
                
                # Corregir C al inicio que debería ser F (común en OCR)
                # Solo si el resultado es un nombre conocido
                if palabra_corregida.startswith('C') and len(palabra_corregida) > 3:
                    posible_f = 'F' + palabra_corregida[1:]
                    if posible_f in self.NOMBRES_COMUNES or posible_f in self.APELLIDOS_COMUNES:
                        palabra_corregida = posible_f
                
                palabras_corregidas.append(palabra_corregida)
        
        return ' '.join(palabras_corregidas)
    
    def _separate_stuck_names(self, texto: str) -> str:
        """
        Separa nombres pegados como LUISFERNANDO → LUIS FERNANDO.
        Usa diccionario de nombres comunes mexicanos.
        Solo separa si ambas partes tienen sentido como nombres.
        """
        if not texto or ' ' in texto:
            # Ya tiene espacios o está vacío
            return texto
        
        texto = texto.upper()
        
        # Si es un nombre compuesto conocido, separarlo directamente
        compuestos = {
            'LUISFERNANDO': 'LUIS FERNANDO',
            'JUANCARLOS': 'JUAN CARLOS',
            'JOSEMARIA': 'JOSE MARIA',
            'MARIAELENA': 'MARIA ELENA',
            'MARIAFERNANDA': 'MARIA FERNANDA',
            'JUANPABLO': 'JUAN PABLO',
            'JOSEMIGUEL': 'JOSE MIGUEL',
            'JOSEANTONIO': 'JOSE ANTONIO',
            'MARIALUISA': 'MARIA LUISA',
            'MARIADELCARMEN': 'MARIA DEL CARMEN',
            'MARIANAJOCELYN': 'MARIANA JOCELYN',
            'MARIAJOSE': 'MARIA JOSE',
            'JUANMANUEL': 'JUAN MANUEL',
            # Nombres femeninos compuestos
            'NORMAANGELICA': 'NORMA ANGELICA',
            'MARIAGUADALUPE': 'MARIA GUADALUPE',
            'ANAPATRICIA': 'ANA PATRICIA',
            'ROSAISELA': 'ROSA ISELA',
            'ROSAELENA': 'ROSA ELENA',
            'ANALUISA': 'ANA LUISA',
            'ANAMARIA': 'ANA MARIA',
            'LUZMARIA': 'LUZ MARIA',
            'MARIAISABEL': 'MARIA ISABEL',
            'MARIAELISA': 'MARIA ELISA',
            'MARIAESTHER': 'MARIA ESTHER',
            'MARIATERESA': 'MARIA TERESA',
            'MARIALETICIA': 'MARIA LETICIA',
            'MARIAPATRICIA': 'MARIA PATRICIA',
            'MARIASOCORRO': 'MARIA SOCORRO',
            'MARIADOLORES': 'MARIA DOLORES',
            'MARIACONSUELO': 'MARIA CONSUELO',
        }
        if texto in compuestos:
            return compuestos[texto]
        
        # Mínimo 3 caracteres para considerar como nombre válido
        MIN_NOMBRE_LEN = 3
        
        # Intentar separar usando nombres conocidos (ordenados por longitud, más largos primero)
        nombres_ordenados = sorted(self.NOMBRES_COMUNES, key=len, reverse=True)
        
        for nombre in nombres_ordenados:
            if nombre in texto and texto != nombre and len(nombre) >= 4:
                # Encontrar posición y separar
                pos = texto.find(nombre)
                if pos > 0:
                    # Nombre está al final: LUISFERNANDO → LUIS FERNANDO
                    antes = texto[:pos]
                    despues = texto[pos:]
                    # Validar que ambas partes tengan sentido
                    if (len(antes) >= MIN_NOMBRE_LEN and antes.isalpha() and
                        len(despues) >= MIN_NOMBRE_LEN and despues.isalpha()):
                        # Verificar que 'antes' sea un nombre conocido o tenga estructura de nombre
                        if antes in self.NOMBRES_COMUNES or len(antes) >= 4:
                            return f"{antes} {despues}"
                elif pos == 0 and len(texto) > len(nombre):
                    # Nombre está al inicio: FERNANDOLUIS → FERNANDO LUIS
                    despues = texto[len(nombre):]
                    # Validar que la parte restante tenga sentido
                    if (len(despues) >= MIN_NOMBRE_LEN and despues.isalpha()):
                        # Verificar que 'despues' sea un nombre conocido o tenga estructura de nombre
                        if despues in self.NOMBRES_COMUNES or len(despues) >= 4:
                            return f"{nombre} {despues}"
        
        return texto
    
    def _clean_name_garbage(self, nombre: str) -> str:
        """
        Limpia basura OCR del nombre.
        Elimina secuencias de caracteres repetidos, palabras duplicadas y basura.
        """
        if not nombre:
            return nombre
        
        # Primero corregir errores OCR comunes (1CO -> ICO, etc.)
        nombre = self._correct_ocr_name(nombre)
        
        palabras = nombre.upper().split()
        palabras_limpias = []
        palabras_vistas = set()  # Para detectar duplicados
        
        for palabra in palabras:
            # Eliminar si tiene más de 2 caracteres repetidos consecutivos (AAAA, OOOO)
            if re.search(r'(.)\1{2,}', palabra):
                continue
            # Eliminar si tiene patrón de basura OCR (muchas vocales seguidas)
            if re.search(r'[AEIOU]{3,}', palabra) and len(palabra) > 6:
                continue
            # Eliminar si es muy larga (>12 chars) - nombres mexicanos rara vez son tan largos
            if len(palabra) > 12 and palabra not in self.NOMBRES_COMUNES:
                continue
            # Eliminar palabras que son claramente basura o etiquetas
            if palabra in self.NO_NOMBRES:
                continue
            # Eliminar si coincide con patrón SEXO + letra(s) (SEXOAM, SEXOM, etc.)
            if re.match(r'^[BS5]EXO[A-Z]{0,2}$', palabra):
                continue
            # Eliminar si contiene "DOMICILIO" o partes de él
            if 'DOMIC' in palabra or 'DOMCL' in palabra or 'DOMCI' in palabra:
                continue
            # Eliminar si empieza con DOM y no es un nombre
            if palabra.startswith('DOM') and palabra not in {'DOMINGO', 'DOMINGA'}:
                continue
            # Eliminar si tiene mezcla rara de consonantes sin vocales
            if len(palabra) > 6 and not re.search(r'[AEIOU]', palabra):
                continue
            # Eliminar si parece código o número mezclado con letras (ej: 1CO, K1CO)
            if re.search(r'\d', palabra):
                continue
            # Eliminar palabras muy cortas que parecen basura OCR (1-2 chars que no son preposiciones)
            if len(palabra) <= 2 and palabra not in {'DE', 'LA', 'EL'}:
                continue
            # Eliminar si termina en I y es casi igual a una palabra ya vista (duplicado OCR)
            # Ejemplo: FACUNDO y FACUNDOI
            palabra_base = palabra.rstrip('I') if palabra.endswith('I') and len(palabra) > 3 else palabra
            if palabra_base in palabras_vistas:
                continue
            # Eliminar si es duplicado exacto
            if palabra in palabras_vistas:
                continue
            # Eliminar palabras muy cortas que no son nombres comunes
            if len(palabra) < 3 and palabra not in {'DE', 'LA', 'EL'}:
                continue
            # Eliminar si tiene patrón PAO + basura (común en OCR malo)
            if palabra.startswith('PAO') and len(palabra) > 5 and palabra not in {'PAOLA', 'PAOLO'}:
                continue
            
            palabras_limpias.append(palabra)
            palabras_vistas.add(palabra)
            # También agregar versión sin I final para detectar duplicados
            if palabra.endswith('I') and len(palabra) > 3:
                palabras_vistas.add(palabra[:-1])
        
        # =========================================================================
        # ELIMINAR SUBSTRINGS Y ECOS: 
        # 1. Si una palabra es substring de otra, eliminarla
        # 2. Si una palabra es "eco" del final de la anterior (ARELLANO -> LANO)
        # 3. Basura OCR específica conocida
        # =========================================================================
        # Basura OCR específica conocida
        BASURA_CONOCIDA = {'LANOID', 'NOID', 'OID', 'LLANO', 'LANO'}
        
        if len(palabras_limpias) > 1:
            palabras_finales = []
            for i, palabra in enumerate(palabras_limpias):
                # Filtrar basura conocida
                if palabra in BASURA_CONOCIDA:
                    continue
                    
                es_basura = False
                
                # Verificar si es substring de otra palabra
                for j, otra in enumerate(palabras_limpias):
                    if i != j and len(otra) > len(palabra) >= 3:
                        if palabra in otra:
                            es_basura = True
                            break
                
                # Verificar si es "eco" del final de la palabra anterior
                # Ejemplo: ARELLANO (prev) -> LANO (curr) o LANOID
                if not es_basura and i > 0 and len(palabra) < 6:
                    prev = palabras_limpias[i-1]
                    # Si la palabra anterior termina con los primeros chars de la actual
                    if len(prev) > len(palabra) and prev.endswith(palabra[:min(4, len(palabra))]):
                        es_basura = True
                    # Si la palabra actual empieza igual que el final de la anterior
                    elif len(prev) >= 4 and palabra.startswith(prev[-4:]):
                        es_basura = True
                
                if not es_basura:
                    palabras_finales.append(palabra)
            palabras_limpias = palabras_finales
        
        return ' '.join(palabras_limpias)
    
    def _validate_extracted_name(self, nombre_data: NameData) -> NameData:
        """
        Valida que el nombre extraído tenga sentido.
        Rechaza nombres con palabras sospechosas (basura OCR).
        
        Returns:
            NameData validado o NameData vacío si es inválido
        """
        if not nombre_data or not nombre_data.nombre_completo:
            return nombre_data
        
        palabras = nombre_data.nombre_completo.split()
        
        # Debe tener al menos 2 palabras (apellido + nombre)
        if len(palabras) < 2:
            return NameData()
        
        # Verificar que al menos una palabra sea un apellido/nombre conocido
        palabras_validas = 0
        for palabra in palabras:
            palabra_upper = palabra.upper()
            if (palabra_upper in self.APELLIDOS_COMUNES or 
                palabra_upper in self.NOMBRES_COMUNES or
                len(palabra_upper) >= 4):  # Palabras largas probablemente son válidas
                palabras_validas += 1
        
        # Si menos del 50% de palabras son válidas, rechazar
        if palabras_validas < len(palabras) * 0.5:
            return NameData()
        
        # Verificar que no haya palabras muy cortas sospechosas (excepto DE, LA, etc.)
        preposiciones = {'DE', 'LA', 'EL', 'LOS', 'LAS', 'DEL'}
        for palabra in palabras:
            if len(palabra) <= 3 and palabra.upper() not in preposiciones:
                # Palabra muy corta que no es preposición - sospechosa
                if palabra.upper() not in self.NOMBRES_COMUNES:
                    # No es un nombre corto conocido (ANA, EVA, etc.)
                    return NameData()
        
        return nombre_data
    
    def _build_name_data_smart(self, palabras: List[str]) -> NameData:
        """
        Construye NameData de manera inteligente cuando solo hay 2 tokens.
        
        Si solo hay 2 tokens y el segundo parece nombre (o no cuadra como apellido),
        deja apellido_materno=None y nombre=token2 (para forzar corrección con MRZ).
        
        Args:
            palabras: Lista de palabras del nombre
            
        Returns:
            NameData construido de manera inteligente
        """
        if not palabras:
            return NameData()
        
        if len(palabras) == 2:
            # Solo 2 tokens: verificar si el segundo parece nombre
            token1 = palabras[0].upper()
            token2 = palabras[1].upper()
            
            # Si el segundo token es un nombre común, asumir que es nombre, no apellido materno
            # Esto evita forzar "EMILIANO" como apellido materno cuando debería ser nombre
            if token2 in self.NOMBRES_COMUNES:
                # Segundo token es nombre conocido -> paterno=token1, nombre=token2
                return NameData(
                    apellido_paterno=palabras[0],
                    apellido_materno=None,  # No forzar como apellido materno
                    nombre=palabras[1],  # Es nombre, no apellido
                    nombre_completo=" ".join(palabras)
                )
            # Si ambos tokens son apellidos comunes, puede ser apellido repetido (CRUZ CRUZ)
            # En ese caso, mantener ambos como apellidos (el parser normal está bien)
        
        # Para 3+ tokens, usar lógica normal
        return NameData(
            apellido_paterno=palabras[0] if len(palabras) >= 1 else None,
            apellido_materno=palabras[1] if len(palabras) >= 2 else None,
            nombre=" ".join(palabras[2:]) if len(palabras) >= 3 else None,
            nombre_completo=" ".join(palabras)
        )
    
    def is_name_suspicious(self, nombre: str) -> bool:
        """
        Detecta si un nombre extraído es sospechoso (contiene basura OCR).
        Se usa como bandera de desconfianza para decidir confiar en MRZ.
        
        Args:
            nombre: Nombre completo a evaluar
            
        Returns:
            True si el nombre es sospechoso (contiene basura OCR conocida)
        """
        if not nombre:
            return True
        
        nombre_upper = nombre.upper()
        palabras = nombre_upper.split()
        
        # Palabras basura conocidas que indican contaminación
        BASURA_CRITICA = {
            'ACNINANOC', 'CONANIÑCA', 'CONANINCA',  # Texto invertido
            'JIOEAA', 'ZIIAIACA', 'IOEAA', 'IIAIACA', 'AIACA', 'OEAA',  # Basura OCR severa
            'SEXOAM', 'SEXOM', 'SEXOH', 'BEXOH', 'BEXO', '5EXOH',  # Variantes SEXO
            'OMBRE', 'OMICILIO',  # Etiquetas mal leídas
            # Variantes OCR de "NOMBRE"
            'NOMEPE', 'NOMERE', 'NOMBPE', 'NOMBBE', 'NOMORE', 'NOMPRE',
            'NOMRE', 'NOMBE', 'NOMPE', 'NOMER', 'NOMBR', 'NOMBEE', 'NOMBFE',
            'HOMBRE', 'MOMBRE', 'ROMBRE', 'POMBRE', 'NOMDRE', 'NOMRRE',
        }
        
        # Verificar si contiene basura crítica
        for palabra in palabras:
            if palabra in BASURA_CRITICA:
                return True
            # Verificar patrones de basura
            if re.search(r'(.)\1{3,}', palabra):  # 4+ caracteres repetidos
                return True
            if re.search(r'[AEIOU]{4,}', palabra) and len(palabra) > 6:  # Muchas vocales seguidas
                return True
            # Verificar si es muy larga y no es nombre conocido
            if len(palabra) > 15 and palabra not in self.NOMBRES_COMUNES and palabra not in self.APELLIDOS_COMUNES:
                return True
            # Verificar si parece variante de "NOMBRE" (NOM + 2-4 letras que no es nombre válido)
            if re.match(r'^NOM[A-Z]{2,4}$', palabra) and palabra not in self.NOMBRES_COMUNES:
                return True
        
        # Verificar primera palabra: si parece etiqueta mal leída, es sospechoso
        if palabras:
            primera = palabras[0]
            for patron in self.PATRONES_PRIMERA_PALABRA_BASURA:
                if re.match(patron, primera):
                    return True
        
        # Verificar si tiene muy pocas palabras válidas
        palabras_validas = 0
        for palabra in palabras:
            if (palabra in self.APELLIDOS_COMUNES or 
                palabra in self.NOMBRES_COMUNES or
                (len(palabra) >= 4 and palabra.isalpha())):
                palabras_validas += 1
        
        # Si menos del 50% son válidas, es sospechoso
        if len(palabras) > 0 and palabras_validas < len(palabras) * 0.5:
            return True
        
        return False
    
    def extract_front(self, ocr_result: OCRResult) -> FrontData:
        """Extrae todos los campos del frente de INE."""
        texto = ocr_result.combined_text
        detections = ocr_result.detections
        
        # Convertir detections a formato dict para compatibilidad
        det_dicts = [{"text": d.text, "confidence": d.confidence, "bbox": d.bbox} 
                     for d in detections]
        
        nombre_data = self.extract_name(texto, det_dicts)
        domicilio_data = self.parse_address(texto)
        curp = self.extract_curp(texto)
        
        # NUEVO: Validar nombre con CURP (detecta nombres incompletos)
        if nombre_data and curp:
            nombre_data = self._validate_name_with_curp(nombre_data, curp, texto)
        
        # NUEVO: Separar nombres compuestos pegados (ej: CARLOSOCTAVIO → CARLOS OCTAVIO)
        if nombre_data and nombre_data.nombre:
            nombre_separado = self._split_compound_name(nombre_data.nombre)
            if nombre_separado != nombre_data.nombre:
                nombre_data.nombre = nombre_separado
                # Reconstruir nombre completo
                partes = [nombre_data.apellido_paterno, nombre_data.apellido_materno, nombre_data.nombre]
                nombre_data.nombre_completo = ' '.join([p for p in partes if p])
        
        # Corregir orden del nombre usando CURP si está disponible
        if nombre_data and curp:
            nombre_data = self._correct_name_order_with_curp(nombre_data, curp)
        
        # NUEVO: Validar y corregir CURP usando el nombre (detecta F→E, etc.)
        if curp and nombre_data:
            curp = self._validate_and_correct_curp_with_name(curp, nombre_data)
        
        return FrontData(
            nombre=nombre_data,
            sexo=self.extract_sexo(texto, curp),  # MEJORADO: Pasar CURP para fallback
            curp=curp,
            clave_elector=self.extract_clave_elector(texto),
            fecha_nacimiento=self.extract_fecha_nacimiento(texto),
            domicilio=domicilio_data,
            seccion=self.extract_seccion(texto),
            vigencia=self.extract_vigencia(texto),
            anio_registro=self.extract_anio_registro(texto),
            confianza_ocr=ocr_result.confidence * 100
        )
    
    def _extract_name_from_zone_strict(self, texto_zona: str) -> Optional[NameData]:
        """
        Extrae nombre de zona específica con limpieza MUY estricta.
        Solo acepta letras y espacios, rechaza cualquier basura.
        Detecta contaminación de otros campos (domicilio, sexo, etc.)
        
        Args:
            texto_zona: Texto de la zona 'nombre' del OCR
            
        Returns:
            NameData si se extrajo un nombre válido, None si contaminado o inválido
        """
        if not texto_zona:
            return None
        
        # Limpiar: solo letras, espacios y acentos
        texto = texto_zona.upper()
        
        # =====================================================================
        # ELIMINAR ENCABEZADO DE INE: Detectar y quitar texto del header
        # El encabezado típico es: "INSTITUTO NACIONAL ELECTORAL MÉXICO CREDENCIAL PARA VOTAR"
        # =====================================================================
        ENCABEZADO_PALABRAS = {
            'INSTITUTO', 'NACIONAL', 'ELECTORAL', 'MEXICO', 'MÉXICO', 
            'CREDENCIAL', 'PARA', 'VOTAR', 'FEDERAL', 'ELEC', 'IEXICO',
            'ESTADOS', 'UNIDOS', 'MEXICANOS',
        }
        
        # Si el texto contiene palabras del encabezado, intentar limpiar
        palabras_raw = texto.split()
        palabras_sin_encabezado = []
        encontro_nombre_label = False
        
        for i, palabra in enumerate(palabras_raw):
            palabra_limpia = re.sub(r'[^A-ZÁÉÍÓÚÑÜ]', '', palabra)
            
            # Si encontramos "NOMBRE", todo lo que sigue es el nombre real
            if palabra_limpia == 'NOMBRE':
                encontro_nombre_label = True
                continue
            
            # Si ya encontramos NOMBRE, agregar todo lo que sigue (excepto SEXO)
            if encontro_nombre_label:
                # Filtrar SEXO y variantes
                if re.match(r'^[BS5]?EXO[HMF]?$', palabra_limpia):
                    continue
                if palabra_limpia in {'SEXO', 'SEXOH', 'SEXOM', 'BEXOH', 'H', 'M'}:
                    continue
                palabras_sin_encabezado.append(palabra_limpia)
            else:
                # Antes de NOMBRE, filtrar palabras del encabezado
                if palabra_limpia not in ENCABEZADO_PALABRAS and len(palabra_limpia) >= 2:
                    # Verificar que no sea parte del encabezado
                    if palabra_limpia not in self.NO_NOMBRES:
                        palabras_sin_encabezado.append(palabra_limpia)
        
        # Si encontramos NOMBRE y hay palabras después, usar esas
        if encontro_nombre_label and palabras_sin_encabezado:
            texto = ' '.join(palabras_sin_encabezado)
        elif not encontro_nombre_label:
            # No encontramos NOMBRE, usar texto original pero filtrar encabezado
            texto = ' '.join([p for p in palabras_raw if re.sub(r'[^A-ZÁÉÍÓÚÑÜ]', '', p) not in ENCABEZADO_PALABRAS])
        
        # =====================================================================
        # DETECCIÓN DE CONTAMINACIÓN: Si hay señales de otros campos, invalidar
        # =====================================================================
        for señal in self.SEÑALES_CONTAMINACION:
            if señal in texto:
                # Zona contaminada con domicilio/datos - forzar fallback
                return None
        
        # Detectar patrones de domicilio pegados (ej: "OZUNASN" de "OZUNA S/N")
        # Si hay tokens que terminan en "SN" y tienen >6 chars, probablemente es domicilio
        palabras_raw = texto.split()
        for palabra in palabras_raw:
            palabra_limpia = re.sub(r'[^A-Z]', '', palabra)
            if len(palabra_limpia) > 6 and palabra_limpia.endswith('SN'):
                return None  # Contaminación de domicilio
            # Detectar códigos postales pegados (5 dígitos)
            if re.search(r'\d{5}', palabra):
                return None
        
        # Reemplazar saltos de línea con espacios
        texto = texto.replace('\n', ' ').replace('\r', ' ')
        
        # Eliminar todo excepto letras y espacios
        texto_limpio = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', '', texto)
        texto_limpio = ' '.join(texto_limpio.split())  # Normalizar espacios
        
        if not texto_limpio or len(texto_limpio) < 5:
            return None
        
        # Aplicar correcciones OCR
        texto_limpio = self._correct_ocr_name(texto_limpio)
        
        # Filtrar palabras que NO son nombres
        palabras = texto_limpio.split()
        palabras_validas = []
        
        for i, palabra in enumerate(palabras):
            # Rechazar palabras en NO_NOMBRES
            if palabra in self.NO_NOMBRES:
                continue
            # Rechazar palabras muy cortas (< 2 chars)
            if len(palabra) < 2:
                continue
            # Rechazar si tiene patrones de basura OCR
            if re.search(r'(.)\1{2,}', palabra):  # Letras repetidas 3+ veces
                continue
            # Rechazar palabras MUY largas (>12 chars) - probablemente domicilio pegado
            if len(palabra) > 12:
                continue
            # Rechazar si termina en abreviatura de estado (VER, PUE, OAX, etc.)
            if len(palabra) > 5 and palabra[-3:] in {'VER', 'PUE', 'OAX', 'GRO', 'JAL', 'MEX', 'GTO'}:
                continue
            # NUEVO: Rechazar variantes de "NOMBRE" (NOM + letras que no forman nombre válido)
            if re.match(r'^NOM[A-Z]{2,4}$', palabra) and palabra not in self.NOMBRES_COMUNES:
                continue
            # NUEVO: Rechazar si es primera palabra y parece etiqueta mal leída
            if i == 0 or len(palabras_validas) == 0:
                es_etiqueta_basura = False
                for patron in self.PATRONES_PRIMERA_PALABRA_BASURA:
                    if re.match(patron, palabra):
                        es_etiqueta_basura = True
                        break
                if es_etiqueta_basura:
                    continue
            # Aceptar palabras de 4+ chars o nombres cortos conocidos
            if len(palabra) >= 4:
                palabras_validas.append(palabra)
            elif palabra in self.NOMBRES_COMUNES or palabra in self.APELLIDOS_COMUNES:
                palabras_validas.append(palabra)
            # Aceptar palabras de 3 chars si son nombres conocidos
            elif len(palabra) == 3 and palabra in self.NOMBRES_COMUNES:
                palabras_validas.append(palabra)
        
        # Necesitamos al menos 2 palabras para un nombre válido
        if len(palabras_validas) < 2:
            return None
        
        # Limitar a 5 palabras máximo
        palabras_validas = palabras_validas[:5]
        
        # Validar que al menos una palabra sea apellido/nombre conocido
        tiene_apellido_conocido = any(p in self.APELLIDOS_COMUNES for p in palabras_validas)
        tiene_nombre_conocido = any(p in self.NOMBRES_COMUNES for p in palabras_validas)
        
        if not tiene_apellido_conocido and not tiene_nombre_conocido:
            # Ninguna palabra conocida - verificar que sean palabras "razonables"
            # (al menos 5 caracteres cada una para ser seguro)
            if not all(len(p) >= 5 for p in palabras_validas):
                return None
        
        nombre_completo = ' '.join(palabras_validas)
        
        return NameData(
            apellido_paterno=palabras_validas[0] if len(palabras_validas) >= 1 else None,
            apellido_materno=palabras_validas[1] if len(palabras_validas) >= 2 else None,
            nombre=' '.join(palabras_validas[2:]) if len(palabras_validas) >= 3 else None,
            nombre_completo=nombre_completo
        )
    
    def extract_front_with_zones(self, ocr_result: OCRResult, zone_results: Dict[str, OCRResult]) -> FrontData:
        """
        Extrae campos del frente usando OCR por zonas para mayor precisión.
        PRIORIZA zona 'nombre' sobre texto global para evitar contaminación.
        
        Args:
            ocr_result: Resultado OCR de imagen completa
            zone_results: Resultados OCR por zona (de run_ocr_by_zones)
        """
        texto_completo = ocr_result.combined_text
        detections = ocr_result.detections
        
        # Convertir detections a formato dict
        det_dicts = [{"text": d.text, "confidence": d.confidence, "bbox": d.bbox} 
                     for d in detections]
        
        # =====================================================================
        # EXTRACCIÓN DE NOMBRE - Priorizar zona específica con fallback inteligente
        # =====================================================================
        nombre_data = None
        nombre_zona_incompleto = False
        
        # PASO 1: Intentar extraer de zona 'nombre' con limpieza estricta
        if 'nombre' in zone_results:
            texto_zona_nombre = zone_results['nombre'].combined_text
            nombre_data = self._extract_name_from_zone_strict(texto_zona_nombre)
            
            # Detectar si la zona trajo solo apellidos (≤2 tokens = probablemente incompleto)
            # EXCEPCIÓN: Si los 2 tokens son IGUALES, es apellido repetido (CRUZ CRUZ) - NO es incompleto
            if nombre_data and nombre_data.nombre_completo:
                tokens_zona = nombre_data.nombre_completo.split()
                num_tokens = len(tokens_zona)
                
                if num_tokens <= 2:
                    # Verificar si es apellido repetido (ej: CRUZ CRUZ)
                    es_apellido_repetido = (
                        num_tokens == 2 and 
                        tokens_zona[0].upper() == tokens_zona[1].upper() and
                        tokens_zona[0].upper() in self.APELLIDOS_COMUNES
                    )
                    
                    if es_apellido_repetido:
                        # Apellido repetido es válido, NO marcar como incompleto
                        # Pero intentar buscar el nombre (tercera palabra) en OCR global
                        nombre_zona_incompleto = False
                        
                        # NUEVO: Buscar nombre en OCR global para completar apellido repetido
                        nombre_global_temp = self.extract_name(texto_completo, det_dicts)
                        if nombre_global_temp and nombre_global_temp.nombre_completo:
                            palabras_global = nombre_global_temp.nombre_completo.split()
                            # Si global tiene 3+ tokens y los primeros 2 coinciden con zona
                            if len(palabras_global) >= 3:
                                # Verificar que los apellidos coincidan
                                if (palabras_global[0].upper() == tokens_zona[0].upper() or
                                    palabras_global[1].upper() == tokens_zona[0].upper()):
                                    # Combinar: apellidos de zona + nombre de global
                                    nombre_combinado = tokens_zona + palabras_global[2:]
                                    nombre_data = NameData(
                                        apellido_paterno=tokens_zona[0],
                                        apellido_materno=tokens_zona[1],
                                        nombre=' '.join(palabras_global[2:]),
                                        nombre_completo=' '.join(nombre_combinado)
                                    )
                    else:
                        nombre_zona_incompleto = True
        
        # PASO 2: Si zona falló o está incompleta, intentar con OCR global/coordenadas
        if not nombre_data or not nombre_data.nombre_completo or nombre_zona_incompleto:
            # Intentar extraer por coordenadas (más preciso para nombres multilínea)
            nombre_global = self.extract_name(texto_completo, det_dicts)
            
            if nombre_global and nombre_global.nombre_completo:
                nombre_global = self._validate_extracted_name(nombre_global)
                
                if nombre_global and nombre_global.nombre_completo:
                    # Filtrar palabras de NO_NOMBRES
                    palabras = nombre_global.nombre_completo.split()
                    palabras_limpias = [p for p in palabras if p not in self.NO_NOMBRES]
                    
                    if len(palabras_limpias) >= 2:
                        nombre_global = NameData(
                            apellido_paterno=palabras_limpias[0] if len(palabras_limpias) >= 1 else None,
                            apellido_materno=palabras_limpias[1] if len(palabras_limpias) >= 2 else None,
                            nombre=' '.join(palabras_limpias[2:]) if len(palabras_limpias) >= 3 else None,
                            nombre_completo=' '.join(palabras_limpias)
                        )
                        
                        # Usar global si tiene más tokens que zona (más completo)
                        tokens_global = len(palabras_limpias)
                        tokens_zona = len(nombre_data.nombre_completo.split()) if nombre_data and nombre_data.nombre_completo else 0
                        
                        if tokens_global > tokens_zona:
                            nombre_data = nombre_global
                        elif tokens_zona == tokens_global and nombre_zona_incompleto:
                            # Si tienen mismos tokens pero zona estaba incompleta, preferir global
                            # porque tiene mejor orden por coordenadas
                            nombre_data = nombre_global
        
        # PASO 3: Si aún no hay nombre, intentar método de texto de zona (menos estricto)
        if not nombre_data or not nombre_data.nombre_completo:
            if 'nombre' in zone_results:
                texto_nombre = zone_results['nombre'].combined_text
                nombre_data = self._extract_name_from_text(texto_nombre)
                if nombre_data and nombre_data.nombre_completo:
                    nombre_data = self._validate_extracted_name(nombre_data)
        
        # =====================================================================
        # EXTRACCIÓN DE OTROS CAMPOS
        # =====================================================================
        
        # Extraer domicilio: preferir zona 'domicilio'
        domicilio_data = None
        if 'domicilio' in zone_results and zone_results['domicilio'].confidence > 0.6:
            texto_dom = zone_results['domicilio'].combined_text
            domicilio_data = self.parse_address(texto_dom)
        
        if not domicilio_data or not domicilio_data.domicilio_completo:
            domicilio_data = self.parse_address(texto_completo)
        
        # Extraer CURP y clave elector: preferir zona 'datos'
        curp = None
        clave_elector = None
        if 'datos' in zone_results:
            texto_datos = zone_results['datos'].combined_text
            curp = self.extract_curp(texto_datos)
            clave_elector = self.extract_clave_elector(texto_datos)
        
        # NUEVO: Si no se encontró CURP en zona datos, buscar en zona inferior
        if not curp and 'inferior' in zone_results:
            texto_inf = zone_results['inferior'].combined_text
            curp = self.extract_curp(texto_inf)
        
        # NUEVO: Si aún no hay CURP, buscar en todas las zonas
        if not curp:
            for zone_name, zone_result in zone_results.items():
                if zone_name not in ['datos', 'inferior']:  # Ya revisadas
                    curp_temp = self.extract_curp(zone_result.combined_text)
                    if curp_temp:
                        curp = curp_temp
                        break
        
        if not curp:
            curp = self.extract_curp(texto_completo)
        if not clave_elector:
            clave_elector = self.extract_clave_elector(texto_completo)
        
        # Extraer fechas y sección: preferir zona 'fechas'
        fecha_nacimiento = None
        seccion = None
        if 'fechas' in zone_results:
            texto_fechas = zone_results['fechas'].combined_text
            fecha_nacimiento = self.extract_fecha_nacimiento(texto_fechas)
            seccion = self.extract_seccion(texto_fechas)
        
        if not fecha_nacimiento:
            fecha_nacimiento = self.extract_fecha_nacimiento(texto_completo)
        if not seccion:
            seccion = self.extract_seccion(texto_completo)
        
        # Extraer vigencia: preferir zona 'inferior'
        vigencia = None
        anio_registro = None
        if 'inferior' in zone_results:
            texto_inf = zone_results['inferior'].combined_text
            vigencia = self.extract_vigencia(texto_inf)
            anio_registro = self.extract_anio_registro(texto_inf)
        
        if not vigencia:
            vigencia = self.extract_vigencia(texto_completo)
        if not anio_registro:
            anio_registro = self.extract_anio_registro(texto_completo)
        
        # Detectar tipo de INE (IFE vs INE) y año de emisión
        tipo_ine = self.detect_ine_type(texto_completo)
        anio_emision = self.extract_anio_emision(texto_completo, vigencia)
        
        # Detectar modelo específico de INE (C, D, E, F, G, H)
        modelo_ine = self.detect_modelo_ine(
            texto_completo, 
            vigencia=vigencia, 
            anio_emision=anio_emision,
            clave_elector=clave_elector,
            tipo_ine=tipo_ine
        )
        
        # =====================================================================
        # CORRECCIÓN DE ORDEN DE NOMBRE USANDO CURP
        # Si tenemos CURP, verificar y corregir el orden del nombre
        # =====================================================================
        
        # NUEVO: Validar nombre con CURP (detecta nombres incompletos)
        if nombre_data and curp:
            nombre_data = self._validate_name_with_curp(nombre_data, curp, texto_completo)
        
        # NUEVO: Separar nombres compuestos pegados (ej: CARLOSOCTAVIO → CARLOS OCTAVIO)
        if nombre_data and nombre_data.nombre:
            nombre_separado = self._split_compound_name(nombre_data.nombre)
            if nombre_separado != nombre_data.nombre:
                nombre_data.nombre = nombre_separado
                # Reconstruir nombre completo
                partes = [nombre_data.apellido_paterno, nombre_data.apellido_materno, nombre_data.nombre]
                nombre_data.nombre_completo = ' '.join([p for p in partes if p])
        
        if nombre_data and curp:
            nombre_data = self._correct_name_order_with_curp(nombre_data, curp)
        
        # NUEVO: Validar y corregir CURP usando el nombre (detecta F→E, etc.)
        if curp and nombre_data:
            curp = self._validate_and_correct_curp_with_name(curp, nombre_data)
        
        return FrontData(
            nombre=nombre_data,
            sexo=self.extract_sexo(texto_completo, curp),  # MEJORADO: Pasar CURP para fallback
            curp=curp,
            clave_elector=clave_elector,
            fecha_nacimiento=fecha_nacimiento,
            domicilio=domicilio_data,
            seccion=seccion,
            vigencia=vigencia,
            anio_registro=anio_registro,
            anio_emision=anio_emision,
            tipo_ine=tipo_ine,
            modelo_ine=modelo_ine,
            confianza_ocr=ocr_result.confidence * 100
        )
    
    def detect_ine_type(self, texto: str) -> Optional[str]:
        """
        Detecta el tipo de credencial: IFE (modelos C/D) o INE (modelos E/F/G/H).
        
        - IFE: "INSTITUTO FEDERAL ELECTORAL" (vigencia casi expirada o expirada)
        - INE: "INSTITUTO NACIONAL ELECTORAL" (vigentes)
        
        Returns:
            "IFE" para modelos C/D, "INE" para modelos E+, None si no detectado
        """
        texto_upper = texto.upper()
        
        # Buscar "FEDERAL" vs "NACIONAL"
        if 'FEDERAL' in texto_upper or 'IFE' in texto_upper:
            return "IFE"
        elif 'NACIONAL' in texto_upper or 'INE' in texto_upper:
            return "INE"
        
        # Heurística por vigencia: IFE típicamente tiene vigencias antes de 2014
        # INE empezó a emitir en 2014
        vigencia_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', texto_upper)
        if vigencia_match:
            anio_inicio = int(vigencia_match.group(1))
            if anio_inicio < 2014:
                return "IFE"
            else:
                return "INE"
        
        return None
    
    def detect_modelo_ine(self, texto: str, vigencia: str = None, anio_emision: str = None, 
                          clave_elector: str = None, tipo_ine: str = None) -> Optional[str]:
        """
        Detecta el modelo específico de la credencial INE/IFE.
        
        Modelos y características:
        - Modelo C (IFE): 2001-2008, vigencia 6 años, sin hologramas avanzados
        - Modelo D (IFE): 2008-2013, vigencia 10 años, código de barras 2D
        - Modelo E (INE): 2014-2018, primera INE, vigencia 10 años
        - Modelo F (INE): 2019-2020, mejoras de seguridad
        - Modelo G (INE): 2020-2023, QR code, nuevos hologramas
        - Modelo H (INE): 2024+, última versión con más seguridad
        
        Returns:
            Letra del modelo: "C", "D", "E", "F", "G", "H" o None
        """
        texto_upper = texto.upper()
        
        # Extraer año de emisión si no se proporciona
        if not anio_emision and vigencia:
            vigencia_match = re.search(r'(\d{4})', vigencia)
            if vigencia_match:
                anio_emision = vigencia_match.group(1)
        
        # Si no hay año de emisión, intentar extraerlo del texto
        if not anio_emision:
            # Buscar patrón de vigencia YYYY - YYYY
            vigencia_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', texto_upper)
            if vigencia_match:
                anio_emision = vigencia_match.group(1)
        
        # Determinar modelo por año de emisión
        if anio_emision:
            try:
                anio = int(anio_emision)
                
                # Modelo H: 2024 en adelante
                if anio >= 2024:
                    return "H"
                # Modelo G: 2020-2023
                elif anio >= 2020:
                    return "G"
                # Modelo F: 2019
                elif anio == 2019:
                    return "F"
                # Modelo E: 2014-2018
                elif 2014 <= anio <= 2018:
                    return "E"
                # Modelo D: 2008-2013
                elif 2008 <= anio <= 2013:
                    return "D"
                # Modelo C: 2001-2007
                elif 2001 <= anio <= 2007:
                    return "C"
                # Modelos anteriores (A, B) - muy raros
                elif anio < 2001:
                    return "B"  # O anterior
            except ValueError:
                pass
        
        # Heurística adicional: tipo de institución
        if tipo_ine == "IFE":
            # IFE solo emitió modelos C y D
            # Si tiene vigencia de 10 años, probablemente es D
            if vigencia:
                vigencia_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', vigencia)
                if vigencia_match:
                    inicio, fin = int(vigencia_match.group(1)), int(vigencia_match.group(2))
                    if fin - inicio == 10:
                        return "D"
                    elif fin - inicio == 6:
                        return "C"
            return "D"  # Default para IFE
        elif tipo_ine == "INE":
            # INE emite modelos E, F, G, H
            # Sin más info, asumir modelo reciente
            return "G"  # Default para INE sin año específico
        
        return None
    
    def extract_anio_emision(self, texto: str, vigencia: str = None) -> Optional[str]:
        """
        Extrae el año de emisión de la credencial.
        
        Puede estar explícito o inferirse de la vigencia (año inicio).
        """
        texto_upper = texto.upper()
        
        # Buscar "EMISION" o "EMISIÓN" seguido de año
        emision_match = re.search(r'EMISI[OÓ]N\s*:?\s*(\d{4})', texto_upper)
        if emision_match:
            return emision_match.group(1)
        
        # Inferir de vigencia (primer año)
        if vigencia:
            vigencia_match = re.search(r'(\d{4})', vigencia)
            if vigencia_match:
                return vigencia_match.group(1)
        
        # Buscar en texto general
        vigencia_match = re.search(r'(\d{4})\s*[-–]\s*\d{4}', texto_upper)
        if vigencia_match:
            return vigencia_match.group(1)
        
        return None
    
    def _correct_name_order_with_curp(self, nombre_data: NameData, curp: str) -> NameData:
        """
        Corrige el orden del nombre usando el CURP como referencia.
        También detecta y elimina basura OCR que no coincide con el CURP.
        
        El CURP tiene el formato: AABB910909HSLLNL00
        - Posición 0: Primera letra del apellido paterno
        - Posición 1: Primera vocal del apellido paterno
        - Posición 2: Primera letra del apellido materno
        - Posición 3: Primera letra del nombre
        
        Si el nombre extraído no coincide con el CURP, intenta reordenar.
        Si hay palabras que no coinciden con ninguna letra del CURP, las elimina.
        """
        if not nombre_data or not nombre_data.nombre_completo or not curp:
            return nombre_data
        
        curp = curp.upper().strip()
        if len(curp) < 4:
            return nombre_data
        
        palabras = nombre_data.nombre_completo.split()
        if len(palabras) < 2:
            return nombre_data
        
        # Extraer letras del CURP
        curp_ap = curp[0]  # Primera letra apellido paterno
        curp_am = curp[2]  # Primera letra apellido materno
        curp_nom = curp[3]  # Primera letra nombre
        
        # Verificar si el orden actual es correcto
        ap_actual = palabras[0][0] if palabras[0] else ''
        am_actual = palabras[1][0] if len(palabras) > 1 and palabras[1] else ''
        nom_actual = palabras[2][0] if len(palabras) > 2 and palabras[2] else ''
        
        # Si el orden actual coincide con CURP, no hacer nada
        if ap_actual == curp_ap and am_actual == curp_am:
            if len(palabras) < 3 or nom_actual == curp_nom:
                return nombre_data
        
        # Intentar encontrar el orden correcto
        # Buscar qué palabra empieza con cada letra del CURP
        palabra_ap = None
        palabra_am = None
        palabras_nom = []
        
        palabras_usadas = set()
        
        # Buscar apellido paterno (primera letra = curp[0])
        for i, p in enumerate(palabras):
            if p and p[0] == curp_ap and i not in palabras_usadas:
                # Verificar que no sea basura conocida
                if p not in self.NO_NOMBRES:
                    palabra_ap = p
                    palabras_usadas.add(i)
                    break
        
        # Buscar apellido materno (primera letra = curp[2])
        for i, p in enumerate(palabras):
            if p and p[0] == curp_am and i not in palabras_usadas:
                if p not in self.NO_NOMBRES:
                    palabra_am = p
                    palabras_usadas.add(i)
                    break
        
        # El resto son nombres - filtrar basura
        for i, p in enumerate(palabras):
            if i not in palabras_usadas:
                # Solo agregar si no es basura conocida
                if p not in self.NO_NOMBRES:
                    palabras_nom.append(p)
        
        # Si no encontramos apellido paterno o materno, intentar sin filtro de NO_NOMBRES
        if not palabra_ap:
            for i, p in enumerate(palabras):
                if p and p[0] == curp_ap and i not in palabras_usadas:
                    palabra_ap = p
                    palabras_usadas.add(i)
                    break
        
        if not palabra_am:
            for i, p in enumerate(palabras):
                if p and p[0] == curp_am and i not in palabras_usadas:
                    palabra_am = p
                    palabras_usadas.add(i)
                    break
        
        # Verificar que encontramos al menos apellidos
        if not palabra_ap or not palabra_am:
            # Si no encontramos coincidencias, retornar original pero filtrar basura
            palabras_filtradas = [p for p in palabras if p not in self.NO_NOMBRES]
            if len(palabras_filtradas) >= 2 and palabras_filtradas != palabras:
                return NameData(
                    apellido_paterno=palabras_filtradas[0] if len(palabras_filtradas) >= 1 else None,
                    apellido_materno=palabras_filtradas[1] if len(palabras_filtradas) >= 2 else None,
                    nombre=' '.join(palabras_filtradas[2:]) if len(palabras_filtradas) >= 3 else None,
                    nombre_completo=' '.join(palabras_filtradas)
                )
            return nombre_data
        
        # Filtrar nombres que no empiezan con la letra correcta si hay varios
        if palabras_nom and curp_nom:
            # Mantener palabras que empiezan con la letra del nombre O que son continuación
            palabras_nom_filtradas = []
            encontro_nombre = False
            for p in palabras_nom:
                if p and p[0] == curp_nom:
                    palabras_nom_filtradas.append(p)
                    encontro_nombre = True
                elif encontro_nombre:
                    # Después del primer nombre, aceptar otros (nombres compuestos)
                    palabras_nom_filtradas.append(p)
            
            if palabras_nom_filtradas:
                palabras_nom = palabras_nom_filtradas
        
        # Construir nombre corregido
        partes = [palabra_ap, palabra_am] + palabras_nom
        nombre_completo = ' '.join(partes)
        
        return NameData(
            apellido_paterno=palabra_ap,
            apellido_materno=palabra_am,
            nombre=' '.join(palabras_nom) if palabras_nom else None,
            nombre_completo=nombre_completo
        )
    
    def _extract_name_from_text(self, texto: str) -> NameData:
        """Extrae nombre de un texto de zona específica (sin buscar etiquetas)."""
        texto = texto.upper().strip()
        
        # Detectar contaminación temprana - si hay señales de otros campos, retornar vacío
        for señal in self.SEÑALES_CONTAMINACION:
            if señal in texto:
                return NameData()
        
        # Corregir errores OCR comunes primero
        texto = self._correct_ocr_name(texto)
        
        # Limpiar basura
        texto = self._clean_name_garbage(texto)
        
        palabras = [p for p in texto.split() if p not in self.NO_NOMBRES and len(p) >= 2]
        
        # Separar nombres pegados
        palabras_separadas = []
        for p in palabras:
            separado = self._separate_stuck_names(p)
            palabras_separadas.extend(separado.split())
        
        # Filtrar de nuevo después de separar (por si se generaron tokens de NO_NOMBRES)
        palabras_filtradas = []
        for i, p in enumerate(palabras_separadas):
            if p in self.NO_NOMBRES:
                continue
            # Rechazar palabras >12 chars (probablemente domicilio pegado)
            if len(p) > 12:
                continue
            # Rechazar si termina en abreviatura de estado
            if len(p) > 5 and p[-3:] in {'VER', 'PUE', 'OAX', 'GRO', 'JAL', 'MEX', 'GTO'}:
                continue
            # NUEVO: Rechazar variantes de "NOMBRE" (NOM + letras que no forman nombre válido)
            if re.match(r'^NOM[A-Z]{2,4}$', p) and p not in self.NOMBRES_COMUNES:
                continue
            # NUEVO: Si es la primera palabra, verificar que no sea etiqueta mal leída
            if i == 0 or len(palabras_filtradas) == 0:
                es_basura_primera = False
                for patron in self.PATRONES_PRIMERA_PALABRA_BASURA:
                    if re.match(patron, p):
                        es_basura_primera = True
                        break
                if es_basura_primera:
                    continue
            palabras_filtradas.append(p)
        
        palabras = palabras_filtradas[:5]  # Máximo 5 palabras
        
        if not palabras:
            return NameData()
        
        # NUEVO: Construir NameData de manera inteligente (maneja casos con 2 tokens)
        return self._build_name_data_smart(palabras)
    
    def extract_back(self, ocr_result: OCRResult) -> BackData:
        """Extrae datos del MRZ del reverso."""
        texto = ocr_result.combined_text
        
        mrz_data = self.extract_mrz(texto)
        curp = self.extract_curp(texto)
        cic = self.extract_cic(texto)
        ocr_vertical = self.extract_ocr_vertical(texto)
        
        return BackData(
            mrz=mrz_data,
            curp=curp,
            cic=cic,
            ocr_vertical=ocr_vertical,
            confianza_ocr=ocr_result.confidence * 100
        )
    
    def extract_back_with_zones(self, ocr_result: OCRResult, zone_results: Dict[str, OCRResult]) -> BackData:
        """
        Extrae datos del reverso usando OCR por zonas para mayor precisión.
        
        Args:
            ocr_result: Resultado OCR de imagen completa
            zone_results: Resultados OCR por zona (de run_ocr_by_zones)
        """
        texto_completo = ocr_result.combined_text
        
        # Extraer MRZ: preferir zona 'mrz' si tiene buena confianza
        mrz_data = None
        if 'mrz' in zone_results and zone_results['mrz'].confidence > 0.6:
            texto_mrz = zone_results['mrz'].combined_text
            mrz_data = self.extract_mrz(texto_mrz)
        
        if not mrz_data or not mrz_data.nombre_completo:
            mrz_data = self.extract_mrz(texto_completo)
        
        # Extraer CURP: preferir zona 'datos_extra' del reverso
        curp = None
        if 'datos_extra' in zone_results and zone_results['datos_extra'].confidence > 0.6:
            texto_datos = zone_results['datos_extra'].combined_text
            curp = self.extract_curp(texto_datos)
        
        if not curp:
            curp = self.extract_curp(texto_completo)
        
        # Extraer CIC (Código de Identificación de Credencial) - 9 dígitos
        cic = self.extract_cic(texto_completo)
        
        # Extraer OCR vertical (Identificador ciudadano) - 13 dígitos
        ocr_vertical = self.extract_ocr_vertical(texto_completo)
        
        return BackData(
            mrz=mrz_data,
            curp=curp,
            cic=cic,
            ocr_vertical=ocr_vertical,
            confianza_ocr=ocr_result.confidence * 100
        )
    
    def extract_cic(self, texto: str) -> Optional[str]:
        """
        Extrae CIC (Código de Identificación de Credencial) del reverso.
        Es un número de 9 dígitos que identifica la credencial.
        
        El CIC aparece en el reverso de la INE, generalmente:
        - Cerca de la etiqueta "CIC" o "IDMEX"
        - Como número aislado de 9 dígitos
        - NO es parte del MRZ ni del CURP
        """
        if not texto:
            return None
        
        texto_upper = texto.upper()
        
        # ESTRATEGIA 1: Buscar cerca de etiqueta "CIC"
        patron_cic = r'\bCIC\s*:?\s*(\d{9})\b'
        match = re.search(patron_cic, texto_upper)
        if match:
            return match.group(1)
        
        # ESTRATEGIA 2: Buscar cerca de "IDMEX" (común en INEs)
        patron_idmex = r'\bIDMEX\s*(\d{9})\b'
        match = re.search(patron_idmex, texto_upper)
        if match:
            return match.group(1)
        
        # ESTRATEGIA 3: Buscar 9 dígitos aislados
        # Excluir números que son parte de CURP, MRZ o fechas
        patron_9dig = r'(?<!\d)(\d{9})(?!\d)'
        matches = re.findall(patron_9dig, texto)
        
        # Filtrar candidatos
        for m in matches:
            # Excluir si parece fecha (empieza con 19 o 20)
            if m.startswith('19') or m.startswith('20'):
                continue
            # Excluir si está en una línea de MRZ (contiene <<<)
            linea_con_numero = [l for l in texto.split('\n') if m in l]
            if linea_con_numero and '<<<' in linea_con_numero[0]:
                continue
            return m
        
        return None
    
    def extract_ocr_vertical(self, texto: str) -> Optional[str]:
        """
        Extrae el identificador ciudadano (OCR vertical) del reverso.
        Es un número de 13 dígitos que aparece verticalmente en el reverso.
        
        Este número es único por credencial y se usa para validación.
        """
        if not texto:
            return None
        
        # ESTRATEGIA 1: Buscar secuencia de 13 dígitos directa
        patron_ocr = r'(?<!\d)(\d{13})(?!\d)'
        matches = re.findall(patron_ocr, texto)
        
        if matches:
            # Preferir el que NO esté en línea de MRZ
            for m in matches:
                linea_con_numero = [l for l in texto.split('\n') if m in l]
                if linea_con_numero and '<<<' not in linea_con_numero[0]:
                    return m
            return matches[0]
        
        # ESTRATEGIA 2: OCR vertical puede venir con espacios/saltos
        # Buscar patrón de dígitos separados que sumen 13
        lineas = texto.split('\n')
        for i, linea in enumerate(lineas):
            # Buscar líneas que sean solo dígitos (OCR vertical)
            linea_limpia = re.sub(r'\s+', '', linea)
            if linea_limpia.isdigit() and len(linea_limpia) == 13:
                return linea_limpia
        
        # ESTRATEGIA 3: Concatenar dígitos cercanos
        digitos = re.findall(r'\d+', texto)
        concatenado = ''.join(digitos)
        
        # Buscar 13 dígitos consecutivos
        match_13 = re.search(r'(\d{13})', concatenado)
        if match_13:
            candidato = match_13.group(1)
            # Verificar que no sea parte de algo más largo
            if candidato not in texto or texto.count(candidato) == 1:
                return candidato
        
        return None

    def extract_curp(self, texto: str) -> Optional[str]:
        """
        Extrae y valida CURP con corrección de errores OCR.
        Formato: 4 letras + 6 dígitos (fecha) + H/M + 5 letras + 2 alfanuméricos
        """
        texto_upper = texto.upper()
        
        # ESTRATEGIA 1: Buscar cerca de etiqueta "CURP"
        pos_curp = texto_upper.find('CURP')
        if pos_curp != -1:
            # Buscar en los siguientes 30 caracteres después de "CURP"
            zona_curp = texto_upper[pos_curp:pos_curp + 50]
            # Patrón estricto en zona CURP
            patron_zona = r'CURP\s*([A-Z]{4}[0-9]{6}[HM][A-Z]{5}[A-Z0-9]{2})'
            match = re.search(patron_zona, zona_curp)
            if match:
                curp_candidato = match.group(1)
                curp_corregido = self._correct_curp_ocr(curp_candidato)
                if self._validate_curp_checksum(curp_corregido):
                    return curp_corregido
                # Si no pasa checksum pero tiene estructura correcta, retornarlo
                if len(curp_corregido) == 18:
                    return curp_corregido
        
        # ESTRATEGIA 2: Patrón estricto en todo el texto
        patron_estricto = r'\b([A-Z]{4}[0-9]{6}[HM][A-Z]{5}[A-Z0-9]{2})\b'
        matches = re.findall(patron_estricto, texto_upper)
        
        for curp in matches:
            curp_corregido = self._correct_curp_ocr(curp)
            if self._validate_curp_checksum(curp_corregido):
                return curp_corregido
        
        # ESTRATEGIA 3: Patrón flexible (permite caracteres confusos)
        patron_flexible = r'\b([A-Z0-9]{4}[0-9OIDTS]{6}[HM][A-Z0-9]{5}[A-Z0-9]{2})\b'
        matches = re.findall(patron_flexible, texto_upper)
        
        mejor_candidato = None
        mejor_score = 0
        
        for match in matches:
            curp_corregido = self._correct_curp_ocr(match)
            if self._validate_curp_checksum(curp_corregido):
                return curp_corregido
            
            # Calcular score de candidato (longitud correcta + estructura)
            score = 0
            if len(curp_corregido) == 18:
                score += 10
            if re.match(r'^[A-Z]{4}', curp_corregido):
                score += 5
            if re.match(r'^[A-Z]{4}[0-9]{6}', curp_corregido):
                score += 5
            if curp_corregido[10] in 'HM':
                score += 5
            
            if score > mejor_score:
                mejor_score = score
                mejor_candidato = curp_corregido
        
        # ESTRATEGIA 4: Buscar secuencias de 18 caracteres alfanuméricos
        # que podrían ser CURP con errores OCR severos
        patron_18chars = r'\b([A-Z0-9]{18})\b'
        matches_18 = re.findall(patron_18chars, texto_upper)
        
        for match in matches_18:
            # Verificar si tiene estructura parecida a CURP
            if (len(match) == 18 and 
                match[10] in 'HMN' and  # N puede ser M mal leído
                re.match(r'^[A-Z0-9]{4}[0-9OIDTS]{6}', match)):
                
                curp_corregido = self._correct_curp_ocr(match)
                if self._validate_curp_checksum(curp_corregido):
                    return curp_corregido
                
                # Si no pasa checksum pero tiene mejor estructura que el mejor candidato
                score = 0
                if len(curp_corregido) == 18:
                    score += 10
                if re.match(r'^[A-Z]{4}', curp_corregido):
                    score += 5
                if re.match(r'^[A-Z]{4}[0-9]{6}', curp_corregido):
                    score += 5
                if curp_corregido[10] in 'HM':
                    score += 5
                
                if score > mejor_score:
                    mejor_score = score
                    mejor_candidato = curp_corregido
        
        return mejor_candidato
    
    def _correct_curp_ocr(self, curp: str) -> str:
        """
        Corrige errores comunes de OCR en CURP.
        
        Estructura CURP (18 caracteres):
        - Pos 0-3: 4 letras (iniciales apellidos + nombre)
        - Pos 4-9: 6 dígitos (fecha YYMMDD)
        - Pos 10: 1 letra (sexo H/M)
        - Pos 11-12: 2 letras (estado)
        - Pos 13-15: 3 letras (consonantes internas)
        - Pos 16: 1 dígito o letra (homoclave - diferenciador)
        - Pos 17: 1 dígito (dígito verificador)
        """
        if not curp or len(curp) < 18:
            return curp
        
        curp = curp.upper()
        
        # Parte 1: Primeros 4 caracteres (DEBEN ser letras)
        reemplazos_a_letras = {'0': 'O', '1': 'I', '8': 'B', '7': 'T', '5': 'S', '6': 'G', '9': 'Q'}
        curp1 = curp[0:4]
        for old, new in reemplazos_a_letras.items():
            curp1 = curp1.replace(old, new)
        
        # Parte 2: Caracteres 4-9 (DEBEN ser dígitos - fecha YYMMDD)
        reemplazos_a_numeros = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'T': '7', 'S': '5', 'B': '8', 'G': '6', 'Q': '9', 'Z': '2'}
        curp2 = curp[4:10] if len(curp) >= 10 else curp[4:]
        for old, new in reemplazos_a_numeros.items():
            curp2 = curp2.replace(old, new)
        
        # Parte 3: Carácter 10 (DEBE ser H o M - sexo)
        curp3 = curp[10] if len(curp) > 10 else ""
        if curp3 == 'N':
            curp3 = 'M'
        elif curp3 not in 'HM':
            if curp3 in '0O':
                curp3 = 'M'
            elif curp3 in '1I':
                curp3 = 'H'
        
        # Parte 4: Caracteres 11-12 (DEBEN ser letras - código de estado)
        curp4_estado = curp[11:13] if len(curp) >= 13 else curp[11:] if len(curp) > 11 else ""
        for old, new in reemplazos_a_letras.items():
            curp4_estado = curp4_estado.replace(old, new)
        
        # Parte 5: Caracteres 13-15 (DEBEN ser letras - consonantes internas)
        curp5_consonantes = curp[13:16] if len(curp) >= 16 else curp[13:] if len(curp) > 13 else ""
        for old, new in reemplazos_a_letras.items():
            curp5_consonantes = curp5_consonantes.replace(old, new)
        
        # Parte 6: Carácter 16 (homoclave - puede ser letra O dígito)
        # Este es el diferenciador para personas con mismos datos
        # Típicamente es un dígito (0-9) pero puede ser letra
        curp6_homoclave = curp[16] if len(curp) > 16 else ""
        # NO corregir automáticamente - mantener como está
        
        # Parte 7: Carácter 17 (DEBE ser dígito - dígito verificador)
        curp7_verificador = curp[17] if len(curp) > 17 else ""
        # Corregir O -> 0 ya que el verificador SIEMPRE es dígito
        if curp7_verificador == 'O':
            curp7_verificador = '0'
        elif curp7_verificador == 'I' or curp7_verificador == 'L':
            curp7_verificador = '1'
        elif curp7_verificador == 'S':
            curp7_verificador = '5'
        elif curp7_verificador == 'B':
            curp7_verificador = '8'
        elif curp7_verificador == 'G':
            curp7_verificador = '6'
        elif curp7_verificador == 'T':
            curp7_verificador = '7'
        elif curp7_verificador == 'Q':
            curp7_verificador = '9'
        elif curp7_verificador == 'Z':
            curp7_verificador = '2'
        
        resultado = curp1 + curp2 + curp3 + curp4_estado + curp5_consonantes + curp6_homoclave + curp7_verificador
        
        # Asegurar que tenga exactamente 18 caracteres
        if len(resultado) > 18:
            resultado = resultado[:18]
        elif len(resultado) < 18:
            resultado = curp[:18] if len(curp) >= 18 else curp
        
        # Validación adicional: si el checksum no pasa, intentar corregir homoclave
        if not self._validate_curp_checksum(resultado) and len(resultado) == 18:
            # Intentar con homoclave como dígito (O -> 0)
            resultado_alt = resultado[:16] + ('0' if resultado[16] == 'O' else resultado[16]) + resultado[17]
            if self._validate_curp_checksum(resultado_alt):
                return resultado_alt
            
            # Intentar con homoclave como letra (0 -> O)
            resultado_alt2 = resultado[:16] + ('O' if resultado[16] == '0' else resultado[16]) + resultado[17]
            if self._validate_curp_checksum(resultado_alt2):
                return resultado_alt2
        
        return resultado
    
    def _validate_and_correct_curp_with_name(self, curp: str, nombre_data: NameData) -> str:
        """
        Valida y corrige CURP usando el nombre como referencia.
        
        Detecta errores comunes de OCR en las primeras 4 letras del CURP:
        - Posición 1-2: Primera letra + primera vocal interna del apellido paterno
        - Posición 3: Primera letra del apellido materno
        - Posición 4: Primera letra del nombre
        
        Errores comunes:
        - F confundida con E (PEMF → PEME)
        - I confundida con L (PELI → PELL)
        - O confundida con 0 (POLO → P0LO)
        
        Args:
            curp: CURP extraído por OCR
            nombre_data: Datos del nombre extraídos
        
        Returns:
            CURP corregido
        """
        if not curp or len(curp) < 18:
            return curp
        
        if not nombre_data or not nombre_data.apellido_paterno:
            return curp
        
        curp_upper = curp.upper()
        
        # Extraer componentes del nombre
        ap = (nombre_data.apellido_paterno or "").upper().strip()
        am = (nombre_data.apellido_materno or "").upper().strip()
        nombre = (nombre_data.nombre or "").upper().strip()
        
        if not ap or len(ap) < 2:
            return curp
        
        # Calcular las 4 primeras letras esperadas del CURP
        # Pos 1: Primera letra del apellido paterno
        letra1_esperada = ap[0]
        
        # Pos 2: Primera vocal interna del apellido paterno
        vocales = 'AEIOU'
        letra2_esperada = None
        for i in range(1, len(ap)):
            if ap[i] in vocales:
                letra2_esperada = ap[i]
                break
        
        # Si no hay vocal interna, usar X
        if not letra2_esperada:
            letra2_esperada = 'X'
        
        # Pos 3: Primera letra del apellido materno (o X si no hay)
        letra3_esperada = am[0] if am else 'X'
        
        # Pos 4: Primera letra del nombre (o X si no hay)
        letra4_esperada = nombre[0] if nombre else 'X'
        
        # Comparar con CURP actual
        curp_letras = curp_upper[:4]
        letra1_curp = curp_letras[0]
        letra2_curp = curp_letras[1]
        letra3_curp = curp_letras[2]
        letra4_curp = curp_letras[3]
        
        # Detectar y corregir errores
        correcciones = []
        
        # Validar letra 1
        if letra1_curp != letra1_esperada:
            # Verificar si es error OCR común
            if self._are_ocr_similar(letra1_curp, letra1_esperada):
                correcciones.append((0, letra1_esperada))
        
        # Validar letra 2
        if letra2_curp != letra2_esperada:
            if self._are_ocr_similar(letra2_curp, letra2_esperada):
                correcciones.append((1, letra2_esperada))
        
        # Validar letra 3
        if letra3_curp != letra3_esperada:
            if self._are_ocr_similar(letra3_curp, letra3_esperada):
                correcciones.append((2, letra3_esperada))
        
        # Validar letra 4 (AQUÍ ESTÁ EL ERROR COMÚN: F → E)
        if letra4_curp != letra4_esperada:
            if self._are_ocr_similar(letra4_curp, letra4_esperada):
                correcciones.append((3, letra4_esperada))
        
        # Aplicar correcciones
        if correcciones:
            curp_list = list(curp_upper)
            for pos, letra_correcta in correcciones:
                curp_list[pos] = letra_correcta
            curp_corregido = ''.join(curp_list)
            
            # Validar que la corrección mejore el checksum
            if self._validate_curp_checksum(curp_corregido):
                return curp_corregido
            
            # Si no pasa checksum pero las correcciones son válidas, retornar corregido
            return curp_corregido
        
        return curp
    
    def _are_ocr_similar(self, char1: str, char2: str) -> bool:
        """
        Verifica si dos caracteres son similares en OCR (se confunden fácilmente).
        
        Pares comunes de confusión:
        - F ↔ E ↔ P
        - I ↔ L ↔ 1 ↔ J
        - O ↔ 0 ↔ Q ↔ D
        - S ↔ 5
        - B ↔ 8
        - G ↔ 6
        - Z ↔ 2
        - X ↔ E (caso: XICO por EVELIN)
        """
        confusion_pairs = [
            {'F', 'E', 'P'},  # F se confunde con E y P
            {'I', 'L', '1', 'J'},
            {'O', '0', 'Q', 'D'},
            {'S', '5'},
            {'B', '8'},
            {'G', '6'},
            {'Z', '2'},
            {'U', 'V'},
            {'C', 'G'},
            {'M', 'N'},
            {'X', 'E'},  # NUEVO: X se confunde con E (caso XICO/EVELIN)
        ]
        
        for pair in confusion_pairs:
            if char1 in pair and char2 in pair:
                return True
        
        return False
    
    def _validate_curp_checksum(self, curp: str) -> bool:
        """Valida CURP con dígito verificador RENAPO."""
        curp = curp.upper().strip()
        if not re.match(r'^[A-Z0-9]{18}$', curp):
            return False
        
        diccionario = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
        suma = 0.0
        
        for i in range(17):
            pos = diccionario.find(curp[i])
            if pos == -1:
                return False
            suma += pos * (18 - i)
        
        digito = 10 - (int(suma) % 10)
        if digito == 10:
            digito = 0
        
        ultimo = curp[17]
        if ultimo.isdigit():
            return int(ultimo) == digito
        return True  # Algunos CURPs antiguos tienen letra
    
    def _validate_name_with_curp(self, nombre_data: NameData, curp: str, texto: str) -> NameData:
        """
        Valida que el nombre extraído sea consistente con el CURP.
        Si no coincide, intenta re-extraer el nombre usando el CURP como guía.
        
        Esta función detecta casos donde el OCR extrajo mal el nombre (ej: solo 2 palabras
        cuando debería haber 3) y usa el CURP para buscar las palabras faltantes en el texto.
        
        Args:
            nombre_data: Nombre extraído por OCR
            curp: CURP extraído
            texto: Texto completo del OCR para buscar palabras faltantes
            
        Returns:
            NameData corregido o el original si no se puede mejorar
        """
        if not nombre_data or not curp or len(curp) < 4:
            return nombre_data
        
        # Calcular primeras 4 letras esperadas del CURP basándose en el nombre actual
        expected_prefix = self._calculate_curp_prefix(nombre_data)
        actual_prefix = curp[:4].upper()
        
        # Si coinciden, el nombre está bien
        if expected_prefix == actual_prefix:
            return nombre_data
        
        # No coinciden - intentar re-extraer nombre usando CURP como guía
        # Extraer las letras que deberían estar en el CURP
        letra1_curp = curp[0]  # Primera letra apellido paterno
        letra2_curp = curp[1]  # Primera vocal interna apellido paterno
        letra3_curp = curp[2]  # Primera letra apellido materno
        letra4_curp = curp[3]  # Primera letra nombre
        
        # Buscar en el texto palabras que coincidan con estas letras
        palabras_candidatas = self._find_name_words_in_text(texto, letra1_curp, letra2_curp, letra3_curp, letra4_curp)
        
        if palabras_candidatas and len(palabras_candidatas) >= 3:
            # Construir nuevo NameData con las palabras encontradas
            return NameData(
                apellido_paterno=palabras_candidatas[0],
                apellido_materno=palabras_candidatas[1],
                nombre=' '.join(palabras_candidatas[2:]),
                nombre_completo=' '.join(palabras_candidatas)
            )
        
        # Si no se pudo mejorar, retornar original
        return nombre_data
    
    def _calculate_curp_prefix(self, nombre_data: NameData) -> str:
        """
        Calcula las primeras 4 letras del CURP basándose en el nombre.
        
        Returns:
            String de 4 letras (ej: "PEMF")
        """
        if not nombre_data:
            return "XXXX"
        
        ap = (nombre_data.apellido_paterno or "").upper().strip()
        am = (nombre_data.apellido_materno or "").upper().strip()
        nombre = (nombre_data.nombre or "").upper().strip()
        
        # Letra 1: Primera letra del apellido paterno
        letra1 = ap[0] if ap else 'X'
        
        # Letra 2: Primera vocal interna del apellido paterno
        vocales = 'AEIOU'
        letra2 = 'X'
        if ap and len(ap) > 1:
            for i in range(1, len(ap)):
                if ap[i] in vocales:
                    letra2 = ap[i]
                    break
        
        # Letra 3: Primera letra del apellido materno
        letra3 = am[0] if am else 'X'
        
        # Letra 4: Primera letra del nombre
        letra4 = nombre[0] if nombre else 'X'
        
        return letra1 + letra2 + letra3 + letra4
    
    def _find_name_words_in_text(self, texto: str, letra1: str, letra2: str, letra3: str, letra4: str) -> Optional[List[str]]:
        """
        Busca en el texto palabras que coincidan con las letras del CURP.
        
        Args:
            texto: Texto completo del OCR
            letra1: Primera letra del apellido paterno (CURP pos 0)
            letra2: Primera vocal interna del apellido paterno (CURP pos 1)
            letra3: Primera letra del apellido materno (CURP pos 2)
            letra4: Primera letra del nombre (CURP pos 3)
            
        Returns:
            Lista de palabras [apellido_paterno, apellido_materno, nombre] o None
        """
        # Buscar zona de nombre en el texto
        texto_upper = texto.upper()
        
        # Buscar después de "NOMBRE" y antes de "DOMICILIO"
        pos_nombre = texto_upper.find('NOMBRE')
        pos_domicilio = texto_upper.find('DOMICILIO')
        
        if pos_nombre == -1:
            pos_nombre = 0
        else:
            pos_nombre += 6
        
        if pos_domicilio == -1:
            pos_domicilio = len(texto_upper)
        
        zona_nombre = texto_upper[pos_nombre:pos_domicilio]
        
        # Extraer palabras alfabéticas de la zona
        palabras = re.findall(r'\b([A-ZÁÉÍÓÚÑ]{2,})\b', zona_nombre)
        
        # Filtrar palabras que NO son nombres
        palabras_validas = [p for p in palabras if p not in self.NO_NOMBRES and len(p) >= 3]
        
        if len(palabras_validas) < 3:
            return None
        
        # Buscar apellido paterno (debe empezar con letra1 y tener letra2 como vocal interna)
        apellido_paterno = None
        idx_ap = -1
        vocales = 'AEIOU'
        
        for i, palabra in enumerate(palabras_validas):
            if palabra[0] == letra1:
                # Verificar vocal interna
                tiene_vocal = False
                for j in range(1, len(palabra)):
                    if palabra[j] in vocales:
                        if palabra[j] == letra2:
                            tiene_vocal = True
                            break
                        # Si tiene otra vocal como primera vocal interna, no es match
                        break
                
                if tiene_vocal or letra2 == 'X':
                    apellido_paterno = palabra
                    idx_ap = i
                    break
        
        if not apellido_paterno or idx_ap == -1:
            return None
        
        # Buscar apellido materno (debe empezar con letra3, después del apellido paterno)
        apellido_materno = None
        idx_am = -1
        
        for i in range(idx_ap + 1, len(palabras_validas)):
            if palabras_validas[i][0] == letra3:
                apellido_materno = palabras_validas[i]
                idx_am = i
                break
        
        if not apellido_materno or idx_am == -1:
            return None
        
        # Buscar nombre (debe empezar con letra4, después del apellido materno)
        nombres = []
        
        for i in range(idx_am + 1, len(palabras_validas)):
            palabra = palabras_validas[i]
            # Primera palabra del nombre debe empezar con letra4
            if i == idx_am + 1:
                if palabra[0] == letra4:
                    nombres.append(palabra)
            else:
                # Palabras adicionales del nombre (nombres compuestos)
                nombres.append(palabra)
        
        if not nombres:
            return None
        
        # Retornar [apellido_paterno, apellido_materno, nombre1, nombre2, ...]
        return [apellido_paterno, apellido_materno] + nombres

    def _split_compound_name(self, nombre: str) -> str:
        """
        Separa nombres compuestos que están pegados sin espacio.
        
        Ejemplos:
        - "CARLOSOCTAVIO" → "CARLOS OCTAVIO"
        - "JUANCARLOS" → "JUAN CARLOS"
        - "MARIAJOSE" → "MARIA JOSE"
        - "LUISMIGUEL" → "LUIS MIGUEL"
        
        Estrategia:
        1. Busca nombres comunes dentro de la cadena pegada
        2. Si encuentra 2+ nombres, los separa
        3. Si no encuentra, retorna el original
        
        Args:
            nombre: Nombre potencialmente pegado
            
        Returns:
            Nombre con espacios si se detectaron nombres compuestos
        """
        if not nombre or ' ' in nombre:
            # Ya tiene espacios o está vacío
            return nombre
        
        nombre_upper = nombre.upper().strip()
        
        # Lista de nombres comunes mexicanos para detectar
        nombres_comunes_mexicanos = {
            'CARLOS', 'JUAN', 'JOSE', 'LUIS', 'MIGUEL', 'PEDRO', 'JORGE', 'FRANCISCO',
            'ANTONIO', 'JESUS', 'MANUEL', 'DAVID', 'DANIEL', 'RICARDO', 'ROBERTO',
            'FERNANDO', 'EDUARDO', 'ALBERTO', 'ALEJANDRO', 'SERGIO', 'RAUL', 'JAVIER',
            'MARIA', 'GUADALUPE', 'ROSA', 'ANA', 'MARTHA', 'PATRICIA', 'LAURA',
            'CARMEN', 'TERESA', 'ELENA', 'SILVIA', 'VERONICA', 'GABRIELA', 'ADRIANA',
            'OCTAVIO', 'EVELIN', 'FELIPA', 'ANGEL', 'MARTIN', 'PABLO', 'DIEGO',
            'ANDRES', 'ARTURO', 'ENRIQUE', 'GERARDO', 'HECTOR', 'OSCAR', 'VICTOR',
            'CRISTINA', 'DIANA', 'ELIZABETH', 'FERNANDA', 'ISABEL', 'JULIA', 'LETICIA',
            'MONICA', 'NANCY', 'OLGA', 'PAOLA', 'SANDRA', 'SUSANA', 'YOLANDA'
        }
        
        # Intentar encontrar nombres dentro de la cadena
        nombres_encontrados = []
        posiciones = []
        
        for nombre_comun in sorted(nombres_comunes_mexicanos, key=len, reverse=True):
            # Buscar el nombre común en la cadena
            idx = nombre_upper.find(nombre_comun)
            if idx != -1:
                # Verificar que no se solape con nombres ya encontrados
                solapa = False
                for pos_inicio, pos_fin, _ in posiciones:
                    if not (idx + len(nombre_comun) <= pos_inicio or idx >= pos_fin):
                        solapa = True
                        break
                
                if not solapa:
                    nombres_encontrados.append(nombre_comun)
                    posiciones.append((idx, idx + len(nombre_comun), nombre_comun))
        
        # Si encontramos 2 o más nombres, separarlos
        if len(nombres_encontrados) >= 2:
            # Ordenar por posición
            posiciones.sort(key=lambda x: x[0])
            nombres_ordenados = [p[2] for p in posiciones]
            return ' '.join(nombres_ordenados)
        
        # Si solo encontramos 1 nombre pero la cadena es más larga, intentar separar
        if len(nombres_encontrados) == 1 and len(nombre_upper) > len(nombres_encontrados[0]) + 3:
            pos_inicio, pos_fin, nombre_encontrado = posiciones[0]
            
            # Hay texto antes del nombre encontrado
            if pos_inicio > 2:
                texto_antes = nombre_upper[:pos_inicio]
                if texto_antes in nombres_comunes_mexicanos:
                    return texto_antes + ' ' + nombre_encontrado
            
            # Hay texto después del nombre encontrado
            if pos_fin < len(nombre_upper) - 2:
                texto_despues = nombre_upper[pos_fin:]
                if texto_despues in nombres_comunes_mexicanos:
                    return nombre_encontrado + ' ' + texto_despues
        
        # No se pudo separar, retornar original
        return nombre

    def extract_clave_elector(self, texto: str, img: np.ndarray = None) -> Optional[str]:
        """
        Extrae Clave de Elector con múltiples estrategias (MEJORADO).
        Formato: 6 letras + 8 dígitos + 1 letra + 3 dígitos = 18 caracteres
        Ejemplo: HRPDLS07111320H600, CSBNMR06052630M500
        
        Mejoras implementadas:
        - Estrategia 1: Patrón estricto sin espacios
        - Estrategia 2: Patrón con espacios opcionales
        - Estrategia 3: Búsqueda con etiqueta "CLAVE DE ELECTOR"
        - Estrategia 4: Patrón relajado con validación
        - Estrategia 5: OCR focalizado en región específica (si se proporciona imagen)
        """
        if not texto:
            return None
        
        texto_upper = texto.upper().replace('\n', ' ')
        
        # Estrategia 1: Patrón estricto (sin espacios)
        patron_estricto = r'\b[A-Z]{6}\d{8}[HM]\d{3}\b'
        match = re.search(patron_estricto, texto_upper)
        if match:
            return match.group(0)
        
        # Estrategia 2: Patrón con espacios opcionales
        # Ejemplo: "GOMJUA 85010312 H 400" o "GOMJUA85010312H400"
        patron_espacios = r'\b[A-Z]{6}\s?\d{8}\s?[HM]\s?\d{3}\b'
        match = re.search(patron_espacios, texto_upper)
        if match:
            return match.group(0).replace(' ', '')
        
        # Estrategia 3: Buscar con etiqueta "CLAVE DE ELECTOR" o "CLAVE ELECTOR"
        patron_etiqueta = r'CLAVE\s*(?:DE\s*)?ELECTOR[:\s]*([A-Z0-9\s]{18,25})'
        match = re.search(patron_etiqueta, texto_upper, re.IGNORECASE)
        if match:
            candidato = match.group(1).strip().replace(' ', '').upper()
            # Validar que el candidato tenga el formato correcto
            if re.match(patron_estricto, candidato):
                return candidato
        
        # Estrategia 4: Buscar patrón relajado y validar estructura
        # Permite algunos errores de OCR (O→0, I→1, etc.)
        patron_relajado = r'\b[A-Z0-9]{6}\d{8}[HM]\d{3}\b'
        matches = re.finditer(patron_relajado, texto_upper)
        for match in matches:
            candidato = match.group(0)
            # Validar que los primeros 6 caracteres sean mayormente letras
            primeros_6 = candidato[:6]
            if sum(c.isalpha() for c in primeros_6) >= 5:  # Al menos 5 de 6 son letras
                # Corregir OCR
                corrected = self._correct_clave_elector_ocr(candidato)
                if re.match(patron_estricto, corrected):
                    return corrected
        
        # Estrategia 5: OCR focalizado en región específica (si se proporciona imagen)
        if img is not None:
            try:
                h, w = img.shape[:2]
                # La clave de elector suele estar en la parte inferior derecha
                # o cerca del campo de vigencia
                roi = img[int(h*0.65):h, int(w*0.45):w]
                
                # Preprocesar ROI para mejor OCR
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                # Aumentar contraste
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                # Binarizar
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # OCR con configuración optimizada para texto alfanumérico
                texto_roi = pytesseract.image_to_string(
                    binary, 
                    lang='spa',
                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789HM'
                ).upper()
                
                # Buscar patrón en ROI
                match = re.search(patron_estricto, texto_roi)
                if match:
                    return match.group(0)
            except Exception:
                pass  # Si falla OCR focalizado, continuar
        
        return None
    
    def _correct_clave_elector_ocr(self, clave: str) -> str:
        """Corrige errores OCR en clave de elector."""
        if len(clave) != 18:
            return clave
        
        clave = clave.upper()
        
        # Primeras 6 posiciones deben ser letras
        parte1 = clave[:6]
        for old, new in {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}.items():
            parte1 = parte1.replace(old, new)
        
        # Posiciones 6-13 deben ser dígitos (8 dígitos)
        parte2 = clave[6:14]
        for old, new in {'O': '0', 'I': '1', 'D': '0', 'T': '7', 'S': '5', 'B': '8'}.items():
            parte2 = parte2.replace(old, new)
        
        # Posición 14 debe ser letra
        parte3 = clave[14]
        for old, new in {'0': 'O', '1': 'I', '8': 'B'}.items():
            parte3 = parte3.replace(old, new)
        
        # Posiciones 15-17 deben ser dígitos (3 dígitos)
        parte4 = clave[15:18]
        for old, new in {'O': '0', 'I': '1', 'D': '0', 'S': '5', 'B': '8'}.items():
            parte4 = parte4.replace(old, new)
        
        return parte1 + parte2 + parte3 + parte4
    
    def extract_fecha_nacimiento(self, texto: str) -> Optional[str]:
        """Extrae fecha de nacimiento en formato DD/MM/YYYY."""
        import datetime
        
        patrones = [
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{2}-\d{2}-\d{4})',
            r'(\d{2})\s*/\s*(\d{2})\s*/\s*(\d{4})',
        ]
        
        for patron in patrones:
            matches = re.findall(patron, texto)
            for match in matches:
                if isinstance(match, tuple):
                    fecha_str = f"{match[0]}/{match[1]}/{match[2]}"
                else:
                    fecha_str = match.replace('-', '/')
                
                try:
                    dia, mes, anio = map(int, fecha_str.split('/'))
                    if 1900 <= anio <= datetime.datetime.now().year and 1 <= mes <= 12 and 1 <= dia <= 31:
                        datetime.date(anio, mes, dia)  # Validar fecha
                        return f"{dia:02d}/{mes:02d}/{anio}"
                except ValueError:
                    continue
        
        return None
    
    def extract_sexo(self, texto: str, curp: Optional[str] = None) -> Optional[str]:
        """
        Extrae sexo (H/M) de la INE con fallback a CURP (MEJORADO).
        
        Estrategias:
        1. Buscar "SEXO H" o "SEXO M" en el texto
        2. Buscar H o M aislado cerca de la palabra SEXO
        3. Fallback a CURP (posición 11) si está disponible
        
        Args:
            texto: Texto extraído por OCR
            curp: CURP extraído (opcional, para fallback)
        
        Returns:
            'H' o 'M' o None
        """
        if not texto:
            # Si no hay texto pero hay CURP, usar fallback
            if curp and len(curp) >= 11:
                sexo_curp = curp[10]
                if sexo_curp in ['H', 'M']:
                    return sexo_curp
            return None
        
        texto_upper = texto.upper()
        
        # Estrategia 1: Buscar "SEXO H" o "SEXO M" en el texto
        match = re.search(r'SEXO\s*[:\s]*([HM])\b', texto_upper)
        if match:
            return match.group(1)
        
        # Estrategia 2: Buscar H o M aislado cerca de la palabra SEXO
        # Ejemplo: "SEXO\nH" o "SEXO: M"
        match = re.search(r'SEXO.{0,10}([HM])\b', texto_upper, re.DOTALL)
        if match:
            return match.group(1)
        
        # Estrategia 3: Fallback a CURP (posición 11)
        if curp and len(curp) >= 11:
            sexo_curp = curp[10]
            if sexo_curp in ['H', 'M']:
                return sexo_curp
        
        return None
    
    def extract_seccion(self, texto: str) -> Optional[str]:
        """Extrae número de sección electoral (4 dígitos)."""
        texto_upper = texto.upper()
        
        # Patrones directos con etiqueta SECCIÓN - más confiables
        patrones = [
            r'SECCI[OÓ]N\s*[:\-]?\s*(\d{4})',
            r'SECCION\s*[:\-]?\s*(\d{4})',
            r'SECC[A-Z]*\s*[:\-]?\s*(\d{4})',
            r'SECCIÓN\s*[:\-]?\s*(\d{4})',
            r'SEC[CG]?\s*[:\-]?\s*(\d{4})',  # SEC, SECC, SECG (OCR error)
            r'SECC?I[OÓ0]N\s*(\d{4})',  # SECCION con O como 0
        ]
        
        for patron in patrones:
            matches = re.findall(patron, texto_upper)
            if matches:
                seccion = matches[0]
                # Validar que NO sea un año (1900-2100)
                if not (1900 <= int(seccion) <= 2100):
                    return seccion
        
        # Buscar cerca de la etiqueta SECCIÓN (más variantes)
        pos_seccion = -1
        for etiqueta in ['SECCIÓN', 'SECCION', 'SECC']:
            pos = texto_upper.find(etiqueta)
            if pos != -1:
                pos_seccion = pos
                break
        
        if pos_seccion != -1:
            # Buscar 4 dígitos en los siguientes 20 caracteres (zona más pequeña)
            zona = texto_upper[pos_seccion:pos_seccion + 25]
            match = re.search(r'(\d{4})', zona)
            if match:
                seccion = match.group(1)
                num = int(seccion)
                # Validar: NO es año, NO empieza con 19/20
                if 1 <= num <= 9999 and not (1900 <= num <= 2100):
                    return seccion
        
        # Buscar patrón en clave de elector (posición 14-17 contiene sección)
        # Formato: XXXXXX00000000SSSS donde SSSS es sección
        # Ejemplo: RYFCAN01050430H200 -> sección no está aquí, está después
        # La clave tiene 18 chars, la sección está DESPUÉS
        clave_match = re.search(r'CLAVE\s*DE\s*ELECTOR\s*([A-Z]{6}\d{8}[HM]\d{3})', texto_upper)
        if clave_match:
            # La sección suele estar cerca de la clave pero no dentro
            pos_clave = texto_upper.find(clave_match.group(0))
            zona_despues = texto_upper[pos_clave:pos_clave + 80]
            # Buscar 4 dígitos que no sean año
            nums = re.findall(r'\b(\d{4})\b', zona_despues)
            for n in nums:
                if not (1900 <= int(n) <= 2100):
                    return n
        
        # Buscar patrón específico: número de 4 dígitos entre VIGENCIA y fin de línea
        # que NO sea parte del rango de vigencia (YYYY - YYYY)
        vigencia_match = re.search(r'VIGENCIA\s*(\d{4})\s*[-–]\s*(\d{4})', texto_upper)
        if vigencia_match:
            # La sección suele estar ANTES de VIGENCIA en la misma zona
            pos_vigencia = texto_upper.find('VIGENCIA')
            if pos_vigencia > 50:
                zona_antes = texto_upper[pos_vigencia - 50:pos_vigencia]
                nums = re.findall(r'\b(\d{4})\b', zona_antes)
                for n in nums:
                    if not (1900 <= int(n) <= 2100):
                        return n
        
        # Último recurso: buscar 4 dígitos que parezcan sección (típicamente 1000-9999, no años)
        # Secciones comunes: 2310, 3613, etc.
        all_nums = re.findall(r'\b(\d{4})\b', texto_upper)
        for n in all_nums:
            num = int(n)
            # Excluir años (1900-2100) y códigos postales (típicamente 5 dígitos)
            if 1000 <= num <= 9999 and not (1900 <= num <= 2100):
                # Verificar que no sea parte de CURP o clave de elector
                pos = texto_upper.find(n)
                if pos > 0:
                    contexto = texto_upper[max(0, pos-20):pos+20]
                    # Si está cerca de FECHA o NACIMIENTO, es probablemente una fecha
                    if 'FECHA' not in contexto and 'NACIMIENTO' not in contexto:
                        return n
        
        return None
    
    def extract_vigencia(self, texto: str) -> Optional[str]:
        """
        Extrae periodo de vigencia.
        Soporta formatos:
        - YYYY - YYYY (rango completo)
        - VIGENCIA YYYY (año único)
        - YYYY (año único sin etiqueta)
        """
        texto_upper = texto.upper()
        
        # Patrón 1: Rango completo YYYY - YYYY
        patron_rango = r'(\d{4})\s*[-–]\s*(\d{4})'
        matches = re.findall(patron_rango, texto)
        if matches:
            return f"{matches[0][0]} - {matches[0][1]}"
        
        # Patrón 2: VIGENCIA + año único
        patron_vigencia = r'VIGENCIA\s*:?\s*(\d{4})'
        match = re.search(patron_vigencia, texto_upper)
        if match:
            return match.group(1)
        
        # Patrón 3: Año único cerca de contexto de vigencia (2025-2035 range típico)
        patron_anio = r'\b(202[0-9]|203[0-9])\b'
        matches = re.findall(patron_anio, texto)
        # Si hay un año en rango típico de vigencia y está cerca de "VIGENCIA"
        if matches and 'VIGENCIA' in texto_upper:
            # Tomar el año más alto (fecha de expiración)
            return max(matches)
        
        return None
    
    def extract_anio_registro(self, texto: str) -> Optional[str]:
        """Extrae año de registro."""
        texto_upper = texto.upper()
        
        patrones = [
            r'A[ÑN]O\s*DE\s*REGISTRO\s*(\d{4})\s*(\d{2})',
            r'REGISTRO\s*(\d{4})\s*(\d{2})',
        ]
        
        for patron in patrones:
            matches = re.findall(patron, texto_upper)
            if matches:
                return f"{matches[0][0]} {matches[0][1]}"
        
        return None

    def extract_name(self, texto: str, detections: List[dict] = None) -> NameData:
        """
        Extrae nombre completo usando coordenadas espaciales.
        Busca texto debajo de "NOMBRE" y arriba de "DOMICILIO".
        """
        nombre_data = NameData()
        texto_upper = texto.upper()
        
        # Método 1: Usar coordenadas de PaddleOCR
        if detections and len(detections) > 0:
            nombre_coords = self._extract_name_by_coordinates(detections)
            if nombre_coords and nombre_coords.nombre_completo:
                if len(nombre_coords.nombre_completo.split()) >= 3:
                    # Filtro final: eliminar tokens de NO_NOMBRES
                    nombre_coords = self._filter_name_tokens(nombre_coords)
                    if nombre_coords and nombre_coords.nombre_completo:
                        return nombre_coords
        
        # Método 2: Búsqueda por texto
        nombre_texto = self._extract_name_by_text(texto_upper)
        
        # Elegir el mejor resultado
        if detections and nombre_coords and nombre_coords.nombre_completo:
            palabras_coords = len(nombre_coords.nombre_completo.split())
            palabras_texto = len(nombre_texto.nombre_completo.split()) if nombre_texto.nombre_completo else 0
            
            if palabras_coords >= palabras_texto:
                return self._filter_name_tokens(nombre_coords)
        
        # Filtro final para resultado de texto
        if nombre_texto and nombre_texto.nombre_completo:
            nombre_texto = self._filter_name_tokens(nombre_texto)
        
        return nombre_texto if nombre_texto and nombre_texto.nombre_completo else nombre_data
    
    def _filter_name_tokens(self, nombre_data: NameData) -> Optional[NameData]:
        """
        Filtro final para eliminar tokens de NO_NOMBRES del nombre.
        Esto cierra el bug de 'KICO', 'NOMEPE' y similares al 100%.
        """
        if not nombre_data or not nombre_data.nombre_completo:
            return nombre_data
        
        palabras = nombre_data.nombre_completo.split()
        palabras_limpias = []
        
        for i, palabra in enumerate(palabras):
            # Rechazar si está en NO_NOMBRES
            if palabra in self.NO_NOMBRES:
                continue
            # Rechazar variantes de "NOMBRE" (NOM + letras que no forman nombre válido)
            if re.match(r'^NOM[A-Z]{2,4}$', palabra) and palabra not in self.NOMBRES_COMUNES:
                continue
            # Rechazar si es primera palabra y parece etiqueta mal leída
            if i == 0 or len(palabras_limpias) == 0:
                es_etiqueta = False
                for patron in self.PATRONES_PRIMERA_PALABRA_BASURA:
                    if re.match(patron, palabra):
                        es_etiqueta = True
                        break
                if es_etiqueta:
                    continue
            # Rechazar palabras cortas (<=4) que no son nombres/apellidos conocidos
            if len(palabra) <= 4:
                if palabra not in self.NOMBRES_COMUNES and palabra not in self.APELLIDOS_COMUNES:
                    continue
            palabras_limpias.append(palabra)
        
        # Necesitamos al menos 2 palabras para un nombre válido
        if len(palabras_limpias) < 2:
            return None
        
        return NameData(
            apellido_paterno=palabras_limpias[0] if len(palabras_limpias) >= 1 else None,
            apellido_materno=palabras_limpias[1] if len(palabras_limpias) >= 2 else None,
            nombre=' '.join(palabras_limpias[2:]) if len(palabras_limpias) >= 3 else None,
            nombre_completo=' '.join(palabras_limpias)
        )
    
    def _extract_name_by_coordinates(self, detections: List[dict]) -> Optional[NameData]:
        """
        Extrae nombre usando coordenadas de detecciones.
        Ordena por Y (línea) primero, luego por X (posición horizontal).
        El nombre en INE está en formato vertical:
        - Línea 1: Apellido Paterno
        - Línea 2: Apellido Materno
        - Línea 3: Nombre(s)
        """
        if not detections:
            return None
        
        # Eliminar duplicados
        detections = self._filter_duplicate_detections(detections)
        
        # Buscar posición de etiquetas
        pos_nombre = None
        pos_domicilio = None
        pos_sexo = None
        
        for det in detections:
            texto = det.get("text", "").upper().strip()
            bbox = det.get("bbox", [])
            
            if not bbox or len(bbox) < 4:
                continue
            
            try:
                y_centro = (bbox[0][1] + bbox[2][1]) / 2
                x_inicio = bbox[0][0]
                x_fin = bbox[1][0] if len(bbox) > 1 else x_inicio + 100
            except:
                continue
            
            if texto == "NOMBRE":
                pos_nombre = {"y": y_centro, "x": x_inicio, "x_fin": x_fin}
            elif "DOMICILIO" in texto:
                pos_domicilio = {"y": y_centro}
            elif texto in ["SEXO", "SEXOH", "SEXOM"] or texto.startswith("SEXO"):
                pos_sexo = {"y": y_centro, "x": x_inicio}
        
        if not pos_nombre:
            return None
        
        # Definir zona del nombre
        # Y: desde la etiqueta NOMBRE hasta DOMICILIO
        y_min = pos_nombre["y"]
        y_max = pos_domicilio["y"] if pos_domicilio else float('inf')
        
        # X: desde la etiqueta NOMBRE hacia la derecha, pero NO donde está SEXO
        x_min = pos_nombre["x"] - 50  # Un poco a la izquierda de NOMBRE
        x_max_sexo = pos_sexo["x"] - 20 if pos_sexo else float('inf')  # Antes de SEXO
        
        # Buscar textos en la zona
        textos_en_zona = []
        
        for det in detections:
            texto = det.get("text", "").upper().strip()
            bbox = det.get("bbox", [])
            
            if not bbox or len(bbox) < 4 or not texto:
                continue
            
            try:
                y_centro = (bbox[0][1] + bbox[2][1]) / 2
                x_inicio = bbox[0][0]
            except:
                continue
            
            # Filtrar por zona Y (debajo de NOMBRE, arriba de DOMICILIO)
            if y_centro <= y_min or y_centro >= y_max:
                continue
            
            texto_limpio = texto.replace(".", "").replace(",", "").strip()
            
            # Filtrar basura
            if len(texto_limpio) < 2 or texto_limpio in self.NO_NOMBRES:
                continue
            
            if any(c.isdigit() for c in texto_limpio):
                continue
            
            # Excluir si está en la zona de SEXO (a la derecha)
            if pos_sexo:
                # Si está en la misma línea que SEXO y a su derecha, excluir
                if abs(y_centro - pos_sexo["y"]) < 30 and x_inicio >= pos_sexo["x"] - 50:
                    continue
            
            # Excluir si está muy a la derecha (zona de SEXO/foto derecha)
            if x_inicio > x_max_sexo:
                continue
            
            textos_en_zona.append({"texto": texto_limpio, "y": y_centro, "x": x_inicio})
        
        if not textos_en_zona:
            return None
        
        # Agrupar por líneas y ordenar (incluye separación de nombres pegados)
        palabras = self._group_and_sort_words(textos_en_zona)
        
        if not palabras:
            return None
        
        # Limpiar basura OCR del nombre completo
        nombre_completo = " ".join(palabras)
        nombre_completo = self._clean_name_garbage(nombre_completo)
        palabras = nombre_completo.split() if nombre_completo else []
        
        if not palabras:
            return None
        
        # NUEVO: Construir nombre de manera inteligente (maneja casos con 2 tokens)
        return self._build_name_data_smart(palabras)
    
    def _filter_duplicate_detections(self, detections: List[dict]) -> List[dict]:
        """
        Elimina detecciones duplicadas SOLO si están en la misma posición.
        IMPORTANTE: NO eliminar apellidos duplicados como "CRUZ CRUZ" que están en líneas diferentes.
        """
        vistos = {}
        resultado = []
        
        for det in detections:
            texto = det.get("text", "").upper().strip()
            bbox = det.get("bbox", [])
            
            if not texto or len(texto) < 2:
                continue
            
            try:
                y_centro = (bbox[0][1] + bbox[2][1]) / 2 if bbox and len(bbox) >= 4 else 0
                x_centro = (bbox[0][0] + bbox[1][0]) / 2 if bbox and len(bbox) >= 4 else 0
            except:
                y_centro = 0
                x_centro = 0
            
            if texto in vistos:
                # Solo considerar duplicado si está MUY cerca (misma posición exacta)
                # Usar tolerancia más estricta: 20px en Y y 30px en X
                if abs(y_centro - vistos[texto]["y"]) < 20 and abs(x_centro - vistos[texto]["x"]) < 30:
                    continue
            
            vistos[texto] = {"y": y_centro, "x": x_centro}
            resultado.append(det)
        
        return resultado
    
    def _group_and_sort_words(self, textos: List[dict]) -> List[str]:
        """
        Agrupa palabras por línea y ordena.
        IMPORTANTE: Permite apellidos repetidos (ej: CRUZ CRUZ) cuando están en líneas diferentes.
        """
        TOLERANCIA_Y = 25
        
        textos.sort(key=lambda x: x["y"])
        
        lineas = []
        linea_actual = []
        y_linea = None
        
        for item in textos:
            if y_linea is None:
                y_linea = item["y"]
                linea_actual = [item]
            elif abs(item["y"] - y_linea) <= TOLERANCIA_Y:
                linea_actual.append(item)
            else:
                if linea_actual:
                    lineas.append(linea_actual)
                linea_actual = [item]
                y_linea = item["y"]
        
        if linea_actual:
            lineas.append(linea_actual)
        
        # Ordenar palabras dentro de cada línea
        for linea in lineas:
            linea.sort(key=lambda x: x["x"])
        
        # Extraer palabras con separación de nombres pegados
        # CAMBIO: Usar dict con posición Y para permitir apellidos repetidos en líneas diferentes
        palabras = []
        vistos_por_linea = {}  # {palabra: [y1, y2, ...]} para detectar duplicados en MISMA línea
        
        for idx_linea, linea in enumerate(lineas):
            y_linea_actual = linea[0]["y"] if linea else 0
            
            for item in linea:
                palabra = item["texto"].upper().strip()
                
                # Corregir errores OCR comunes primero
                palabra = self._correct_ocr_name(palabra)
                
                if not palabra.isalpha() or len(palabra) < 2:
                    continue
                
                # Verificar si es duplicado en la MISMA línea (error OCR real)
                # Pero permitir duplicados en líneas DIFERENTES (apellidos repetidos legítimos)
                if palabra in vistos_por_linea:
                    # Verificar si ya apareció en esta misma línea (tolerancia Y)
                    es_duplicado_misma_linea = any(
                        abs(y_linea_actual - y_prev) < TOLERANCIA_Y 
                        for y_prev in vistos_por_linea[palabra]
                    )
                    if es_duplicado_misma_linea:
                        continue  # Duplicado real en misma línea, omitir
                    # Si está en línea diferente, es apellido repetido legítimo - permitir
                
                # Registrar posición Y de esta palabra
                if palabra not in vistos_por_linea:
                    vistos_por_linea[palabra] = []
                vistos_por_linea[palabra].append(y_linea_actual)
                
                # Intentar separar nombres pegados
                palabra_separada = self._separate_stuck_names(palabra)
                if ' ' in palabra_separada:
                    # Se separó en múltiples palabras
                    for p in palabra_separada.split():
                        if p.isalpha() and len(p) >= 2:
                            palabras.append(p)
                else:
                    palabras.append(palabra)
        
        # Limpiar basura
        palabras_limpias = []
        for p in palabras:
            if p not in self.NO_NOMBRES and len(palabras_limpias) < 5:
                palabras_limpias.append(p)
        
        return palabras_limpias
    
    def _extract_name_by_text(self, texto_upper: str) -> NameData:
        """Extrae nombre por búsqueda de texto."""
        nombre_data = NameData()
        
        pos_nombre = texto_upper.find('NOMBRE')
        if pos_nombre == -1:
            pos_nombre = 0
        else:
            pos_nombre += 6
        
        fin_nombre = len(texto_upper)
        for etiqueta in ['DOMICILIO', 'CLAVE DE ELECTOR', 'CURP', 'FECHA DE NACIMIENTO']:
            pos = texto_upper.find(etiqueta, pos_nombre)
            if pos != -1 and pos < fin_nombre:
                fin_nombre = pos
        
        bloque = texto_upper[pos_nombre:fin_nombre]
        bloque = ' '.join(bloque.split())
        
        palabras_candidatas = re.findall(r'\b([A-ZÁÉÍÓÚÑ]{2,})\b', bloque)
        
        palabras_nombre = []
        for palabra in palabras_candidatas:
            if palabra in self.NO_NOMBRES:
                continue
            # Rechazar palabras >12 chars (probablemente domicilio pegado)
            if len(palabra) > 12:
                continue
            # Rechazar si termina en abreviatura de estado
            if len(palabra) > 5 and palabra[-3:] in {'VER', 'PUE', 'OAX', 'GRO', 'JAL', 'MEX', 'GTO'}:
                continue
            
            if len(palabras_nombre) < 5:
                # Intentar separar nombres pegados
                palabra_separada = self._separate_stuck_names(palabra)
                if ' ' in palabra_separada:
                    # Se separó en múltiples palabras - filtrar cada una
                    for p in palabra_separada.split():
                        if p not in self.NO_NOMBRES and len(palabras_nombre) < 5:
                            if len(p) <= 12:  # También filtrar las separadas
                                palabras_nombre.append(p)
                else:
                    palabras_nombre.append(palabra)
        
        # Limpiar basura OCR del nombre completo
        if palabras_nombre:
            nombre_completo = " ".join(palabras_nombre)
            nombre_completo = self._clean_name_garbage(nombre_completo)
            palabras_nombre = nombre_completo.split() if nombre_completo else []
        
        if palabras_nombre:
            nombre_data.apellido_paterno = palabras_nombre[0] if len(palabras_nombre) >= 1 else None
            nombre_data.apellido_materno = palabras_nombre[1] if len(palabras_nombre) >= 2 else None
            nombre_data.nombre = " ".join(palabras_nombre[2:]) if len(palabras_nombre) >= 3 else None
            nombre_data.nombre_completo = " ".join(palabras_nombre)
        
        return nombre_data

    def _reconstruir_lineas_mrz(self, mrz_clean: str) -> Optional[List[str]]:
        """
        Reconstruye líneas MRZ INE (típicamente 3) desde mrz_clean (solo desde IDMEX...).
        
        Estrategia (orden de prioridad):
        1) Si hay \n, usarlo (limpiando y tomando desde IDMEX/1DMEX).
        2) Si NO hay \n:
           2.1) Anclar línea 3 por patrón de nombres: APELLIDO<APELLIDO<<NOMBRES
           2.2) En el prefijo (líneas 1+2 pegadas), anclar inicio de línea 2 por patrón fecha+sexo.
        3) Fallback final: corte por longitud típica (~90) con tolerancia.
        
        Args:
            mrz_clean: MRZ limpio validado (solo desde IDMEX...)
            
        Returns:
            Lista con 2-3 líneas o None si no se puede reconstruir
        """
        if not mrz_clean:
            return None
        
        # Normalizar charset (mantener \n si existiera)
        s = mrz_clean.upper()
        s = re.sub(r'[^A-Z0-9<\n]', '', s)
        s = re.sub(r'\n+', '\n', s).strip()
        
        # --- 1) Si hay saltos de línea, usarlos ---
        if '\n' in s:
            lines = [ln.strip() for ln in s.split('\n') if ln.strip()]
            # Quedarse desde IDMEX/1DMEX si hay basura antes
            idx = next((i for i, ln in enumerate(lines)
                        if ln.startswith('IDMEX') or ln.startswith('1DMEX')), None)
            if idx is not None:
                lines = lines[idx:]
            else:
                # NUEVO: Si ninguna línea empieza con IDMEX, buscar IDMEX dentro de líneas y recortar
                lines_recortadas = []
                for ln in lines:
                    # Buscar IDMEX/1DMEX dentro de la línea
                    pos_idmex = ln.find('IDMEX')
                    if pos_idmex == -1:
                        pos_idmex = ln.find('1DMEX')
                    if pos_idmex != -1:
                        # Recortar desde IDMEX
                        ln_recortada = ln[pos_idmex:]
                        if len(ln_recortada) >= 30:
                            lines_recortadas.append(ln_recortada)
                    elif len(ln) >= 30:
                        # Si no tiene IDMEX pero es larga, mantener (puede ser línea 2 o 3)
                        lines_recortadas.append(ln)
                if lines_recortadas:
                    lines = lines_recortadas
            
            lines = [ln for ln in lines if len(ln) >= 30]  # Filtrar ruido corto
            return lines[:3] if len(lines) >= 2 else None
        
        # --- 2) Sin \n: heurística por anclas ---
        # 2.1) Anclar línea 3 (nombres)
        patron_linea3 = r'[A-Z]{2,}<[A-Z]{2,}<<[A-Z<]{2,}'
        m3 = re.search(patron_linea3, s)
        if not m3:
            # No hay nombres => muy arriesgado reconstruir
            # Fallback a longitud típica
            return self._reconstruir_mrz_por_longitud(s)
        
        start3 = m3.start()
        pref = s[:start3]
        line3 = s[start3:].strip()
        
        # 2.2) Dentro de pref (líneas 1+2 pegadas), ubicar inicio de línea 2
        # Después de "IDMEX"+10 dígitos suele venir línea 2 o parte de ella
        pos_after_idmex = 14
        search_from = min(len(pref), pos_after_idmex)
        
        # Patrón típico en línea 2: YYMMDD + (check opcional) + SEXO(H/M)
        # Ej: 0711135H... (puede tener < intercalados)
        # También manejar variantes OCR donde H/M se lee como N/K
        m2 = re.search(r'\d{6}[0-9<]{0,3}[HMNK]', pref[search_from:])
        if m2:
            start2 = search_from + m2.start()
            line1 = pref[:start2].strip()
            line2 = pref[start2:].strip()
        else:
            # Si no se detecta, fallback: split por longitud en pref
            line1, line2 = self._split_pref_por_longitud(pref)
        
        # Validación mínima: línea 1 debe empezar por IDMEX/1DMEX
        if not (line1.startswith('IDMEX') or line1.startswith('1DMEX')):
            # Si se movió el split mal, intentar arreglar moviendo el corte al primer YYMMDD...
            m2b = re.search(r'\d{6}', pref)
            if m2b and m2b.start() > 10:
                line1 = pref[:m2b.start()].strip()
                line2 = pref[m2b.start():].strip()
        
        lines = [ln for ln in [line1, line2, line3] if ln and len(ln) >= 20]
        return lines if len(lines) >= 2 else None
    
    def _split_pref_por_longitud(self, pref: str) -> Tuple[str, str]:
        """
        Split auxiliar para separar pref (líneas 1+2) por longitud típica,
        sin tocar línea 3. Pref suele estar cerca de 2*90 pero puede variar.
        
        Args:
            pref: Prefijo que contiene líneas 1+2 pegadas
            
        Returns:
            Tupla (linea1, linea2)
        """
        LONG = 90
        TOL = 15
        if len(pref) <= LONG:
            return pref.strip(), ""
        
        # Buscar el mejor corte cercano a LONG (más cercano a 90, o último válido)
        # Esto evita cortar demasiado pronto cuando hay varios dígitos seguidos
        best = LONG
        start = max(30, LONG - TOL)
        end = min(len(pref), LONG + TOL)
        
        candidatos = []
        for i in range(start, end):
            if pref[i-1].isdigit() or pref[i-1] == '<':
                # Guardar candidato con su distancia a LONG
                distancia = abs(i - LONG)
                candidatos.append((distancia, i))
        
        if candidatos:
            # Elegir el más cercano a LONG (menor distancia)
            candidatos.sort(key=lambda x: x[0])
            best = candidatos[0][1]
        else:
            # Si no hay candidatos válidos, usar LONG directamente
            best = LONG
        
        return pref[:best].strip(), pref[best:].strip()
    
    def _reconstruir_mrz_por_longitud(self, s: str) -> Optional[List[str]]:
        """
        Último recurso: cortar en ~90 chars cuando no se pueden usar anclas.
        
        Args:
            s: Texto MRZ sin saltos de línea
            
        Returns:
            Lista de líneas o None si no se puede reconstruir
        """
        LONG = 90
        TOL = 15
        t = s.replace('\n', '').strip()
        if len(t) < 60:
            return None
        
        cortes = []
        # Corte 1: buscar el mejor punto de corte (más cercano a LONG)
        c1 = min(LONG, len(t))
        candidatos1 = []
        for i in range(max(30, LONG - TOL), min(LONG + TOL, len(t))):
            if t[i-1].isdigit() or t[i-1] == '<':
                distancia = abs(i - LONG)
                candidatos1.append((distancia, i))
        
        if candidatos1:
            candidatos1.sort(key=lambda x: x[0])
            c1 = candidatos1[0][1]
        
        cortes.append(t[:c1])
        rest = t[c1:]
        
        if len(rest) < 30:
            return cortes if len(cortes) >= 1 else None
        
        # Corte 2: buscar el mejor punto de corte (más cercano a LONG)
        c2 = min(LONG, len(rest))
        candidatos2 = []
        for i in range(max(30, LONG - TOL), min(LONG + TOL, len(rest))):
            if rest[i-1].isdigit() or rest[i-1] == '<':
                distancia = abs(i - LONG)
                candidatos2.append((distancia, i))
        
        if candidatos2:
            candidatos2.sort(key=lambda x: x[0])
            c2 = candidatos2[0][1]
        cortes.append(rest[:c2])
        rest2 = rest[c2:]
        if rest2:
            cortes.append(rest2)
        
        cortes = [x.strip() for x in cortes if len(x.strip()) >= 30]
        return cortes[:3] if len(cortes) >= 2 else None
    
    def _validate_and_clean_mrz_text(self, text: str) -> Optional[str]:
        """
        Valida y limpia texto MRZ (misma lógica que OCREngine._validate_and_clean_mrz).
        Usado para guardar MRZ limpio en extract_mrz.
        """
        if not text:
            return None
        
        # Normalizar: solo A-Z, 0-9, < y saltos de línea
        text_upper = text.upper()
        text_clean = re.sub(r'[^A-Z0-9<\n]', '', text_upper)
        
        # Localizar última ocurrencia de IDMEX
        idx_idmex = text_clean.rfind('IDMEX')
        if idx_idmex == -1:
            idx_idmex = text_clean.rfind('1DMEX')
            if idx_idmex == -1:
                return None
        
        # Recortar desde IDMEX
        mrz_candidate = text_clean[idx_idmex:]
        
        # Validar estructura
        if not re.search(r'^(IDMEX|1DMEX)\d{10}', mrz_candidate):
            return None
        
        pos_despues_idmex = 14
        if len(mrz_candidate) > pos_despues_idmex:
            ventana = mrz_candidate[pos_despues_idmex:pos_despues_idmex + 100]
            if 'MEX' not in ventana:
                return None
        
        count_lt = mrz_candidate.count('<')
        if count_lt < 18:
            return None
        
        patron_linea3 = r'[A-Z]{2,}<[A-Z]{2,}<<[A-Z<]{2,}'
        if not re.search(patron_linea3, mrz_candidate):
            return None
        
        if len(mrz_candidate) > 30:
            tramo_final = mrz_candidate[-30:]
            if not re.search(r'<{3,}', tramo_final):
                if not re.search(r'<{5,}', mrz_candidate):
                    return None
        else:
            if not re.search(r'<{3,}', mrz_candidate):
                return None
        
        if len(mrz_candidate) < 100:
            return None
        
        mrz_candidate = re.sub(r'\n+', '\n', mrz_candidate)
        mrz_candidate = mrz_candidate.strip()
        
        return mrz_candidate
    
    def extract_mrz(self, texto: str) -> MRZData:
        """
        Parsea las 3 líneas del MRZ mexicano.
        
        Formato:
        Línea 1: IDMEX + número_doc(10) + << + otros
        Línea 2: fecha_nac(YYMMDD) + check + sexo + fecha_exp + MEX + ...
        Línea 3: APELLIDO<APELLIDO<<NOMBRE<NOMBRE
        
        También maneja variantes OCR donde < se lee como K, L, o espacios.
        """
        mrz_data = MRZData()
        
        # Normalizar texto: quitar espacios extras pero preservar estructura de líneas
        texto_original = texto.upper()
        texto_limpio = texto_original.replace(' ', '')
        
        # =====================================================================
        # EXTRAER LÍNEAS RAW DEL MRZ (para debugging)
        # =====================================================================
        lineas_mrz = []
        for linea in texto_original.split('\n'):
            linea_limpia = linea.strip().upper()
            linea_sin_espacios = linea_limpia.replace(' ', '')
            
            # Una línea MRZ típicamente:
            # - Contiene < o K (OCR error de <)
            # - Empieza con IDMEX
            # - Es alfanumérica de 20+ caracteres
            # - Contiene MEX
            es_mrz = (
                '<' in linea_limpia or 
                linea_sin_espacios.startswith('IDMEX') or 
                'MEX<' in linea_sin_espacios or
                'MEX' in linea_sin_espacios and len(linea_sin_espacios) > 20 or
                re.match(r'^[A-Z0-9<]{20,}$', linea_sin_espacios) or
                # Línea de nombre: APELLIDO<APELLIDO<<NOMBRE
                re.match(r'^[A-Z]+<[A-Z]+<<[A-Z]+', linea_sin_espacios)
            )
            
            if es_mrz and len(linea_sin_espacios) >= 15:
                lineas_mrz.append(linea_limpia)
        
        # Guardar líneas raw (pueden estar contaminadas) - mantener tal cual para debugging
        mrz_data.lineas_raw = lineas_mrz if lineas_mrz else []
        
        # NUEVO: Validar y guardar MRZ limpio (solo desde IDMEX...)
        mrz_clean = self._validate_and_clean_mrz_text(texto)
        if mrz_clean:
            mrz_data.lineas_clean = mrz_clean
            
            # Reconstruir líneas estándar (2-3 líneas) a partir de lineas_clean
            # El MRZ típicamente tiene 3 líneas de ~90 caracteres cada una
            lineas_estandar = self._reconstruir_lineas_mrz(mrz_clean)
            if lineas_estandar:
                # Usar líneas estándar reconstruidas para parsing (más confiable)
                texto_limpio = ''.join(lineas_estandar)  # Unir sin espacios
            else:
                # Fallback: usar texto limpio sin saltos de línea
                texto_limpio = mrz_clean.replace('\n', '')
        else:
            # Si no hay MRZ limpio, usar texto original (comportamiento anterior)
            texto_limpio = texto_original.replace(' ', '')
        
        # =====================================================================
        # CORREGIR ERRORES OCR COMUNES EN MRZ
        # =====================================================================
        # K -> < (muy común)
        # L -> < (a veces)
        # Espacios -> nada
        texto_mrz = texto_limpio.replace('K', '<')
        
        # =====================================================================
        # 1. BUSCAR NÚMERO DE DOCUMENTO (IDMEX + 10 dígitos)
        # =====================================================================
        match_doc = re.search(r'IDMEX(\d{10})', texto_mrz)
        if match_doc:
            mrz_data.documento_tipo = "ID"
            mrz_data.pais = "MEX"
            mrz_data.numero_documento = match_doc.group(1)
        
        # =====================================================================
        # 2. BUSCAR NOMBRE EN FORMATO MRZ
        # Formato: APELLIDO1<APELLIDO2<<NOMBRE1<NOMBRE2
        # =====================================================================
        # Patrón principal: AP1<AP2<<NOMBRE (con posibles < extras al final)
        # Límite de 25 chars por componente para evitar capturar texto externo al MRZ
        patron_nombre = r'([A-Z]{2,25})<([A-Z]{2,25})<<([A-Z]{2,25})(?:<([A-Z]{2,20}))?(?:<+)?'
        match_nombre = re.search(patron_nombre, texto_mrz)
        
        if match_nombre:
            ap1 = match_nombre.group(1)
            ap2 = match_nombre.group(2)
            nombre1 = match_nombre.group(3)
            nombre2 = match_nombre.group(4) if match_nombre.group(4) else ""
            
            mrz_data.apellido_paterno = ap1
            mrz_data.apellido_materno = ap2
            mrz_data.nombres = f"{nombre1} {nombre2}".strip() if nombre2 else nombre1
            
            partes = [ap1, ap2, nombre1]
            if nombre2:
                partes.append(nombre2)
            mrz_data.nombre_completo = " ".join(partes)
        else:
            # Fallback: buscar patrón más flexible
            # A veces el OCR lee mal los < y quedan como espacios o se pierden
            patron_flexible = r'([A-Z]{3,})\s*[<\s]+\s*([A-Z]{3,})\s*[<\s]{2,}\s*([A-Z]{2,})'
            match_flex = re.search(patron_flexible, texto_original)
            if match_flex:
                mrz_data.apellido_paterno = match_flex.group(1)
                mrz_data.apellido_materno = match_flex.group(2)
                mrz_data.nombres = match_flex.group(3)
                mrz_data.nombre_completo = f"{match_flex.group(1)} {match_flex.group(2)} {match_flex.group(3)}"
        
        # =====================================================================
        # 3. BUSCAR FECHA Y SEXO
        # Formato línea 2: YYMMDD + check_digit + H/M + ...
        # =====================================================================
        patron_fecha_sexo = r'(\d{2})(\d{2})(\d{2})\d([HM])'
        matches_fecha = re.findall(patron_fecha_sexo, texto_mrz)
        
        for yy, mm, dd, sexo in matches_fecha:
            try:
                mm_int, dd_int, yy_int = int(mm), int(dd), int(yy)
                if 1 <= mm_int <= 12 and 1 <= dd_int <= 31:
                    año = f"19{yy}" if yy_int > 30 else f"20{yy}"
                    mrz_data.fecha_nacimiento = f"{dd}/{mm}/{año}"
                    mrz_data.sexo = sexo
                    break
            except:
                continue
        
        # Fallback para fecha/sexo
        if not mrz_data.fecha_nacimiento:
            patron_simple = r'(\d{6})([HM])'
            matches = re.findall(patron_simple, texto_mrz)
            for fecha_str, sexo in matches:
                yy, mm, dd = fecha_str[0:2], fecha_str[2:4], fecha_str[4:6]
                try:
                    mm_int, dd_int, yy_int = int(mm), int(dd), int(yy)
                    if 1 <= mm_int <= 12 and 1 <= dd_int <= 31:
                        año = f"19{yy}" if yy_int > 30 else f"20{yy}"
                        mrz_data.fecha_nacimiento = f"{dd}/{mm}/{año}"
                        mrz_data.sexo = sexo
                        break
                except:
                    continue
        
        return mrz_data
    
    def parse_address(self, texto: str) -> AddressData:
        """
        Parsea domicilio separando componentes usando heurística posicional.
        
        Estructura típica de INE (modelos E/F/G/H):
        - Línea 1: CALLE + NÚMERO
        - Línea 2: COLONIA + CP (a veces)
        - Línea 3: MUNICIPIO, ESTADO + CP
        
        Maneja palabras pegadas por OCR.
        """
        address = AddressData()
        texto_upper = texto.upper()
        
        # Buscar bloque de domicilio
        patron = r'DOMICILIO\s*\n?\s*([\s\S]*?)(?=\s*(?:CLAVE|CURP|FECHA|AÑO|SECCI[OÓ]N|VIGENCIA|REGISTRO|NACIMIENTO|ELECTOR|$))'
        matches = re.findall(patron, texto_upper, re.IGNORECASE)
        
        if not matches:
            # Intentar extraer sin etiqueta DOMICILIO
            return address
        
        domicilio_raw = matches[0].strip()
        domicilio_raw = re.sub(r'\b(SEXO|H|M|NOMBRE)\b', '', domicilio_raw)
        
        # Separar palabras pegadas primero
        domicilio_raw = self._separate_stuck_words(domicilio_raw)
        
        # Dividir por líneas (el OCR preserva saltos de línea)
        lineas = [l.strip() for l in domicilio_raw.split('\n') if len(l.strip()) > 2]
        
        # Si no hay saltos de línea, intentar dividir por patrones conocidos
        if len(lineas) <= 1:
            lineas = self._split_address_by_patterns(domicilio_raw)
        
        # Aplicar separación de palabras pegadas a cada línea
        lineas = [self._separate_stuck_words(l) for l in lineas]
        
        domicilio_completo = ' '.join(domicilio_raw.split())
        # Aplicar separación de palabras pegadas al domicilio completo
        domicilio_completo = self._separate_stuck_words(domicilio_completo)
        address.domicilio_completo = domicilio_completo
        
        # Extraer código postal (5 dígitos) - buscar primero
        cp_match = re.search(r'\b(\d{5})\b', domicilio_completo)
        if cp_match:
            address.codigo_postal = cp_match.group(1)
        
        # Extraer estado (abreviatura al final)
        for abrev, nombre_completo in self.ESTADOS_ABREV.items():
            # Buscar abreviatura al final o antes de punto
            patron_estado = rf'[,.\s]{abrev}\.?\s*$|[,.\s]{abrev}\.'
            if re.search(patron_estado, domicilio_completo):
                address.estado = abrev
                break
        
        # =========================================================================
        # HEURÍSTICA POSICIONAL: Usar estructura de líneas
        # =========================================================================
        if len(lineas) >= 3:
            # Línea 1: Calle + número
            linea_calle = lineas[0]
            calle_parsed = self._parse_calle(linea_calle)
            if calle_parsed:
                address.calle = calle_parsed.get('calle')
                address.numero_exterior = calle_parsed.get('numero')
                address.numero_interior = calle_parsed.get('interior')
            
            # Línea 2 (y posiblemente parte de 3): Colonia
            linea_colonia = lineas[1]
            # Si hay más de 3 líneas, la colonia puede estar en 2 líneas
            if len(lineas) > 3:
                linea_colonia = ' '.join(lineas[1:-1])
            
            colonia_parsed = self._parse_colonia(linea_colonia)
            if colonia_parsed:
                address.colonia = colonia_parsed
            
            # Última línea: Municipio, Estado
            linea_final = lineas[-1]
            mun_estado = self._parse_municipio_estado(linea_final, address.codigo_postal, address.estado)
            if mun_estado:
                address.municipio = mun_estado.get('municipio')
                if not address.estado and mun_estado.get('estado'):
                    address.estado = mun_estado.get('estado')
        
        elif len(lineas) == 2:
            # Solo 2 líneas: Calle en línea 1, resto en línea 2
            calle_parsed = self._parse_calle(lineas[0])
            if calle_parsed:
                address.calle = calle_parsed.get('calle')
                address.numero_exterior = calle_parsed.get('numero')
            
            # Línea 2 tiene colonia + municipio + estado
            self._parse_combined_line(lineas[1], address)
        
        else:
            # Una sola línea o texto sin estructura - usar regex tradicional
            self._parse_flat_address(domicilio_completo, address)
        
        # =========================================================================
        # POST-PROCESAMIENTO: Limpiar campos extraídos
        # =========================================================================
        # Aplicar separación de palabras pegadas a campos individuales
        if address.calle:
            address.calle = self._separate_stuck_words(address.calle)
        if address.colonia:
            address.colonia = self._separate_stuck_words(address.colonia)
        if address.municipio:
            address.municipio = self._separate_stuck_words(address.municipio)
        
        return address
    
    def _split_address_by_patterns(self, texto: str) -> List[str]:
        """
        Divide dirección sin saltos de línea usando heurísticas inteligentes.
        
        Estrategia:
        1. Si hay COL/FRACC, dividir ahí
        2. Si hay CP (5 dígitos), usar como punto de división
        3. Si hay número + texto largo, asumir calle + resto
        4. Buscar patrones de municipio/estado al final
        """
        lineas = []
        texto = texto.strip()
        
        # ESTRATEGIA 1: Buscar inicio de colonia (COL, FRACC, UNIDAD, etc.)
        col_match = re.search(r'\b(COL(?:ONIA)?|FRACC?(?:IONAMIENTO)?|UNIDAD|BARRIO|RESIDENCIAL|INFONAVIT)\s+', texto, re.IGNORECASE)
        if col_match:
            # Todo antes de COL es calle
            calle = texto[:col_match.start()].strip()
            if calle and len(calle) > 3:
                lineas.append(calle)
            resto = texto[col_match.start():].strip()
            
            # Buscar CP para dividir colonia de municipio
            cp_match = re.search(r'\b(\d{5})\b', resto)
            if cp_match:
                colonia = resto[:cp_match.end()].strip()
                municipio = resto[cp_match.end():].strip()
                if colonia:
                    lineas.append(colonia)
                if municipio and len(municipio) > 2:
                    lineas.append(municipio)
            else:
                lineas.append(resto)
            return lineas if lineas else [texto]
        
        # ESTRATEGIA 2: Sin COL explícito - usar CP como divisor principal
        cp_match = re.search(r'\b(\d{5})\b', texto)
        if cp_match:
            antes_cp = texto[:cp_match.start()].strip()
            despues_cp = texto[cp_match.end():].strip()
            
            # Intentar dividir "antes_cp" en calle y colonia
            # Buscar número de calle (1-4 dígitos seguidos de espacio o fin)
            num_calle_match = re.search(r'^(.+?)\s+(\d{1,4})\s+(.+)$', antes_cp)
            if num_calle_match:
                # Patrón: CALLE NUM COLONIA
                calle = f"{num_calle_match.group(1)} {num_calle_match.group(2)}"
                colonia = num_calle_match.group(3)
                lineas.append(calle.strip())
                lineas.append(f"{colonia} {cp_match.group(1)}".strip())
            else:
                # No hay número claro, todo antes del CP es calle+colonia
                lineas.append(antes_cp)
            
            # Después del CP es municipio/estado
            if despues_cp and len(despues_cp) > 2:
                lineas.append(despues_cp)
            
            return lineas if lineas else [texto]
        
        # ESTRATEGIA 3: Sin CP - buscar estado al final
        for abrev in self.ESTADOS_ABREV.keys():
            patron = rf'(.+?)\s*[,.]?\s*({abrev})\.?\s*$'
            match = re.search(patron, texto, re.IGNORECASE)
            if match:
                contenido = match.group(1).strip()
                # Intentar dividir contenido en calle y colonia
                num_match = re.search(r'^(.+?)\s+(\d{1,4})\s+(.+)$', contenido)
                if num_match:
                    lineas.append(f"{num_match.group(1)} {num_match.group(2)}")
                    lineas.append(num_match.group(3))
                else:
                    lineas.append(contenido)
                lineas.append(match.group(2))
                return lineas
        
        # ESTRATEGIA 4: Fallback - buscar número de calle y dividir
        num_match = re.search(r'^(.+?)\s+(\d{1,4})\s+(.+)$', texto)
        if num_match:
            lineas.append(f"{num_match.group(1)} {num_match.group(2)}")
            lineas.append(num_match.group(3))
            return lineas
        
        # Sin estructura detectada
        return [texto]
    
    def _parse_calle(self, texto: str) -> Optional[dict]:
        """Extrae calle y número de una línea."""
        if not texto:
            return None
        
        resultado = {'calle': None, 'numero': None, 'interior': None}
        texto_original = texto
        
        # Limpiar prefijos comunes pero guardarlos
        prefijo = ""
        prefijo_match = re.match(r'^(C\s+|CALLE\s+|AV\.?\s+|AVENIDA\s+|PRIV\.?\s+|PRIVADA\s+|BLVD?\.?\s+|BOULEVARD\s+|AND\.?\s+|ANDADOR\s+|CERRADA\s+|CDA\.?\s+)', texto, re.IGNORECASE)
        if prefijo_match:
            prefijo = prefijo_match.group(1).strip() + " "
            texto = texto[prefijo_match.end():].strip()
        
        # Buscar número exterior con posible interior
        # Patrones: "NOMBRE 123", "NOMBRE NUM 123", "NOMBRE #123", "NOMBRE 123 INT 4", "NOMBRE 123-A"
        num_match = re.search(r'\s+(?:NUM\.?\s*)?#?(\d+)(?:\s*[-]?\s*([A-Z]))?(?:\s+(?:INT\.?|INTERIOR)\s*(\d+|[A-Z]))?$', texto, re.IGNORECASE)
        if num_match:
            resultado['numero'] = num_match.group(1)
            if num_match.group(2):
                resultado['numero'] += num_match.group(2)
            if num_match.group(3):
                resultado['interior'] = num_match.group(3)
            resultado['calle'] = (prefijo + texto[:num_match.start()]).strip()
        else:
            # Buscar S/N (sin número)
            sn_match = re.search(r'\s+S/?N\s*$', texto, re.IGNORECASE)
            if sn_match:
                resultado['calle'] = (prefijo + texto[:sn_match.start()]).strip()
                resultado['numero'] = 'S/N'
            else:
                # Sin número detectado
                resultado['calle'] = (prefijo + texto).strip()
        
        return resultado if resultado['calle'] else None
    
    def _parse_colonia(self, texto: str) -> Optional[str]:
        """Extrae nombre de colonia de una línea."""
        if not texto:
            return None
        
        # Quitar prefijo COL/COLONIA/FRACC pero mantener el nombre
        texto = re.sub(r'^(COL(?:ONIA)?\s+|FRACC?(?:IONAMIENTO)?\s+|UNIDAD\s+|BARRIO\s+|RESIDENCIAL\s+|INFONAVIT\s+)', '', texto, flags=re.IGNORECASE)
        
        # Quitar CP si está presente (al final o en medio)
        texto = re.sub(r'\s*\d{5}\s*', ' ', texto)
        
        # Quitar estado si está al final
        for abrev in self.ESTADOS_ABREV.keys():
            texto = re.sub(rf'\s*[,.]?\s*{abrev}\.?\s*$', '', texto, flags=re.IGNORECASE)
        
        # Limpiar espacios múltiples y puntuación al final
        texto = re.sub(r'\s+', ' ', texto).strip(' ,.')
        
        return texto if texto and len(texto) > 2 else None
    
    def _parse_municipio_estado(self, texto: str, cp: str, estado_conocido: str) -> Optional[dict]:
        """Extrae municipio y estado de la última línea."""
        if not texto:
            return None
        
        resultado = {'municipio': None, 'estado': None}
        
        # Quitar CP si está presente
        if cp:
            texto = texto.replace(cp, '').strip()
        texto = re.sub(r'\b\d{5}\b', '', texto).strip()
        
        # Buscar estado al final (puede ser abreviatura o nombre completo)
        for abrev, nombre in self.ESTADOS_ABREV.items():
            # Buscar abreviatura
            patron_abrev = rf'[,.\s]({abrev})\.?\s*$'
            match = re.search(patron_abrev, texto, re.IGNORECASE)
            if match:
                resultado['estado'] = abrev
                texto = texto[:match.start()].strip()
                break
            # Buscar nombre completo
            if nombre.upper() in texto.upper():
                resultado['estado'] = abrev
                texto = re.sub(rf'\s*,?\s*{nombre}\s*$', '', texto, flags=re.IGNORECASE).strip()
                break
        
        # Lo que queda es el municipio
        texto = texto.strip(' ,.')
        if texto and len(texto) > 2:
            resultado['municipio'] = texto.upper()
        
        return resultado if resultado['municipio'] or resultado['estado'] else None
    
    def _parse_combined_line(self, texto: str, address: AddressData):
        """Parsea línea combinada (colonia + municipio + estado)."""
        # Buscar colonia con prefijo
        col_match = re.search(r'(COL(?:ONIA)?|FRACC?|UNIDAD|BARRIO)\s+([A-ZÁÉÍÓÚÑ\s0-9]+?)(?:\s+\d{5}|,|$)', texto, re.IGNORECASE)
        if col_match:
            address.colonia = col_match.group(2).strip()
        else:
            # Sin prefijo COL - asumir que todo antes del CP es colonia
            if address.codigo_postal:
                cp_pos = texto.find(address.codigo_postal)
                if cp_pos > 0:
                    address.colonia = texto[:cp_pos].strip(' ,.')
        
        # Buscar municipio (después del CP)
        if address.codigo_postal:
            cp_pos = texto.find(address.codigo_postal)
            if cp_pos >= 0:
                resto = texto[cp_pos + 5:].strip()
                # Quitar estado del final
                for abrev in self.ESTADOS_ABREV.keys():
                    resto = re.sub(rf'\s*[,.]?\s*{abrev}\.?\s*$', '', resto, flags=re.IGNORECASE)
                if resto and len(resto) > 2:
                    address.municipio = resto.strip(' ,.').upper()
    
    def _parse_flat_address(self, texto: str, address: AddressData):
        """Parsea dirección sin estructura de líneas (fallback inteligente)."""
        # PASO 1: Extraer colonia si hay prefijo
        col_match = re.search(r'(COL(?:ONIA)?|FRACC?|UNIDAD|BARRIO)\s+([A-ZÁÉÍÓÚÑ\s0-9]+?)(?:\s+\d{5}|,|\s+[A-Z]{2,4}\.?\s*$)', texto, re.IGNORECASE)
        if col_match:
            address.colonia = col_match.group(2).strip()
            pos_col = col_match.start()
        else:
            pos_col = -1
        
        # PASO 2: Extraer calle (todo antes de COL o antes del CP si no hay COL)
        if pos_col > 5:
            calle_texto = texto[:pos_col].strip()
        elif address.codigo_postal:
            cp_pos = texto.find(address.codigo_postal)
            if cp_pos > 5:
                calle_texto = texto[:cp_pos].strip()
            else:
                calle_texto = None
        else:
            calle_texto = None
        
        if calle_texto:
            # Extraer número de la calle
            num_match = re.search(r'\s+(\d{1,4})(?:\s*[-]?([A-Z]))?\s*$', calle_texto, re.IGNORECASE)
            if num_match:
                address.numero_exterior = num_match.group(1)
                if num_match.group(2):
                    address.numero_exterior += num_match.group(2)
                address.calle = calle_texto[:num_match.start()].strip()
            else:
                address.calle = calle_texto
        
        # PASO 3: Extraer municipio (después del CP, antes del estado)
        if address.estado and address.codigo_postal:
            patron_mun = rf'{address.codigo_postal}\s+([A-ZÁÉÍÓÚÑ\s]+?)[,.]?\s*{address.estado}'
            match_mun = re.search(patron_mun, texto, re.IGNORECASE)
            if match_mun:
                address.municipio = match_mun.group(1).strip().upper()
    
    def _separate_stuck_words(self, texto: str) -> str:
        """Separa palabras pegadas por OCR en direcciones."""
        # Primero normalizar S/N (sin número)
        texto = re.sub(r'S\s*/\s*N', 'S/N', texto)
        texto = re.sub(r'SN\b', 'S/N', texto)
        
        # =====================================================================
        # DICCIONARIO DE PALABRAS COMUNES EN DOMICILIOS
        # =====================================================================
        PALABRAS_CALLE = [
            'SIN', 'NOMBRE', 'NUMERO', 'CALLE', 'AVENIDA', 'PRIVADA', 'CERRADA',
            'ANDADOR', 'CALLEJON', 'BOULEVARD', 'CAMINO', 'CARRETERA', 'PROLONGACION',
            'CIRCUITO', 'RETORNO', 'CERRO', 'LOMA', 'LLANO', 'VALLE', 'MONTE',
            'RAFAEL', 'MIGUEL', 'JOSE', 'JUAN', 'PEDRO', 'FRANCISCO', 
            'BENITO', 'JUAREZ', 'HIDALGO', 'MORELOS', 'ALLENDE', 'ALDAMA',
            'GUERRERO', 'VICTORIA', 'MADERO', 'CARRANZA', 'OBREGON',
            'ZAPATA', 'VILLA', 'CUAUHTEMOC', 'AZTECA', 'MAYA', 'OLMECA',
            'PRINCIPAL', 'NACIONAL', 'FEDERAL', 'ESTATAL', 'MUNICIPAL',
        ]
        
        PALABRAS_COLONIA = [
            'COLONIA', 'COL', 'FRACCIONAMIENTO', 'FRACC', 'UNIDAD', 'BARRIO',
            'RESIDENCIAL', 'INFONAVIT', 'FOVISSSTE', 'EJIDO', 'RANCHO',
            'HACIENDA', 'CENTRO', 'POPULAR', 'INDUSTRIAL', 'AGRICOLA',
            'LAS', 'LOS', 'EL', 'LA', 'DEL', 'DE', 'SAN', 'SANTA',
            'FLORES', 'JARDINES', 'LOMAS', 'VISTA', 'BELLA', 'HERMOSA',
            'NUEVA', 'NUEVO', 'VIEJA', 'VIEJO', 'ALTA', 'ALTO', 'BAJA', 'BAJO',
            'NORTE', 'SUR', 'ORIENTE', 'PONIENTE', 'ESTE', 'OESTE',
            'SUCHIL', 'PALMAS', 'PINOS', 'CEDROS', 'ROBLES', 'ENCINOS',
        ]
        
        PALABRAS_MUNICIPIO = [
            'TECPAN', 'GALEANA', 'ACAPULCO', 'CHILPANCINGO', 'IGUALA', 'TAXCO',
            'ZIHUATANEJO', 'COYUCA', 'ATOYAC', 'PETATLAN', 'BENITO',
            'MUNICIPIO', 'CIUDAD', 'VILLA', 'PUEBLO',
        ]
        
        # Combinar todas las palabras conocidas
        TODAS_PALABRAS = set(PALABRAS_CALLE + PALABRAS_COLONIA + PALABRAS_MUNICIPIO)
        
        # =====================================================================
        # SEPARACIÓN INTELIGENTE DE PALABRAS PEGADAS
        # =====================================================================
        
        # Función auxiliar para separar una palabra pegada
        def separar_palabra(palabra: str) -> str:
            """Intenta separar una palabra pegada en sus componentes."""
            if len(palabra) < 6:
                return palabra
            
            palabra_upper = palabra.upper()
            mejor_separacion = palabra
            
            # Intentar encontrar palabras conocidas dentro de la palabra pegada
            for i in range(3, len(palabra_upper) - 2):
                parte1 = palabra_upper[:i]
                parte2 = palabra_upper[i:]
                
                # Verificar si ambas partes son palabras conocidas o tienen sentido
                p1_conocida = parte1 in TODAS_PALABRAS or len(parte1) >= 3
                p2_conocida = parte2 in TODAS_PALABRAS or len(parte2) >= 3
                
                if p1_conocida and p2_conocida:
                    # Priorizar si alguna es palabra conocida
                    if parte1 in TODAS_PALABRAS or parte2 in TODAS_PALABRAS:
                        return f"{parte1} {parte2}"
            
            # Buscar patrones específicos conocidos
            patrones_separacion = [
                # C + nombre de calle
                (r'^C([A-Z]{3,})$', r'C \1'),
                # COL + nombre de colonia
                (r'^COL([A-Z]{3,})$', r'COL \1'),
                # LAS/LOS/EL/LA + palabra
                (r'^(LAS|LOS|EL|LA)([A-Z]{3,})$', r'\1 \2'),
                # Palabra + EL/LA/LOS/LAS + palabra
                (r'^([A-Z]{3,})(EL|LA|LOS|LAS)([A-Z]{3,})$', r'\1 \2 \3'),
                # SIN + NOMBRE
                (r'^SIN(NOMBRE)$', r'SIN \1'),
                (r'^SINNOMBRE$', r'SIN NOMBRE'),
                # Colonia pegada: LASFLORES -> LAS FLORES
                (r'^LAS(FLORES|PALMAS|LOMAS|ROSAS|MARGARITAS|BUGAMBILIAS)$', r'LAS \1'),
                (r'^LOS(PINOS|CEDROS|ROBLES|LAURELES|OLIVOS|ALAMOS)$', r'LOS \1'),
                (r'^EL(SUCHIL|PARAISO|EDEN|MIRADOR|REFUGIO|PORVENIR)$', r'EL \1'),
                (r'^LA(LOMA|CIMA|CUMBRE|ESPERANZA|GLORIA|PAZ)$', r'LA \1'),
                # Patrones compuestos: LASFLORESELSUCHIL -> LAS FLORES EL SUCHIL
                (r'^LAS(FLORES)(EL)(SUCHIL)$', r'LAS \1 \2 \3'),
                (r'^([A-Z]+)(EL)([A-Z]+)$', r'\1 \2 \3'),
            ]
            
            for patron, reemplazo in patrones_separacion:
                resultado = re.sub(patron, reemplazo, palabra_upper)
                if resultado != palabra_upper:
                    return resultado
            
            return palabra
        
        # =====================================================================
        # APLICAR SEPARACIONES CONOCIDAS PRIMERO
        # =====================================================================
        
        # Separar C (calle) pegada a nombre: CRAFAEL -> C RAFAEL
        for nombre in PALABRAS_CALLE:
            texto = re.sub(rf'\bC{nombre}\b', f'C {nombre}', texto, flags=re.IGNORECASE)
        
        # Separar palabras pegadas con S/N: OZUNAS/N -> OZUNA S/N
        texto = re.sub(r'([A-Z]{2,})S/N', r'\1 S/N', texto, flags=re.IGNORECASE)
        
        # CDEL -> C DEL
        texto = re.sub(r'\bCDEL([A-Z])', r'C DEL \1', texto, flags=re.IGNORECASE)
        # CDELA -> C DE LA
        texto = re.sub(r'\bCDELA([A-Z])', r'C DE LA \1', texto, flags=re.IGNORECASE)
        
        # COL + palabra
        texto = re.sub(r'\bCOL([A-Z]{2,})', r'COL \1', texto, flags=re.IGNORECASE)
        # COLEL -> COL EL
        texto = re.sub(r'\bCOL\s*EL([A-Z])', r'COL EL \1', texto, flags=re.IGNORECASE)
        # COLLA -> COL LA
        texto = re.sub(r'\bCOLLA([A-Z])', r'COL LA \1', texto, flags=re.IGNORECASE)
        
        # AV + palabra
        texto = re.sub(r'\bAV([A-Z]{2,})', r'AV \1', texto, flags=re.IGNORECASE)
        # AVDE -> AV DE
        texto = re.sub(r'\bAVDE([A-Z])', r'AV DE \1', texto, flags=re.IGNORECASE)
        
        # PRIV + palabra
        texto = re.sub(r'\bPRIV([A-Z]{2,})', r'PRIV \1', texto, flags=re.IGNORECASE)
        
        # =====================================================================
        # SEPARACIONES ESPECÍFICAS PARA COLONIAS PEGADAS
        # =====================================================================
        
        # SINNOMBRE -> SIN NOMBRE
        texto = re.sub(r'\bSINNOMBRE\b', 'SIN NOMBRE', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bCSINNOMBRE\b', 'C SIN NOMBRE', texto, flags=re.IGNORECASE)
        
        # Patrones de colonia pegada: LASFLORESELSUCHIL -> LAS FLORES EL SUCHIL
        texto = re.sub(r'\bLASFLORES\b', 'LAS FLORES', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bELSUCHIL\b', 'EL SUCHIL', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLASFLORESELSUCHIL\b', 'LAS FLORES EL SUCHIL', texto, flags=re.IGNORECASE)
        
        # Más patrones comunes de colonias
        texto = re.sub(r'\bLASPALMAS\b', 'LAS PALMAS', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLASLOMAS\b', 'LAS LOMAS', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLOSPINOS\b', 'LOS PINOS', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLOSCEDROS\b', 'LOS CEDROS', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bELPARAISO\b', 'EL PARAISO', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bELMIRADOR\b', 'EL MIRADOR', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLALOMA\b', 'LA LOMA', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLACIMA\b', 'LA CIMA', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bSANJUAN\b', 'SAN JUAN', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bSANTAMARIA\b', 'SANTA MARIA', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bVISTAHERMOSA\b', 'VISTA HERMOSA', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bBELLAVISTA\b', 'BELLA VISTA', texto, flags=re.IGNORECASE)
        
        # Patrones genéricos: LAS + palabra larga
        texto = re.sub(r'\bLAS([A-Z]{4,})\b', r'LAS \1', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLOS([A-Z]{4,})\b', r'LOS \1', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bEL([A-Z]{4,})\b', r'EL \1', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bLA([A-Z]{4,})\b', r'LA \1', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bSAN([A-Z]{3,})\b', r'SAN \1', texto, flags=re.IGNORECASE)
        texto = re.sub(r'\bSANTA([A-Z]{3,})\b', r'SANTA \1', texto, flags=re.IGNORECASE)
        
        # =====================================================================
        # SEPARAR NÚMERO PEGADO A PALABRA
        # =====================================================================
        texto = re.sub(r'([A-Z])(\d+)', r'\1 \2', texto)
        texto = re.sub(r'(\d+)([A-Z])', r'\1 \2', texto)
        
        # =====================================================================
        # PROCESAR PALABRAS RESTANTES QUE PUEDAN ESTAR PEGADAS
        # =====================================================================
        palabras = texto.split()
        palabras_procesadas = []
        
        for palabra in palabras:
            if len(palabra) > 10:  # Solo procesar palabras muy largas
                separada = separar_palabra(palabra)
                palabras_procesadas.append(separada)
            else:
                palabras_procesadas.append(palabra)
        
        return ' '.join(palabras_procesadas)


# ============================================================================
# VALIDATOR - Validación cruzada frente/reverso
# ============================================================================

class Validator:
    """
    Valida y cruza datos entre frente y reverso de INE.
    Usa MRZ como ground truth para corrección.
    """
    
    # Pesos para cálculo de match score
    PESO_CURP = 0.40
    PESO_FECHA = 0.25
    PESO_SEXO = 0.15
    PESO_NOMBRE = 0.20
    
    # Umbrales de clasificación
    UMBRAL_CONFIABLE = 85
    UMBRAL_PARCIAL = 70
    
    def validate_curp(self, curp: str) -> Tuple[bool, str]:
        """
        Valida CURP con dígito verificador RENAPO.
        
        Returns:
            (es_valido, curp_corregido)
        """
        if not curp or len(curp) != 18:
            return False, curp or ""
        
        curp = curp.upper().strip()
        
        # Intentar corregir errores OCR
        curp_corregido = self._correct_curp_ocr(curp)
        
        # Validar checksum
        if self._validate_curp_checksum(curp_corregido):
            return True, curp_corregido
        
        return False, curp
    
    def _correct_curp_ocr(self, curp: str) -> str:
        """Corrige errores OCR en CURP."""
        if len(curp) < 18:
            return curp
        
        # Correcciones por posición
        reemplazos_letras = {'0': 'O', '1': 'I', '8': 'B', '7': 'T', '5': 'S'}
        reemplazos_numeros = {'O': '0', 'D': '0', 'I': '1', 'T': '7', 'S': '5'}
        
        # Posiciones 0-3: letras
        parte1 = curp[0:4]
        for old, new in reemplazos_letras.items():
            parte1 = parte1.replace(old, new)
        
        # Posiciones 4-9: números (fecha)
        parte2 = curp[4:10]
        for old, new in reemplazos_numeros.items():
            parte2 = parte2.replace(old, new)
        
        # Posición 10: H/M (no modificar)
        parte3 = curp[10]
        
        # Posiciones 11-15: letras
        parte4 = curp[11:16]
        for old, new in reemplazos_letras.items():
            parte4 = parte4.replace(old, new)
        
        # Posiciones 16-17: homoclave
        parte5 = curp[16:18]
        
        return parte1 + parte2 + parte3 + parte4 + parte5
    
    def _validate_curp_checksum(self, curp: str) -> bool:
        """Valida checksum de CURP."""
        if not re.match(r'^[A-Z0-9]{18}$', curp):
            return False
        
        diccionario = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
        suma = 0
        
        for i in range(17):
            pos = diccionario.find(curp[i])
            if pos == -1:
                return False
            suma += pos * (18 - i)
        
        digito = 10 - (suma % 10)
        if digito == 10:
            digito = 0
        
        ultimo = curp[17]
        return int(ultimo) == digito if ultimo.isdigit() else True
    
    def cross_validate(self, front: FrontData, back: BackData) -> MatchResult:
        """
        Valida cruzadamente frente y reverso.
        
        Returns:
            MatchResult con porcentaje, resultado y detalles
        """
        validaciones = []
        score_total = 0.0
        
        # 1. Comparar CURP (40%)
        curp_score, curp_detalle = self._compare_curp(front.curp, back.curp, back.mrz)
        validaciones.append(curp_detalle)
        score_total += curp_score * self.PESO_CURP
        
        # 2. Comparar fecha de nacimiento (25%)
        fecha_score, fecha_detalle = self._compare_fecha(front.fecha_nacimiento, back.mrz.fecha_nacimiento)
        validaciones.append(fecha_detalle)
        score_total += fecha_score * self.PESO_FECHA
        
        # 3. Comparar sexo (15%)
        sexo_score, sexo_detalle = self._compare_sexo(front.sexo, back.mrz.sexo)
        validaciones.append(sexo_detalle)
        score_total += sexo_score * self.PESO_SEXO
        
        # 4. Comparar nombre (20%)
        # NUEVO: Si fecha/sexo/CURP coinciden fuerte, confiar más en MRZ para nombre
        campos_fuertes_match = (fecha_score >= 0.9 and sexo_score >= 0.9) or \
                              (curp_score >= 0.8 and fecha_score >= 0.8)
        
        # NUEVO: Detectar si nombre del frente es sospechoso (basura OCR)
        extractor = get_field_extractor()
        nombre_sospechoso = False
        if front.nombre and front.nombre.nombre_completo:
            nombre_sospechoso = extractor.is_name_suspicious(front.nombre.nombre_completo)
        
        nombre_corregido = None
        forzar_mrz = False
        
        if front.nombre and back.mrz:
            # Si hay coincidencias fuertes O nombre es sospechoso, forzar corrección con MRZ
            if campos_fuertes_match or nombre_sospechoso:
                forzar_mrz = True
                corrected = self.correct_name_with_mrz(front.nombre, back.mrz, forzar_mrz=True)
            else:
                corrected = self.correct_name_with_mrz(front.nombre, back.mrz, forzar_mrz=False)
            
            nombre_corregido = corrected.nombre_completo if corrected else None
        
        nombre_score, nombre_detalle = self._compare_nombre(
            nombre_corregido or (front.nombre.nombre_completo if front.nombre else None),
            back.mrz.nombre_completo
        )
        
        # Si campos fuertes coinciden pero nombre tiene baja similitud, boost el score
        if campos_fuertes_match and nombre_score < 0.7 and nombre_corregido:
            # Si confiamos en MRZ y hay corrección aplicada, dar crédito parcial
            nombre_score = max(nombre_score, 0.75)  # Mínimo 75% si hay corrección con campos fuertes
            nombre_detalle["score_boosted"] = True
            nombre_detalle["razon"] = "Campos fuertes coinciden, confiando en MRZ para nombre"
        
        # Agregar info de corrección al detalle
        if nombre_corregido and front.nombre:
            nombre_detalle["nombre_corregido"] = nombre_corregido
            nombre_detalle["nombre_original"] = front.nombre.nombre_completo
            nombre_detalle["forzar_mrz"] = forzar_mrz
            nombre_detalle["nombre_sospechoso"] = nombre_sospechoso
        
        validaciones.append(nombre_detalle)
        score_total += nombre_score * self.PESO_NOMBRE
        
        # Calcular porcentaje final
        porcentaje = int(score_total * 100)
        
        # Clasificar resultado
        if porcentaje >= self.UMBRAL_CONFIABLE:
            resultado = "MATCH_CONFIABLE"
            mensaje = "Los datos del frente y reverso coinciden con alta confianza"
            puede_aprobar = True
        elif porcentaje >= self.UMBRAL_PARCIAL:
            resultado = "MATCH_PARCIAL"
            mensaje = "Los datos coinciden parcialmente, revisar campos discrepantes"
            puede_aprobar = False
        else:
            resultado = "NO_MATCH"
            mensaje = "Los datos del frente y reverso no coinciden"
            puede_aprobar = False
        
        return MatchResult(
            porcentaje_match=porcentaje,
            resultado=resultado,
            mensaje=mensaje,
            validaciones_detalle=validaciones,
            puede_aprobar_automatico=puede_aprobar
        )
    
    def _compare_curp(self, curp_frente: Optional[str], curp_reverso: Optional[str], 
                      mrz: MRZData) -> Tuple[float, dict]:
        """Compara CURP entre frente y reverso."""
        detalle = {
            "campo": "CURP",
            "frente": curp_frente,
            "reverso": curp_reverso,
            "coincide": False,
            "score": 0.0
        }
        
        if not curp_frente:
            return 0.0, detalle
        
        # Si hay CURP en reverso, comparar directamente
        if curp_reverso:
            if curp_frente.upper() == curp_reverso.upper():
                detalle["coincide"] = True
                detalle["score"] = 1.0
                return 1.0, detalle
            
            # Comparar con tolerancia a errores OCR
            similitud = SequenceMatcher(None, curp_frente.upper(), curp_reverso.upper()).ratio()
            if similitud >= 0.9:
                detalle["coincide"] = True
                detalle["score"] = similitud
                return similitud, detalle
        
        # Validar CURP con datos del MRZ (cuando no hay CURP en reverso)
        if mrz.fecha_nacimiento and mrz.sexo:
            curp_valido = self._validate_curp_with_mrz(curp_frente, mrz)
            if curp_valido:
                # CURP coherente con MRZ - dar score alto (0.85) en vez de casi cero
                detalle["coincide"] = True
                detalle["score"] = 0.85
                detalle["validado_con_mrz"] = True
                return 0.85, detalle
        
        # Si no hay CURP en reverso y no se pudo validar con MRZ, dar score muy bajo
        if not curp_reverso:
            detalle["score"] = 0.1  # Score mínimo para indicar que existe pero no se pudo validar
            return 0.1, detalle
        
        return 0.0, detalle
    
    def _validate_curp_with_mrz(self, curp: str, mrz: MRZData) -> bool:
        """Valida CURP contra datos del MRZ."""
        if not curp or len(curp) < 18:
            return False
        
        # Extraer fecha de CURP (posiciones 4-9: YYMMDD)
        curp_fecha = curp[4:10]
        
        # Convertir fecha MRZ a formato YYMMDD
        if mrz.fecha_nacimiento:
            try:
                partes = mrz.fecha_nacimiento.split('/')
                if len(partes) == 3:
                    dd, mm, yyyy = partes
                    yy = yyyy[2:4]
                    mrz_fecha = f"{yy}{mm}{dd}"
                    
                    if curp_fecha == mrz_fecha:
                        # Verificar sexo
                        curp_sexo = curp[10]
                        if curp_sexo == mrz.sexo:
                            return True
            except:
                pass
        
        return False
    
    def _compare_fecha(self, fecha_frente: Optional[str], fecha_mrz: Optional[str]) -> Tuple[float, dict]:
        """Compara fecha de nacimiento."""
        detalle = {
            "campo": "Fecha Nacimiento",
            "frente": fecha_frente,
            "reverso": fecha_mrz,
            "coincide": False,
            "score": 0.0
        }
        
        if not fecha_frente or not fecha_mrz:
            return 0.0, detalle
        
        # Normalizar fechas
        fecha_f = self._normalize_date(fecha_frente)
        fecha_m = self._normalize_date(fecha_mrz)
        
        if fecha_f == fecha_m:
            detalle["coincide"] = True
            detalle["score"] = 1.0
            return 1.0, detalle
        
        return 0.0, detalle
    
    def _normalize_date(self, fecha: str) -> str:
        """Normaliza fecha a formato DD/MM/YYYY."""
        fecha = fecha.replace('-', '/').replace('.', '/')
        partes = fecha.split('/')
        if len(partes) == 3:
            return f"{int(partes[0]):02d}/{int(partes[1]):02d}/{partes[2]}"
        return fecha
    
    def _compare_sexo(self, sexo_frente: Optional[str], sexo_mrz: Optional[str]) -> Tuple[float, dict]:
        """Compara sexo."""
        detalle = {
            "campo": "Sexo",
            "frente": sexo_frente,
            "reverso": sexo_mrz,
            "coincide": False,
            "score": 0.0
        }
        
        if not sexo_frente or not sexo_mrz:
            return 0.0, detalle
        
        if sexo_frente.upper() == sexo_mrz.upper():
            detalle["coincide"] = True
            detalle["score"] = 1.0
            return 1.0, detalle
        
        return 0.0, detalle
    
    def _compare_nombre(self, nombre_frente: Optional[str], nombre_mrz: Optional[str]) -> Tuple[float, dict]:
        """Compara nombre usando similitud de secuencia."""
        detalle = {
            "campo": "Nombre",
            "frente": nombre_frente,
            "reverso": nombre_mrz,
            "coincide": False,
            "score": 0.0
        }
        
        if not nombre_frente or not nombre_mrz:
            return 0.0, detalle
        
        # Limpiar nombre del frente antes de comparar (eliminar basura como SEXOAM)
        nombre_frente_limpio = self._clean_name_for_comparison(nombre_frente)
        
        # Normalizar nombres
        nombre_f = self._normalize_name(nombre_frente_limpio)
        nombre_m = self._normalize_name(nombre_mrz)
        
        # Actualizar detalle con nombre limpio
        detalle["frente_limpio"] = nombre_frente_limpio
        
        # Calcular similitud
        similitud = SequenceMatcher(None, nombre_f, nombre_m).ratio()
        
        # UMBRAL para considerar match (ajustado para ser más flexible)
        UMBRAL_MATCH_NOMBRE = 0.75
        
        detalle["score"] = similitud
        # FIX: match debe basarse en umbral de similitud, no en igualdad estricta
        detalle["coincide"] = similitud >= UMBRAL_MATCH_NOMBRE
        
        # Si similitud alta, retornar directamente
        if similitud >= 0.8:
            return similitud, detalle
        
        # NUEVO: Validación token-based para detectar subconjuntos (ej: CRUZ EMILIANO ⊂ CRUZ CRUZ EMILIANO)
        partes_f = nombre_f.split()
        partes_m = nombre_m.split()
        
        # Normalizar tokens: quitar duplicados por posición pero mantener orden
        partes_f_norm = []
        for i, p in enumerate(partes_f):
            if i == 0 or p != partes_f[i-1]:  # No agregar si es igual al anterior
                partes_f_norm.append(p)
        
        partes_m_norm = []
        for i, p in enumerate(partes_m):
            if i == 0 or p != partes_m[i-1]:  # No agregar si es igual al anterior
                partes_m_norm.append(p)
        
        # Verificar si tokens del frente son subconjunto de MRZ en el mismo orden
        # Ej: CRUZ EMILIANO ⊂ CRUZ CRUZ EMILIANO
        is_subset = False
        if len(partes_f_norm) > 0 and len(partes_m_norm) >= len(partes_f_norm):
            # Buscar si todos los tokens de frente están en MRZ en orden
            i_mrz = 0
            matches = 0
            for token_f in partes_f_norm:
                # Buscar token en MRZ desde donde quedamos
                found = False
                for j in range(i_mrz, len(partes_m_norm)):
                    if partes_m_norm[j] == token_f:
                        matches += 1
                        i_mrz = j + 1
                        found = True
                        break
                if not found:
                    break
            
            # Si todos los tokens del frente están en MRZ en orden, es subconjunto
            if matches == len(partes_f_norm):
                is_subset = True
                # Ajustar score: subconjunto válido pero con penalización por incompletitud
                similitud_ajustada = max(similitud, 0.80)  # Mínimo 80% si es subconjunto válido
                detalle["coincide"] = True
                detalle["es_subconjunto"] = True
                detalle["score"] = similitud_ajustada
                detalle["razon"] = "Tokens del frente son subconjunto de MRZ (frente incompleto)"
                return similitud_ajustada, detalle
        
        # Si la similitud es media-baja, verificar coincidencias token-based (maneja truncamiento)
        if len(partes_f_norm) >= 2 and len(partes_m_norm) >= 2:
            # Verificar apellidos (deben coincidir exactamente)
            apellidos_match = (partes_f_norm[0] == partes_m_norm[0] and 
                             (len(partes_f_norm) > 1 and len(partes_m_norm) > 1 and 
                              partes_f_norm[1] == partes_m_norm[1]))
            
            if apellidos_match:
                # Apellidos coinciden - verificar nombres con lógica de prefijo (maneja truncamiento)
                nombres_f = partes_f_norm[2:] if len(partes_f_norm) > 2 else []
                nombres_m = partes_m_norm[2:] if len(partes_m_norm) > 2 else []
                
                # Comparar nombres: si uno es prefijo del otro, considerarlo match
                nombres_match = False
                if nombres_f and nombres_m:
                    # Comparar primer nombre (puede estar truncado en MRZ)
                    nom_f = nombres_f[0] if nombres_f else ""
                    nom_m = nombres_m[0] if nombres_m else ""
                    
                    if nom_f and nom_m:
                        # Si uno es prefijo del otro (ej: FERNAN vs FERNANDO), es match
                        if nom_f.startswith(nom_m) or nom_m.startswith(nom_f):
                            nombres_match = True
                        # O si son muy similares
                        elif SequenceMatcher(None, nom_f, nom_m).ratio() >= 0.85:
                            nombres_match = True
                
                if nombres_match or (not nombres_f and not nombres_m):
                    # Apellidos + nombres coinciden (o ambos sin nombres) - dar crédito alto
                    detalle["coincide"] = True
                    detalle["apellidos_match"] = True
                    detalle["nombres_match"] = nombres_match
                    return max(similitud, 0.8), detalle
                else:
                    # Solo apellidos coinciden - dar crédito parcial
                    detalle["coincide"] = True
                    detalle["apellidos_match"] = True
                    return max(similitud, 0.7), detalle
        
        return similitud, detalle
    
    def _clean_name_for_comparison(self, nombre: str) -> str:
        """Limpia nombre del frente eliminando basura antes de comparar."""
        if not nombre:
            return nombre
        
        # Palabras basura conocidas
        BASURA = {
            'SEXO', 'SEXOH', 'SEXOM', 'SEXOF', 'SEXOAM', 'BEXOH', 'BEXO', 'BEXOM', 'BEXOAM',
            '5EXOH', '5EXO', 'NOMBRE', 'OMBRE', 'DOMICILIO', 'OMICILIO',
            'CREDENCIAL', 'VOTAR', 'PARA', 'INE', 'IFE', 'ELECTORAL',
        }
        
        # Patrón regex para SEXO + cualquier cantidad de letras/dígitos pegados
        # Incluye dígitos porque OCR puede meter 0/8 en lugar de O
        patron_sexo = re.compile(r'^[BS58]EX[O0][A-Z0-9]*$', re.IGNORECASE)
        
        # Prefijos de campos que pueden pegarse al nombre
        PREFIJOS_CAMPO = ['NOMBRE', 'OMBRE', 'DOMICILIO', 'OMICILIO']
        
        partes = nombre.upper().split()
        partes_limpias = []
        
        for parte in partes:
            # Saltar palabras basura conocidas
            if parte in BASURA:
                continue
            # Saltar variantes de SEXO (SEXO, SEXOH, SEXOAM, 5EX0AM, 8EXO, etc.)
            if patron_sexo.match(parte):
                continue
            
            # Intentar rescatar apellido de campo pegado (NOMBREGARCIA -> GARCIA)
            rescatado = None
            for prefijo in PREFIJOS_CAMPO:
                if parte.startswith(prefijo) and len(parte) > len(prefijo):
                    resto = parte[len(prefijo):]
                    # Solo rescatar si el resto parece nombre válido (3+ letras, solo letras)
                    if len(resto) >= 3 and resto.isalpha():
                        rescatado = resto
                        break
            
            if rescatado:
                partes_limpias.append(rescatado)
            else:
                partes_limpias.append(parte)
        
        return ' '.join(partes_limpias)
    
    def _normalize_name(self, nombre: str) -> str:
        """Normaliza nombre para comparación."""
        nombre = nombre.upper().strip()
        # Eliminar acentos
        replacements = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'N'}
        for old, new in replacements.items():
            nombre = nombre.replace(old, new)
        # Eliminar caracteres especiales
        nombre = re.sub(r'[^A-Z\s]', '', nombre)
        # Normalizar espacios
        return ' '.join(nombre.split())
    
    def correct_name_with_mrz(self, front_name: NameData, mrz_name: MRZData, forzar_mrz: bool = False) -> CorrectedName:
        """
        Corrige nombre del frente usando MRZ como ground truth.
        El MRZ es más confiable ortográficamente - si hay alta similitud, confiar en MRZ.
        
        Args:
            front_name: Nombre extraído del frente
            mrz_name: Datos del MRZ
            forzar_mrz: Si True, confiar en MRZ incluso con baja similitud (cuando fecha/sexo/CURP coinciden)
        """
        front_completo = front_name.nombre_completo if front_name else ""
        mrz_completo = mrz_name.nombre_completo if mrz_name else ""
        
        if not mrz_completo:
            return CorrectedName(
                nombre_completo=front_completo or "",
                apellido_paterno=front_name.apellido_paterno if front_name else None,
                apellido_materno=front_name.apellido_materno if front_name else None,
                nombres=front_name.nombre if front_name else None,
                fuente="Frontal",
                confianza=0.7,
                correccion_aplicada=False
            )
        
        palabras_front = len(front_completo.split()) if front_completo else 0
        palabras_mrz = len(mrz_completo.split()) if mrz_completo else 0
        
        # Calcular similitud primero
        similitud = SequenceMatcher(None, 
                                    self._normalize_name(front_completo),
                                    self._normalize_name(mrz_completo)).ratio()
        
        # Si son prácticamente idénticos (>99%), usar frente
        if similitud > 0.99:
            return CorrectedName(
                nombre_completo=front_completo,
                apellido_paterno=front_name.apellido_paterno if front_name else None,
                apellido_materno=front_name.apellido_materno if front_name else None,
                nombres=front_name.nombre if front_name else None,
                fuente="Frontal (validado MRZ)",
                confianza=1.0,
                correccion_aplicada=False
            )
        
        # NUEVO: Si forzar_mrz=True (campos fuertes coinciden), confiar en MRZ incluso con baja similitud
        if forzar_mrz:
            return CorrectedName(
                nombre_completo=mrz_completo,
                apellido_paterno=mrz_name.apellido_paterno,
                apellido_materno=mrz_name.apellido_materno,
                nombres=mrz_name.nombres,
                fuente="MRZ (forzado por coincidencia de campos fuertes)",
                confianza=max(similitud, 0.75),  # Mínimo 75% si se fuerza
                correccion_aplicada=True
            )
        
        # Si similitud alta (>85%), CONFIAR EN MRZ para corrección ortográfica
        # Esto corrige typos como ARELANO -> ARELLANO
        if similitud > 0.85:
            return CorrectedName(
                nombre_completo=mrz_completo,
                apellido_paterno=mrz_name.apellido_paterno,
                apellido_materno=mrz_name.apellido_materno,
                nombres=mrz_name.nombres,
                fuente="MRZ (corrección ortográfica)",
                confianza=similitud,
                correccion_aplicada=True
            )
        
        # Si MRZ tiene más palabras, usarlo como ground truth
        if palabras_mrz > palabras_front:
            return CorrectedName(
                nombre_completo=mrz_completo,
                apellido_paterno=mrz_name.apellido_paterno,
                apellido_materno=mrz_name.apellido_materno,
                nombres=mrz_name.nombres,
                fuente="MRZ (más completo)",
                confianza=0.95,
                correccion_aplicada=True
            )
        
        # Discrepancia significativa (<85%), usar MRZ como ground truth solo si no hay opción
        return CorrectedName(
            nombre_completo=mrz_completo,
            apellido_paterno=mrz_name.apellido_paterno,
            apellido_materno=mrz_name.apellido_materno,
            nombres=mrz_name.nombres,
            fuente="MRZ (ground truth)",
            confianza=0.85,
            correccion_aplicada=True
        )
    
    def calculate_match_score(self, front: FrontData, back: BackData) -> int:
        """
        Calcula score de match ponderado.
        
        Pesos:
        - CURP: 40%
        - Fecha nacimiento: 25%
        - Sexo: 15%
        - Nombre: 20%
        """
        result = self.cross_validate(front, back)
        return result.porcentaje_match
    
    def validate_curp_date_coherence(self, curp: str, fecha_nacimiento: str) -> bool:
        """
        Valida que la fecha en CURP coincida con fecha_nacimiento.
        """
        if not curp or len(curp) < 10 or not fecha_nacimiento:
            return False
        
        # Extraer fecha de CURP (posiciones 4-9: YYMMDD)
        curp_yy = curp[4:6]
        curp_mm = curp[6:8]
        curp_dd = curp[8:10]
        
        try:
            # Parsear fecha de nacimiento
            partes = fecha_nacimiento.replace('-', '/').split('/')
            if len(partes) != 3:
                return False
            
            dd, mm, yyyy = partes
            yy = yyyy[2:4]
            
            return curp_yy == yy and curp_mm == mm and curp_dd == dd
        except:
            return False


# ============================================================================
# API PRINCIPAL - Funciones de alto nivel
# ============================================================================

# Instancias globales
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


def extraer_datos_ine_frente(img: np.ndarray, use_zones: bool = True) -> Dict:
    """
    Extrae todos los datos del frente de la INE.
    OPTIMIZADO v3: Zonas primero (rápido), OCR completo solo como fallback.

    Args:
        img: Imagen en formato numpy array (BGR)
        use_zones: Si usar OCR por zonas para mayor precisión (default: True)

    Returns:
        Diccionario con todos los campos extraídos, incluyendo nombre_raw_ocr
    """
    engine = get_ocr_engine()
    extractor = get_field_extractor()

    nombre_raw_ocr = None
    front_data = None
    ocr_result = None
    zone_results_guardados = None  # Fix 2: reutilizar en fallback

    if use_zones:
        try:
            # RUTA RÁPIDA: Solo OCR por zonas (~6s vs ~62s del OCR completo)
            zone_results = engine.run_ocr_by_zones(img, tipo="frente")

            if zone_results:
                combined_text = " ".join(
                    r.combined_text for r in zone_results.values() if r and r.combined_text
                )
                all_detections = [
                    d for r in zone_results.values() if r for d in r.detections
                ]
                avg_conf = float(np.mean([d.confidence for d in all_detections])) if all_detections else 0.7
                ocr_result = OCRResult(
                    combined_text=combined_text,
                    detections=all_detections,
                    confidence=avg_conf,
                    engine="paddleocr"
                )

                front_data = extractor.extract_front_with_zones(ocr_result, zone_results)

                if 'nombre' in zone_results:
                    nombre_raw_ocr = zone_results['nombre'].combined_text.upper()
                    nombre_raw_ocr = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', '', nombre_raw_ocr)
                    nombre_raw_ocr = ' '.join(nombre_raw_ocr.split())

                tiene_nombre = bool(front_data.nombre.nombre_completo)
                tiene_id = bool(front_data.curp or front_data.clave_elector)
                campos_criticos_ok = tiene_nombre and tiene_id
                if not campos_criticos_ok:
                    zone_results_guardados = zone_results  # Fix 2: guardar antes de borrar
                    zone_results = None
                    front_data = None
                    ocr_result = None
        except Exception:
            zone_results = None
            front_data = None
            ocr_result = None

    # FALLBACK: OCR completo si zonas fallaron o no se usan
    if front_data is None:
        ocr_result = engine.run_ocr(img, fast_mode=True)
        if ocr_result.confidence < 0.6 or len(ocr_result.combined_text.strip()) < 30:
            ocr_result_full = engine.run_ocr(img, fast_mode=False)
            if (ocr_result_full.confidence > ocr_result.confidence or
                    len(ocr_result_full.combined_text) > len(ocr_result.combined_text) * 1.3):
                ocr_result = ocr_result_full

        nombre_raw_global = extractor.extract_name(ocr_result.combined_text, None)
        nombre_raw_ocr = nombre_raw_global.nombre_completo if nombre_raw_global else None

        try:
            # Fix 2: reutilizar zonas del primer intento si existen (ahorra ~5s)
            zone_results = zone_results_guardados or engine.run_ocr_by_zones(img, tipo="frente")
            if zone_results:
                front_data = extractor.extract_front_with_zones(ocr_result, zone_results)
                if 'nombre' in zone_results:
                    nombre_raw_ocr = zone_results['nombre'].combined_text.upper()
                    nombre_raw_ocr = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', '', nombre_raw_ocr)
                    nombre_raw_ocr = ' '.join(nombre_raw_ocr.split())
            else:
                front_data = extractor.extract_front(ocr_result)
        except Exception:
            front_data = extractor.extract_front(ocr_result)
    
    # Fix 1: Cross-validar orden apellido/nombre usando CURP como ancla.
    # El CURP codifica: pos[0]=inicial_ap_paterno, pos[2]=inicial_ap_materno, pos[3]=inicial_nombre.
    # Si el apellido_paterno no coincide con curp[0] pero el materno sí → swap.
    curp_val = front_data.curp or ""
    ap = front_data.nombre.apellido_paterno or ""
    am = front_data.nombre.apellido_materno or ""
    nom = front_data.nombre.nombre or ""
    if len(curp_val) >= 4 and ap and am:
        c0 = curp_val[0].upper()   # inicial apellido paterno
        c2 = curp_val[2].upper()   # inicial apellido materno
        if ap[0].upper() != c0 and am[0].upper() == c0:
            # swap
            front_data.nombre.apellido_paterno, front_data.nombre.apellido_materno = am, ap
            partes = [p for p in [am, ap, nom] if p]
            front_data.nombre.nombre_completo = ' '.join(partes)

    # Convertir a diccionario para compatibilidad
    return {
        "tipo": "INE_FRENTE",
        "nombre": {
            "apellido_paterno": front_data.nombre.apellido_paterno,
            "apellido_materno": front_data.nombre.apellido_materno,
            "nombre": front_data.nombre.nombre,
            "nombre_completo": front_data.nombre.nombre_completo
        },
        "nombre_raw_ocr": nombre_raw_ocr,  # Nombre tal como lo leyó el OCR (sin correcciones MRZ)
        "domicilio": {
            "calle": front_data.domicilio.calle,
            "numero_exterior": front_data.domicilio.numero_exterior,
            "numero_interior": front_data.domicilio.numero_interior,
            "colonia": front_data.domicilio.colonia,
            "codigo_postal": front_data.domicilio.codigo_postal,
            "municipio": front_data.domicilio.municipio,
            "estado": front_data.domicilio.estado,
            "domicilio_completo": front_data.domicilio.domicilio_completo
        },
        "sexo": front_data.sexo,
        "curp": front_data.curp,
        "clave_elector": front_data.clave_elector,
        "fecha_nacimiento": front_data.fecha_nacimiento,
        "anio_registro": front_data.anio_registro,
        "anio_emision": front_data.anio_emision,
        "tipo_ine": front_data.tipo_ine,  # "IFE" o "INE"
        "modelo_ine": front_data.modelo_ine,  # Modelo específico: "C", "D", "E", "F", "G", "H"
        "seccion": front_data.seccion,
        "vigencia": front_data.vigencia,
        "texto_crudo": ocr_result.combined_text,
        "confianza_ocr": front_data.confianza_ocr
    }


def extraer_datos_ine_reverso(img: np.ndarray) -> Dict:
    """
    Extrae datos del reverso de la INE (principalmente MRZ).
    OPTIMIZADO: Usa fast_mode con fallback inteligente.
    
    Args:
        img: Imagen en formato numpy array (BGR)
    
    Returns:
        Diccionario con datos del MRZ
    """
    engine = get_ocr_engine()
    extractor = get_field_extractor()

    # RUTA RÁPIDA: Solo OCR por zonas (MRZ + datos_extra)
    zone_results = engine.run_ocr_by_zones(img, tipo="reverso")

    if zone_results:
        combined_text = " ".join(
            r.combined_text for r in zone_results.values() if r and r.combined_text
        )
        all_detections = [d for r in zone_results.values() if r for d in r.detections]
        avg_conf = float(np.mean([d.confidence for d in all_detections])) if all_detections else 0.7
        ocr_result = OCRResult(
            combined_text=combined_text,
            detections=all_detections,
            confidence=avg_conf,
            engine="paddleocr"
        )
        back_data = extractor.extract_back_with_zones(ocr_result, zone_results)

        # Validar: si no hay MRZ, caer a fallback completo
        has_mrz = back_data.mrz.nombre_completo or back_data.mrz.numero_documento
        if has_mrz:
            return {
                "tipo": "INE_REVERSO",
                "mrz": {
                    "lineas_raw": back_data.mrz.lineas_raw,
                    "lineas_clean": back_data.mrz.lineas_clean,
                    "documento_tipo": back_data.mrz.documento_tipo,
                    "pais": back_data.mrz.pais,
                    "numero_documento": back_data.mrz.numero_documento,
                    "nombre_completo": back_data.mrz.nombre_completo,
                    "apellido_paterno": back_data.mrz.apellido_paterno,
                    "apellido_materno": back_data.mrz.apellido_materno,
                    "nombres": back_data.mrz.nombres,
                    "fecha_nacimiento": back_data.mrz.fecha_nacimiento,
                    "sexo": back_data.mrz.sexo,
                    "fecha_expiracion": back_data.mrz.fecha_expiracion
                },
                "curp": back_data.curp,
                "cic": back_data.cic,
                "ocr_vertical": back_data.ocr_vertical,
                "texto_crudo": combined_text,
                "confianza_ocr": back_data.confianza_ocr
            }

    # FALLBACK: OCR completo solo si zonas no encontraron MRZ
    ocr_result = engine.run_ocr(img, fast_mode=True)

    has_mrz_text = '<' in ocr_result.combined_text or 'IDMEX' in ocr_result.combined_text.upper()
    if not has_mrz_text or ocr_result.confidence < 0.5:
        ocr_result_full = engine.run_ocr(img, fast_mode=False)
        has_mrz_full = '<' in ocr_result_full.combined_text or 'IDMEX' in ocr_result_full.combined_text.upper()
        if has_mrz_full or ocr_result_full.confidence > ocr_result.confidence:
            ocr_result = ocr_result_full

    if not zone_results and (not ocr_result.combined_text or len(ocr_result.combined_text.strip()) < 20):
        # Extraer zona MRZ manualmente y hacer OCR directo
        h, w = img.shape[:2]
        mrz_region = img[int(h * 0.55):h, :]  # Tercio inferior más amplio
        
        if mrz_region.size > 0:
            gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY) if len(mrz_region.shape) == 3 else mrz_region
            
            best_text = ""
            best_conf = 0.0
            
            # OPTIMIZADO: Solo 2 variantes en lugar de 6
            # Variante 1: Original
            text, detections = engine._run_paddleocr(mrz_region, use_cls=True)
            conf = np.mean([d.confidence for d in detections]) if detections else 0.0
            has_mrz = '<' in text or 'IDMEX' in text.upper() or 'MEX' in text.upper()
            
            if text.strip() and has_mrz:
                best_text = text
                best_conf = conf
            
            # Variante 2: CLAHE (solo si primera falló)
            if not best_text or best_conf < 0.6:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(gray)
                clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
                
                text2, detections2 = engine._run_paddleocr(clahe_bgr, use_cls=True)
                conf2 = np.mean([d.confidence for d in detections2]) if detections2 else 0.0
                has_mrz2 = '<' in text2 or 'IDMEX' in text2.upper() or 'MEX' in text2.upper()
                
                if text2.strip() and (has_mrz2 or conf2 > best_conf):
                    best_text = text2
                    best_conf = conf2
            
            # OPTIMIZADO: Tesseract solo como último recurso y solo 1 intento
            if not best_text:
                try:
                    tess_text = pytesseract.image_to_string(
                        gray, 
                        lang='spa',
                        config='--psm 6 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                    )
                    if tess_text.strip() and ('<' in tess_text or 'IDMEX' in tess_text.upper()):
                        best_text = tess_text.upper()
                        best_conf = 0.7
                except:
                    pass
            
            if best_text:
                ocr_result = OCRResult(
                    combined_text=best_text,
                    detections=[],
                    confidence=best_conf,
                    engine="paddleocr"
                )
    
    # Usar extract_back_with_zones para combinar resultados
    back_data = extractor.extract_back_with_zones(ocr_result, zone_results)
    
    return {
        "tipo": "INE_REVERSO",
        "mrz": {
            "lineas_raw": back_data.mrz.lineas_raw,  # Líneas raw del OCR (pueden estar contaminadas)
            "lineas_clean": back_data.mrz.lineas_clean,  # MRZ limpio validado (solo desde IDMEX...)
            "documento_tipo": back_data.mrz.documento_tipo,
            "pais": back_data.mrz.pais,
            "numero_documento": back_data.mrz.numero_documento,
            "nombre_completo": back_data.mrz.nombre_completo,
            "apellido_paterno": back_data.mrz.apellido_paterno,
            "apellido_materno": back_data.mrz.apellido_materno,
            "nombres": back_data.mrz.nombres,
            "fecha_nacimiento": back_data.mrz.fecha_nacimiento,
            "sexo": back_data.mrz.sexo,
            "fecha_expiracion": back_data.mrz.fecha_expiracion
        },
        "curp": back_data.curp,
        "cic": back_data.cic,  # Código de Identificación de Credencial (9 dígitos)
        "ocr_vertical": back_data.ocr_vertical,  # Identificador ciudadano (13 dígitos)
        "texto_crudo": ocr_result.combined_text,
        "confianza_ocr": back_data.confianza_ocr
    }


def extraer_datos_ine(img: np.ndarray, tipo: str = "auto") -> Dict:
    """
    Función principal para extraer datos de una INE.
    
    Args:
        img: Imagen en formato numpy array (BGR)
        tipo: "frente", "reverso" o "auto" para detección automática
    
    Returns:
        Diccionario con todos los datos extraídos
    """
    if tipo == "auto":
        # Detectar automáticamente si es frente o reverso
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        try:
            text = pytesseract.image_to_string(gray, lang='spa', config='--psm 6')
            if "IDMEX" in text.upper() or "<<<" in text:
                tipo = "reverso"
            else:
                tipo = "frente"
        except:
            tipo = "frente"
    
    if tipo == "reverso":
        return extraer_datos_ine_reverso(img)
    else:
        return extraer_datos_ine_frente(img)


def validar_cruzado_ine(img_frente: np.ndarray, img_reverso: np.ndarray) -> Dict:
    """
    Valida cruzadamente frente y reverso de INE.
    
    Args:
        img_frente: Imagen del frente de INE
        img_reverso: Imagen del reverso de INE
    
    Returns:
        Diccionario con resultado de validación y datos combinados
    """
    engine = get_ocr_engine()
    extractor = get_field_extractor()
    validator = get_validator()
    
    # Extraer datos de ambos lados
    ocr_frente = engine.run_ocr(img_frente)
    ocr_reverso = engine.run_ocr(img_reverso)
    
    front_data = extractor.extract_front(ocr_frente)
    back_data = extractor.extract_back(ocr_reverso)
    
    # Validación cruzada
    match_result = validator.cross_validate(front_data, back_data)
    
    # Corregir nombre con MRZ si es necesario
    nombre_corregido = validator.correct_name_with_mrz(front_data.nombre, back_data.mrz)
    
    return {
        "validacion": {
            "porcentaje_match": match_result.porcentaje_match,
            "resultado": match_result.resultado,
            "mensaje": match_result.mensaje,
            "puede_aprobar_automatico": match_result.puede_aprobar_automatico,
            "validaciones_detalle": match_result.validaciones_detalle
        },
        "datos_frente": extraer_datos_ine_frente(img_frente),
        "datos_reverso": extraer_datos_ine_reverso(img_reverso),
        "nombre_corregido": {
            "nombre_completo": nombre_corregido.nombre_completo,
            "apellido_paterno": nombre_corregido.apellido_paterno,
            "apellido_materno": nombre_corregido.apellido_materno,
            "nombres": nombre_corregido.nombres,
            "fuente": nombre_corregido.fuente,
            "confianza": nombre_corregido.confianza,
            "correccion_aplicada": nombre_corregido.correccion_aplicada
        }
    }


def validar_ocr_mejorado(img: np.ndarray) -> Tuple[bool, Dict]:
    """
    Versión mejorada de validar_ocr que además extrae datos.
    Compatible con API existente.
    
    Returns:
        Tuple de (es_valido, datos_extraidos)
    """
    try:
        datos = extraer_datos_ine(img, tipo="auto")
        
        texto = datos.get("texto_crudo", "")
        confianza = datos.get("confianza_ocr", 0)
        
        tiene_texto = len(texto) > 40
        
        palabras_clave = ["NOMBRE", "CURP", "ELECTOR", "VIGENCIA", 
                          "INSTITUTO", "NACIONAL", "IDMEX", "MEX"]
        tiene_palabras = any(pal in texto.upper() for pal in palabras_clave)
        
        confianza_ok = confianza > 20
        
        es_valido = tiene_texto and (tiene_palabras or confianza_ok)
        
        return es_valido, datos
    
    except Exception as e:
        return False, {"error": str(e)}


# ============================================================================
# FUNCIONES LEGACY - Compatibilidad con código existente
# ============================================================================

def ocr_combinado(img: np.ndarray, pipeline: str = "default") -> Dict:
    """
    Función legacy para compatibilidad.
    Ejecuta OCR combinado y retorna en formato antiguo.
    """
    engine = get_ocr_engine()
    result = engine.run_ocr(img)
    
    return {
        "combined_text": result.combined_text,
        "individual_results": [],
        "detections": [{"text": d.text, "confidence": d.confidence, "bbox": d.bbox} 
                       for d in result.detections],
        "primary_engine": result.engine
    }


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


# ============================================================================
# TEST LOCAL
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python ocr_avanzado.py <ruta_imagen> [tipo]")
        print("  tipo: frente, reverso, auto (default: auto)")
        sys.exit(1)
    
    img_path = sys.argv[1]
    tipo = sys.argv[2] if len(sys.argv) > 2 else "auto"
    
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: No se pudo cargar la imagen {img_path}")
        sys.exit(1)
    
    print(f"Procesando: {img_path}")
    print(f"Tipo: {tipo}")
    print("=" * 50)
    
    datos = extraer_datos_ine(img, tipo=tipo)
    
    print(f"\nTipo detectado: {datos.get('tipo')}")
    print(f"Confianza OCR: {datos.get('confianza_ocr'):.1f}%")
    
    if datos.get('tipo') == 'INE_FRENTE':
        print(f"\n--- DATOS EXTRAÍDOS ---")
        nombre = datos.get('nombre', {})
        print(f"Nombre: {nombre.get('nombre_completo', 'N/A')}")
        print(f"  - Apellido Paterno: {nombre.get('apellido_paterno', 'N/A')}")
        print(f"  - Apellido Materno: {nombre.get('apellido_materno', 'N/A')}")
        print(f"  - Nombre(s): {nombre.get('nombre', 'N/A')}")
        print(f"Sexo: {datos.get('sexo', 'N/A')}")
        print(f"CURP: {datos.get('curp', 'N/A')}")
        print(f"Clave Elector: {datos.get('clave_elector', 'N/A')}")
        print(f"Fecha Nacimiento: {datos.get('fecha_nacimiento', 'N/A')}")
        print(f"Sección: {datos.get('seccion', 'N/A')}")
        print(f"Vigencia: {datos.get('vigencia', 'N/A')}")
        domicilio = datos.get('domicilio', {})
        print(f"Domicilio: {domicilio.get('domicilio_completo', 'N/A')}")
        print(f"  - Calle: {domicilio.get('calle', 'N/A')}")
        print(f"  - Colonia: {domicilio.get('colonia', 'N/A')}")
        print(f"  - CP: {domicilio.get('codigo_postal', 'N/A')}")
        print(f"  - Estado: {domicilio.get('estado', 'N/A')}")
    else:
        print(f"\n--- DATOS MRZ ---")
        mrz = datos.get('mrz', {})
        print(f"Nombre: {mrz.get('nombre_completo', 'N/A')}")
        print(f"  - Apellido Paterno: {mrz.get('apellido_paterno', 'N/A')}")
        print(f"  - Apellido Materno: {mrz.get('apellido_materno', 'N/A')}")
        print(f"  - Nombres: {mrz.get('nombres', 'N/A')}")
        print(f"Documento: {mrz.get('numero_documento', 'N/A')}")
        print(f"Fecha Nacimiento: {mrz.get('fecha_nacimiento', 'N/A')}")
        print(f"Sexo: {mrz.get('sexo', 'N/A')}")
        print(f"CURP: {datos.get('curp', 'N/A')}")
    
    print(f"\n--- TEXTO CRUDO (primeros 500 chars) ---")
    print(datos.get('texto_crudo', '')[:500])


# ============================================================================
# FUNCIONES LEGACY ADICIONALES - Para compatibilidad con scripts existentes
# ============================================================================

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
