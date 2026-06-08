"""Extracción de zonas específicas de la INE."""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from .preprocessor import Preprocessor

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
        'domicilio': (0.32, 0.50, 0.95, 0.62), # Zona de domicilio (extendida a 0.62 para capturar CP/colonia)
        'datos': (0.32, 0.60, 0.70, 0.85),     # CURP, Clave elector (x=0.70: ensanchar baja CURP -8pp)
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


