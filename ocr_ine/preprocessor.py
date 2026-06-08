"""Preprocesamiento adaptativo de imagen para OCR."""
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional

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


