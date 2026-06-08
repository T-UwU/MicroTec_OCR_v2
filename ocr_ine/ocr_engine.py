"""Motor OCR con múltiples pasadas (PaddleOCR + Tesseract)."""
import cv2
import numpy as np
import pytesseract
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from .models import Detection, OCRResult
from .preprocessor import Preprocessor
from .zone_extractor import INEZoneExtractor
from .paddle_loader import get_paddleocr_reader, is_paddleocr_available

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
    
    # Modo de realce de zona. Con PP-OCRv5 el sharpen ya NO aporta (precisión
    # idéntica con y sin: 85%) y solo añade latencia → "none". (Con el v3 antiguo
    # sí ayudaba +2pp; el rec de v5 es robusto y no necesita realce de bordes.)
    zone_enhance_mode = "none"

    def _enhance_zone(self, img: np.ndarray) -> np.ndarray:
        """
        Realce de zona antes del OCR. Despacha según `zone_enhance_mode`.

        Modos disponibles (probados empíricamente contra ground truth):
        - none      : passthrough (default; mejor precisión/velocidad)
        - clahe     : CLAHE LAB (empeora frente -4%)
        - upscale   : upscale 2x (neutro, 2x más lento)
        - denoise   : bilateral filter (preserva bordes)
        - sharpen   : unsharp mask
        - gamma     : corrección gamma adaptativa
        - antiglare : reducción de reflejos
        - otsu      : binarización Otsu
        - gray_sharp: grayscale + sharpen suave
        """
        mode = self.zone_enhance_mode
        if mode == "none":
            return img
        try:
            if mode == "clahe":
                return self.preprocessor.enhance_contrast(img)
            if mode == "upscale":
                h = img.shape[0]
                if 0 < h < 256:
                    s = min(256 / h, 2.0)
                    if s > 1.05:
                        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                return img
            if mode == "denoise":
                if len(img.shape) == 3:
                    return cv2.bilateralFilter(img, 5, 50, 50)
                return cv2.bilateralFilter(img, 5, 50, 50)
            if mode == "sharpen":
                blur = cv2.GaussianBlur(img, (0, 0), 2.0)
                return cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            if mode == "gamma":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                mean = float(np.mean(gray))
                g = 1.4 if mean < 110 else (0.8 if mean > 180 else 1.0)
                if g == 1.0:
                    return img
                inv = 1.0 / g
                table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
                return cv2.LUT(img, table)
            if mode == "antiglare":
                return self.preprocessor._anti_reflejo(img)
            if mode == "otsu":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            if mode == "gray_sharp":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
                sharp = cv2.addWeighted(gray, 1.3, blur, -0.3, 0)
                return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
        except Exception:
            return img
        return img

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

                # OPT v5: en reverso, si el MRZ ya se resolvió (Tesseract), saltar
                # 'datos_extra' (PaddleOCR ~3s, CURP/CIC secundarios) — gran ahorro.
                if zone_name == 'datos_extra' and 'mrz' in results:
                    continue

                # Realzar la zona (sharpen) antes del OCR. Se EXCLUYE 'datos'
                # (CURP) porque el sharpen baja la precisión de CURP -2pp.
                if zone_name == 'datos':
                    zone_enh = zone_img
                else:
                    zone_enh = self._enhance_zone(zone_img)

                # Otras zonas: una sola pasada sin cls (rápido)
                text, detections = self._run_paddleocr(zone_enh, use_cls=False)
                avg_conf = np.mean([d.confidence for d in detections]) if detections else 0.0

                # Solo retry con cls para zona 'nombre' si resultado muy malo
                if zone_name == 'nombre' and (not text.strip() or avg_conf < 0.5):
                    text_cls, detections_cls = self._run_paddleocr(zone_enh, use_cls=True)
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
    
    # Realce del MRZ antes de binarizar. PROBADO contra ground truth:
    # "sharpen" EMPEORA el MRZ -3% (num_doc -8, ap_paterno -6) porque la
    # fuente OCR-B sobre fondo limpio ya se binariza óptimo con Otsu y el
    # sharpen mete ruido. Se deja en "none".
    mrz_enhance = "none"

    def _run_tesseract_mrz(self, img: np.ndarray) -> str:
        """
        OCR rápido para MRZ usando Tesseract con whitelist A-Z0-9<.
        Tesseract tarda ~0.3-0.5s vs ~30s de PaddleOCR para líneas MRZ anchas.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            if self.mrz_enhance == "sharpen":
                blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
                gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
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
        Ejecuta PaddleOCR (PP-OCRv5, API 3.x) en una imagen.

        API 3.x: reader.predict(img) → lista de dicts con 'rec_texts',
        'rec_scores' y 'rec_polys' (en vez del antiguo [[bbox,(text,conf)]]).
        El parámetro use_cls se ignora (la orientación de línea se configura
        en el constructor en 3.x); se mantiene por compatibilidad de firma.
        """
        reader = get_paddleocr_reader()
        if reader is None:
            return "", []

        try:
            # Convertir a BGR si es escala de grises
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            results = reader.predict(img)

            text_parts = []
            detections = []

            for res in (results or []):
                # res es dict-like (PaddleOCR 3.x)
                try:
                    textos = res['rec_texts']
                except Exception:
                    textos = res.get('rec_texts', []) if hasattr(res, 'get') else []
                try:
                    scores = res['rec_scores']
                except Exception:
                    scores = res.get('rec_scores', []) if hasattr(res, 'get') else []
                try:
                    polys = res['rec_polys']
                except Exception:
                    polys = res.get('rec_polys', []) if hasattr(res, 'get') else []

                for i, text in enumerate(textos):
                    conf = float(scores[i]) if i < len(scores) else 0.9
                    bbox = polys[i] if i < len(polys) else []
                    # Normalizar bbox a lista de [x, y] (puede venir como np.ndarray)
                    if hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    text_parts.append(text)
                    detections.append(Detection(text=text, confidence=conf, bbox=bbox))

            return " ".join(text_parts), detections
        except Exception:
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


