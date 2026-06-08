"""API pública: funciones de extracción de datos de INE."""
import cv2
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from .models import (Detection, OCRResult, NameData, AddressData,
                     FrontData, MRZData, BackData, MatchResult,
                     CorrectedName, INEData)
from .factory import get_ocr_engine, get_field_extractor, get_validator

# Substrings de etiquetas/campos que indican contaminación del campo nombre
_LABELS_CONTAM = (
    'FECHA', 'NACIM', 'VIGEN', 'DOMICIL', 'CLAVE', 'ELECTOR', 'SECCION',
    'SECCIÓN', 'REGISTRO', 'CURP', 'SEXO', 'EMISION', 'EMISIÓN', 'ESTADO',
    'MUNICIPIO', 'LOCALIDAD', 'OMBRE', 'ANOD', 'TLAJOMULCO',
)


def _token_contaminado(tok: str) -> bool:
    """True si el token parece etiqueta o ruido de zona vecina."""
    t = tok.upper()
    if len(t) > 18:  # token anormalmente largo = palabras pegadas/ruido
        return True
    return any(lbl in t for lbl in _LABELS_CONTAM)


def _es_etiqueta_nombre(tok: str) -> bool:
    """True si el token es la etiqueta 'NOMBRE' mal leída por OCR (IOMBRE, OMBRE...)."""
    t = (tok or "").upper()
    return t.endswith("OMBRE") or t in ("NOMBRE", "NOMERE", "NOMBPE", "NOMBRF")


def _token_implausible(tok: str) -> bool:
    """True si un token de nombre es ruido OCR (1 letra, o sin vocales)."""
    t = re.sub(r'[^A-ZÑ]', '', tok.upper())
    if len(t) <= 1:
        return True
    vocales = sum(1 for c in t if c in 'AEIOU')
    # un nombre real tiene al menos una vocal; ruido tipo "IMIOILIC" / "COFICIL"
    # se cae si la proporción de vocales es muy baja para su longitud
    if len(t) >= 4 and vocales == 0:
        return True
    return False


def _limpiar_nombres(nombres: str) -> str:
    """
    Limpia el campo nombres: corta en la primera palabra contaminada (etiqueta
    de zona vecina), descarta tokens de ruido OCR al final, y limita a 3 tokens.
    """
    if not nombres:
        return nombres
    limpios = []
    for tk in nombres.split():
        if _token_contaminado(tk):
            break  # ruido de otra zona, lo que sigue es basura
        if limpios and _token_implausible(tk):
            break  # ya hay un nombre válido; token final implausible = ruido OCR
        limpios.append(tk)
    return ' '.join(limpios[:3])  # nombres compuestos rara vez >3


def _fecha_desde_curp(curp: str) -> Optional[str]:
    """
    Deriva la fecha de nacimiento (DD/MM/AAAA) desde el CURP.
    CURP pos 4-9 = YYMMDD. El siglo se infiere de la homoclave (pos 16):
    dígito → nacido antes de 2000 (19YY), letra → 2000+ (20YY).
    """
    if not curp or len(curp) < 17:
        return None
    yy, mm, dd = curp[4:6], curp[6:8], curp[8:10]
    if not (yy.isdigit() and mm.isdigit() and dd.isdigit()):
        return None
    mm_i, dd_i = int(mm), int(dd)
    if not (1 <= mm_i <= 12 and 1 <= dd_i <= 31):
        return None
    homoclave = curp[16]
    siglo = "19" if homoclave.isdigit() else "20"
    return f"{dd}/{mm}/{siglo}{yy}"


def _fecha_norm(f: str) -> str:
    """Normaliza una fecha a dígitos DDMMYYYY para comparar."""
    return re.sub(r'\D', '', str(f or ''))


def extraer_datos_ine_frente(img: np.ndarray, use_zones: bool = True) -> Dict:
    """
    Extrae todos los datos del frente de la INE.

    OPTIMIZADO v5: con PP-OCRv5 la estrategia de IMAGEN COMPLETA (1 sola pasada
    OCR) supera a la de zonas (5 pasadas) tanto en precisión (+7pp) como en
    velocidad (1.7x), porque el detector de v5 no necesita recortes para leer
    bien. Las zonas quedan como FALLBACK si la imagen completa falla.

    Args:
        img: Imagen en formato numpy array (BGR)
        use_zones: Si True, permite el fallback por zonas (default: True)

    Returns:
        Diccionario con todos los campos extraídos, incluyendo nombre_raw_ocr
    """
    engine = get_ocr_engine()
    extractor = get_field_extractor()

    nombre_raw_ocr = None
    front_data = None

    # RUTA PRINCIPAL (v5): OCR de imagen completa en una sola pasada.
    try:
        ocr_result = engine.run_ocr(img, fast_mode=True)
        front_data = extractor.extract_front(ocr_result)
        nombre_raw_global = extractor.extract_name(ocr_result.combined_text, None)
        nombre_raw_ocr = nombre_raw_global.nombre_completo if nombre_raw_global else None
    except Exception:
        front_data = None
        ocr_result = None

    # FALLBACK por zonas: solo si la imagen completa no obtuvo campos críticos.
    if use_zones:
        tiene_nombre = bool(front_data and front_data.nombre.nombre_completo)
        tiene_id = bool(front_data and (front_data.curp or front_data.clave_elector))
        if not (tiene_nombre and tiene_id):
            try:
                zone_results = engine.run_ocr_by_zones(img, tipo="frente")
                if zone_results:
                    combined_text = " ".join(
                        r.combined_text for r in zone_results.values() if r and r.combined_text
                    )
                    all_detections = [d for r in zone_results.values() if r for d in r.detections]
                    avg_conf = float(np.mean([d.confidence for d in all_detections])) if all_detections else 0.7
                    ocr_zone = OCRResult(combined_text=combined_text, detections=all_detections,
                                         confidence=avg_conf, engine="paddleocr")
                    fd_zone = extractor.extract_front_with_zones(ocr_zone, zone_results)
                    # Usar el resultado por zonas si recupera más campos críticos
                    z_ok = bool(fd_zone.nombre.nombre_completo) and bool(fd_zone.curp or fd_zone.clave_elector)
                    if z_ok or front_data is None:
                        front_data = fd_zone
                        ocr_result = ocr_zone
                        if 'nombre' in zone_results:
                            nombre_raw_ocr = zone_results['nombre'].combined_text.upper()
                            nombre_raw_ocr = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', '', nombre_raw_ocr)
                            nombre_raw_ocr = ' '.join(nombre_raw_ocr.split())
            except Exception:
                pass

    if front_data is None:  # último recurso
        ocr_result = engine.run_ocr(img, fast_mode=False)
        front_data = extractor.extract_front(ocr_result)

    # Fix 6: si apellido_paterno es la etiqueta "NOMBRE" mal leída (IOMBRE...),
    # todo el nombre quedó corrido un lugar → desplazar: materno→paterno,
    # 1er token de nombres→materno, resto→nombres.
    if _es_etiqueta_nombre(front_data.nombre.apellido_paterno or ""):
        am_old = front_data.nombre.apellido_materno or ""
        nom_old = (front_data.nombre.nombre or "").split()
        front_data.nombre.apellido_paterno = am_old
        front_data.nombre.apellido_materno = nom_old[0] if nom_old else None
        front_data.nombre.nombre = ' '.join(nom_old[1:]) if len(nom_old) > 1 else None
        partes = [p for p in [front_data.nombre.apellido_paterno,
                              front_data.nombre.apellido_materno,
                              front_data.nombre.nombre] if p]
        front_data.nombre.nombre_completo = ' '.join(partes)

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

    # Fix 2: limpiar contaminación del campo nombres (etiquetas / zonas vecinas)
    nom_actual = front_data.nombre.nombre or ""
    nom_limpio = _limpiar_nombres(nom_actual)
    if nom_limpio != nom_actual:
        front_data.nombre.nombre = nom_limpio
        partes = [p for p in [front_data.nombre.apellido_paterno,
                              front_data.nombre.apellido_materno, nom_limpio] if p]
        front_data.nombre.nombre_completo = ' '.join(partes)

    # Fix 5: fallback de CP. Si parse_address no lo encontró, buscar en el texto
    # completo un número de 5 díg aislado que no sea año (el CP es el único así).
    if not front_data.domicilio.codigo_postal and ocr_result:
        for m in re.finditer(r'(?<!\d)(\d{5})(?!\d)', ocr_result.combined_text):
            cp = m.group(1)
            if not (19000 <= int(cp) <= 21000):  # descartar rangos tipo año*10
                front_data.domicilio.codigo_postal = cp
                break

    # Fix 3: votación de fecha de nacimiento con el CURP como ancla.
    # El CURP (validado por checksum) es la fuente más confiable de la fecha.
    # Si el campo fecha falta o discrepa del CURP, usar la del CURP.
    fecha_curp = _fecha_desde_curp(curp_val)
    if fecha_curp:
        fecha_campo = front_data.fecha_nacimiento
        if not fecha_campo or _fecha_norm(fecha_campo) != _fecha_norm(fecha_curp):
            front_data.fecha_nacimiento = fecha_curp

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

