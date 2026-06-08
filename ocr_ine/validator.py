"""Validación cruzada frente/reverso y corrección con MRZ."""
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from .models import (NameData, FrontData, MRZData, BackData,
                     MatchResult, CorrectedName)

# Proxy perezoso para evitar import circular con factory
def get_field_extractor():
    from .factory import get_field_extractor as _gfe
    return _gfe()

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

