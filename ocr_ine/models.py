"""Modelos de datos (dataclasses) del OCR de INE."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


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


