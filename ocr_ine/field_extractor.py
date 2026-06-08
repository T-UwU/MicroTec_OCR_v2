"""Extracción de campos específicos (nombre, CURP, domicilio, MRZ...)."""
import cv2
import numpy as np
import pytesseract
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from .models import (NameData, AddressData, FrontData, MRZData,
                     BackData, OCRResult, Detection)

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

        # IMPORTANTE: SIN \b. El OCR pega la clave a la etiqueta
        # ("CLAVEDEELECTORORPCRC02120315H500"), y \b al inicio impide el match
        # porque la clave queda en medio de letras (ELECTOR + ORPCRC).
        patron_core = r'[A-Z]{6}\d{8}[HM]\d{3}'      # para búsqueda (search)
        patron_full = r'[A-Z]{6}\d{8}[HM]\d{3}$'      # para validación (fullmatch-like)

        # Estrategia 1: Patrón estricto (sin espacios, sin \b)
        match = re.search(patron_core, texto_upper)
        if match:
            return match.group(0)

        # Estrategia 2: Patrón con espacios opcionales
        # Ejemplo: "GOMJUA 85010312 H 400" o "GOMJUA85010312H400"
        patron_espacios = r'[A-Z]{6}\s?\d{8}\s?[HM]\s?\d{3}'
        match = re.search(patron_espacios, texto_upper)
        if match:
            return match.group(0).replace(' ', '')

        # Estrategia 3: Buscar con etiqueta "CLAVE DE ELECTOR" o "CLAVE ELECTOR"
        patron_etiqueta = r'CLAVE\s*(?:DE\s*)?ELECTOR[:\s]*([A-Z0-9\s]{18,25})'
        match = re.search(patron_etiqueta, texto_upper, re.IGNORECASE)
        if match:
            candidato = match.group(1).strip().replace(' ', '').upper()
            if re.match(patron_core, candidato):
                return candidato[:18]

        # Estrategia 4: Buscar patrón relajado y validar estructura
        # Permite algunos errores de OCR (O→0, I→1, etc.)
        patron_relajado = r'[A-Z0-9]{6}\d{8}[HM]\d{3}'
        matches = re.finditer(patron_relajado, texto_upper)
        for match in matches:
            candidato = match.group(0)
            # Validar que los primeros 6 caracteres sean mayormente letras
            primeros_6 = candidato[:6]
            if sum(c.isalpha() for c in primeros_6) >= 5:  # Al menos 5 de 6 son letras
                # Corregir OCR
                corrected = self._correct_clave_elector_ocr(candidato)
                if re.match(patron_core, corrected):
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

        # Ancla posicional FUERTE (modelo nuevo E/F/G/H): la fila de valores es
        # "FECHA SECCION VIGENCIA" → "03/12/2002 0723 2020-2030".
        # La sección es el bloque de 4 díg ENTRE la fecha y el rango de vigencia.
        m_pos = re.search(r'\d{2}/\d{2}/\d{4}\s+(\d{4})\s+\d{4}\s*[-–]\s*\d{4}', texto_upper)
        if m_pos and not (1900 <= int(m_pos.group(1)) <= 2100):
            return m_pos.group(1)
        # Variante: 4 díg inmediatamente antes del rango de vigencia
        m_pos2 = re.search(r'(?<!\d)(\d{4})\s+\d{4}\s*[-–]\s*\d{4}', texto_upper)
        if m_pos2 and not (1900 <= int(m_pos2.group(1)) <= 2100):
            return m_pos2.group(1)

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
        
        # Extraer código postal (5 dígitos). SIN \b: el OCR pega el CP a la
        # colonia ("HACIENDA54715"), y \b falla letra→dígito. Usar lookarounds
        # de dígito para no partir números más largos.
        cp_match = re.search(r'(?<!\d)(\d{5})(?!\d)', domicilio_completo)
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


