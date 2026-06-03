# OCR Avanzado Versión "2.0"

`ocr_avanzado_v2.py` es un módulo de OCR para extraer información de credenciales INE mexicanas a partir de imágenes del frente, reverso o ambos lados. Está diseñado para correr en CPU, sin depender de GPU, y combina preprocesamiento con OpenCV, OCR por zonas, PaddleOCR, Tesseract y validación cruzada entre frente y reverso.

La versión v2 está enfocada en reducir tiempos de procesamiento en lotes, especialmente cuando se procesan muchas INEs. Para lograrlo, cambia la estrategia del flujo original, prioriza zonas específicas de la credencial y usa OCR completo solo cuando los campos críticos no se pueden extraer con suficiente confianza.

## Características principales

- Extracción de datos del frente de la INE.
- Extracción de datos del reverso, principalmente MRZ, CIC y OCR vertical.
- Detección automática de frente o reverso con `extraer_datos_ine(img, tipo="auto")`.
- Validación cruzada entre frente y reverso con `validar_cruzado_ine`.
- Corrección parcial de errores comunes de OCR en CURP y nombres.
- OCR por zonas para acelerar el procesamiento.
- Fallbacks inteligentes a OCR completo cuando las zonas no son suficientes.
- Procesamiento MRZ optimizado con Tesseract antes de usar PaddleOCR.
- Compatibilidad con API legacy mediante funciones como `ocr_combinado`, `extraer_curp`, `extraer_clave_elector`, `extraer_fecha_nacimiento`, `extraer_sexo`, `extraer_mrz`, `extraer_nombre_completo` y `extraer_domicilio`.
- Compatibilidad adicional con Windows para localizar Tesseract si está instalado en la ruta estándar.
- Shim de compatibilidad para NumPy 2.x, necesario porque algunas versiones de PaddlePaddle todavía dependen de `np.sctypes`.

## Arquitectura general

El módulo está dividido en capas internas.

| Componente | Responsabilidad |
|---|---|
| `Preprocessor` | Corrige orientación, detecta región de tarjeta, ajusta perspectiva, escala imagen, mejora contraste y genera variantes procesadas. |
| `INEZoneExtractor` | Recorta zonas específicas del frente y reverso, como nombre, domicilio, datos, fechas, inferior, MRZ y datos extra. |
| `OCREngine` | Ejecuta PaddleOCR y Tesseract, combina resultados, filtra duplicados, maneja OCR por zonas y aplica early exit. |
| `FieldExtractor` | Convierte texto OCR en campos estructurados como nombre, CURP, domicilio, vigencia, sección, MRZ, CIC y OCR vertical. |
| `Validator` | Valida CURP, compara frente y reverso, calcula porcentaje de match y puede corregir nombre usando MRZ como referencia. |
| Data models | Usa `dataclasses` como `FrontData`, `BackData`, `MRZData`, `NameData`, `AddressData`, `OCRResult` y `MatchResult`. |

## Requisitos

Este módulo usa Python y depende de librerías de visión por computadora y OCR.

```bash
pip install opencv-python numpy pillow pytesseract paddleocr paddlepaddle
```

## Configuración de CPU

```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FLAGS_use_cuda'] = '0'
os.environ['FLAGS_use_gpu'] = '0'
os.environ['PADDLEOCR_USE_GPU'] = '0'
```

## Uso rápido

### Extraer datos del frente

```python
import cv2
from ocr_avanzado_v2 import extraer_datos_ine_frente

img = cv2.imread("ine_frente.jpg")
datos = extraer_datos_ine_frente(img)

print(datos["nombre"])
print(datos["curp"])
print(datos["vigencia"])
```

### Extraer datos del reverso

```python
import cv2
from ocr_avanzado_v2 import extraer_datos_ine_reverso

img = cv2.imread("ine_reverso.jpg")
datos = extraer_datos_ine_reverso(img)

print(datos["mrz"])
print(datos["cic"])
print(datos["ocr_vertical"])
```

### Detección automática de tipo de imagen

```python
import cv2
from ocr_avanzado_v2 import extraer_datos_ine

img = cv2.imread("ine.jpg")
datos = extraer_datos_ine(img, tipo="auto")

print(datos["tipo"])
```

### Validación cruzada frente y reverso

```python
import cv2
from ocr_avanzado_v2 import validar_cruzado_ine

frente = cv2.imread("ine_frente.jpg")
reverso = cv2.imread("ine_reverso.jpg")

resultado = validar_cruzado_ine(frente, reverso)

print(resultado["validacion"])
print(resultado["nombre_corregido"])
```

## Campos extraídos

### Frente

La función `extraer_datos_ine_frente` devuelve un diccionario con esta estructura general:

| Campo | Descripción |
|---|---|
| `tipo` | Tipo de documento procesado, normalmente `INE_FRENTE`. |
| `nombre` | Objeto con `apellido_paterno`, `apellido_materno`, `nombre` y `nombre_completo`. |
| `nombre_raw_ocr` | Nombre leído directamente por OCR antes de correcciones adicionales. |
| `domicilio` | Objeto con calle, número exterior, número interior, colonia, código postal, municipio, estado y domicilio completo. |
| `sexo` | Sexo detectado en la credencial. |
| `curp` | CURP extraída y corregida cuando es posible. |
| `clave_elector` | Clave de elector detectada. |
| `fecha_nacimiento` | Fecha de nacimiento normalizada cuando se puede interpretar. |
| `anio_registro` | Año de registro. |
| `anio_emision` | Año de emisión. |
| `tipo_ine` | Identificación general como `IFE` o `INE`. |
| `modelo_ine` | Modelo específico de credencial, como C, D, E, F, G o H. |
| `seccion` | Sección electoral. |
| `vigencia` | Vigencia detectada. |
| `texto_crudo` | Texto OCR completo o combinado. |
| `confianza_ocr` | Confianza estimada del resultado. |

### Reverso

La función `extraer_datos_ine_reverso` devuelve un diccionario con esta estructura general:

| Campo | Descripción |
|---|---|
| `tipo` | Tipo de documento procesado, normalmente `INE_REVERSO`. |
| `mrz.lineas_raw` | Líneas MRZ tal como fueron leídas por OCR. |
| `mrz.lineas_clean` | MRZ limpio y validado desde `IDMEX` cuando se logra reconstruir. |
| `mrz.documento_tipo` | Tipo de documento detectado en MRZ. |
| `mrz.pais` | País detectado, normalmente `MEX`. |
| `mrz.numero_documento` | Número de documento del MRZ. |
| `mrz.nombre_completo` | Nombre completo reconstruido desde MRZ. |
| `mrz.apellido_paterno` | Apellido paterno desde MRZ. |
| `mrz.apellido_materno` | Apellido materno desde MRZ. |
| `mrz.nombres` | Nombres desde MRZ. |
| `mrz.fecha_nacimiento` | Fecha de nacimiento desde MRZ. |
| `mrz.sexo` | Sexo desde MRZ. |
| `mrz.fecha_expiracion` | Fecha de expiración desde MRZ. |
| `curp` | CURP extraída del reverso si aparece. |
| `cic` | Código de Identificación de Credencial. |
| `ocr_vertical` | Identificador ciudadano vertical. |
| `texto_crudo` | Texto OCR completo o combinado. |
| `confianza_ocr` | Confianza estimada del resultado. |

## Flujo de procesamiento en v2

Para el frente, v2 intenta primero una ruta rápida por zonas. Procesa zonas como `nombre`, `datos`, `domicilio`, `fechas` e `inferior`. Si encuentra nombre y al menos un identificador crítico, como CURP o clave de elector, se queda con ese resultado. Si no, cae al OCR completo y reutiliza zonas ya calculadas para no repetir trabajo innecesario.

Para el reverso, v2 intenta primero OCR por zonas y prioriza el MRZ. Si el MRZ produce nombre completo o número de documento, devuelve directamente el resultado. Si no hay MRZ suficiente, ejecuta OCR completo como fallback.

La zona MRZ tiene una ruta especial. Primero intenta Tesseract con whitelist `A-Z0-9<`, después intenta Tesseract con CLAHE y solo si eso falla usa PaddleOCR. Esta decisión reduce bastante el tiempo de reverso, porque evita usar PaddleOCR sobre líneas MRZ anchas cuando Tesseract puede resolverlas rápido.

## Validación cruzada

`validar_cruzado_ine(img_frente, img_reverso)` compara CURP, fecha de nacimiento, sexo y nombre entre frente y reverso. El validador asigna pesos internos a los campos y devuelve un porcentaje de match, un resultado general y una bandera para saber si puede aprobarse automáticamente.

| Campo validado | Peso interno |
|---|---:|
| CURP | 40% |
| Fecha de nacimiento | 25% |
| Sexo | 15% |
| Nombre | 20% |

La función también intenta corregir el nombre usando MRZ como referencia cuando hay coincidencias fuertes en campos más confiables, como CURP, fecha o sexo.

## Cambios del OG al v2

| Área | OG | v2 |
|---|---|---|
| Estrategia del frente | Ejecuta OCR completo en `fast_mode` primero y después usa zonas si hay buena confianza. | Ejecuta OCR por zonas primero y solo cae a OCR completo si faltan campos críticos. |
| Estrategia del reverso | Ejecuta zonas para MRZ, pero también corre OCR completo en la ruta principal. | Devuelve el resultado desde zonas si el MRZ ya tiene nombre completo o número de documento. OCR completo queda como fallback. |
| MRZ | Usa PaddleOCR como ruta principal para la zona MRZ, con CLAHE si el primer intento falla. | Usa Tesseract con whitelist primero, luego CLAHE + Tesseract, y después PaddleOCR solo como fallback. |
| Validación MRZ | Requiere al menos 18 caracteres `<` y longitud mínima de 100. | Baja el mínimo a 10 caracteres `<` y longitud mínima de 85 para aceptar MRZ más cortos leídos por Tesseract. |
| Tamaño mínimo de escalado | `MIN_SIZE = 1200`. | `MIN_SIZE = 900`, reduciendo upscaling y tiempo de reconocimiento. |
| Compatibilidad NumPy 2.x | No incluye shim para `np.sctypes`. | Añade shim para `np.sctypes` cuando no existe, mejorando compatibilidad con PaddlePaddle. |
| Compatibilidad Windows | No configura ruta local de Tesseract. | Intenta usar `C:\Program Files\Tesseract-OCR\tesseract.exe` si Tesseract no está en `PATH`. |
| Reutilización de zonas | Puede recalcular zonas en fallback. | Guarda y reutiliza zonas del primer intento para ahorrar tiempo. |
| Corrección de apellidos | No aplica swap con CURP como ancla al final del frente. | Compara iniciales de CURP con apellido paterno y materno, y hace swap si detecta orden invertido. |
| Patrón de nombre MRZ | Permite capturas más amplias en componentes del nombre. | Limita componentes a rangos de longitud para evitar capturar texto externo al MRZ. |
| Objetivo principal | Mayor contexto textual y mejor precisión promedio. | Mejor velocidad en procesamiento por lotes, aceptando pérdida de precisión en algunos campos. |

## Resultado (En dispositivo no optimizado)

### Velocidad

| Métrica | OG | v2 | Ganancia |
|---|---:|---:|---:|
| Frente avg | 11.7s | 7.4s | 1.6x |
| Reverso avg | 7.7s | 3.8s | 2.0x |
| 100 imgs total | 16.1 min | 9.3 min | 1.7x más rápido |

### Precisión frente

| Campo | OG | v2 |
|---|---:|---:|
| `apellido_paterno` | 94% | 88% |
| `apellido_materno` | 86% | 80% |
| `nombre(s)` | 66% | 44% |
| `curp` | 92% | 92% empate |
| `clave_elector` | 2% | 4% casi nada |
| `fecha_nacimiento` | 96% | 84% |
| `vigencia` | 100% | 84% |
| `sección` | 50% | 40% |
| `CP` | 54% | 32% |

### Precisión reverso MRZ

| Campo | OG | v2 |
|---|---:|---:|
| `apellidos` | 96% | 88% |
| `nombres MRZ` | 86% | 58% |
| `num. documento` | 94% | 92% |
| `fecha nacimiento` | 98% | 92% |
| `sexo` | 100% | 95% |
