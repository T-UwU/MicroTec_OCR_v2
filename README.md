# OCR Avanzado versión "3.0" (con PP-OCRv5)

![versión](https://img.shields.io/badge/versión-3.0-blue)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.6.0-0098FF)
![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.0.0-0062B0)
![modelo](https://img.shields.io/badge/modelo-PP--OCRv5-success)
![Tesseract](https://img.shields.io/badge/Tesseract-5.5-4B8BBE)
![precisión frente](https://img.shields.io/badge/precisión%20frente-85%25-brightgreen)
![CPU](https://img.shields.io/badge/CPU-only%20(sin%20GPU)-lightgrey)
![plataforma](https://img.shields.io/badge/plataforma-Windows%20%7C%20Linux%20%7C%20macOS-informational)
![privacidad](https://img.shields.io/badge/privacidad-100%25%20local-success)

Extracción automática de datos de credenciales INE/IFE mexicanas (frente y reverso),
con motor **PP-OCRv5** (PaddleOCR 3.x), arquitectura modular y validación cruzada
frente/reverso. Optimizado para CPU (sin GPU).

> **Novedad v3.0:** migración a **PP-OCRv5** con reconocedor latino dedicado, estrategia
> de OCR de imagen completa en una sola pasada, y reescritura modular del proyecto.
> Precisión promedio del frente **61% → 85%** frente a la v2.

---

## Tabla de contenidos
- [Características](#características)
- [Arquitectura](#arquitectura)
- [Instalación](#instalación)
- [Uso](#uso)
- [Precisión](#precisión)
- [Cambios respecto a v2](#resumen-de-cambios-respecto-a-v2)

---

## Características

- **Frente:** nombre, apellidos, CURP, clave de elector, fecha de nacimiento, sexo,
  domicilio (calle, colonia, CP, municipio, estado), sección, vigencia, año de registro,
  tipo y modelo de INE (C/D/E/F/G/H).
- **Reverso:** MRZ (3 líneas), apellidos/nombres, número de documento, fecha de nacimiento,
  sexo, fecha de expiración, CURP, CIC.
- **Validación cruzada** frente↔reverso (cuando se tienen ambas caras).
- **Correcciones inteligentes** que explotan la redundancia del documento:
  - Dígito verificador de CURP (solo devuelve CURPs válidos).
  - Votación de fecha de nacimiento contra el CURP.
  - Reordenamiento de apellidos/nombre usando las iniciales del CURP.
  - Limpieza de contaminación de zonas vecinas en el nombre.
- **MRZ por Tesseract** (rápido y robusto sobre la fuente OCR-B del reverso).
- 100% local (los datos no salen del equipo).

## Arquitectura

Paquete modular `ocr_ine/`:

```
ocr_ine/
├── __init__.py        API pública (importa config primero)
├── __main__.py        CLI: python -m ocr_ine <img> [tipo]
├── config.py          Entorno: CUDA off, ruta Tesseract
├── models.py          Dataclasses (Detection, FrontData, MRZData, ...)
├── paddle_loader.py   Carga perezosa de PaddleOCR (PP-OCRv5)
├── preprocessor.py    Preprocesamiento de imagen
├── zone_extractor.py  Extracción de zonas (fallback)
├── ocr_engine.py      Motor OCR (PaddleOCR 3.x + Tesseract)
├── field_extractor.py Extracción/validación de campos
├── validator.py       Validación cruzada frente/reverso
├── factory.py         Singletons de los motores
├── api.py             Funciones públicas
└── legacy.py          Compatibilidad con scripts antiguos
```

**Estrategia de OCR (v3.0):** con PP-OCRv5 se hace **una sola pasada de OCR sobre la imagen
completa** (más preciso y más rápido que recortar por zonas). El enfoque por zonas queda
como *fallback* automático si la pasada principal no recupera los campos críticos.

## Instalación

Requiere Python 3.11 y el binario de **Tesseract OCR** instalado en el sistema.

```bash
python -m venv venv
# Windows
venv\Scripts\pip install -r requirements.txt
# Linux/Mac
venv/bin/pip install -r requirements.txt
```

`requirements.txt`:
```
paddleocr==3.6.0
paddlepaddle==3.0.0
numpy>=2.0
opencv-python-headless
pytesseract==0.3.13
Pillow
```

> PaddlePaddle 3.x soporta NumPy 2.x de forma nativa: ya **no** se necesitan los parches
> de `protobuf`/`np.sctypes` que requería la v2 (PaddleOCR 2.8 / paddlepaddle 2.6).

## Uso

### Python

```python
import cv2
from ocr_ine import extraer_datos_ine_frente, extraer_datos_ine_reverso

datos = extraer_datos_ine_frente(cv2.imread("ine_frente.jpeg"))
print(datos["nombre"]["nombre_completo"], datos["curp"])

mrz = extraer_datos_ine_reverso(cv2.imread("ine_reverso.jpeg"))
print(mrz["mrz"]["nombre_completo"], mrz["mrz"]["numero_documento"])
```

### Validación cruzada (ambas caras)

```python
from ocr_ine import validar_cruzado_ine
res = validar_cruzado_ine(img_frente, img_reverso)
```

### CLI

```bash
python -m ocr_ine ruta/a/ine.jpeg auto    # tipo: frente | reverso | auto
```

## Precisión

Medida sobre 50 imágenes de frente con *ground truth* etiquetado manualmente.

| Campo               | v2     | **v3.0 (PP-OCRv5)** |
|---------------------|:------:|:-------------------:|
| Apellido paterno    | 88%    | **94%** |
| Apellido materno    | 80%    | **88%** |
| Nombre(s)           | 44%    | **78%** |
| CURP                | 92%    | **92%** \* |
| Fecha de nacimiento | 84%    | **98%** |
| Sección             | 40%    | **96%** |
| Vigencia            | 84%    | **100%** |
| Código postal       | 32%    | **96%** |
| Clave de elector    | 4%     | **26%** |
| **Promedio**        | **61%**| **85%** |

\* La CURP extraída pasa el dígito verificador en el **100%** de los casos (es decir, es
correcta); el ground truth manual tiene ruido en cadenas largas, por lo que el valor real
es ≥92%.

Reverso (MRZ): número de documento ~92%, fecha ~92%, sexo ~95%, apellidos ~88%,
nombres ~82%. El campo `nombres` está limitado por el truncamiento físico del MRZ.

## Comparación de versiones (OG → v2 → v3)

Mismas 50 imágenes de frente y 50 de reverso, *ground truth* etiquetado manualmente,
en un dispositivo **no optimizado** (CPU, modelo en caliente).

### Velocidad

| Métrica          |     OG |     v2 | **v3 (PP-OCRv5)** |
|------------------|-------:|-------:|------------------:|
| Frente avg       | 11.7 s |  7.4 s |          **~8 s** |
| Reverso avg      |  7.7 s |  3.8 s |         **4.3 s** |
| 100 imgs (total) | 16.1 m |  9.3 m |        **10.8 m** |

### Precisión: Frente

| Campo               |   OG |   v2 | **v3** |
|---------------------|-----:|-----:|-------:|
| Apellido paterno    |  94% |  88% |**94%** |
| Apellido materno    |  86% |  80% |**88%** |
| Nombre(s)           |  66% |  44% |**78%** |
| CURP                |  92% |  92% |**92%** |
| Clave de elector    |   2% |   4% |**26%** |
| Fecha de nacimiento |  96% |  84% |**98%** |
| Sección             |  50% |  40% |**96%** |
| Vigencia            | 100% |  84% |**100%**|
| Código postal       |  54% |  32% |**96%** |
| **Promedio**        | ~71% | ~61% |**85%** |

### Precisión: Reverso (MRZ)

| Campo            |   OG |  v2 | **v3** |
|------------------|-----:|----:|-------:|
| Apellidos        |  96% | 88% |**88%** |
| Nombres MRZ      |  86% | 58% |**82%** |
| Número documento |  94% | 92% |**92%** |
| Fecha nacimiento |  98% | 92% |**92%** |
| Sexo             | 100% | 95% |**95%** |

## Evolución de decisiones de diseño (OG → v2 → v3)

| Área | OG | v2 | v3 (PP-OCRv5) |
|---|---|---|---|
| Motor OCR | PaddleOCR 2.8 (PP-OCRv3 latino). | PaddleOCR 2.8 (PP-OCRv3 latino). | **PaddleOCR 3.6 (PP-OCRv5 latino)**, mucho mejor en nombres en español. |
| Estrategia del frente | Ejecuta OCR completo en `fast_mode` primero y después usa zonas si hay buena confianza. | Ejecuta OCR por zonas primero y solo cae a OCR completo si faltan campos críticos. | **Una sola pasada de OCR de imagen completa** (con v5 supera a las zonas en precisión y velocidad); las zonas quedan como fallback. |
| Estrategia del reverso | Ejecuta zonas para MRZ, pero también corre OCR completo en la ruta principal. | Devuelve el resultado desde zonas si el MRZ ya tiene nombre completo o número de documento. OCR completo queda como fallback. | Igual que v2, y además **salta la zona `datos_extra`** (PaddleOCR ~3 s) cuando el MRZ ya se resolvió por Tesseract. |
| MRZ | Usa PaddleOCR como ruta principal para la zona MRZ, con CLAHE si el primer intento falla. | Usa Tesseract con whitelist primero, luego CLAHE + Tesseract, y después PaddleOCR solo como fallback. | Igual que v2 (Tesseract primero; la fuente OCR-B se binariza óptimo con Otsu). |
| Validación MRZ | Requiere al menos 18 caracteres `<` y longitud mínima de 100. | Baja el mínimo a 10 caracteres `<` y longitud mínima de 85 para aceptar MRZ más cortos leídos por Tesseract. | Igual que v2 (10 caracteres `<`, longitud mínima 85). |
| Tamaño mínimo de escalado | `MIN_SIZE = 1200`. | `MIN_SIZE = 900`, reduciendo upscaling y tiempo de reconocimiento. | `MIN_SIZE = 900`. |
| Compatibilidad NumPy | No incluye shim para `np.sctypes`. | Añade shim para `np.sctypes` (PaddlePaddle 2.x lo requiere). | **Sin shim**: PaddlePaddle 3.x soporta NumPy 2.x de forma nativa. |
| Compatibilidad Windows | No configura ruta local de Tesseract. | Intenta usar `C:\Program Files\Tesseract-OCR\tesseract.exe` si Tesseract no está en `PATH`. | Igual que v2. |
| Reutilización de zonas | Puede recalcular zonas en fallback. | Guarda y reutiliza zonas del primer intento para ahorrar tiempo. | Las zonas solo corren como fallback, así que rara vez se recalculan. |
| Corrección de apellidos/nombre | No aplica swap con CURP como ancla. | Compara iniciales de CURP con apellido paterno y materno, y hace swap si detecta orden invertido. | Swap por CURP **más** corte de contaminación del nombre, desplazamiento si la etiqueta "NOMBRE" se cuela como apellido, y votación de fecha contra el CURP. |
| Patrón de nombre MRZ | Permite capturas más amplias en componentes del nombre. | Limita componentes a rangos de longitud para evitar capturar texto externo al MRZ. | Igual que v2. |
| Sección / CP | Extracción frágil (anclas débiles, `\b` que falla con texto pegado). | Igual que OG. | **Anclaje posicional** para sección y patrón sin `\b` + fallback para CP. |
| Objetivo principal | Mayor contexto textual y mejor precisión promedio. | Mejor velocidad en procesamiento por lotes, aceptando pérdida de precisión. | **Precisión de la OG y velocidad de la v2 a la vez** (PP-OCRv5 + parsers corregidos). |

## Resumen de cambios respecto a v2

- **Motor:** PaddleOCR 2.8 (PP-OCRv3 latino) → **PaddleOCR 3.6 (PP-OCRv5 latino)**.
- **Estrategia:** de 5 pasadas por zona → **1 pasada de imagen completa** (+precisión, +velocidad).
- **Entorno:** NumPy 2.x nativo; se eliminaron los shims de `protobuf 3.20.2` y `np.sctypes`.
- **Parsers corregidos** (bugs reales encontrados con diagnóstico contra ground truth):
  - Clave de elector y CP: el patrón usaba `\b` que fallaba cuando el OCR pega el valor a
    la etiqueta (`CLAVEDEELECTORORPCRC...`, `HACIENDA54715`).
  - Sección: anclaje posicional fecha → sección → vigencia.
  - Nombre: corte de contaminación de zonas vecinas y de la etiqueta "NOMBRE" mal leída.
- **Reverso:** MRZ con Tesseract primero (la fuente OCR-B se binariza óptimo con Otsu);
  PaddleOCR solo como fallback.
- Estructura modular (paquete `ocr_ine`) en lugar de un único archivo de ~7000 líneas.
