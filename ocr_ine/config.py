"""
Configuración crítica del entorno OCR (PaddleOCR 3.x / PP-OCRv5).
DEBE importarse antes de cualquier uso de PaddleOCR.
- Suprime CUDA/GPU
- Configura la ruta de Tesseract en Windows
NOTA: PaddlePaddle 3.x soporta NumPy 2.x nativo → ya NO se necesita el shim
de np.sctypes que requería la versión 2.x.
"""
import os
import warnings
import platform

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FLAGS_use_cuda'] = '0'
os.environ['FLAGS_use_gpu'] = '0'
os.environ['PADDLEOCR_USE_GPU'] = '0'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import pytesseract

# Ruta de Tesseract en Windows si no está en PATH
if platform.system() == 'Windows':
    import shutil
    if not shutil.which('tesseract'):
        _tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.isfile(_tess_path):
            pytesseract.pytesseract.tesseract_cmd = _tess_path

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
