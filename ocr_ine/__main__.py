"""
CLI del paquete ocr_ine.
Uso: python -m ocr_ine <ruta_imagen> [tipo]
  tipo: frente, reverso, auto (default: auto)
"""
import sys
import cv2

from .api import extraer_datos_ine


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m ocr_ine <ruta_imagen> [tipo]")
        print("  tipo: frente, reverso, auto (default: auto)")
        sys.exit(1)

    img_path = sys.argv[1]
    tipo = sys.argv[2] if len(sys.argv) > 2 else "auto"

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {img_path}")
        sys.exit(1)

    print(f"Procesando: {img_path}")
    print(f"Tipo: {tipo}")
    print("=" * 50)

    datos = extraer_datos_ine(img, tipo=tipo)

    print(f"\nTipo detectado: {datos.get('tipo')}")
    print(f"Confianza OCR: {datos.get('confianza_ocr'):.1f}%")

    if datos.get('tipo') == 'INE_FRENTE':
        print(f"\n--- DATOS EXTRAÍDOS ---")
        nombre = datos.get('nombre', {})
        print(f"Nombre: {nombre.get('nombre_completo', 'N/A')}")
        print(f"  - Apellido Paterno: {nombre.get('apellido_paterno', 'N/A')}")
        print(f"  - Apellido Materno: {nombre.get('apellido_materno', 'N/A')}")
        print(f"  - Nombre(s): {nombre.get('nombre', 'N/A')}")
        print(f"Sexo: {datos.get('sexo', 'N/A')}")
        print(f"CURP: {datos.get('curp', 'N/A')}")
        print(f"Clave Elector: {datos.get('clave_elector', 'N/A')}")
        print(f"Fecha Nacimiento: {datos.get('fecha_nacimiento', 'N/A')}")
        print(f"Sección: {datos.get('seccion', 'N/A')}")
        print(f"Vigencia: {datos.get('vigencia', 'N/A')}")
        domicilio = datos.get('domicilio', {})
        print(f"Domicilio: {domicilio.get('domicilio_completo', 'N/A')}")
        print(f"  - Calle: {domicilio.get('calle', 'N/A')}")
        print(f"  - Colonia: {domicilio.get('colonia', 'N/A')}")
        print(f"  - CP: {domicilio.get('codigo_postal', 'N/A')}")
        print(f"  - Estado: {domicilio.get('estado', 'N/A')}")
    else:
        print(f"\n--- DATOS MRZ ---")
        mrz = datos.get('mrz', {})
        print(f"Nombre: {mrz.get('nombre_completo', 'N/A')}")
        print(f"  - Apellido Paterno: {mrz.get('apellido_paterno', 'N/A')}")
        print(f"  - Apellido Materno: {mrz.get('apellido_materno', 'N/A')}")
        print(f"  - Nombres: {mrz.get('nombres', 'N/A')}")
        print(f"Documento: {mrz.get('numero_documento', 'N/A')}")
        print(f"Fecha Nacimiento: {mrz.get('fecha_nacimiento', 'N/A')}")
        print(f"Sexo: {mrz.get('sexo', 'N/A')}")
        print(f"CURP: {datos.get('curp', 'N/A')}")

    print(f"\n--- TEXTO CRUDO (primeros 500 chars) ---")
    print(datos.get('texto_crudo', '')[:500])


if __name__ == "__main__":
    main()
