"""Publica una copia verificada con salidas (excepción acordada en #54)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from pdi_tp import publicaciones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo", type=Path)
    parser.add_argument("--origen", required=True, help="Ruta del notebook limpio, dentro del repo")
    parser.add_argument("--nombre", help="Nombre de la copia en ejecutados/")
    parser.add_argument("--tipo", choices=["ejecucion_completa", "consulta_guardada"], default="ejecucion_completa")
    parser.add_argument("--familia", choices=["clasificadores", "yolo26n"])
    parser.add_argument("--omitir-exportacion-csv", action="store_true", help="Solo test: omitir la celda adicional de descarga de CSV de Colab")
    args = parser.parse_args()
    nombre = args.nombre or Path(args.origen).name
    if Path(nombre).name != nombre or not nombre.endswith(".ipynb"):
        parser.error("--nombre debe ser un nombre .ipynb, sin directorios")
    nb = json.loads(args.archivo.read_text(encoding="utf-8"))
    if args.omitir_exportacion_csv:
        esperada = ('from google.colab import files\n\n'
                    'predicciones.to_csv("/content/predicciones_test_resnet50.csv", index=False)\n'
                    'files.download("/content/predicciones_test_resnet50.csv")')
        if Path(args.origen).name != "18_evaluacion_test.ipynb" or publicaciones.texto(nb["cells"][-1]["source"]).strip() != esperada:
            parser.error("La última celda no es la exportación adicional conocida del test de ResNet50")
        nb["cells"].pop()
        nb.setdefault("metadata", {}).setdefault("ejecucion_entrega", {})["preparacion"] = (
            "se omitió únicamente la celda adicional que descargaba el CSV de ResNet50 en Colab; "
            "SHA-256 del archivo recibido: " + publicaciones.huella(args.archivo.read_bytes()))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    copia = publicaciones.preparar(nb, raiz=RAIZ, origen=args.origen, commit=commit, tipo=args.tipo, familia=args.familia)
    destino = RAIZ / "ejecutados" / nombre
    destino.parent.mkdir(exist_ok=True)
    destino.write_bytes(publicaciones.serializar(copia))
    print(f"Publicada {destino.relative_to(RAIZ)} ({destino.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
