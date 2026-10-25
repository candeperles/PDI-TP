"""Descarga el dataset en Windows, Linux o Colab, sin requerir Bash.

Uso: python scripts/download_data.py
Si está instalado gh, se utiliza su autenticación; si no, se usa la URL pública.
"""
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "candeperles/PDI-TP"
TAG = "dataset-v1"
ASSET = "grupo_5.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def descargar() -> None:
    destino = DATA_DIR / "grupo_5"
    if destino.exists():
        if not (destino / "images").is_dir() or not (destino / "labels").is_dir():
            raise RuntimeError(f"Dataset incompleto en {destino}; revisar antes de descargar de nuevo.")
        print(f"El dataset ya existe: {destino}. No se modifica.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    # Solo se publica la carpeta final cuando la descarga y extracción terminan.
    with tempfile.TemporaryDirectory(prefix="descarga-", dir=DATA_DIR) as temporal:
        temporal = Path(temporal)
        archivo = temporal / ASSET
        print(f"Descargando {ASSET} ({TAG})...", flush=True)
        if shutil.which("gh"):
            subprocess.run([
                "gh", "release", "download", TAG, "--repo", REPO,
                "--pattern", ASSET, "--dir", str(temporal),
            ], check=True)
        else:
            url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
            with urllib.request.urlopen(url, timeout=120) as respuesta, archivo.open("wb") as salida:
                shutil.copyfileobj(respuesta, salida)

        extraido = temporal / "extraido"
        extraido.mkdir()
        with zipfile.ZipFile(archivo) as zip_dataset:
            # Evita que entradas del ZIP escriban fuera de la carpeta temporal.
            for entrada in zip_dataset.infolist():
                ruta = (extraido / entrada.filename).resolve()
                if not ruta.is_relative_to(extraido.resolve()):
                    raise ValueError(f"Ruta inválida en el ZIP: {entrada.filename}")
            zip_dataset.extractall(extraido)
        grupo = extraido / "grupo_5"
        if not (grupo / "images").is_dir() or not (grupo / "labels").is_dir():
            raise RuntimeError("El ZIP no contiene grupo_5/images y grupo_5/labels.")
        grupo.rename(destino)
    print(f"Dataset disponible en {destino}")


if __name__ == "__main__":
    descargar()
