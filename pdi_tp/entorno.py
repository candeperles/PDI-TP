"""Prepara el entorno de trabajo, en Colab o en local.

- Descarga el dataset si no está (scripts/download_data.sh).
- En Colab monta Google Drive y usa la carpeta compartida del equipo para los resultados,
  así no se pierden si se corta la sesión.
- Avisa si no hay GPU.
"""
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVE_MOUNT = Path("/content/drive")
# Carpeta compartida del equipo (ver README: cada integrante agrega un acceso directo en "Mi unidad").
DRIVE_DIR = DRIVE_MOUNT / "MyDrive" / "PDI-TP"


def en_colab() -> bool:
    return "google.colab" in sys.modules


@dataclass
class Rutas:
    integrante: str
    repo: Path
    datos: Path
    imagenes: Path
    labels: Path
    resultados_equipo: Path  # resultados de todos los integrantes
    resultados: Path  # resultados propios: resultados_equipo / integrante


def preparar(integrante: str, usar_drive: bool = True) -> Rutas:
    integrante = integrante.strip().lower()
    if not integrante or integrante == "nombre":
        raise ValueError("Poné tu nombre en INTEGRANTE, se usa para separar tus resultados.")

    datos = REPO_ROOT / "data" / "grupo_5"
    if not datos.is_dir():
        subprocess.run(["bash", str(REPO_ROOT / "scripts" / "download_data.sh")], check=True)

    if en_colab() and usar_drive:
        from google.colab import drive

        drive.mount(str(DRIVE_MOUNT))
        if not DRIVE_DIR.is_dir():
            raise FileNotFoundError(
                f"No existe {DRIVE_DIR}. Agregá un acceso directo a la carpeta compartida "
                "'PDI-TP' en tu 'Mi unidad' (ver README)."
            )
        resultados_equipo = DRIVE_DIR / "resultados"
    else:
        resultados_equipo = REPO_ROOT / "resultados"

    resultados = resultados_equipo / integrante
    resultados.mkdir(parents=True, exist_ok=True)

    rutas = Rutas(
        integrante=integrante,
        repo=REPO_ROOT,
        datos=datos,
        imagenes=datos / "images",
        labels=datos / "labels",
        resultados_equipo=resultados_equipo,
        resultados=resultados,
    )
    _resumen(rutas)
    return rutas


def _resumen(rutas: Rutas) -> None:
    n_imagenes = len(list(rutas.imagenes.glob("*.png")))
    print(f"Dataset:    {rutas.datos} ({n_imagenes} imágenes)")
    print(f"Resultados: {rutas.resultados}")
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        print(f"GPU:        {gpu}")
    elif en_colab():
        print("GPU:        NO HAY. Entorno de ejecución → Cambiar tipo de entorno → T4.")
    else:
        print("GPU:        no hay GPU NVIDIA (corriendo en local)")
