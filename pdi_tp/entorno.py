"""Prepara el entorno de trabajo, en Colab o en local.

- Descarga el dataset si no está (scripts/download_data.py).
- En Colab monta Google Drive (`drive.mount`) y usa la carpeta compartida del equipo, así los
  resultados no se pierden si se corta la sesión. Es el único mecanismo de acceso a Drive del repo.
- Los resultados se organizan por modelo y modo (ver pdi_tp/resultados.py), no por integrante.
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
    resultados_equipo: Path  # raíz de resultados: <raíz>/<modelo>/<modo>/ (ver pdi_tp/resultados.py)
    resultados: Path  # igual a resultados_equipo; se conserva por compatibilidad
    analisis: Path | None = None  # salidas de los notebooks de datos (validación, revisión)
    cache: Path | None = None  # archivos intermedios reutilizables (features, folds); no son resultados


def preparar(integrante: str, usar_drive: bool = True) -> Rutas:
    integrante = integrante.strip().lower()
    if not integrante or integrante == "nombre":
        raise ValueError("Poné tu nombre en INTEGRANTE: queda registrado como autor de cada corrida.")

    datos = REPO_ROOT / "data" / "grupo_5"
    if not datos.is_dir():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "download_data.py")], check=True)
    if not all((datos / carpeta).is_dir() for carpeta in ("images", "labels")):
        raise FileNotFoundError(f"Dataset incompleto o inaccesible en {datos}: se esperan images/ y labels/.")

    if en_colab() and usar_drive:
        from google.colab import drive

        drive.mount(str(DRIVE_MOUNT))
        if not DRIVE_DIR.is_dir():
            raise FileNotFoundError(
                f"No existe {DRIVE_DIR}. Agregá un acceso directo a la carpeta compartida "
                "'PDI-TP' en tu 'Mi unidad' (ver README)."
            )
        base = DRIVE_DIR
    else:
        base = REPO_ROOT / "resultados_locales"

    resultados = base / "resultados"
    for carpeta in (resultados, base / "analisis", base / "cache"):
        carpeta.mkdir(parents=True, exist_ok=True)

    rutas = Rutas(
        integrante=integrante,
        repo=REPO_ROOT,
        datos=datos,
        imagenes=datos / "images",
        labels=datos / "labels",
        resultados_equipo=resultados,
        resultados=resultados,
        analisis=base / "analisis",
        cache=base / "cache",
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
