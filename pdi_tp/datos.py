"""Lectura del dataset: imágenes PNG y etiquetas YOLO (clase x_centro y_centro ancho alto, normalizados)."""
from pathlib import Path

import numpy as np
from PIL import Image

from .entorno import Rutas


def listar_imagenes(rutas: Rutas) -> list[Path]:
    return sorted(rutas.imagenes.glob("*.png"))


def label_de(rutas: Rutas, imagen: Path) -> Path:
    return rutas.labels / f"{imagen.stem}.txt"


def leer_cajas(label: Path) -> np.ndarray:
    """Devuelve un array (n, 5) con [clase, xc, yc, w, h]. Vacío (0, 5) si la imagen no tiene paneles."""
    cajas = np.loadtxt(label, ndmin=2) if label.stat().st_size else np.empty((0, 5))
    return cajas.reshape(-1, 5)


def tiene_paneles(label: Path) -> bool:
    return len(leer_cajas(label)) > 0


def leer_imagen(imagen: Path) -> np.ndarray:
    # Algunas PNG vienen en modo paleta (P) en vez de RGB: se convierten todas a RGB.
    return np.asarray(Image.open(imagen).convert("RGB"))


def dibujar_cajas(ax, rutas: Rutas, imagen: Path, colores=("red", "yellow", "cyan")) -> None:
    from matplotlib.patches import Rectangle

    img = leer_imagen(imagen)
    alto, ancho = img.shape[:2]
    ax.imshow(img)
    for clase, xc, yc, w, h in leer_cajas(label_de(rutas, imagen)):
        x0, y0 = (xc - w / 2) * ancho, (yc - h / 2) * alto
        ax.add_patch(Rectangle((x0, y0), w * ancho, h * alto, fill=False, lw=1.5,
                               edgecolor=colores[int(clase) % len(colores)]))
    ax.set_axis_off()
