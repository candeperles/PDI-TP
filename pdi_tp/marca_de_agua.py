"""Marca de agua de Google Maps en las imágenes de Jamaica (issue #7).

Las capturas de Jamaica (`kingston`, `kingston2` y `geo`) son de Google Maps, y algunos recortes traen la marca
de agua al pie. Las otras fuentes no la tienen. El test de Kingston también va a ser de Google Maps, así que un
modelo podría aprender "marca de agua → Jamaica → sin paneles" en vez de mirar los paneles.

La marca tiene tres partes:

- **Barra de atribución**: franja gris oscura semitransparente con "Imagery ©2025 Airbus, CNES / Airbus, Maxar
  Technologies" en letras claras. En estos recortes de 256 px empieza en la fila 226–228 y llega hasta abajo.
- **Logo "Google"**: letras blancas con contorno oscuro, abajo a la izquierda, a la altura de la barra.
- **"© 2025 Google" casi transparente**, repetido sobre toda la imagen. Es muy tenue y no se detecta acá.

Los umbrales se calibraron mirando las 172 imágenes de Jamaica y comprobando que ninguna de las otras 728 dé
positivo (ver notebooks/11_marca_de_agua.ipynb). Es un detector a mano: puede fallar en recortes donde solo
queda un pedacito de la barra o del logo.
"""
import numpy as np
from scipy import ndimage

# Barra: borde superior marcado, entre estas filas, y al menos este largo en píxeles.
FILAS_BORDE_BARRA = (222, 231)
LARGO_MIN_BARRA = 25
# Logo: zona donde se busca y mínimo de píxeles blancos pegados a un contorno oscuro.
ZONA_LOGO = (slice(212, 254), slice(0, 140))
PIXELES_MIN_LOGO = 80
# Zona que cubre la barra y el logo en todos los recortes: de esta fila para abajo.
FILA_INICIO_MARCA = 222


def _barra(imagen: np.ndarray) -> tuple[int, int | None]:
    """(largo, fila) de la corrida horizontal más larga que parece el borde superior de la barra."""
    luminancia = imagen.mean(2)
    saturacion = np.ptp(imagen, axis=2)
    mejor = (0, None)
    for fila in range(FILAS_BORDE_BARRA[0] - 26, 252):
        arriba = luminancia[fila - 3:fila].mean(0)
        abajo = luminancia[fila:fila + 2].mean(0)
        gris = (saturacion[fila:fila + 2] < 12).all(0)
        borde = gris & (abajo < 60) & (abajo < 0.65 * arriba)
        etiquetas, n = ndimage.label(borde)
        if n:
            largo = int(np.bincount(etiquetas.ravel())[1:].max())
            if largo > mejor[0]:
                mejor = (largo, fila)
    return mejor


def _logo(imagen: np.ndarray) -> int:
    """Píxeles casi blancos con un píxel casi negro a 2 px o menos, en la zona del logo."""
    zona = imagen[ZONA_LOGO]
    blanco = zona.min(2) > 225
    cerca_de_negro = ndimage.binary_dilation(zona.max(2) < 50, iterations=2)
    return int((blanco & cerca_de_negro).sum())


def detectar(imagen: np.ndarray) -> dict:
    """Detecta la barra de atribución y el logo en una imagen RGB de 256×256 (uint8 o float en 0..255)."""
    imagen = np.asarray(imagen, dtype=float)
    if imagen.shape != (256, 256, 3):
        raise ValueError(f"Se espera una imagen RGB de 256×256, no {imagen.shape}.")
    largo, fila = _barra(imagen)
    pixeles_logo = _logo(imagen)
    barra = largo >= LARGO_MIN_BARRA and fila is not None and FILAS_BORDE_BARRA[0] <= fila <= FILAS_BORDE_BARRA[1]
    logo = pixeles_logo >= PIXELES_MIN_LOGO
    return {"marca": bool(barra or logo), "barra": bool(barra), "logo": bool(logo),
            "barra_largo": largo, "barra_fila": fila, "logo_pixeles": pixeles_logo}
