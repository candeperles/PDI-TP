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
positivo (ver notebooks/04_marca_de_agua.ipynb). Es un detector a mano: puede fallar en recortes donde solo
queda un pedacito de la barra o del logo.

Para medir si un modelo ya entrenado usa la marca (notebooks/13_marca_de_agua_efecto.ipynb) hay dos pruebas, cada
una con su control, que hace lo mismo con una franja sin marca:

- **Tapar**: en las imágenes de Jamaica con marca se reemplaza la franja de abajo por el reflejo de las filas de
  arriba (`tapar`). Control: lo mismo en las de Jamaica sin marca. Si el modelo usa la marca como "sin paneles",
  la probabilidad sube más al tapar la marca que en el control.
- **Pegar**: sobre las imágenes con paneles de las otras fuentes se pega la franja de abajo de una imagen de
  Jamaica con marca (`pegar_franja`). Control: la franja de una imagen de Jamaica sin marca. Las mismas imágenes
  reciben las dos franjas, así que la comparación es pareada. Los donantes son siempre imágenes sin paneles, para
  no pegar paneles.
"""
import numpy as np
import pandas as pd
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


def tapar(imagen: np.ndarray, desde: int = FILA_INICIO_MARCA) -> np.ndarray:
    """Copia de la imagen con las filas `desde` en adelante reemplazadas por el reflejo de las de arriba.

    Se refleja en vez de pintar un color plano para no agregar un borde nuevo que la red pueda usar.
    """
    imagen = np.asarray(imagen)
    alto = imagen.shape[0] - desde
    if not 0 < alto <= desde:
        raise ValueError(f"No se puede reflejar desde la fila {desde} en una imagen de {imagen.shape[0]} filas.")
    tapada = imagen.copy()
    tapada[desde:] = imagen[desde - alto:desde][::-1]
    return tapada


def pegar_franja(destino: np.ndarray, donante: np.ndarray, desde: int = FILA_INICIO_MARCA) -> np.ndarray:
    """Copia de `destino` con las filas `desde` en adelante tomadas de `donante`."""
    destino, donante = np.asarray(destino), np.asarray(donante)
    if destino.shape != donante.shape:
        raise ValueError(f"Las imágenes tienen tamaños distintos: {destino.shape} y {donante.shape}.")
    pegada = destino.copy()
    pegada[desde:] = donante[desde:]
    return pegada


def armar_pruebas(tabla: pd.DataFrame, semilla: int = 42) -> pd.DataFrame:
    """Una fila por imagen modificada: prueba, grupo, archivo, donante (o vacío), split y y_real.

    `tabla` tiene una fila por imagen con `archivo`, `split`, `jamaica`, `marca` y `con_paneles` (la del
    notebook 04). Tapar usa todas las imágenes de Jamaica (las de train el modelo ya las vio: se reportan aparte).
    Pegar usa las de val con paneles de las otras fuentes, y cada una recibe un donante con marca y uno sin marca,
    elegidos al azar entre las imágenes de Jamaica sin paneles.
    """
    faltan = {"archivo", "split", "jamaica", "marca", "con_paneles"} - set(tabla.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    jamaica = tabla[tabla["jamaica"] & tabla["split"].notna()]
    tapadas = jamaica.assign(prueba="tapar", grupo=np.where(jamaica["marca"], "con marca", "control"), donante="")
    destinos = tabla[~tabla["jamaica"] & (tabla["split"] == "val") & (tabla["con_paneles"] == 1)]
    sin_paneles = tabla[tabla["jamaica"] & (tabla["con_paneles"] == 0)]
    rng = np.random.default_rng(semilla)
    pegadas = []
    for grupo, con_marca in (("con marca", True), ("control", False)):
        donantes = sin_paneles.loc[sin_paneles["marca"] == con_marca, "archivo"].to_numpy()
        if not len(donantes):
            raise ValueError(f"No hay imágenes de Jamaica sin paneles para el grupo {grupo!r}.")
        pegadas.append(destinos.assign(prueba="pegar", grupo=grupo, donante=rng.choice(donantes, len(destinos))))
    columnas = ["prueba", "grupo", "archivo", "donante", "split", "con_paneles"]
    pruebas = pd.concat([tapadas, *pegadas])[columnas].rename(columns={"con_paneles": "y_real"})
    return pruebas.reset_index(drop=True)


def modificar(prueba: pd.Series, leer) -> np.ndarray:
    """La imagen modificada de una fila de `armar_pruebas`. `leer(nombre)` devuelve la imagen original."""
    if prueba["prueba"] == "tapar":
        return tapar(leer(prueba["archivo"]))
    if prueba["prueba"] == "pegar":
        return pegar_franja(leer(prueba["archivo"]), leer(prueba["donante"]))
    raise ValueError(f"Prueba desconocida: {prueba['prueba']!r}")


def _intervalo(valores: np.ndarray, rng, n: int, nivel: float) -> tuple[float, float]:
    medias = valores[rng.integers(0, len(valores), size=(n, len(valores)))].mean(1)
    return tuple(np.percentile(medias, [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100]))


def efecto(resultados: pd.DataFrame, *, n: int = 2000, semilla: int = 42, nivel: float = 0.95) -> pd.DataFrame:
    """Cuánto cambia la probabilidad por la marca, para cada corrida y prueba.

    `resultados` tiene una fila por corrida e imagen modificada: `corrida`, `prueba`, `grupo`, `archivo`, `split`,
    `original` y `modificada` (probabilidad o puntaje antes y después) y `umbral`. Para cada grupo se reporta el
    cambio medio (modificada − original) y cuántas imágenes cambian de predicción. El efecto de la marca es la
    diferencia entre el grupo "con marca" y su control, con un intervalo bootstrap: pareado en pegar (mismas
    imágenes), independiente en tapar (imágenes distintas). En tapar se espera que sea positivo y en pegar
    negativo si el modelo usa la marca; si el intervalo incluye el 0, no hay evidencia de que la use.
    """
    datos = resultados.assign(cambio=resultados["modificada"] - resultados["original"],
                              cambia_prediccion=(resultados["original"] >= resultados["umbral"])
                              != (resultados["modificada"] >= resultados["umbral"]))
    rng = np.random.default_rng(semilla)
    filas = []
    for (corrida, prueba, conjunto), parte in datos.groupby(["corrida", "prueba", "split"], sort=False):
        marca, control = parte[parte["grupo"] == "con marca"], parte[parte["grupo"] == "control"]
        if marca.empty or control.empty:
            continue
        fila = {"corrida": corrida, "prueba": prueba, "split": conjunto,
                "n_con_marca": len(marca), "n_control": len(control),
                "cambio_con_marca": marca["cambio"].mean(), "cambio_control": control["cambio"].mean(),
                "cambian_con_marca": int(marca["cambia_prediccion"].sum()),
                "cambian_control": int(control["cambia_prediccion"].sum())}
        if prueba == "pegar":
            pares = marca.set_index("archivo")["cambio"] - control.set_index("archivo")["cambio"]
            if pares.isna().any():
                raise ValueError(f"{corrida}: en pegar, cada imagen tiene que tener su control.")
            efecto_, (inf, sup) = pares.mean(), _intervalo(pares.to_numpy(), rng, n, nivel)
        else:
            cambio_marca, cambio_control = marca["cambio"].to_numpy(), control["cambio"].to_numpy()
            efecto_ = cambio_marca.mean() - cambio_control.mean()
            remuestreo = (cambio_marca[rng.integers(0, len(cambio_marca), size=(n, len(cambio_marca)))].mean(1)
                          - cambio_control[rng.integers(0, len(cambio_control), size=(n, len(cambio_control)))].mean(1))
            inf, sup = np.percentile(remuestreo, [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100])
        fila.update(efecto=efecto_, efecto_inf=inf, efecto_sup=sup)
        filas.append(fila)
    return pd.DataFrame(filas)
