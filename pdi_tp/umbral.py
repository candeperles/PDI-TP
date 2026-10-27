"""Umbral de decisión y métricas sobre Jamaica, como el test de Kingston (issue #7).

El test va a tener mayoría de imágenes sin paneles, al revés de train (85 % con paneles). Por eso:

- El umbral se elige maximizando la **balanced accuracy** (promedio del acierto en cada clase), que no
  depende de la proporción de clases. La F1 macro sí depende: se reporta pero no se usa para elegir.
- Elegir el umbral y medir sobre las mismas imágenes da un número optimista. La estimación honesta deja
  afuera un grupo espacial por vez: elige el umbral con el resto y lo aplica al grupo que quedó afuera.

Hoy las únicas predicciones de Jamaica son las de val (23 imágenes en 9 grupos). Las funciones reciben
cualquier tabla de predicciones, así que sirven igual para las predicciones fuera de fold de la validación
cruzada sobre Jamaica cuando existan (ver docs/evaluacion_jamaica.md).

Uso en un notebook:

    tabla = umbral.tabla_jamaica(resultados.cargar_predicciones(rutas))
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import split

GRILLA = np.round(np.arange(0.05, 0.951, 0.05), 2)  # umbrales candidatos


def balanced_accuracy(y, pred) -> float:
    """Promedio del acierto de cada clase presente (si falta una clase, el acierto de la otra)."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    aciertos = [(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()]
    return float(np.mean(aciertos)) if aciertos else float("nan")


def elegir_umbral(y, probabilidad, grilla=GRILLA) -> float:
    """Umbral de `grilla` que maximiza la balanced accuracy.

    Si empatan varios, se toma el tramo contiguo más largo de umbrales empatados (ante igual largo, el más
    cercano a 0.5) y se devuelve su punto medio: así el umbral no queda pegado al borde de una sola imagen.
    """
    grilla = np.asarray(grilla, dtype=float)
    p = np.asarray(probabilidad, dtype=float)
    puntajes = np.array([balanced_accuracy(y, p >= u) for u in grilla])
    empatados = np.flatnonzero(np.isclose(puntajes, np.nanmax(puntajes)))
    tramos = np.split(empatados, np.flatnonzero(np.diff(empatados) > 1) + 1)
    medios = [(grilla[t[0]] + grilla[t[-1]]) / 2 for t in tramos]
    mejor = max(range(len(tramos)), key=lambda i: (len(tramos[i]), -abs(medios[i] - 0.5)))
    return float(medios[mejor])


def evaluar_umbral_por_grupos(y, probabilidad, grupos, grilla=GRILLA) -> tuple[np.ndarray, np.ndarray]:
    """Predicción honesta: para cada grupo, umbral elegido con los demás grupos y aplicado a ese grupo.

    Devuelve (predicción, umbral usado para cada imagen).
    """
    y, p, grupos = np.asarray(y), np.asarray(probabilidad, dtype=float), np.asarray(grupos)
    umbrales = np.empty(len(p))
    for g in np.unique(grupos):
        afuera = grupos == g
        umbrales[afuera] = elegir_umbral(y[~afuera], p[~afuera], grilla)
    return (p >= umbrales).astype(int), umbrales


def _metricas(y, pred) -> dict:
    from sklearn.metrics import f1_score

    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {
        "acierto_con": float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
        "acierto_sin": float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan"),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "f1_macro": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
    }


def grupos_jamaica(archivos) -> np.ndarray:
    """Grupo espacial de cada imagen de Jamaica, calculado junto con todas las de Jamaica del split.

    Se agrupa con el split completo (no solo con `archivos`) para que dos imágenes de val unidas a través de
    una vecina de train queden en el mismo grupo. Ningún grupo mezcla Jamaica con el resto del dataset.
    """
    nombres = [Path(a).stem for a in archivos]
    del_split = []
    for nombre in ("train", "val"):
        lista = split.SPLITS_DIR / f"{nombre}.txt"
        if lista.is_file():
            del_split += [Path(a).stem for a in lista.read_text().split() if split.es_jamaica(a)]
    todos = list(dict.fromkeys(nombres + del_split))
    ids = dict(zip(todos, split.grupos_espaciales(todos)))
    return np.array([ids[n] for n in nombres])


def tabla_jamaica(predicciones: pd.DataFrame, grilla=GRILLA) -> pd.DataFrame:
    """Una fila por modelo/modo con las métricas solo sobre Jamaica, con umbral 0.5 y con umbral elegido.

    `predicciones`: salida de `resultados.cargar_predicciones(rutas)` (o cualquier tabla con modelo, modo,
    archivo, y_real y probabilidad). Columnas:
    - `n`, `n_con`, `n_sin`: imágenes de Jamaica evaluadas.
    - `<métrica>_05`: con umbral 0.5.
    - `umbral`: elegido con todas las imágenes de Jamaica; es el que se usaría en el test.
    - `<métrica>_cv`: estimación honesta del umbral elegido (dejando afuera un grupo espacial por vez).
    - `umbral_cv_min` / `umbral_cv_max`: cuánto varía el umbral elegido según el grupo que queda afuera.
    """
    jamaica = predicciones[predicciones["archivo"].map(split.es_jamaica)]
    filas = []
    for (modelo, modo), g in jamaica.groupby(["modelo", "modo"], sort=True):
        y, p = g["y_real"].to_numpy().astype(int), g["probabilidad"].to_numpy(dtype=float)
        pred_cv, umbrales_cv = evaluar_umbral_por_grupos(y, p, grupos_jamaica(g["archivo"]), grilla)
        fila = {"modelo": modelo, "modo": modo, "n": len(y), "n_con": int(y.sum()), "n_sin": int((1 - y).sum())}
        fila.update({f"{k}_05": v for k, v in _metricas(y, p >= 0.5).items()})
        fila["umbral"] = elegir_umbral(y, p, grilla)
        fila.update({f"{k}_cv": v for k, v in _metricas(y, pred_cv).items()})
        fila.update(umbral_cv_min=float(umbrales_cv.min()), umbral_cv_max=float(umbrales_cv.max()))
        filas.append(fila)
    if not filas:
        return pd.DataFrame(columns=["modelo", "modo", "n", "n_con", "n_sin", "umbral"])
    return (pd.DataFrame(filas).sort_values(["balanced_accuracy_cv", "f1_macro_cv"], ascending=False)
            .reset_index(drop=True))
