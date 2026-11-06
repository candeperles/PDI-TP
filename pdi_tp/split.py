"""División train/val compartida por todo el equipo.

Las imágenes vienen de 4 fuentes y muchas son recortes vecinos de una misma imagen grande
(p. ej. PV01 son tiles contiguos de la misma instalación). Si recortes vecinos quedan unos en
train y otros en val, la validación sobreestima el rendimiento. Por eso:

1. Se agrupan las imágenes por cercanía espacial (según lo que codifica cada nombre de archivo).
2. Se divide por grupos, estratificando por fuente + con/sin paneles, para que val tenga las
   mismas proporciones que el total.

El resultado queda en splits/train.txt y splits/val.txt (un nombre de archivo por línea).
Se genera una sola vez con scripts/make_split.py y se versiona: todos usan los mismos archivos.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .datos import label_de, listar_imagenes, tiene_paneles
from .entorno import Rutas

SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"
ESTRATEGIA_FOLDS_JAMAICA = "sgkf_sin_shuffle_ids_permutados_v1"

_HD = re.compile(r"solarpanels_hd_(\d+)__x0_(\d+)_y0_(\d+)_dxdy_(\d+)_\d+_\d+$")
_PV01 = re.compile(r"PV01_(\d+)_(\d+)$")
_KINGSTON = re.compile(r"kingston_(\d+)_\d+_\d+_\d+$")
_KINGSTON2 = re.compile(r"kingston2_(\d+)_(\d+)_\d+$")
_GEO = re.compile(r"\d+_unknown_(-?[\d.]+)_(-?[\d.]+)_\d+_\d+$")
# Rectángulo que contiene a Jamaica: (latitud mín., máx.), (longitud mín., máx.).
_JAMAICA = ((17.6, 18.6), (-78.5, -76.1))


def fuente(nombre: str) -> str:
    if nombre.startswith("solarpanels_hd"):
        return "solarpanels_hd"
    if nombre.startswith("PV01"):
        return "PV01"
    if nombre.startswith("kingston2"):
        return "kingston2"
    if nombre.startswith("kingston_"):
        return "kingston"
    return "geo"


def es_jamaica(nombre: str) -> bool:
    """Imágenes de Jamaica (kingston, kingston2 y geo): capturas tipo Google Maps, como las del test."""
    nombre = Path(nombre).stem
    if fuente(nombre) in {"kingston", "kingston2"}:
        return True
    if m := _GEO.match(nombre):
        (lat_min, lat_max), (lon_min, lon_max) = _JAMAICA
        return lat_min <= float(m[1]) <= lat_max and lon_min <= float(m[2]) <= lon_max
    return False


def _ubicacion(nombre: str) -> tuple[str, float, float, float, float]:
    """(zona, x, y, umbral_x, umbral_y): dos imágenes de la misma zona son vecinas si
    |Δx| <= umbral_x y |Δy| <= umbral_y. Los umbrales son conservadores: ante la duda, se agrupa."""
    if m := _HD.match(nombre):
        conjunto, x0, y0, lado = m.groups()
        # Recortes de lado x lado px: se superponen si sus orígenes están a menos de un lado.
        return f"hd{conjunto}", int(x0), int(y0), int(lado), int(lado)
    if m := _PV01.match(nombre):
        # Tiles contiguos avanzan de a ~2-3 unidades.
        return "PV01", int(m[1]), int(m[2]), 5, 5
    if m := _KINGSTON.match(nombre):
        # Mismo número = misma imagen grande; números consecutivos podrían ser contiguos.
        return "kingston", int(m[1]), 0, 1, 0
    if m := _KINGSTON2.match(nombre):
        return "kingston2", int(m[1]), int(m[2]), 3, 0
    if m := _GEO.match(nombre):
        # Latitud/longitud: 0.005° ≈ 500 m.
        return "geo", float(m[1]), float(m[2]), 0.005, 0.005
    raise ValueError(f"Nombre de archivo no reconocido: {nombre}")


def grupos_espaciales(nombres: list[str]) -> np.ndarray:
    """Asigna un id de grupo a cada imagen: componentes conexas del grafo 'son vecinas'."""
    ubic = [_ubicacion(n) for n in nombres]
    padre = list(range(len(nombres)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i, (zi, xi, yi, ux, uy) in enumerate(ubic):
        for j in range(i + 1, len(ubic)):
            zj, xj, yj, _, _ = ubic[j]
            if zi == zj and abs(xi - xj) <= ux and abs(yi - yj) <= uy:
                padre[raiz(i)] = raiz(j)
    raices = [raiz(i) for i in range(len(nombres))]
    _, ids = np.unique(raices, return_inverse=True)
    return ids


def tabla_imagenes(rutas: Rutas) -> pd.DataFrame:
    imagenes = listar_imagenes(rutas)
    nombres = [p.stem for p in imagenes]
    return pd.DataFrame({
        "archivo": [p.name for p in imagenes],
        "fuente": [fuente(n) for n in nombres],
        "grupo": grupos_espaciales(nombres),
        "con_paneles": [tiene_paneles(label_de(rutas, p)) for p in imagenes],
    })


def armar_split(rutas: Rutas, val: float = 0.2, semilla: int = 42) -> pd.DataFrame:
    """Devuelve la tabla de imágenes con una columna 'split' ('train' / 'val')."""
    from sklearn.model_selection import StratifiedGroupKFold

    tabla = tabla_imagenes(rutas)
    estrato = tabla["fuente"] + "_" + tabla["con_paneles"].map({True: "si", False: "no"})
    total = estrato.value_counts(normalize=True)

    # De los k folds posibles, se elige como val el que mejor reproduce las proporciones del total.
    kfold = StratifiedGroupKFold(n_splits=round(1 / val), shuffle=True, random_state=semilla)
    mejor = min(
        (idx_val for _, idx_val in kfold.split(tabla, estrato, groups=tabla["grupo"])),
        key=lambda idx: (estrato.iloc[idx].value_counts(normalize=True)
                         .reindex(total.index, fill_value=0) - total).abs().sum(),
    )
    tabla["split"] = "train"
    tabla.loc[tabla.index[mejor], "split"] = "val"
    return tabla


def guardar_split(tabla: pd.DataFrame) -> None:
    SPLITS_DIR.mkdir(exist_ok=True)
    for nombre in ("train", "val"):
        archivos = sorted(tabla.loc[tabla["split"] == nombre, "archivo"])
        (SPLITS_DIR / f"{nombre}.txt").write_text("\n".join(archivos) + "\n")


def cargar_split(rutas: Rutas) -> dict[str, list[Path]]:
    """{'train': [...], 'val': [...]} con las rutas a las imágenes."""
    if not (SPLITS_DIR / "train.txt").exists():
        raise FileNotFoundError("No hay split. Generarlo con: python scripts/make_split.py")
    return {
        nombre: [rutas.imagenes / a for a in (SPLITS_DIR / f"{nombre}.txt").read_text().split()]
        for nombre in ("train", "val")
    }


def folds_jamaica(rutas: Rutas, n: int = 5, semilla: int = 42) -> list[dict]:
    """Particiones adicionales para el #43; no escribe ni modifica el split oficial.

    Cada elemento contiene `fold` (1..n) y tres manifiestos: `train`, `validacion`
    y `evaluacion`. Todas las imágenes de Jamaica se evalúan una vez, por grupos
    espaciales. Train contiene el train no jamaicano y los otros n-1 folds.
    La validación que elige checkpoints es SOLO el val oficial no jamaicano:
    el fold evaluado nunca decide una época ni una parada temprana.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from . import clasificacion

    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("Se requieren al menos dos folds.")
    tabla, _ = clasificacion.preparar_tabla(rutas)
    tabla = tabla.sort_values("archivo").reset_index(drop=True).copy()
    oficiales = cargar_split(rutas)
    esperados = {p.name: conjunto for conjunto, archivos in oficiales.items() for p in archivos}
    if (sum(map(len, oficiales.values())) != len(esperados)
            or tabla.archivo.duplicated().any() or set(tabla.archivo) != set(esperados)
            or not tabla.split.eq(tabla.archivo.map(esperados)).all()):
        raise ValueError("El manifiesto debe coincidir exactamente con train/val oficiales; no admite test.")
    if not tabla.y_original.isin([0, 1]).all():
        raise ValueError("Se requieren etiquetas binarias.")
    # Recalcular juntos: una cadena de vecinos puede atravesar train y val.
    tabla["grupo"] = grupos_espaciales([Path(a).stem for a in tabla.archivo])
    jamaica = tabla.archivo.map(es_jamaica)
    seleccion = tabla.loc[jamaica].reset_index(drop=True)
    if seleccion.pendiente_revision.any():
        raise ValueError("Resolver las etiquetas pendientes de Jamaica antes de hacer CV.")
    por_clase = seleccion.groupby("y_original").grupo.nunique()
    if set(por_clase.index) != {0, 1} or (por_clase < n).any():
        raise ValueError("No hay suficientes grupos de ambas clases para los folds pedidos.")
    train_fijo = tabla.loc[~jamaica & tabla.split.eq("train")]
    validacion = tabla.loc[~jamaica & tabla.split.eq("val")]
    if train_fijo.empty or validacion.empty:
        raise ValueError("Se requiere train y val no jamaicano para separar checkpoints de evaluación.")
    # Desempatar por IDs permutados mantiene alineados grupos y recuentos de clase.
    # En sklearn 1.7.2, shuffle=True permuta los recuentos pero no sus índices de grupo.
    ids = np.unique(seleccion.grupo)
    orden = dict(zip(ids, np.random.default_rng(semilla).permutation(ids)))
    grupos_cv = seleccion.grupo.map(orden)
    particiones = StratifiedGroupKFold(n_splits=n, shuffle=False)
    folds = []
    for numero, (idx_train, idx_eval) in enumerate(particiones.split(
            seleccion, seleccion.y_original, grupos_cv), 1):
        train = pd.concat([train_fijo, seleccion.iloc[idx_train]], ignore_index=True)
        evaluacion = seleccion.iloc[idx_eval]
        conjuntos = [train, validacion, evaluacion]
        if any(set(a.grupo) & set(b.grupo)
               for i, a in enumerate(conjuntos) for b in conjuntos[i + 1:]):
            raise ValueError("Hay grupos espaciales compartidos entre entrenamiento, validación y evaluación.")
        if any(set(t.y_original) != {0, 1} for t in conjuntos):
            raise ValueError("Alguna partición no contiene ambas clases; revisar el diseño de los folds.")
        folds.append(dict(fold=numero, **{
            nombre: t.sort_values("archivo").reset_index(drop=True)
            for nombre, t in zip(("train", "validacion", "evaluacion"), conjuntos)
        }))
    return folds
