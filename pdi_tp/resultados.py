"""Forma única de guardar los resultados de un modelo. Todos los notebooks que entrenan la usan.

Cada corrida queda en una carpeta por **modelo** y **modo** (sin nombres de personas):

    resultados/<modelo>/<modo>/
        config.json            qué se corrió: modelo, modo, parámetros, fecha y commit del repo
        metricas.json          métricas sobre val (las que pide la consigna), por fuente, solo Jamaica
                               (ver docs/evaluacion_jamaica.md) + extras del notebook
        predicciones_val.csv   predicción contra el real, una fila por imagen de val:
                               archivo, fuente, y_real, probabilidad, prediccion, acierto
        <pesos>                el modelo entrenado (.keras, .pt, .joblib, ...), salvo corridas sin pesos
        ...                    opcionales: historia.csv, curvas.png, matriz_confusion.png, checkpoints

Uso típico en un notebook:

    carpeta = resultados.carpeta_corrida(rutas, "mobilenetv2", "imagenet_congelado", reiniciar=REPETIR)
    ...entrenar, guardando los pesos dentro de `carpeta`...
    resultados.guardar_corrida(rutas, carpeta, archivos=archivos_val, probabilidad=prob, params=config)

`guardar_corrida` calcula las métricas siempre de la misma manera, verifica que la carpeta cumpla el
formato y frena con un error si algo falta. `y_real` sale de las etiquetas del dataset, no del notebook.
"""
import json
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import datos, split
from .entorno import Rutas

ARCHIVOS_OBLIGATORIOS = ("config.json", "metricas.json", "predicciones_val.csv")
COLUMNAS_PREDICCIONES = ("archivo", "fuente", "y_real", "probabilidad", "prediccion", "acierto")
EXTENSIONES_PESOS = (".keras", ".h5", ".pt", ".pth", ".joblib", ".pkl", ".onnx")
METRICAS_OBLIGATORIAS = ("accuracy", "precision", "recall", "f1", "recall_sin_paneles", "f1_macro",
                         "balanced_accuracy", "matriz_confusion", "umbral", "n")
NOMBRE_VALIDO = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
ARCHIVADA = "__anterior_"  # sufijo de corridas reemplazadas (nunca se borran)


def _validar_nombre(tipo: str, valor: str) -> str:
    if not NOMBRE_VALIDO.fullmatch(valor) or ARCHIVADA in valor:
        raise ValueError(f"{tipo} inválido: {valor!r}. Usar minúsculas, números, '-', '_' o '.'.")
    return valor


def ruta_corrida(rutas: Rutas, modelo: str, modo: str) -> Path:
    """Carpeta de una corrida, sin crearla."""
    return rutas.resultados_equipo / _validar_nombre("modelo", modelo) / _validar_nombre("modo", modo)


def carpeta_corrida(rutas: Rutas, modelo: str, modo: str, reiniciar: bool = False) -> Path:
    """Crea (si hace falta) y devuelve la carpeta de la corrida.

    Con `reiniciar=True` y una carpeta con contenido, la anterior se renombra a
    `<modo>__anterior_<fecha>` (no se borra nada) y se empieza una vacía.
    Sin reiniciar, una carpeta incompleta se reutiliza (permite reanudar checkpoints).
    """
    carpeta = ruta_corrida(rutas, modelo, modo)
    if reiniciar and carpeta.is_dir() and any(carpeta.iterdir()):
        archivo = carpeta.with_name(f"{modo}{ARCHIVADA}{datetime.now():%Y%m%d_%H%M%S}")
        carpeta.rename(archivo)
        print(f"Corrida anterior conservada en {archivo}")
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def corrida_completa(carpeta: Path) -> bool:
    return Path(carpeta).is_dir() and not verificar_corrida(carpeta)


def _metricas(y: np.ndarray, p: np.ndarray, umbral: float) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    pred = (p >= umbral).astype(int)
    with warnings.catch_warnings():
        # Por fuente puede haber una sola clase (PV01 no tiene imágenes sin paneles).
        warnings.simplefilter("ignore", UserWarning)
        metricas = {
            "n": int(len(y)),
            "umbral": float(umbral),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "recall_sin_paneles": recall_score(y, pred, pos_label=0, zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else None,
            # filas = real, columnas = predicho, orden [sin paneles, con paneles]
            "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        }
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in metricas.items()}


def metricas_jamaica(predicciones: pd.DataFrame, umbral: float) -> dict | None:
    """Métricas solo sobre las imágenes de val de Jamaica (el test es de Kingston). None si no hay ninguna."""
    jamaica = predicciones["archivo"].map(split.es_jamaica).to_numpy(dtype=bool)
    if not jamaica.any():
        return None
    return _metricas(predicciones["y_real"].to_numpy()[jamaica],
                     predicciones["probabilidad"].to_numpy()[jamaica], umbral)


def _commit(rutas: Rutas):
    try:
        return subprocess.run(["git", "-C", str(rutas.repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def guardar_corrida(rutas: Rutas, carpeta: Path, archivos, probabilidad, params: dict, *,
                    umbral: float = 0.5, metricas_extra: dict | None = None, sin_pesos: bool = False) -> dict:
    """Guarda config, métricas y predicciones de val en `carpeta` y verifica el formato.

    - `archivos`: las imágenes de val (rutas o nombres), en el mismo orden que `probabilidad`.
      Tienen que ser exactamente las del split compartido.
    - `probabilidad`: p(con paneles) en [0, 1] para cada imagen.
    - Los pesos del modelo tienen que estar guardados dentro de `carpeta` antes de llamar
      (salvo `sin_pesos=True`, p. ej. un baseline o un zero-shot).
    """
    carpeta = Path(carpeta)
    modelo, modo = carpeta.parent.name, carpeta.name
    if carpeta.parent.parent != rutas.resultados_equipo:
        raise ValueError(f"{carpeta} no es una carpeta de resultados: usar resultados.carpeta_corrida(...).")

    nombres = [Path(a).name for a in archivos]
    p = np.asarray(probabilidad, dtype=float).reshape(-1)
    val = [Path(a).name for a in split.cargar_split(rutas)["val"]]
    if len(nombres) != len(set(nombres)) or set(nombres) != set(val):
        raise ValueError(f"Las predicciones tienen que ser exactamente las {len(val)} imágenes de val del split.")
    if len(p) != len(nombres) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("`probabilidad` tiene que tener un valor en [0, 1] por imagen.")

    y = np.array([int(datos.tiene_paneles(rutas.labels / f"{Path(n).stem}.txt")) for n in nombres])
    pred = (p >= umbral).astype(int)
    predicciones = pd.DataFrame({
        "archivo": nombres,
        "fuente": [split.fuente(Path(n).stem) for n in nombres],
        "y_real": y,
        "probabilidad": p,
        "prediccion": pred,
        "acierto": (pred == y).astype(int),
    })

    metricas = _metricas(y, p, umbral)
    metricas["por_fuente"] = {f: _metricas(g["y_real"].to_numpy(), g["probabilidad"].to_numpy(), umbral)
                              for f, g in predicciones.groupby("fuente")}
    metricas["jamaica"] = metricas_jamaica(predicciones, umbral)
    if metricas_extra:
        metricas["extra"] = metricas_extra

    config = {
        "modelo": modelo,
        "modo": modo,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "commit": _commit(rutas),
        "sin_pesos": bool(sin_pesos),
        "params": params,
    }

    predicciones.to_csv(carpeta / "predicciones_val.csv", index=False)
    (carpeta / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=str))
    (carpeta / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))

    problemas = verificar_corrida(carpeta)
    if problemas:
        raise RuntimeError(f"La corrida {modelo}/{modo} no cumple el formato:\n- " + "\n- ".join(problemas))
    print(f"Guardado {modelo}/{modo}: f1_macro={metricas['f1_macro']:.3f}  "
          f"recall_sin_paneles={metricas['recall_sin_paneles']:.3f}  → {carpeta}")
    return metricas


def verificar_corrida(carpeta: Path) -> list[str]:
    """Lista de problemas de formato de una carpeta de corrida (vacía = cumple)."""
    carpeta = Path(carpeta)
    problemas = [f"falta {a}" for a in ARCHIVOS_OBLIGATORIOS if not (carpeta / a).is_file()]
    if problemas:
        return problemas
    try:
        config = json.loads((carpeta / "config.json").read_text())
        metricas = json.loads((carpeta / "metricas.json").read_text())
        predicciones = pd.read_csv(carpeta / "predicciones_val.csv")
    except (ValueError, OSError) as error:
        return [f"archivo ilegible: {error}"]
    if (config.get("modelo"), config.get("modo")) != (carpeta.parent.name, carpeta.name):
        problemas.append("config.json no coincide con la carpeta <modelo>/<modo>")
    problemas += [f"metricas.json sin '{m}'" for m in METRICAS_OBLIGATORIAS if m not in metricas]
    faltan = [c for c in COLUMNAS_PREDICCIONES if c not in predicciones.columns]
    if faltan:
        problemas.append(f"predicciones_val.csv sin columnas {faltan}")
    elif predicciones.empty or predicciones["archivo"].duplicated().any():
        problemas.append("predicciones_val.csv vacío o con imágenes repetidas")
    if not config.get("sin_pesos") and not any(f.suffix in EXTENSIONES_PESOS for f in carpeta.iterdir()):
        problemas.append(f"faltan los pesos del modelo ({', '.join(EXTENSIONES_PESOS)})")
    return problemas


def _corridas(rutas: Rutas):
    for metricas in sorted(rutas.resultados_equipo.glob("*/*/metricas.json")):
        carpeta = metricas.parent
        if ARCHIVADA not in carpeta.name:
            yield carpeta


def cargar_corridas(rutas: Rutas) -> pd.DataFrame:
    """Todas las corridas del equipo, una fila por modelo/modo, con las métricas en columnas.

    Las columnas `jamaica.<métrica>` se recalculan desde predicciones_val.csv: así también las tienen las
    corridas guardadas antes de que existieran.
    """
    filas = []
    for carpeta in _corridas(rutas):
        config = json.loads((carpeta / "config.json").read_text())
        metricas = json.loads((carpeta / "metricas.json").read_text())
        fila = {"modelo": config["modelo"], "modo": config["modo"], "fecha": config.get("fecha"),
                "completa": not verificar_corrida(carpeta)}
        fila.update({k: v for k, v in metricas.items() if k not in ("por_fuente", "jamaica", "extra")})
        if (carpeta / "predicciones_val.csv").is_file():
            predicciones = pd.read_csv(carpeta / "predicciones_val.csv")
            jamaica = metricas_jamaica(predicciones, metricas.get("umbral", 0.5)) or {}
            fila.update({f"jamaica.{k}": v for k, v in jamaica.items() if k != "umbral"})
        fila.update({f"params.{k}": v for k, v in config.get("params", {}).items()})
        filas.append(fila)
    if not filas:
        return pd.DataFrame(columns=["modelo", "modo", "fecha", "completa"])
    return pd.DataFrame(filas).sort_values(["modelo", "modo"]).reset_index(drop=True)


def cargar_predicciones(rutas: Rutas) -> pd.DataFrame:
    """Predicciones de val de todas las corridas, en formato largo (una fila por imagen y corrida)."""
    tablas = [pd.read_csv(c / "predicciones_val.csv").assign(modelo=c.parent.name, modo=c.name)
              for c in _corridas(rutas)]
    return pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame(columns=COLUMNAS_PREDICCIONES)
