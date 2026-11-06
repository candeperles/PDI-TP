"""Comparación final: clasificadores (CNN) y detectores usados como clasificadores, sobre el mismo val.

Todo sale de los resultados guardados con el formato común (pdi_tp/resultados.py); acá no se entrena nada.

- Solo se comparan corridas **vigentes**: las CNN con la receta común actual (`params.receta`) y los
  detectores con imgsz=640 (con su propia receta, la misma para YOLOv8n y YOLO11n: ver 08 y 09).
- Cada corrida se mide con su propio umbral (0.5 las CNN, 0.25 los YOLO). El puntaje de un YOLO es la
  máxima confianza de sus cajas, no una probabilidad calibrada: el AUC no depende del umbral y es la
  comparación más justa entre familias.
- Val tiene 162 imágenes (20 sin paneles) y Jamaica 23: una imagen mueve mucho las métricas. Por eso cada
  métrica va con un intervalo bootstrap (se remuestrean las imágenes) y las diferencias entre dos corridas
  se miden **pareadas** (las mismas imágenes remuestreadas para las dos). Una diferencia es clara solo si su
  intervalo no incluye el 0.
- Val también eligió los checkpoints: los números son optimistas, y eso no lo corrige el bootstrap.

Uso en un notebook:

    corridas = comparacion.corridas_vigentes(resultados.cargar_corridas(rutas))
    predicciones = resultados.cargar_predicciones(rutas)
    elegidas = corridas.loc[corridas.vigente, "corrida"].tolist()
    comparacion.intervalos(predicciones, elegidas)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import split

RECETA_VIGENTE = "comun_v2"
METRICAS = ("f1_macro", "balanced_accuracy", "recall_sin_paneles", "roc_auc")
COLUMNAS_DETECCION = ("map50", "map50_95", "precision", "recall")


def familia(modelo: str) -> str:
    if modelo.startswith("yolo"):
        return "detector"
    if modelo == "baseline":
        return "referencia"
    return "clasificador"


def corridas_vigentes(corridas: pd.DataFrame, receta: str = RECETA_VIGENTE) -> pd.DataFrame:
    """Una fila por corrida completa, con su familia y si entra en la comparación (`vigente`)."""
    tabla = corridas[corridas["completa"]].copy()
    tabla.insert(0, "corrida", tabla["modelo"] + "/" + tabla["modo"])
    tabla.insert(1, "familia", tabla["modelo"].map(familia))
    # La receta común es de las CNN: los detectores guardan otra cosa en `params.receta` y el baseline no tiene.
    recetas = tabla["params.receta"] if "params.receta" in tabla else pd.Series(None, index=tabla.index)
    tabla.insert(2, "receta", recetas.where(tabla["familia"] == "clasificador"))
    detectores_640 = recetas.map(lambda r: isinstance(r, dict) and r.get("imgsz") == 640)
    tabla.insert(3, "vigente", (tabla["familia"] == "referencia")
                 | ((tabla["familia"] == "clasificador") & (tabla["receta"] == receta))
                 | ((tabla["familia"] == "detector") & detectores_640))
    columnas = ["corrida", "familia", "receta", "vigente", "modelo", "modo", "fecha", "umbral", *METRICAS,
                *[f"jamaica.{m}" for m in METRICAS if f"jamaica.{m}" in tabla]]
    return tabla[[c for c in columnas if c in tabla]].reset_index(drop=True)


def _matrices(predicciones: pd.DataFrame, corridas: list[str], solo_jamaica: bool):
    """y (n,), probabilidad y predicción (corridas, n), alineadas por archivo."""
    tabla = predicciones.assign(corrida=predicciones["modelo"] + "/" + predicciones["modo"])
    tabla = tabla[tabla["corrida"].isin(corridas)]
    if solo_jamaica:
        tabla = tabla[tabla["archivo"].map(split.es_jamaica)]
    faltan = set(corridas) - set(tabla["corrida"])
    if faltan:
        raise ValueError(f"No hay predicciones para: {sorted(faltan)}")
    prob = tabla.pivot(index="archivo", columns="corrida", values="probabilidad")[corridas]
    pred = tabla.pivot(index="archivo", columns="corrida", values="prediccion")[corridas]
    y = tabla.pivot(index="archivo", columns="corrida", values="y_real")[corridas]
    if prob.isna().any().any():
        raise ValueError("Las corridas no tienen predicciones para las mismas imágenes de val.")
    if (y.nunique(axis=1) > 1).any():
        raise ValueError("Las corridas no coinciden en la etiqueta real de alguna imagen.")
    return y.iloc[:, 0].to_numpy(int), prob.to_numpy(float).T, pred.to_numpy(int).T


def _calcular(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    """Métricas para cada remuestreo. y: (B, n); prob y pred: (C, B, n). Devuelve arrays (C, B)."""
    tp = (pred * y).sum(-1)
    tn = ((1 - pred) * (1 - y)).sum(-1)
    fp = (pred * (1 - y)).sum(-1)
    fn = ((1 - pred) * y).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall_con = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        recall_sin = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        f1_con = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_sin = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
        n_con = y.sum(-1)
        n_sin = y.shape[-1] - n_con
        rangos = rankdata(prob, axis=-1)
        auc = ((rangos * y).sum(-1) - n_con * (n_con + 1) / 2) / (n_con * n_sin)
    return {
        "accuracy": (tp + tn) / y.shape[-1],
        "precision": np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0),
        "recall": recall_con,
        "f1": f1_con,
        "f1_macro": (f1_con + f1_sin) / 2,
        "balanced_accuracy": np.nanmean(np.stack([recall_con, recall_sin]), axis=0),
        "recall_sin_paneles": recall_sin,
        "roc_auc": np.where((n_con > 0) & (n_sin > 0), auc, np.nan),
    }


def _remuestreos(y, prob, pred, n: int, semilla: int):
    """Índices bootstrap de imágenes (los mismos para todas las corridas: así las diferencias son pareadas)."""
    indices = np.random.default_rng(semilla).integers(0, len(y), size=(n, len(y)))
    return y[indices], prob[:, indices], pred[:, indices]


def intervalos(predicciones: pd.DataFrame, corridas: list[str], *, solo_jamaica: bool = False,
               n: int = 2000, semilla: int = 42, nivel: float = 0.95) -> pd.DataFrame:
    """Cada métrica de cada corrida con su intervalo bootstrap. Una fila por corrida."""
    y, prob, pred = _matrices(predicciones, corridas, solo_jamaica)
    puntual = _calcular(y[None], prob[:, None], pred[:, None])
    remuestreadas = _calcular(*_remuestreos(y, prob, pred, n, semilla))
    colas = [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100]
    filas = []
    for i, corrida in enumerate(corridas):
        fila = {"corrida": corrida, "n": len(y), "n_sin_paneles": int((y == 0).sum())}
        for metrica in METRICAS:
            inf, sup = np.nanpercentile(remuestreadas[metrica][i], colas)
            fila.update({metrica: puntual[metrica][i, 0], f"{metrica}_inf": inf, f"{metrica}_sup": sup})
        filas.append(fila)
    return pd.DataFrame(filas)


def diferencias(predicciones: pd.DataFrame, corridas: list[str], referencia: str, *,
                metrica: str = "f1_macro", solo_jamaica: bool = False, n: int = 2000, semilla: int = 42,
                nivel: float = 0.95) -> pd.DataFrame:
    """Diferencia pareada `otra − referencia` en una métrica, con su intervalo bootstrap.

    `clara` es True cuando el intervalo no incluye el 0: con estos tamaños, solo esas diferencias se pueden
    afirmar. Las demás pueden ser ruido.
    """
    if metrica not in (*METRICAS, "accuracy", "precision", "recall", "f1"):
        raise ValueError(f"Métrica desconocida: {metrica}")
    if referencia not in corridas:
        corridas = [referencia, *corridas]
    y, prob, pred = _matrices(predicciones, corridas, solo_jamaica)
    puntual = _calcular(y[None], prob[:, None], pred[:, None])[metrica][:, 0]
    remuestreada = _calcular(*_remuestreos(y, prob, pred, n, semilla))[metrica]
    i_ref = corridas.index(referencia)
    colas = [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100]
    filas = []
    for i, corrida in enumerate(corridas):
        if i == i_ref:
            continue
        inf, sup = np.nanpercentile(remuestreada[i] - remuestreada[i_ref], colas)
        filas.append({"corrida": corrida, "referencia": referencia, "metrica": metrica,
                      "diferencia": puntual[i] - puntual[i_ref], "inf": inf, "sup": sup,
                      "clara": bool(inf > 0 or sup < 0)})
    return pd.DataFrame(filas).sort_values("diferencia", ascending=False).reset_index(drop=True)


def tabla_deteccion(resultados_equipo: Path, corridas: list[str]) -> pd.DataFrame:
    """mAP y precisión/recall de cajas de los detectores, sobre val y sobre Jamaica (de metricas.json)."""
    filas = []
    for corrida in corridas:
        metricas = json.loads((Path(resultados_equipo) / corrida / "metricas.json").read_text())
        deteccion = (metricas.get("extra") or {}).get("deteccion")
        if not deteccion:
            continue
        for conjunto, valores in deteccion.items():
            filas.append({"corrida": corrida, "conjunto": conjunto,
                          **{c: valores.get(c) for c in COLUMNAS_DETECCION}})
    return pd.DataFrame(filas, columns=["corrida", "conjunto", *COLUMNAS_DETECCION])
