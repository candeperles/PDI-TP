"""Controles y análisis de experimentos de una variable, sin TensorFlow ni GPU."""
import json

import numpy as np
import pandas as pd

from . import errores, split

METRICAS = ("accuracy", "precision", "recall", "f1", "f1_macro", "balanced_accuracy",
            "recall_sin_paneles", "roc_auc")
TAMANOS = ("<16", "16–32", "32–64", "≥64")


def validar_control(referencia, variante, variables):
    """Rechaza diferencias ajenas a las claves autorizadas, incluso claves ausentes.

    Los diccionarios deben contener los hiperparámetros efectivos, sin metadatos
    de fecha/entorno. Las variables pueden no diferir si se valida la referencia.
    """
    cambios = {c for c in referencia.keys() | variante.keys()
               if c not in referencia or c not in variante or referencia[c] != variante[c]}
    inesperados = cambios - set(variables)
    if inesperados:
        raise ValueError(f"El control cambia parámetros no autorizados: {sorted(inesperados)}")
    return cambios


def tabla_metricas(raiz, corridas, *, por_fuente=False):
    """Lee métricas ya calculadas por guardar_corrida; no registra métricas nuevas."""
    filas = []
    for corrida in corridas:
        valores = json.loads((raiz / corrida / "metricas.json").read_text(encoding="utf-8"))
        conjuntos = valores.get("por_fuente", {}) if por_fuente else {
            "val": valores, "jamaica": valores.get("jamaica")}
        for conjunto, metricas in conjuntos.items():
            if metricas:
                filas.append(dict(corrida=corrida, conjunto=conjunto, n=metricas.get("n"),
                                  **{m: metricas.get(m) for m in METRICAS}))
    return pd.DataFrame(filas)


def aciertos_por_caja(tabla, anotaciones, cajas_por_corrida, *, confianza=0.25, iou_min=0.5):
    """Una fila por referencia de val: lado sqrt(área) original y acierto por corrida.

    Se emparejan TODAS las cajas de cada imagen antes de separar tamaños. Así una
    predicción no puede acertar en dos intervalos. Las imágenes sin objetos se
    conservan en el remuestreo de recall mediante la tabla del manifiesto.
    """
    filas = []
    for imagen in tabla.loc[tabla.split == "val"].itertuples(index=False):
        reales = anotaciones.loc[(anotaciones.archivo == imagen.archivo) & anotaciones.incluida]
        aciertos = {}
        for corrida, propuestas in cajas_por_corrida.items():
            diagnostico = errores.diagnosticar_cajas(reales[errores.COORDENADAS].to_numpy(float),
                propuestas.loc[propuestas.archivo == imagen.archivo], confianza=confianza, iou_min=iou_min)
            aciertos[corrida] = {r for r, _, _ in diagnostico["coincidencias"]}
        for r, caja in enumerate(reales.itertuples(index=False)):
            lado = float(np.sqrt(caja.w * caja.h * imagen.ancho * imagen.alto))
            grupo = TAMANOS[int(np.searchsorted([16, 32, 64], lado, side="right"))]
            filas.append(dict(archivo=imagen.archivo, caja=r, lado=lado, tamano=grupo,
                              **{c: r in indices for c, indices in aciertos.items()}))
    return pd.DataFrame(filas, columns=["archivo", "caja", "lado", "tamano", *cajas_por_corrida])


def recall_por_tamano(tabla, aciertos, corridas, referencia, *, n=2000, semilla=42):
    """Recall y diferencia pareada con IC95%; bootstrap de IMÁGENES, no de cajas.

    Los bins vacíos se reportan como NaN (no como recall cero). Se incluye val y
    Jamaica; el intervalo no corrige la selección del checkpoint sobre val.
    """
    if n < 1 or referencia not in corridas or len(corridas) != len(set(corridas)):
        raise ValueError("Se requieren remuestreos positivos y una referencia entre corridas únicas.")
    filas = []
    for conjunto in ("val", "jamaica"):
        nombres = tabla.loc[tabla.split == "val", "archivo"]
        if conjunto == "jamaica":
            nombres = nombres[nombres.map(split.es_jamaica)]
        if nombres.empty:
            continue
        indices = np.random.default_rng(semilla).integers(len(nombres), size=(n, len(nombres)))
        for tamano in TAMANOS:
            grupo = aciertos.loc[aciertos.tamano == tamano]
            cantidades = grupo.groupby("archivo").size().reindex(nombres, fill_value=0).to_numpy()
            total = int(cantidades.sum())
            denominadores = cantidades[indices].sum(axis=1)
            puntuales, muestras = {}, {}
            for corrida in corridas:
                tp = grupo.groupby("archivo")[corrida].sum().reindex(nombres, fill_value=0).to_numpy(float)
                puntuales[corrida] = float(tp.sum() / total) if total else np.nan
                muestras[corrida] = np.divide(tp[indices].sum(axis=1), denominadores,
                    out=np.full(n, np.nan), where=denominadores > 0)
            for corrida in corridas:
                diferencia = muestras[corrida] - muestras[referencia]
                validos = np.isfinite(diferencia)
                inf, sup = np.percentile(diferencia[validos], [2.5, 97.5]) if validos.any() else (np.nan, np.nan)
                filas.append(dict(conjunto=conjunto, tamano=tamano, corrida=corrida, cajas=total,
                    recall=puntuales[corrida], referencia=referencia,
                    diferencia=puntuales[corrida] - puntuales[referencia], inf=inf, sup=sup,
                    remuestreos_validos=int(validos.sum())))
    return pd.DataFrame(filas)


def tiempos_por_epoca(historial):
    """Duraciones de Ultralytics a partir de `time` acumulado; no estima faltantes.

    Si el reloj retrocede (posible reanudación), el historial requiere revisión.
    """
    tiempos = historial["time"].to_numpy(float)
    duraciones = np.diff(np.r_[0.0, tiempos])
    if len(tiempos) == 0 or not np.isfinite(duraciones).all() or (duraciones <= 0).any():
        raise ValueError("Historial de tiempos vacío, inválido o reiniciado; revisar las sesiones.")
    return duraciones
