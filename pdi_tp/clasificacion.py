"""Manifiesto binario y evaluación compartida, sin dependencia de TensorFlow."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import validacion


def preparar_tabla(rutas):
    """Aplica criterios v1 en memoria; conserva originales, split y metadatos.

    Solo permite el error Z01 documentado. Cualquier error nuevo detiene el paso.
    Los negativos pendientes tienen referencia original, pero no etiqueta confirmada.
    """
    informe = validacion.validar(rutas)
    errores = informe.incidencias.query("nivel == 'error'")
    z01 = "solarpanels_hd_1__x0_13014_y0_24247_dxdy_832_1_2.png"
    permitido = ((errores.archivo == z01) & (errores.linea == 1)
                 & (errores.control == "dimension_caja_no_positiva"))
    caja = informe.cajas.query("archivo == @z01 and linea == 1")
    if not errores.empty and (not permitido.all() or len(caja) != 1
                              or caja.iloc[0].w != 0 or caja.iloc[0].h <= 0):
        raise ValueError("Errores nuevos: revisar antes de entrenar.\n" + errores.to_string(index=False))
    tabla = informe.imagenes.copy()
    tabla["y_original"] = (tabla.n_cajas_validas > 0).astype(int)
    registro = pd.read_csv(rutas.repo / "docs" / "revision_anotaciones.csv")
    pendientes = set(registro.loc[registro.estado == "pendiente_segunda_revision", "archivo"])
    tabla["pendiente_revision"] = tabla.archivo.isin(pendientes)
    tabla["y"] = tabla.y_original.astype("Int64").mask(tabla.pendiente_revision)
    tabla["ruta"] = tabla.archivo.map(lambda n: str(rutas.imagenes / n))
    if tabla.query("split == 'train'").y.isna().any():
        raise ValueError("Hay etiquetas pendientes en train; resolverlas antes de entrenar.")
    cajas = informe.cajas.copy()
    cajas["clase_original"] = cajas.clase.astype(int)
    cajas["clase_derivada"] = 0
    cajas["incluida"] = cajas.valida
    fuentes = tabla.set_index("archivo").fuente
    cajas["fuente"] = cajas.archivo.map(fuentes)
    cajas["confianza_original"] = [
        {0: "alta", 1: "moderada", 2: "baja"}[c] if f == "solarpanels_hd" else None
        for f, c in zip(cajas.fuente, cajas.clase_original)
    ]
    return tabla, cajas


def huella_tabla(tabla):
    columnas = ["archivo", "split", "sha256", "sha256_etiqueta", "y_original", "pendiente_revision"]
    contenido = tabla.sort_values("archivo")[columnas].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(contenido).hexdigest()


def pesos_clase(y):
    y = np.asarray(y, dtype=int)
    valores, cantidades = np.unique(y, return_counts=True)
    if set(valores) != {0, 1}:
        raise ValueError("Train debe contener ambas clases.")
    return {int(c): float(len(y) / (2 * n)) for c, n in zip(valores, cantidades)}


def evaluar(y, probabilidades, umbral=0.5):
    y, p = np.asarray(y), np.asarray(probabilidades, dtype=float).reshape(-1)
    if y.ndim != 1 or len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("Etiquetas binarias no vacías y alineadas con las probabilidades requeridas.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 <= umbral <= 1:
        raise ValueError("Probabilidades y umbral deben estar en [0, 1].")
    pred = (p >= umbral).astype(int)
    reporte = classification_report(y, pred, labels=[0, 1],
                                    target_names=["sin_paneles", "con_paneles"],
                                    output_dict=True, zero_division=0)
    positivo = reporte["con_paneles"]
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                precision=positivo["precision"], recall=positivo["recall"], f1=positivo["f1-score"],
                f1_macro=reporte["macro avg"]["f1-score"],
                matriz_confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                reporte=reporte, umbral=umbral)


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
