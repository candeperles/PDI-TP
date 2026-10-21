"""Registro de experimentos en la carpeta de resultados (Drive en Colab), para compararlos después.

Cada experimento es un JSON en resultados/<integrante>/<nombre>.json con sus parámetros y métricas.
"""
import json
from datetime import datetime
from pathlib import Path

from .entorno import Rutas


def guardar_experimento(rutas: Rutas, nombre: str, params: dict, metricas: dict) -> Path:
    registro = {
        "nombre": nombre,
        "integrante": rutas.integrante,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "metricas": metricas,
    }
    destino = rutas.resultados / f"{nombre}.json"
    destino.write_text(json.dumps(registro, indent=2, ensure_ascii=False, default=str))
    print(f"Guardado: {destino}")
    return destino


def cargar_experimentos(rutas: Rutas):
    """Todos los experimentos del equipo como DataFrame (una fila por experimento)."""
    import pandas as pd

    registros = [json.loads(p.read_text()) for p in sorted(rutas.resultados_equipo.glob("*/*.json"))]
    return pd.json_normalize(registros)
