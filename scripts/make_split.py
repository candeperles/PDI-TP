"""Genera la división train/val compartida (splits/train.txt y splits/val.txt).

Se corre UNA vez y se versiona el resultado; ver pdi_tp/split.py para el criterio.
Uso: python scripts/make_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdi_tp import entorno, split  # noqa: E402

rutas = entorno.preparar(usar_drive=False)
tabla = split.armar_split(rutas)
split.guardar_split(tabla)

print(f"\nGrupos espaciales: {tabla['grupo'].nunique()} (de {len(tabla)} imágenes)")
print(tabla.groupby(["split", "fuente", "con_paneles"]).size().unstack(fill_value=0))
print(tabla["split"].value_counts())
print(f"Guardado en {split.SPLITS_DIR}")
