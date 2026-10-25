# PDI-TP

Paneles solares: clasificación y detección. Consigna: [docs/Consigna_Paneles_Solares.pdf](docs/Consigna_Paneles_Solares.pdf).

## Dataset

The `grupo_5` dataset (900 image/label pairs, ~84MB) is **not committed to this
repo** — it's attached as a zip to the [`dataset-v1` release](https://github.com/candeperles/PDI-TP/releases/tag/dataset-v1)
instead, so clones/pulls stay fast and git history doesn't bloat.

After cloning, fetch it with:

```bash
./scripts/download_data.sh
```

This downloads and extracts the dataset into `data/grupo_5/` (gitignored —
each teammate keeps their own local copy). Requires the [GitHub CLI](https://cli.github.com/)
(`gh`) for authenticated access, or falls back to a plain `curl` download.

If you add to or change the dataset, re-zip `data/grupo_5/` and upload a new
release (e.g. `dataset-v2`), then update the `TAG` in `scripts/download_data.sh`.

## Trabajar en Colab

| Notebook | |
|---|---|
| `notebooks/00_plantilla.ipynb` — base para cada experimento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/00_plantilla.ipynb) |
| `notebooks/01_exploracion.ipynb` — exploración del dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/01_exploracion.ipynb) |

### Una sola vez

1. **Carpeta compartida en Drive.** Una persona crea la carpeta `PDI-TP` en su Drive y la comparte
   con el resto como Editor. Cada integrante entra a *Compartidos conmigo* → clic derecho en `PDI-TP` →
   *Organizar → Agregar acceso directo* → *Mi unidad*. Tiene que quedar en `Mi unidad/PDI-TP`.
2. **GPU.** En cada notebook: *Entorno de ejecución → Cambiar tipo de entorno → T4*.

### Cada vez

- Abrir el notebook con el botón de arriba y correr la celda de **setup** con tu nombre en `INTEGRANTE`.
  Clona o actualiza el repo, descarga el dataset, monta Drive y verifica la GPU.
- Guardar tu copia antes de editar (*Archivo → Guardar una copia en Drive* o *en GitHub*). Así nadie
  pisa el notebook de otro.
- Registrar cada experimento con `resultados.guardar_experimento(rutas, nombre, params, metricas)`.
  Queda en `Mi unidad/PDI-TP/resultados/<integrante>/<nombre>.json`.
- Ver los resultados de todo el equipo con `resultados.cargar_experimentos(rutas)`.
- Guardar los pesos de los modelos en `rutas.resultados` (Drive), no en `/content`, que se borra al
  cortarse la sesión.

### División train/val

Está versionada en `splits/train.txt` y `splits/val.txt` (738 / 162 imágenes); se carga con
`split.cargar_split(rutas)`. Es 80/20 **por grupos espaciales** (recortes vecinos de una misma imagen
grande quedan del mismo lado, para no tener fuga) y estratificada por fuente + con/sin paneles.
Todos usan esta misma división: no regenerarla salvo que el equipo lo decida
(`python scripts/make_split.py`).

El código compartido (setup, lectura del dataset, split, registro de resultados) está en `pdi_tp/`. Si lo
cambiás, hacé push para que el resto lo reciba en el próximo setup.

En local, los notebooks también funcionan (corriendo desde `notebooks/`). Los resultados van a
`resultados/`, que está en `.gitignore`.