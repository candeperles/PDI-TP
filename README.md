# PDI-TP

Paneles solares: clasificación y detección. Consigna: [docs/Consigna_Paneles_Solares.pdf](docs/Consigna_Paneles_Solares.pdf).

## Dataset

El dataset `grupo_5` (900 pares imagen/etiqueta, ~84 MB) **no se sube al repositorio**.
Se distribuye como ZIP en la [release `dataset-v1`](https://github.com/candeperles/PDI-TP/releases/tag/dataset-v1).

Después de clonar, descargarlo con Python (Windows, Linux o Colab):

```bash
python scripts/download_data.py
```

Se descarga y extrae en `data/grupo_5/`, excluida de Git. No requiere instalar paquetes,
Bash ni unzip. Si está disponible `gh`, usa su autenticación; si no, descarga desde la
URL pública. Una carpeta existente no se sobrescribe. La descarga se extrae primero
en una carpeta temporal para que un fallo no deje un dataset parcialmente instalado.
El comando original `./scripts/download_data.sh` sigue disponible como wrapper.

Para publicar una nueva versión del dataset, subir una nueva release y actualizar
`TAG` en `scripts/download_data.py`. Conservar los datos originales.

## Entorno local de exploración

Requiere Python **3.12 o superior**. Verificado en Windows con Python 3.14.
No se instalan frameworks de entrenamiento. Ejecutar desde la raíz del repositorio.

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m notebook
```

Si el lanzador `py` no está disponible, usar la ruta del ejecutable de Python para
crear el entorno. En esta máquina es:

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe" -m venv .venv
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python -m notebook
```

Abrir `notebooks/01_exploracion.ipynb`, completar `INTEGRANTE` en el setup y ejecutar
todas las celdas. También funciona si Jupyter se inicia desde `notebooks/`.

- `requirements.txt`: versiones fijas de las dependencias de exploración, también usadas en Colab.
- `requirements-dev.txt`: agrega Jupyter y las herramientas para ejecutar notebooks localmente.
- `requirements-lock.txt`: snapshot completo, incluidas dependencias transitivas, del entorno
  verificado en Windows/Python 3.14. Para replicarlo allí, instalar este archivo en lugar
  de `requirements-dev.txt`. En otras plataformas usar `requirements-dev.txt`.

### Comprobar la instalación desde cero

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py
```

En Linux/macOS usar `.venv/bin/python`. El verificador ejecuta las 10 celdas de código
de la exploración en un kernel nuevo del mismo entorno, sin modificar el notebook
original ni el split. Guarda la copia ejecutada y las versiones en
`resultados/verificacion/`, fuera de Git. Si una celda falla, conserva el error en
esa copia y termina con error. No ejecuta la plantilla de experimentos ni registra
experimentos ficticios.

La ejecución local verificada reprodujo **900 imágenes, 2669 cajas, 187 grupos
espaciales y 738/162 imágenes en train/val**, sin grupos espaciales compartidos.
La exploración funciona en CPU; no hace falta una GPU para esta etapa.
Esta comprobación no reemplaza la validación detallada de anotaciones del siguiente paso.

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
  Clona o actualiza el repo, instala `requirements.txt`, descarga el dataset, monta Drive y verifica la GPU.
  Si Colab indica que hay que reiniciar por cambios de paquetes, reiniciar la sesión y volver a correr el setup.
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

En local, los notebooks funcionan desde la raíz o desde `notebooks/`. Los resultados van a
`resultados/`, que está en `.gitignore`.

El montaje de Drive y la GPU de Colab requieren una comprobación desde la cuenta de
cada integrante; la verificación local no los comprueba.
