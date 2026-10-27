# Instrucciones técnicas

Detalles de instalación, descarga del dataset, trabajo en local y verificaciones. Para el uso normal
(correr notebooks en Colab y subir cambios) alcanza con el [README](README.md).

## Descarga del dataset

El dataset `grupo_5` (900 pares imagen/etiqueta, ~84 MB) no se sube al repositorio: está como ZIP en la
[release `dataset-v1`](https://github.com/candeperles/PDI-TP/releases/tag/dataset-v1). Los notebooks lo
descargan solos en el setup. A mano, desde la raíz del repo:

```bash
python scripts/download_data.py
```

- Queda en `data/grupo_5/` (`images/` y `labels/`), excluido de Git.
- Funciona en Windows, Linux, macOS y Colab, sin Bash ni `unzip`. Si está `gh`, usa su autenticación; si
  no, descarga desde la URL pública.
- No sobrescribe una carpeta existente. Para volver a bajarlo, borrar `data/grupo_5/` primero.
- Extrae primero en una carpeta temporal, así un fallo no deja el dataset a medias.
- `./scripts/download_data.sh` sigue funcionando (llama al script de Python).

**Publicar una nueva versión del dataset:** subir una nueva release (p. ej. `dataset-v2`) y actualizar
`TAG` en `scripts/download_data.py`. No modificar la `dataset-v1` original.

## Entorno local

Sirve para explorar datos y probar código. Los entrenamientos se corren en Colab con GPU.
Requiere Python **3.12 o superior**. Todos los comandos se corren desde la raíz del repositorio.

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python -m notebook
```

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m notebook
```

Si el lanzador `py` no está disponible, crear el entorno con la ruta del ejecutable de Python, por ejemplo
`& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe" -m venv .venv`.

En Jupyter, abrir el notebook y ejecutar todo. Funciona tanto si Jupyter se
inicia desde la raíz como desde `notebooks/`. En local, los resultados van a `resultados_locales/` dentro del repo
(ignorado por Git), con la misma estructura que en Drive (`resultados/<modelo>/<modo>/`, `analisis/`, `cache/`).

### Dependencias

| Archivo | Para qué |
|---|---|
| `requirements.txt` | Exploración y validación de datos, con versiones fijas. También lo instalan en Colab los notebooks de datos. |
| `requirements-dev.txt` | Lo anterior + Jupyter y herramientas para ejecutar notebooks en local. |
| `requirements-lock.txt` | Snapshot completo (con dependencias transitivas) del entorno verificado en Windows / Python 3.14. En otras plataformas usar `requirements-dev.txt`. |
| `requirements-training.txt` | Entrenamiento con TensorFlow. Lo instalan los notebooks de clasificación en Colab. **No mezclar con `requirements.txt`**: sus versiones de NumPy son incompatibles con TensorFlow. |

El notebook extra (`notebooks/extras/modelos_adicionales.ipynb`) usa PyTorch: en Colab solo instala
`ultralytics`; en local hace falta además `pip install torch torchvision transformers ultralytics`.

## Verificaciones

En Windows reemplazar `.venv/bin/python` por `.\.venv\Scripts\python.exe`.

```bash
.venv/bin/python scripts/verificar_notebooks.py       # normas de los notebooks (ver README)
.venv/bin/python -m pip check                         # dependencias consistentes
.venv/bin/python scripts/check_environment.py         # ejecuta la exploración en un kernel limpio
.venv/bin/python -m unittest discover -s tests -v     # pruebas de pdi_tp, del verificador y de ResNet50
```

El verificador también corre solo antes de cada commit (`git config core.hooksPath .githooks`, una vez por
clon) y en GitHub en cada push.

`check_environment.py` ejecuta el notebook en un kernel nuevo sin modificar el original ni el split. Guarda
la copia ejecutada y las versiones en `resultados/verificacion/` y termina con error si falla una celda.
Debe reproducir **900 imágenes, 2669 cajas, 187 grupos espaciales y 738/162 imágenes en train/val**, sin
grupos espaciales compartidos. Con `--notebook` se verifica otro:

```bash
.venv/bin/python scripts/check_environment.py --notebook 02_validacion_datos.ipynb
.venv/bin/python scripts/check_environment.py --notebook 03_revision_anotaciones.ipynb
```

- **Validación del dataset** (`02_validacion_datos`): correspondencia imagen/etiqueta, lectura y
  dimensiones, formato YOLO, clases, geometría de cajas, duplicados exactos y cobertura del split. No
  modifica los datos. Resultados: [docs/validacion_dataset.md](docs/validacion_dataset.md).
- **Revisión de anotaciones** (`03_revision_anotaciones`): galerías de casos dudosos, cotejadas con
  `docs/revision_anotaciones.csv` (válido para `dataset-v1`; si cambian los datos, repetir la revisión).
  Resultados: [docs/revision_anotaciones.md](docs/revision_anotaciones.md).
- **Cotejo de clases con la fuente original** (requiere red):
  `.venv/bin/python scripts/check_annotation_source.py`.

## Regenerar el split

Solo si el equipo lo decide (cambia la base de comparación de todos los experimentos):

```bash
.venv/bin/python scripts/make_split.py
```

Escribe `splits/train.txt` y `splits/val.txt`. El criterio está documentado en `pdi_tp/split.py`.

## Problemas comunes

| Síntoma | Causa probable y solución |
|---|---|
| `No existe /content/drive/MyDrive/PDI-TP` | Falta el acceso directo a la carpeta compartida, o se montó Drive con otra cuenta. Ver *Puesta en marcha* en el README. |
| "Desaparecieron" los resultados del resto | Colab está usando otra carpeta `PDI-TP` (duplicada en tu Drive) u otra cuenta. Correr `!ls /content/drive/MyDrive` y revisar. En Drive, el panel *Actividad* de la carpeta muestra quién movió o borró algo. |
| `preparar() ya no recibe el nombre del integrante` | El notebook es de una versión vieja: borrar la línea `INTEGRANTE = ...` y llamar `entorno.preparar()`. |
| `La corrida ... no cumple el formato` | `guardar_corrida` encontró algo incompleto (casi siempre: no se guardaron los pesos dentro de la carpeta antes de llamarla). El mensaje dice qué falta. |
| `Las predicciones tienen que ser exactamente las 162 imágenes de val` | Se predijo sobre otro conjunto o faltan imágenes: predecir sobre `split.cargar_split(rutas)["val"]`. |
| El commit se cancela con una lista de ✗ | El verificador encontró notebooks que no cumplen las normas: corregir lo que indica. |
| Colab pide reiniciar la sesión | Normal tras instalar paquetes: reiniciar y correr de nuevo desde el setup. |
| Se cortó la sesión a mitad de un entrenamiento | Lo guardado en Drive se conserva. Varios notebooks retoman o saltean lo ya hecho al volver a correrlos. |
| `ModuleNotFoundError: torch` (local) | El `.venv` de exploración no trae PyTorch: ver *Dependencias*. |
| `git push` a `main` rechazado (*protected branch* / *required status check*) | `main` está protegida: subir a una rama y abrir un PR (ver README, *Subir notebooks y cambios*). |
| PR en rojo | El chequeo `verificar` falló: abrir el detalle en el PR, corregir en la misma rama y volver a subir. |
| El push de la rama se rechaza porque `main` cambió | `git pull --rebase origin main` y `git push --force-with-lease` (solo en tu rama). |
