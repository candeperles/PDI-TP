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

### Ítem 2: clasificación con ResNet50

Usar [05_ResNet50_Colab_v2.ipynb](notebooks/05_ResNet50_Colab_v2.ipynb) en una sesión
nueva de Colab con GPU T4. Completar integrante y corrida; cuando se pida, pegar
la URL de una carpeta existente en Drive. Esta versión guarda primero en disco
local y copia archivos por ID, sin montar Drive ni crear carpetas allí. Comprueba
escritura antes de entrenar y copia checkpoints al terminar cada época. No usar
el mismo nombre de corrida en dos sesiones simultáneas.

Compara ImageNet congelado, ajuste fino de conv5 y pesos aleatorios. Mantiene el
split y la evaluación provisional por N06/N16, incluyendo métricas sin esos casos.
La preparación, exportaciones y 16 pruebas se verificaron localmente; la autorización
de Drive y el entrenamiento completo requieren una sesión real de Colab.

| Notebook | |
|---|---|
| `notebooks/00_plantilla.ipynb` — base para cada experimento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/00_plantilla.ipynb) |
| `notebooks/01_exploracion.ipynb` — exploración del dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/01_exploracion.ipynb) |
| `notebooks/02_validacion_datos.ipynb` — validación de archivos, cajas, duplicados y split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/02_validacion_datos.ipynb) |
| `notebooks/03_revision_anotaciones.ipynb` — casos dudosos, muestra de negativos y criterios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/03_revision_anotaciones.ipynb) |
| `notebooks/04_clasificacion_mobilenetv2.ipynb` — clasificación con MobileNetV2 (congelado, fine-tuning, sin pre-entrenamiento) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/04_clasificacion_mobilenetv2.ipynb) |
| `notebooks/05_ResNet50_Colab_v2.ipynb` — ResNet50, versión recomendada con guardado por ID | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/05_ResNet50_Colab_v2.ipynb) |
| `notebooks/04_clasificacion_ResNet50.ipynb` — clasificación con ResNet50 (congelado, fine-tuning y pesos aleatorios) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/04_clasificacion_ResNet50.ipynb) |

### Extra: modelos adicionales (fuera de la consigna)

No es parte de lo que pide la consigna: es un agregado para comparar los modelos obligatorios con otros
enfoques, usando el mismo split.

| Notebook | |
|---|---|
| `notebooks/extras/modelos_adicionales.ipynb` — **EXTRA**: DINOv2 y CLIP (features congeladas y zero-shot), fine-tuning de ConvNeXt-Tiny / EfficientNetV2-S y YOLO11s como clasificador, con comparador final | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/extras/modelos_adicionales.ipynb) |

Elige todo con validación cruzada de 5 folds dentro de train (agrupada por zona espacial) y evalúa una
sola vez en `val` al final (`EVALUAR_TEST = True`). Guarda resultados y caché en
`Mi unidad/PDI-TP/resultados/<integrante>/modelos_extra/`. En Colab instala solo `ultralytics`, no
`requirements.txt`, para no cambiar las versiones de torch que trae Colab.

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

## Subir cambios al repositorio

Se trabaja directo sobre `main`. En el repo va **el código**; los resultados de cada corrida van a Drive.

| Va a Git | No va a Git |
|---|---|
| Notebooks **sin salidas**, `pdi_tp/`, `scripts/`, `docs/`, `splits/` | `data/` (dataset), `resultados/` (JSON, pesos `.keras`, CSV de predicciones) |
| Experimentos nuevos, como una entrada más en `EXPERIMENTOS` del notebook | Copias ejecutadas de un notebook por experimento (se guardan en Drive) |

### Un experimento nuevo

No se copia el notebook: en `notebooks/04_clasificacion_mobilenetv2.ipynb` se agrega una entrada a
`EXPERIMENTOS` con un nombre único y los valores que cambian respecto de `BASE`, por ejemplo
`"mobilenetv2_imagenet_sin_aug": {"pesos": "imagenet", "modo": "congelado", "augmentation": False}`.
Después se hace push, así queda registrado qué se corrió. El nombre es el del JSON en Drive: no reutilizar
uno existente, porque se pisa el resultado.

### Desde una copia local

```bash
git pull                                   # traer lo último antes de tocar nada
# ... editar ...
git status                                 # revisar que no se cuelen data/ ni resultados/
git add notebooks/04_clasificacion_mobilenetv2.ipynb   # agregar archivos puntuales, no "git add ."
git commit -m "Agregar experimento MobileNetV2 sin augmentation"
git push
```

Si `git push` se rechaza porque alguien subió antes: `git pull --rebase` y de nuevo `git push`.
Si hay un conflicto en un notebook, avisar al equipo antes de resolverlo: un `.ipynb` mal mergeado deja
de abrir.

### Desde Colab

1. *Editar → Borrar todos los resultados*, para no subir las salidas.
2. *Archivo → Guardar una copia en GitHub*, repositorio `candeperles/PDI-TP`, rama `main`, y la **misma
   ruta** del notebook (p. ej. `notebooks/04_clasificacion_mobilenetv2.ipynb`). Escribir un mensaje de commit.
3. Ojo: guarda la versión de Colab tal cual. Si alguien cambió ese notebook en `main` después de que lo
   abriste, sus cambios se pierden. Ante la duda, volver a abrirlo desde el README antes de editar.

Los mensajes de commit van en castellano, en infinitivo y diciendo qué cambia
("Agregar ...", "Corregir ...").

## Validación del dataset

Ejecutar `notebooks/02_validacion_datos.ipynb` después de completar `INTEGRANTE`.
También se puede verificar desde un kernel nuevo en local:

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py --notebook 02_validacion_datos.ipynb
```

En Linux/macOS usar `.venv/bin/python`. El notebook exporta tablas CSV, un resumen
JSON y ejemplos visuales a `rutas.resultados / "validacion_datos"`. Con el comando
anterior quedan en `resultados/verificacion/validacion_datos/`, y la copia ejecutada
en `resultados/verificacion/02_validacion_datos.ejecutado.ipynb`.

Revisa correspondencias, lectura y dimensiones de imágenes, formato YOLO,
clases esperadas, geometría de cajas, duplicados exactos por bytes y píxeles RGB,
cobertura del split y grupos espaciales compartidos. No modifica los datos ni
regenera la división. Las cajas pequeñas son advertencias para revisión.

Resultados y limitaciones de la primera ejecución:
[docs/validacion_dataset.md](docs/validacion_dataset.md).

Pruebas del validador con datos sintéticos, incluidos archivos dañados y fuga entre splits:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Revisión de anotaciones y criterios

Los criterios para la futura preparación están en [docs/criterios_datos.md](docs/criterios_datos.md).
En `solarpanels_hd`, los IDs 0/1/2 indican confianza alta/moderada/baja, según la
documentación original cotejada. Se definió una clase `panel` para el derivado,
conservando la clase y confianza originales como metadatos. No se filtran cajas
pequeñas positivas; se excluirá únicamente la caja de ancho cero. Estas reglas
todavía no se aplicaron a una copia de datos.

La [revisión documentada](docs/revision_anotaciones.md) incluye 17 cajas minoritarias,
37 pequeñas, una nula y 18 imágenes con etiqueta vacía. Los negativos N06/N16 quedan
para una segunda revisión, sin modificar sus etiquetas ni el split.

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py --notebook 03_revision_anotaciones.ipynb
```

El notebook genera las galerías completas y coteja la selección con
`docs/revision_anotaciones.csv`, correspondiente a `dataset-v1`. Si cambian los
datos, hay que repetir la revisión y actualizar el registro. No necesita conexión
adicional una vez instalado el entorno y descargados los datos.

Para repetir el cotejo de las 17 cajas con las etiquetas de Figshare (requiere red):

```powershell
.\.venv\Scripts\python.exe scripts/check_annotation_source.py
```
