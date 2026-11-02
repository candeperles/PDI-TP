# PDI-TP · Paneles solares: clasificación y detección

Trabajo práctico de **Procesamiento de Imágenes** (grupo 5). Consigna:
[docs/Consigna_Paneles_Solares.pdf](docs/Consigna_Paneles_Solares.pdf) · resumen de requisitos:
[docs/alcance.md](docs/alcance.md) · instalación, trabajo en local y verificaciones:
[INSTRUCCIONES.md](INSTRUCCIONES.md) · normas para agentes (Claude, Codex, etc.): [AGENTS.md](AGENTS.md).

## De qué se trata

Encontrar paneles solares en imágenes aéreas sirve, por ejemplo, para que una distribuidora eléctrica sepa
dónde hay generación propia. El mismo dataset se resuelve de dos maneras y se comparan:

1. **Clasificación**: ¿la imagen tiene paneles? (sí / no). Redes de Keras (MobileNetV2, EfficientNetB0,
   ResNet50...), comparando pesos de ImageNet contra aleatorios y extractor congelado contra fine-tuning.
2. **Detección**: ¿dónde está cada panel? YOLOv8n y YOLO11n (Faster R-CNN opcional), entrenando toda la red
   contra congelar una parte.
3. **Comparación**: el detector también se usa como clasificador (si detecta al menos un panel, la imagen
   "tiene paneles") y se compara con los clasificadores sobre las mismas imágenes de validación.

Métricas: matriz de confusión, accuracy, precision, recall y F1 para clasificación; mAP50, mAP50-95 y curva
precisión-recall para detección. Se entrega un informe con comparaciones y conclusiones.

## Receta común de clasificación

Las tres redes se entrenan con la misma receta y cambian solo la arquitectura, para que la comparación valga
(detalle y discusión en el [issue #15](https://github.com/candeperles/PDI-TP/issues/15)). Cada corrida guarda
`receta: "comun_v2"` en su `config.json`.

| | `imagenet_congelado` | `imagenet_finetuning` | `aleatorio` |
|---|---|---|---|
| Qué se entrena | solo la cabeza | todo el extractor menos BatchNorm, desde la mejor cabeza | toda la red desde cero |
| LR (Adam) | 1e-3 | 1e-5 | 1e-3, a la mitad tras 3 épocas sin mejorar |
| Épocas máximas | 30 | 20 adicionales | 60 |
| Early stopping (`val_loss`) | paciencia 10 | paciencia 10 | paciencia 10, cuenta desde la época 20 |
| Momentum de BatchNorm | el de cada red | el de cada red | 0.9 |

Común a todos: entrada 256×256 (el tamaño original), batch 32, dropout 0.3, umbral 0.5, semilla 42,
`aumentacion.google_maps` y pesos de muestra por fuente × clase. El preprocesado es el de cada red.

**Val elige el checkpoint y también se usa para reportar**, así que las métricas son optimistas: decirlo en el
informe. Para volver a correr un notebook sin mezclar resultados, ver
[Dónde quedan los resultados](#dónde-quedan-los-resultados).

## Estado

| Etapa | Notebook | Estado |
|---|---|---|
| Exploración del dataset | `01_exploracion` | Hecho |
| Validación del dataset y revisión de anotaciones | `02_validacion_datos`, `03_revision_anotaciones` | Hecho; criterios en [docs/criterios_datos.md](docs/criterios_datos.md) |
| Clasificación: MobileNetV2 | `04_clasificacion_mobilenetv2` | Receta común v2; **falta entrenar en Colab con `REPETIR = True`** (las corridas del 2/11 usaron la primera versión de la receta) |
| Clasificación: ResNet50 | `05_clasificacion_resnet50` | Receta común v2; **falta entrenar en Colab con `REPETIR = True`** |
| Clasificación: EfficientNetB0 | `06_clasificacion_efficientnet` | Receta común v2; **falta entrenar en Colab con `REPETIR = True`** |
| Receta común de clasificación | — | Hecha en los tres notebooks (ver [Receta común](#receta-común-de-clasificación)); falta entrenar y comparar |
| Marca de agua de Google Maps (issue #7) | `11_marca_de_agua` | **Ubicada:** aparece en 49 de las 172 imágenes de Jamaica y en ninguna de las otras fuentes; ocupa desde la fila 222 para abajo (barra de atribución y logo). En todo el dataset, "marca → sin paneles" acierta el 63 % contra el 13 % sin marca, porque solo aparece en Jamaica. **Falta:** elegir entre tapar esa franja en todas las imágenes o pegar marcas falsas como augmentation, y medir el efecto en Jamaica |
| Detección: YOLOv8n | `08_deteccion_yolov8n` | Hecho: `coco_completo` y `coco_backbone_congelado`, en Drive (27/10) |
| Detección: YOLO11n | `09_deteccion_yolo11n` | Notebook listo, misma receta que YOLOv8n salvo el tamaño de entrada (640 en vez de 256); **falta entrenar en Colab** |
| Detector como clasificador y comparación final | `10_comparacion_final` | Notebook listo: compara todas las corridas vigentes con intervalos bootstrap sobre val y Jamaica. Falta correrlo cuando estén los tres clasificadores y los dos YOLO |
| Informe | — | Pendiente |
| Página interactiva para el cierre | — | Idea: un módulo que muestre cómo se generan las cajas (candidatas → filtrado → finales) y otro con la red calculando en vivo (mapas de activación capa por capa). Se arma al final, con los resultados reales |

## Datos

- **Dataset `grupo_5`**: 900 imágenes aéreas PNG de 256×256 con etiquetas YOLO, de 4 fuentes
  (`solarpanels_hd`, `PV01`, `kingston`, `geo`). ~85 % tiene paneles. No está en el repo: los notebooks
  lo descargan solos (detalles en [INSTRUCCIONES.md](INSTRUCCIONES.md#descarga-del-dataset)).
- **Split train/val compartido**: `splits/train.txt` y `splits/val.txt` (738 / 162). 80/20 **por grupos
  espaciales** (recortes vecinos de una misma imagen grande quedan del mismo lado, para que no haya fuga) y
  estratificado por fuente y con/sin paneles. Se carga con `split.cargar_split(rutas)`. **Todos usan el
  mismo**; no regenerarlo sin acuerdo del equipo (`scripts/make_split.py`).
- **Clases**: en `solarpanels_hd` los IDs 0/1/2 son la *confianza* de la anotación (alta/moderada/baja),
  no tipos de panel. Se trabaja con una sola clase `panel`. Ver [docs/criterios_datos.md](docs/criterios_datos.md).

## Puesta en marcha (una sola vez por persona)

1. **Carpeta compartida en Drive.** Una sola persona crea la carpeta `PDI-TP` y la comparte con el resto
   como Editor. Cada persona entra a *Compartidos conmigo* → clic derecho en `PDI-TP` → *Organizar →
   Agregar acceso directo* → *Mi unidad*. Tiene que quedar exactamente en `Mi unidad/PDI-TP`.
   - **No crear otra carpeta llamada `PDI-TP`** en tu Drive: Drive permite nombres repetidos y Colab puede
     agarrar la equivocada (vas a ver la carpeta vacía y "desaparecen" los resultados del resto).
   - Montar Drive en Colab **con la misma cuenta** a la que se compartió la carpeta.
2. **Si trabajás con git en local (con o sin IA)**, una vez por clon: activar el chequeo previo a cada commit
   e instalar la CLI de GitHub para abrir PRs ([cli.github.com](https://cli.github.com/)):

   ```bash
   git config core.hooksPath .githooks
   ```

   ```bash
   gh auth login
   ```

## Correr un notebook en Colab

1. Abrirlo con su botón **Open in Colab** (tabla de abajo). Siempre abre la última versión de `main`.
2. *Entorno de ejecución → Cambiar tipo de entorno → T4* (GPU).
3. *Entorno de ejecución → Ejecutar todas*. La celda de setup clona o actualiza el repo, instala
   dependencias, descarga el dataset, monta Drive (solo los notebooks que entrenan) y avisa si no hay GPU.
   Si Colab pide reiniciar por
   cambios de paquetes, reiniciar y volver a correr desde el principio.

No hay que poner ningún nombre: los resultados se guardan por modelo y modo, no por persona.

### Notebooks

**Datos**

| Notebook | |
|---|---|
| `01_exploracion` — EDA: balance de clases, fuentes, cajas, split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/01_exploracion.ipynb) |
| `02_validacion_datos` — validación de archivos, cajas, duplicados y split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/02_validacion_datos.ipynb) |
| `03_revision_anotaciones` — casos dudosos, muestra de negativos y criterios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/03_revision_anotaciones.ipynb) |
| `11_marca_de_agua` — dónde aparece la marca de agua de Google Maps, si se asocia con la clase y qué zona tapar (issue #7) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/11_marca_de_agua.ipynb) |

**Clasificación (consigna, ítem 2)**

| Notebook | |
|---|---|
| `04_clasificacion_mobilenetv2` — MobileNetV2: congelado, fine-tuning y sin preentrenamiento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/04_clasificacion_mobilenetv2.ipynb) |
| `05_clasificacion_resnet50` — ResNet50: congelado, fine-tuning de todo el extractor y pesos aleatorios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/05_clasificacion_resnet50.ipynb) |
| `06_clasificacion_efficientnet` — EfficientNetB0: congelado, fine-tuning y pesos aleatorios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/06_clasificacion_efficientnet.ipynb) |

**Detección (consigna)**

| Notebook | |
|---|---|
| `08_deteccion_yolov8n` — YOLOv8n: red completa y backbone congelado, mAP en val/Jamaica y detector como clasificador | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/08_deteccion_yolov8n.ipynb) |
| `09_deteccion_yolo11n` — YOLO11n: lo mismo que el 08, con la misma receta salvo el tamaño de entrada (640 en vez de 256) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/09_deteccion_yolo11n.ipynb) |

Usar una sesión nueva de Colab con GPU T4 y `requirements-detection.txt` (PyTorch).
Solo usa el dataset existente y el split fijo. Guía: [detección con YOLOv8n](docs/deteccion_yolov8n.md); vale también para YOLO11n,
salvo que congelar el backbone es `freeze=11` (módulos 0–10) en vez de `freeze=10` y que entrena a 640 px.

**Comparación final (consigna)**

| Notebook | |
|---|---|
| `10_comparacion_final` — compara clasificadores y detectores usados como clasificadores sobre val y Jamaica, con intervalos bootstrap y diferencias pareadas, más el mAP de los detectores. No entrena: lee los resultados en solo lectura | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/10_comparacion_final.ipynb) |

No necesita GPU. Correrlo cada vez que haya corridas nuevas: solo compara las CNN con `receta: "comun_v2"`.

**Herramienta vieja (no usar para entrenar)**

| Notebook | |
|---|---|
| `07_ordenar_resultados_resnet50` — ⚠️ **HERRAMIENTA VIEJA**: solo ordena en Drive corridas de ResNet50 hechas con la versión anterior (API de Drive, formato ZIP). No entrena ni genera resultados. Es una excepción documentada del verificador | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/07_ordenar_resultados_resnet50.ipynb) |

## Dónde quedan los resultados

**Todos los notebooks guardan igual: una carpeta por modelo y por modo**, nunca por persona. Lo hace
`resultados.guardar_corrida(...)` de [`pdi_tp/resultados.py`](pdi_tp/resultados.py):

```
Mi unidad/PDI-TP/
├── resultados/                         ← solo modelos
│   ├── mobilenetv2/
│   │   ├── imagenet_congelado/
│   │   │   ├── modelo.keras            ← pesos (o .pt / .joblib según el framework)
│   │   │   ├── metricas.json           ← accuracy, precision, recall, F1, F1 macro, matriz, por fuente y solo Jamaica
│   │   │   ├── predicciones_val.csv    ← predicción contra el real: archivo, fuente, y_real,
│   │   │   │                              probabilidad, prediccion, acierto (las 162 imágenes de val)
│   │   │   ├── config.json             ← modelo, modo, parámetros, fecha y commit del repo
│   │   │   └── historia.csv, curvas…   ← opcionales
│   │   ├── imagenet_finetuning/
│   │   └── aleatorio/
│   ├── efficientnetb0/
│   ├── resnet50/
│   └── baseline/siempre_con_paneles/
├── analisis/                           ← tablas auxiliares de notebooks que entrenan (p. ej. manifiesto de ResNet50)
└── cache/                              ← intermedios reutilizables (features, folds); no son resultados
```

**Solo los notebooks que entrenan crean carpetas en Drive**, y recién cuando guardan algo. El EDA, la
validación y la revisión de anotaciones no montan Drive: correrlos no crea nada en la carpeta compartida
(sus reportes quedan solo en la sesión de Colab). La comparación final sí monta Drive, pero en **solo
lectura** (`entorno.preparar(solo_lectura=True)`): no puede crear ni guardar corridas ni escribir archivos.

- **Siempre las mismas métricas, calculadas de la misma manera**, y `y_real` sale de las etiquetas del
  dataset, no del notebook. Si falta algo (pesos, una imagen de val, una columna), `guardar_corrida` frena
  con un error en vez de dejar una carpeta incompleta.
- **Comparar todo el equipo**: `resultados.cargar_corridas(rutas)` (una fila por modelo/modo) y
  `resultados.cargar_predicciones(rutas)` (una fila por imagen y corrida, para comparar sobre las mismas
  imágenes o contra el detector).
- **Repetir un modo**: cada notebook tiene `REPETIR`. Con `False`, los modos ya completos se saltean y uno a
  medias se retoma; con `True` se entrena de nuevo y la carpeta anterior **se renombra**
  (`<modo>__anterior_<fecha>`), no se borra.
- **Modos que se dejan de usar o cambian de nombre** (por ejemplo `mobilenetv2/imagenet_ft_completo`):
  `REPETIR` no los toca, así que hay que renombrar a mano su carpeta en Drive agregándole `__anterior_<fecha>`.
  Toda carpeta con `__anterior_` queda afuera de `cargar_corridas`, `cargar_predicciones` y las tablas de
  comparación.
- **Corridas hechas con el formato anterior** (carpetas por persona, JSON sueltos o el ZIP de ResNet50) no
  aparecen en `cargar_corridas`: para comparar, volver a correrlas con los notebooks actuales.
- **Nada borra archivos de Drive** y todos los notebooks usan el mismo acceso (`drive.mount` dentro de
  `entorno.preparar`). La única excepción es la herramienta vieja `07_ordenar_resultados_resnet50`.

## Subir notebooks y cambios al repositorio

**`main` está protegida: no se puede subir directo.** Todo cambio entra por una **rama** y un **pull request
(PR)**, y GitHub solo deja unirlo a `main` cuando el chequeo `verificar` está en verde. Hoy la regla no aplica a
quien administra el repo; para que valga para todos, activar *Do not allow bypassing the above settings* en
la protección de `main`.
**En el repo va el código; los resultados van a Drive.**

| Va a Git | No va a Git |
|---|---|
| Notebooks **sin salidas** | Notebooks ejecutados (con gráficos y tablas) |
| `pdi_tp/`, `scripts/`, `docs/`, `splits/`, `tests/` | `data/` (dataset), `resultados/` y `resultados_locales/` |
| Experimentos nuevos, como una entrada más en `EXPERIMENTOS` del notebook | Una copia del notebook por experimento |

### Quién abre y quién acepta un PR

- **Lo abre quien hizo el cambio** (o su IA, automáticamente: ver abajo), desde su rama hacia `main`.
- **Lo acepta (merge) quien lo abrió**, cuando el chequeo `verificar` está en verde. Es lo único que exige
  GitHub hoy.
- **Excepción, pedir revisión a otra persona antes del merge** (que la apruebe en el PR) si el cambio toca
  algo que usan todos: `pdi_tp/`, `splits/`, `scripts/`, `.github/`, `.githooks/`,
  `docs/criterios_datos.md`, o un notebook que armó otra persona.
- Al hacer merge se usa *Squash and merge* y se borra la rama.

### Trabajando con una IA (Claude Code, Codex, Cursor...)

No hay que pedírselo: [`AGENTS.md`](AGENTS.md) (y [`CLAUDE.md`](CLAUDE.md), que lo incluye) le indican al agente
que **cada cambio** lo haga solo así: rama nueva desde `main` actualizado → commit → verificador y tests →
push de la rama → PR → esperar el chequeo en verde. **El merge lo confirma la persona** (o pide revisión,
según la regla de arriba). Para eso el agente necesita `git` y, para abrir el PR, la CLI `gh` (ver
*Puesta en marcha*); sin `gh`, sube la rama y te deja el link para abrir el PR desde el navegador.

### Normas que se verifican solas

[`scripts/verificar_notebooks.py`](scripts/verificar_notebooks.py) revisa cada notebook y **rechaza** los que:

1. tienen salidas (*Editar → Borrar todos los resultados* en Colab);
2. piden el nombre de quien los corre (`INTEGRANTE`): ya no se usa;
3. montan Drive a mano, usan la API de Drive, borran archivos o usan la forma vieja de guardar;
4. entrenan un modelo y no guardan con `resultados.guardar_corrida(...)`, o no entrenan y montan Drive igual
   (tienen que usar `entorno.preparar(usar_drive=False)`);
5. no están enlazados en este README con su botón de Colab.

Corre en dos lugares:

- **Antes de cada commit local** (con el hook activado, ver *Puesta en marcha*): si algo no cumple, el commit
  no se hace.
- **En GitHub, en cada push a cualquier rama** ([`.github/workflows/verificar.yml`](.github/workflows/verificar.yml)),
  junto con los tests. Es el chequeo `verificar` que exige `main`: un PR en rojo no se puede unir.

### Desde una copia local (o con una IA)

```bash
git switch main && git pull                           # partir de lo último
git switch -c notebook/mobilenetv2-sin-aug            # rama nueva: <tipo>/<descripcion-corta>
# ... editar ...
python scripts/verificar_notebooks.py                 # (el hook lo corre solo en el commit)
git add notebooks/04_clasificacion_mobilenetv2.ipynb  # archivos puntuales, nunca "git add ."
git commit -m "Agregar experimento MobileNetV2 sin augmentation"
git push -u origin notebook/mobilenetv2-sin-aug
gh pr create --base main --fill                       # abre el PR (o usar el link que imprime git push)
gh pr checks --watch                                  # esperar el chequeo en verde
gh pr merge --squash --delete-branch                  # unir a main (o pedir revisión, ver arriba)
```

Tipos de rama: `notebook/`, `codigo/`, `datos/`, `docs/`, `fix/`. Si mientras tanto cambió `main`,
traerlo a la rama con `git pull --rebase origin main` y volver a subir (`git push --force-with-lease`, solo
sobre **tu** rama, nunca sobre `main`). Si hay conflicto en un `.ipynb`, avisar al equipo antes de
resolverlo: un notebook mal mergeado deja de abrir.

### Desde Colab

Colab no crea ramas, así que primero se crea en GitHub:

1. En GitHub, botón de ramas (arriba a la izquierda, dice `main`) → escribir el nombre, por ejemplo
   `notebook/resnet50-otra-receta` → *Create branch*.
2. Abrir el notebook **desde el botón del README**, editar, correr y verificar.
3. Borrar salidas (*Editar → Borrar todos los resultados*).
4. *Archivo → Guardar una copia en GitHub* → repositorio `candeperles/PDI-TP`, **la rama nueva**, misma ruta del
   archivo y un mensaje de commit.
5. En GitHub aparece *Compare & pull request*: abrir el PR, esperar el chequeo en verde y hacer merge (o
   pedir revisión, ver arriba).

Ojo: Colab sube su versión tal cual. Si alguien cambió ese notebook en `main` después de que lo abriste,
sus cambios se pierden al unir. Si dos personas tienen que tocar el mismo notebook, coordinar antes.

### Experimentos y notebooks nuevos

- **Otro experimento con un modelo que ya tiene notebook**: no copiar el notebook. Agregar una entrada a
  `EXPERIMENTOS` con un nombre de modo único, por ejemplo en MobileNetV2
  `"imagenet_sin_aug": {"pesos": "imagenet", "modo": "congelado", "augmentation": False}`.
  Queda en `resultados/mobilenetv2/imagenet_sin_aug/`.
- **Un modelo nuevo**: partir del notebook más parecido (p. ej. `04_clasificacion_mobilenetv2`), guardarlo como
  `notebooks/NN_<tarea>_<modelo>.ipynb`, guardar cada modo con `resultados.carpeta_corrida(...)` +
  `resultados.guardar_corrida(...)` y sumarlo a la tabla de este README con su botón de Colab
  (`https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/<ruta del notebook>`).
  Si entrena con TensorFlow, instalar `requirements-training.txt`, no `requirements.txt`.
- **Código que usan varios notebooks**: va en `pdi_tp/`. Si se cambia, avisar: lo reciben todos en el
  próximo setup.
- Mensajes de commit en castellano, en infinitivo y diciendo qué cambia ("Agregar ...", "Corregir ...").

## Estructura del repositorio

```
├── AGENTS.md / CLAUDE.md  normas del repo para agentes de IA (y para personas)
├── data/                  dataset (no se versiona; lo baja scripts/download_data.py)
├── docs/                  consigna, alcance, criterios de datos, revisiones y notas por modelo
├── notebooks/             un notebook por etapa / modelo (sin salidas)
├── pdi_tp/                código compartido: entorno (setup y Drive), datos, split, resultados (formato
│                          común), validación, revisión, clasificación
├── scripts/               descarga del dataset, split, verificador de notebooks y otras verificaciones
├── splits/                train.txt / val.txt compartidos
├── tests/                 pruebas de pdi_tp y del verificador
├── .githooks/             chequeo antes de cada commit
├── .github/               chequeo en GitHub en cada push y plantilla de PR
└── requirements*.txt      dependencias (ver INSTRUCCIONES.md)
```

## Documentación

| Documento | Contenido |
|---|---|
| [Consigna_Paneles_Solares.pdf](docs/Consigna_Paneles_Solares.pdf) | Consigna original |
| [INSTRUCCIONES.md](INSTRUCCIONES.md) | Instalación, descarga del dataset, trabajo en local, verificaciones y problemas comunes |
| [AGENTS.md](AGENTS.md) | Normas de trabajo en el repo (para agentes de IA y para el equipo) |
| [alcance.md](docs/alcance.md) | Requisitos de la consigna, opcionales y puntos abiertos |
| [validacion_dataset.md](docs/validacion_dataset.md) | Resultados de la validación del dataset |
| [revision_anotaciones.md](docs/revision_anotaciones.md) | Revisión de cajas raras, pequeñas, nulas y negativos dudosos |
| [criterios_datos.md](docs/criterios_datos.md) | Decisiones de datos: una clase `panel`, cajas pequeñas, negativos N06/N16 |
| [clasificacion_resnet50.md](docs/clasificacion_resnet50.md) | Notas de los experimentos con ResNet50 |
| [evaluacion_jamaica.md](docs/evaluacion_jamaica.md) | Cómo medir sobre Kingston/Jamaica, como el test (propuesta) |

¿Algo no funciona? Ver [problemas comunes](INSTRUCCIONES.md#problemas-comunes).
