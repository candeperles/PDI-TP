# PDI-TP · Paneles solares: clasificación y detección

> [!CAUTION]
> ## ⚠️ LOS RESULTADOS DE CLASIFICACIÓN TODAVÍA NO SON COMPARABLES ENTRE MODELOS
>
> **1. Bug en `02_clasificacion_efficientnet`: el modo "fine-tuning" no hace fine-tuning.**
> En `entrenar()`, `congelado = weights == 'imagenet' and not fine_tune` da `False` cuando `fine_tune=True`,
> así que el extractor arranca **descongelado** y se entrena toda la red con `Adam(1e-3)` desde la primera
> época. No es el esquema de la consigna (primero la cabeza con el extractor congelado, después descongelar
> con LR ~1e-5). Ese experimento mide otra cosa y, con LR alto, probablemente destruye los pesos de ImageNet.
>
> **2. EfficientNet no usa pesos de clase.** Imprime "Pesos de clase" pero nunca los pasa a `fit`. Con el
> desbalance 85/15 queda en desventaja frente a MobileNetV2 y ResNet50, que sí los usan.
>
> **3. Cada modelo se entrena con una receta distinta.** Si un modelo le gana a otro no se puede saber si es
> por la arquitectura o por todo lo demás:
>
> | | EfficientNetB0 | MobileNetV2 | ResNet50 |
> |---|---|---|---|
> | Tamaño de entrada | 256 | 224 | 256 |
> | Batch | 32 | 32 | 16 |
> | Augmentation | flip horizontal, rotación ±18°, zoom 10 % | rotaciones de 90°, flip, brillo/contraste | flip h+v, rotación ±90° |
> | Pesos de clase | **no** | sí | sí |
> | Épocas (cabeza + fine-tuning / aleatorio) | 10 + 10 / 10 | 15 + 15 / 40 | 10 + 10 / 20 |
> | Early stopping (paciencia) | 3 | 6 | 5 |
> | LR aleatorio | 1e-3 | 1e-3 | 1e-4 |
> | Qué se descongela en fine-tuning | todo, desde el inicio (bug) | desde la capa 100, o todo | solo el bloque conv5 |
> | Dropout | 0.3 | 0.2 | 0.3 |
>
> **4. Val elige el modelo y también lo evalúa.** Los tres usan early stopping o el mejor checkpoint según
> `val_loss` y después reportan métricas sobre ese mismo val: los números son optimistas. Como la consigna
> solo pide train/val es aceptable, pero hay que decirlo en el informe y aplicarlo igual en todos.
>
> **Para que la comparación valga:** corregir 1 y 2 y fijar una receta común (tamaño, augmentation, pesos de
> clase, épocas, paciencia y LR) para las tres redes, cambiando solo la arquitectura.
> *La forma de guardar resultados ya está unificada (ver [Dónde quedan los resultados](#dónde-quedan-los-resultados)).*

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

## Estado

| Etapa | Notebook | Estado |
|---|---|---|
| Exploración del dataset | `01_exploracion` | Hecho |
| Validación del dataset y revisión de anotaciones | `02_validacion_datos`, `03_revision_anotaciones` | Hecho; criterios en [docs/criterios_datos.md](docs/criterios_datos.md) |
| Clasificación: EfficientNetB0 | `02_clasificacion_efficientnet` | En curso · **con bugs (ver cartel)** |
| Clasificación: MobileNetV2 | `04_clasificacion_mobilenetv2` | En curso |
| Clasificación: ResNet50 | `05_ResNet50_Colab_v2` | En curso |
| Receta común de clasificación | — | Pendiente (ver cartel) |
| Detección: YOLOv8n / YOLO11n | — | Pendiente |
| Detector como clasificador y comparación final | — | Pendiente |
| Informe | — | Pendiente |
| *Extra (fuera de la consigna)*: modelos adicionales | `extras/modelos_adicionales` | En curso |

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

## Puesta en marcha (una sola vez por integrante)

1. **Carpeta compartida en Drive.** Una sola persona crea la carpeta `PDI-TP` y la comparte con el resto
   como Editor. Cada integrante entra a *Compartidos conmigo* → clic derecho en `PDI-TP` → *Organizar →
   Agregar acceso directo* → *Mi unidad*. Tiene que quedar exactamente en `Mi unidad/PDI-TP`.
   - **No crear otra carpeta llamada `PDI-TP`** en tu Drive: Drive permite nombres repetidos y Colab puede
     agarrar la equivocada (vas a ver la carpeta vacía y "desaparecen" los resultados del resto).
   - Montar Drive en Colab **con la misma cuenta** a la que se compartió la carpeta.
2. **Elegir tu nombre de integrante**: minúsculas, sin espacios ni acentos (`kevo`, `pia`, `cande`,
   `santiago`). Queda registrado en cada corrida como autor; las carpetas **no** llevan nombres de personas.
3. **Si trabajás con git en local**, activar el chequeo previo a cada commit (una vez por clon):

   ```bash
   git config core.hooksPath .githooks
   ```

## Correr un notebook en Colab

1. Abrirlo con su botón **Open in Colab** (tabla de abajo). Siempre abre la última versión de `main`.
2. *Entorno de ejecución → Cambiar tipo de entorno → T4* (GPU).
3. En la primera celda, cambiar `INTEGRANTE = "nombre"` por tu nombre.
4. *Entorno de ejecución → Ejecutar todas*. La celda de setup clona o actualiza el repo, instala
   dependencias, descarga el dataset, monta Drive y avisa si no hay GPU. Si Colab pide reiniciar por
   cambios de paquetes, reiniciar y volver a correr desde el principio.

### Notebooks

**Datos**

| Notebook | |
|---|---|
| `01_exploracion` — EDA: balance de clases, fuentes, cajas, split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/01_exploracion.ipynb) |
| `02_validacion_datos` — validación de archivos, cajas, duplicados y split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/02_validacion_datos.ipynb) |
| `03_revision_anotaciones` — casos dudosos, muestra de negativos y criterios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/03_revision_anotaciones.ipynb) |

**Clasificación (consigna, ítem 2)**

| Notebook | |
|---|---|
| `02_clasificacion_efficientnet` — EfficientNetB0 (**ver bugs en el cartel**) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/02_clasificacion_efficientnet.ipynb) |
| `04_clasificacion_mobilenetv2` — MobileNetV2: congelado, fine-tuning y sin preentrenamiento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/04_clasificacion_mobilenetv2.ipynb) |
| `05_ResNet50_Colab_v2` — ResNet50: congelado, fine-tuning de conv5 y pesos aleatorios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/05_ResNet50_Colab_v2.ipynb) |

**Extra: modelos adicionales (fuera de la consigna)**

No es parte de lo que pide la consigna: es un agregado para comparar los modelos obligatorios con otros
enfoques, usando el mismo split.

| Notebook | |
|---|---|
| `extras/modelos_adicionales` — **EXTRA**: DINOv2 y CLIP (features congeladas y zero-shot), fine-tuning de ConvNeXt-Tiny / EfficientNetV2-S y YOLO11s como clasificador, con comparador final | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/extras/modelos_adicionales.ipynb) |

Elige todo con validación cruzada de 5 folds dentro de train (agrupada por zona espacial) y evalúa una
sola vez en `val` al final (`EVALUAR_TEST = True`).

**Herramienta vieja (no usar para entrenar)**

| Notebook | |
|---|---|
| `06_ordenar_resultados_resnet50` — ⚠️ **HERRAMIENTA VIEJA**: solo ordena en Drive corridas de ResNet50 hechas con la versión anterior (API de Drive, formato ZIP). No entrena ni genera resultados. Es una excepción documentada del verificador | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/06_ordenar_resultados_resnet50.ipynb) |

## Dónde quedan los resultados

**Todos los notebooks guardan igual: una carpeta por modelo y por modo**, nunca por persona. Lo hace
`resultados.guardar_corrida(...)` de [`pdi_tp/resultados.py`](pdi_tp/resultados.py):

```
Mi unidad/PDI-TP/
├── resultados/                         ← solo modelos
│   ├── mobilenetv2/
│   │   ├── imagenet_congelado/
│   │   │   ├── modelo.keras            ← pesos (o .pt / .joblib según el framework)
│   │   │   ├── metricas.json           ← accuracy, precision, recall, F1, F1 macro, matriz, por fuente
│   │   │   ├── predicciones_val.csv    ← predicción contra el real: archivo, fuente, y_real,
│   │   │   │                              probabilidad, prediccion, acierto (las 162 imágenes de val)
│   │   │   ├── config.json             ← modelo, modo, parámetros, integrante, fecha, commit
│   │   │   └── historia.csv, curvas…   ← opcionales
│   │   ├── imagenet_fine_tuning/
│   │   └── aleatorio/
│   ├── efficientnetb0/
│   ├── resnet50/
│   └── baseline/siempre_con_paneles/
├── analisis/                           ← salidas de los notebooks de datos (validación, revisión)
└── cache/                              ← intermedios reutilizables (features, folds); no son resultados
```

- **Siempre las mismas métricas, calculadas de la misma manera**, y `y_real` sale de las etiquetas del
  dataset, no del notebook. Si falta algo (pesos, una imagen de val, una columna), `guardar_corrida` frena
  con un error en vez de dejar una carpeta incompleta.
- **Comparar todo el equipo**: `resultados.cargar_corridas(rutas)` (una fila por modelo/modo) y
  `resultados.cargar_predicciones(rutas)` (una fila por imagen y corrida, para comparar sobre las mismas
  imágenes o contra el detector).
- **Repetir un modo**: cada notebook tiene `REPETIR`. Con `False`, los modos ya completos se saltean y uno a
  medias se retoma; con `True` se entrena de nuevo y la carpeta anterior **se renombra**
  (`<modo>__anterior_<fecha>`), no se borra.
- **Nada borra archivos de Drive** y todos los notebooks usan el mismo acceso (`drive.mount` dentro de
  `entorno.preparar`). La única excepción es la herramienta vieja `06_ordenar_resultados_resnet50`.

## Subir notebooks y cambios al repositorio

Se trabaja directo sobre `main`. **En el repo va el código; los resultados van a Drive.**

| Va a Git | No va a Git |
|---|---|
| Notebooks **sin salidas** y con `INTEGRANTE = "nombre"` | Notebooks ejecutados (con gráficos y tablas) |
| `pdi_tp/`, `scripts/`, `docs/`, `splits/`, `tests/` | `data/` (dataset), `resultados/` y `resultados_locales/` |
| Experimentos nuevos, como una entrada más en `EXPERIMENTOS` del notebook | Una copia del notebook por experimento |

### Normas que se verifican solas

[`scripts/verificar_notebooks.py`](scripts/verificar_notebooks.py) revisa cada notebook y **rechaza** los que:

1. tienen salidas (*Editar → Borrar todos los resultados* en Colab);
2. tienen un nombre fijo en `INTEGRANTE` (dejar `"nombre"`);
3. montan Drive a mano, usan la API de Drive, borran archivos o usan la forma vieja de guardar;
4. entrenan un modelo y no guardan con `resultados.guardar_corrida(...)`;
5. no están enlazados en este README con su botón de Colab.

Corre en dos lugares:

- **Antes de cada commit local** (con el hook activado, ver *Puesta en marcha*): si algo no cumple, el commit
  no se hace.
- **En GitHub, en cada push** ([`.github/workflows/verificar.yml`](.github/workflows/verificar.yml)), junto con
  los tests. Así también se controla lo que se sube desde Colab. Si falla, el commit queda marcado en rojo;
  para que además **bloquee** la subida, quien administra el repo tiene que activar *Settings → Branches →
  Require status checks* sobre `main`.

### Desde Colab

1. Abrir el notebook **desde el botón del README** (así partís de la última versión).
2. Editar, correr y verificar.
3. Borrar salidas y volver a poner `INTEGRANTE = "nombre"`.
4. *Archivo → Guardar una copia en GitHub* → repositorio `candeperles/PDI-TP`, rama `main`, **misma ruta**
   del archivo (p. ej. `notebooks/04_clasificacion_mobilenetv2.ipynb`) y un mensaje de commit.
5. Revisar en GitHub que el chequeo quedó en verde.

Ojo: Colab sube su versión tal cual. Si alguien cambió ese notebook en `main` después de que lo abriste,
sus cambios se pierden. Si dos personas tienen que tocar el mismo notebook, coordinar antes.

### Desde una copia local

```bash
git pull                                              # traer lo último antes de tocar nada
# ... editar ...
python scripts/verificar_notebooks.py                 # (el hook lo corre solo en el commit)
git status                                            # revisar que no se cuelen data/ ni resultados/
git add notebooks/04_clasificacion_mobilenetv2.ipynb  # archivos puntuales, nunca "git add ."
git commit -m "Agregar experimento MobileNetV2 sin augmentation"
git push
```

Si `git push` se rechaza porque alguien subió antes: `git pull --rebase` y de nuevo `git push`. Si hay
conflicto en un `.ipynb`, avisar al equipo antes de resolverlo: un notebook mal mergeado deja de abrir.

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
│   └── extras/            trabajo adicional, fuera de la consigna
├── pdi_tp/                código compartido: entorno (setup y Drive), datos, split, resultados (formato
│                          común), validación, revisión, clasificación
├── scripts/               descarga del dataset, split, verificador de notebooks y otras verificaciones
├── splits/                train.txt / val.txt compartidos
├── tests/                 pruebas de pdi_tp y del verificador
├── .githooks/             chequeo antes de cada commit
├── .github/workflows/     chequeo en GitHub en cada push
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

¿Algo no funciona? Ver [problemas comunes](INSTRUCCIONES.md#problemas-comunes).
