# PDI-TP · Paneles solares: clasificación y detección

Trabajo práctico de **Procesamiento de Imágenes** (grupo 5). Consigna:
[docs/Consigna_Paneles_Solares.pdf](docs/Consigna_Paneles_Solares.pdf) · resumen de requisitos:
[docs/alcance.md](docs/alcance.md) · instalación, trabajo en local y verificaciones:
[INSTRUCCIONES.md](INSTRUCCIONES.md).

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
| Clasificación: EfficientNetB0 | `02_clasificacion_efficientnet` | En curso |
| Clasificación: MobileNetV2 | `04_clasificacion_mobilenetv2` | En curso |
| Clasificación: ResNet50 | `05_ResNet50_Colab_v2` | En curso |
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
   `santiago`). Es el nombre de tu carpeta de resultados; usá siempre el mismo.

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
| `00_plantilla` — base para armar un experimento nuevo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/00_plantilla.ipynb) |
| `01_exploracion` — EDA: balance de clases, fuentes, cajas, split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/01_exploracion.ipynb) |
| `02_validacion_datos` — validación de archivos, cajas, duplicados y split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/02_validacion_datos.ipynb) |
| `03_revision_anotaciones` — casos dudosos, muestra de negativos y criterios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/03_revision_anotaciones.ipynb) |

**Clasificación (consigna, ítem 2)**

| Notebook | |
|---|---|
| `02_clasificacion_efficientnet` — EfficientNetB0 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/02_clasificacion_efficientnet.ipynb) |
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

## Dónde quedan los resultados

Cada integrante guarda en **su propia carpeta** dentro de la carpeta compartida. El nombre sale de
`INTEGRANTE`, así que nadie pisa los resultados de otro:

```
Mi unidad/PDI-TP/
└── resultados/
    ├── kevo/
    │   ├── <experimento>.json           ← parámetros y métricas (resultados.guardar_experimento)
    │   ├── <experimento>.keras          ← pesos del modelo
    │   ├── <experimento>_predicciones_val.csv
    │   └── modelos_extra/               ← notebook extra (comparaciones y caché)
    ├── pia/
    ├── cande/
    └── santiago/
```

- Todo lo que se guarde va a `rutas.resultados` (= `Mi unidad/PDI-TP/resultados/<integrante>/`), nunca solo
  a `/content`, que se borra cuando se corta la sesión.
- Cada experimento se registra con `resultados.guardar_experimento(rutas, nombre, params, metricas)`.
  Con `resultados.cargar_experimentos(rutas)` se leen los de **todo el equipo** en una tabla.
- **Nombres de experimento únicos**: si se repite un nombre, se pisa el JSON anterior.
- **Excepción, `05_ResNet50_Colab_v2`**: no monta Drive; al correrlo pide la URL de una carpeta existente.
  Para mantener el orden, pegar la URL de **tu** carpeta `PDI-TP/resultados/<integrante>/`. Los archivos
  llevan el prefijo `<integrante>__<corrida>`.

## Subir notebooks y cambios al repositorio

Se trabaja directo sobre `main`. **En el repo va el código; los resultados van a Drive.**

| Va a Git | No va a Git |
|---|---|
| Notebooks **sin salidas** y con `INTEGRANTE = "nombre"` | Notebooks ejecutados (con gráficos y tablas) |
| `pdi_tp/`, `scripts/`, `docs/`, `splits/`, `tests/` | `data/` (dataset) y `resultados/` (JSON, `.keras`, CSV) |
| Experimentos nuevos, como una entrada más en `EXPERIMENTOS` del notebook | Una copia del notebook por experimento |

Antes de subir un notebook, siempre:

1. **Borrar las salidas** (*Editar → Borrar todos los resultados* en Colab).
2. **Dejar `INTEGRANTE = "nombre"`**. Si se sube con tu nombre, el próximo que lo corra guarda sus
   resultados en tu carpeta.

### Desde Colab

1. Abrir el notebook **desde el botón del README** (así partís de la última versión).
2. Editar, correr y verificar.
3. Borrar salidas y volver a poner `INTEGRANTE = "nombre"`.
4. *Archivo → Guardar una copia en GitHub* → repositorio `candeperles/PDI-TP`, rama `main`, **misma ruta**
   del archivo (p. ej. `notebooks/04_clasificacion_mobilenetv2.ipynb`) y un mensaje de commit.

Ojo: Colab sube su versión tal cual. Si alguien cambió ese notebook en `main` después de que lo abriste,
sus cambios se pierden. Si dos personas tienen que tocar el mismo notebook, coordinar antes.

### Desde una copia local

```bash
git pull                                              # traer lo último antes de tocar nada
# ... editar ...
git status                                            # revisar que no se cuelen data/ ni resultados/
git add notebooks/04_clasificacion_mobilenetv2.ipynb  # archivos puntuales, nunca "git add ."
git commit -m "Agregar experimento MobileNetV2 sin augmentation"
git push
```

Si `git push` se rechaza porque alguien subió antes: `git pull --rebase` y de nuevo `git push`. Si hay
conflicto en un `.ipynb`, avisar al equipo antes de resolverlo: un notebook mal mergeado deja de abrir.

### Experimentos y notebooks nuevos

- **Otro experimento con un modelo que ya tiene notebook**: no copiar el notebook. Agregar una entrada a
  `EXPERIMENTOS` con un nombre único y lo que cambia respecto de `BASE`, por ejemplo
  `"mobilenetv2_imagenet_sin_aug": {"pesos": "imagenet", "modo": "congelado", "augmentation": False}`.
- **Un modelo nuevo**: partir del notebook más parecido (p. ej. `04_clasificacion_mobilenetv2` para
  clasificación) o de `00_plantilla`. Los dos traen el setup que guarda en tu carpeta. Si entrena con
  TensorFlow, el setup tiene que instalar `requirements-training.txt` y no `requirements.txt`: la plantilla
  instala el de exploración, que fija versiones de NumPy incompatibles con TensorFlow. Guardarlo como
  `notebooks/NN_<tarea>_<modelo>.ipynb` y sumarlo a la tabla de este README con su botón de Colab
  (`https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/<ruta del notebook>`).
- **Código que usan varios notebooks**: va en `pdi_tp/`. Si se cambia, avisar: lo reciben todos en el
  próximo setup.
- Mensajes de commit en castellano, en infinitivo y diciendo qué cambia ("Agregar ...", "Corregir ...").

## Estructura del repositorio

```
├── data/              dataset (no se versiona; lo baja scripts/download_data.py)
├── docs/              consigna, alcance, criterios de datos, revisiones y notas por modelo
├── notebooks/         un notebook por etapa / modelo (sin salidas)
│   └── extras/        trabajo adicional, fuera de la consigna
├── pdi_tp/            código compartido: entorno (setup y Drive), datos, split, resultados,
│                      validación, revisión, clasificación
├── scripts/           descarga del dataset, generación del split, verificaciones
├── splits/            train.txt / val.txt compartidos
├── tests/             pruebas de pdi_tp
└── requirements*.txt  dependencias (ver "Trabajar en local")
```

## Documentación

| Documento | Contenido |
|---|---|
| [Consigna_Paneles_Solares.pdf](docs/Consigna_Paneles_Solares.pdf) | Consigna original |
| [INSTRUCCIONES.md](INSTRUCCIONES.md) | Instalación, descarga del dataset, trabajo en local, verificaciones y problemas comunes |
| [alcance.md](docs/alcance.md) | Requisitos de la consigna, opcionales y puntos abiertos |
| [validacion_dataset.md](docs/validacion_dataset.md) | Resultados de la validación del dataset |
| [revision_anotaciones.md](docs/revision_anotaciones.md) | Revisión de cajas raras, pequeñas, nulas y negativos dudosos |
| [criterios_datos.md](docs/criterios_datos.md) | Decisiones de datos: una clase `panel`, cajas pequeñas, negativos N06/N16 |
| [clasificacion_resnet50.md](docs/clasificacion_resnet50.md) | Notas de los experimentos con ResNet50 |

¿Algo no funciona? Ver [problemas comunes](INSTRUCCIONES.md#problemas-comunes).
