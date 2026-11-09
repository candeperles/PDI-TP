# PDI-TP · Paneles solares: clasificación y detección

Trabajo práctico de Procesamiento de Imágenes, grupo 5. A partir de imágenes aéreas,
resolvemos **si hay paneles solares** con clasificadores y **dónde están** con detectores.
Después usamos los detectores como clasificadores para comparar ambos enfoques.

[Consigna](docs/Consigna_Paneles_Solares.pdf) · [Alcance del trabajo](docs/alcance.md) ·
[Instalación local y problemas comunes](INSTRUCCIONES.md)

## Recorrido de notebooks

El orden sigue el trabajo: **datos y criterios (01–04) → entrenamiento (05–10) → experimentos y
errores (11–14) → comparación en val (15) → CV de Jamaica (16) → elección del modelo final (17) → test final (18)**.
Los modelos de una misma familia se pueden entrenar en paralelo; los análisis requieren sus resultados.

| Notebook | Qué resuelve | Colab |
|---|---|---|
| [01_exploracion](notebooks/01_exploracion.ipynb) | Fuentes, clases, cajas y separación espacial. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/01_exploracion.ipynb) |
| [02_validacion_datos](notebooks/02_validacion_datos.ipynb) | Archivos, etiquetas YOLO, duplicados y split. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/02_validacion_datos.ipynb) |
| [03_revision_anotaciones](notebooks/03_revision_anotaciones.ipynb) | Casos dudosos y criterios para preparar los datos. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/03_revision_anotaciones.ipynb) |
| [04_marca_de_agua](notebooks/04_marca_de_agua.ipynb) | Ubicar la marca de Google Maps y su relación con la clase. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/04_marca_de_agua.ipynb) |
| [05_clasificacion_mobilenetv2](notebooks/05_clasificacion_mobilenetv2.ipynb) | ImageNet congelado, fine-tuning y pesos aleatorios. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/05_clasificacion_mobilenetv2.ipynb) |
| [06_clasificacion_resnet50](notebooks/06_clasificacion_resnet50.ipynb) | Los mismos tres modos con la receta común. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/06_clasificacion_resnet50.ipynb) |
| [07_clasificacion_efficientnet](notebooks/07_clasificacion_efficientnet.ipynb) | Los mismos tres modos con la receta común. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/07_clasificacion_efficientnet.ipynb) |
| [08_deteccion_yolov8n](notebooks/08_deteccion_yolov8n.ipynb) | Detectar paneles a 640 px y usar el detector como clasificador. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/08_deteccion_yolov8n.ipynb) |
| [09_deteccion_yolo11n](notebooks/09_deteccion_yolo11n.ipynb) | Repetir la receta de YOLOv8n con otra arquitectura. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/09_deteccion_yolo11n.ipynb) |
| [10_deteccion_yolo26n](notebooks/10_deteccion_yolo26n.ipynb) | Tercera versión opcional, con su propio entorno. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/10_deteccion_yolo26n.ipynb) |
| [11_variables_clasificacion](notebooks/11_variables_clasificacion.ipynb) | Medir augmentation, pesos, LR y early stopping en MobileNetV2. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/11_variables_clasificacion.ipynb) |
| [12_deteccion_yolov8n_256](notebooks/12_deteccion_yolov8n_256.ipynb) | Comparar 256 contra 640 px manteniendo el resto de la receta. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/12_deteccion_yolov8n_256.ipynb) |
| [13_marca_de_agua_efecto](notebooks/13_marca_de_agua_efecto.ipynb) | Diagnosticar si las CNN y YOLOv8n/YOLO11n usan la marca como atajo. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/13_marca_de_agua_efecto.ipynb) |
| [14_galeria_errores](notebooks/14_galeria_errores.ipynb) | Inspeccionar clasificaciones, cajas, omisiones y falsas detecciones. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/14_galeria_errores.ipynb) |
| [15_comparacion_validacion](notebooks/15_comparacion_validacion.ipynb) | Comparar arquitecturas sobre val y Jamaica; elegir finalistas. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/15_comparacion_validacion.ipynb) |
| [16_validacion_cruzada_jamaica](notebooks/16_validacion_cruzada_jamaica.ipynb) | Comparar finalistas y decidir modelo y umbral con predicciones fuera de fold. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/16_validacion_cruzada_jamaica.ipynb) |
| [17_modelo_final](notebooks/17_modelo_final.ipynb) | Fijar el modelo y el umbral de cada familia con val y la CV, sin mirar el test. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/17_modelo_final.ipynb) |
| [18_evaluacion_test](notebooks/18_evaluacion_test.ipynb) | Evaluar una vez, al final, con la elección ya fijada. | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/18_evaluacion_test.ipynb) |

Los experimentos 11 y 12 usan las referencias entrenadas en el 05 y el 08, respectivamente.
El 15 solo necesita los entrenamientos del 05–10: los notebooks 11–14 no son requisito para compararlos.
Los diagnósticos y la comparación requieren las corridas guardadas de los modelos anteriores.
El 16 parte de referencias completas de los finalistas y reúne los cinco folds antes de comparar un modelo. El 17 fija modelos y umbrales
sin cargar el test, y el 18 se ejecuta **una sola vez, después**: el test no decide ninguno de ellos.

## Puesta en marcha

Cada integrante agrega un **acceso directo a la carpeta compartida** en `Mi unidad/PDI-TP` y
monta Drive con la cuenta que tiene acceso. Usar esa carpeta compartida, sin crear otra con el mismo nombre.
El dataset se descarga automáticamente al preparar el entorno.

## Correr un notebook en Colab

1. Abrir el botón de la tabla y revisar las opciones de la primera celda.
2. Activar **GPU T4** para entrenar. Los notebooks de datos y los que solo leen CSV no necesitan GPU.
3. Ejecutar todas las celdas. El setup prepara código, dependencias y datos; monta Drive cuando corresponde.
4. Si Colab pide reiniciar por cambios de paquetes, reiniciar y ejecutar otra vez desde el setup.

Usar una sesión nueva al cambiar de entorno: **TensorFlow** para CNN,
**Ultralytics 8.3.221** para YOLOv8n/YOLO11n y **8.4.184** para YOLO26n.
No mezclar `requirements.txt` (exploración) con los requisitos de entrenamiento.
El diagnóstico de marca de agua del 13 todavía no admite YOLO26n.

Dejar `REPETIR=False` para conservar corridas completas. `REPETIR=True` archiva y vuelve a entrenar.
La recuperación de una corrida interrumpida depende del notebook: ver la
[guía de ejecución y recuperación](docs/guia_notebooks.md).

## Receta común de clasificación

Las tres redes se entrenan con la misma receta y cambian solo la arquitectura, para que la comparación valga
(detalle y discusión en el [issue #15](https://github.com/candeperles/PDI-TP/issues/15)). Los tres notebooks
implementan `comun_v2`; cada corrida nueva guarda ese valor en `params.receta` dentro de `config.json`.

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

Las etapas de la cabeza y del extractor, junto con los diagramas de las tres redes, están en
[Estructura de los clasificadores](docs/estructura_clasificadores.md).
La receta de detección y las diferencias entre versiones se documentan en
[Detección con YOLO](docs/deteccion_yolov8n.md).

## Datos y evaluación

- **Dataset:** 900 imágenes PNG de 256×256 con anotaciones YOLO; no se versiona en Git.
  Fuentes: `solarpanels_hd`, `PV01`, `kingston`, `kingston2` y `geo`.
- **Split fijo:** 738 imágenes de train y 162 de val, agrupadas por zona espacial para separar recortes vecinos.
  Se carga desde `splits/train.txt` y `splits/val.txt`.
- **Jamaica:** 172 imágenes en total, 23 en val. La CV adicional del 16 evalúa las 172 por grupos
  y conserva el split oficial.
- **Detección:** una sola clase, `panel`. Los IDs 0/1/2 de `solarpanels_hd` representan confianza
  de anotación. Los criterios y las revisiones están en [criterios de datos](docs/criterios_datos.md).
- **Test de Kingston:** 200 imágenes etiquetadas en Drive, `PDI-TP/test_kingston/`, fuera del repo.
  Se reserva para la evaluación final; no se usa para elegir arquitectura, receta o umbral.

Se reportan métricas por clase, F1 macro, balanced accuracy y AUC para clasificación;
mAP50 y mAP50-95 para detección. La comparación de arquitecturas usa CNN con `comun_v2` y detectores
a 640 px; los controles de variables y resolución se analizan por separado.
Val elige checkpoints y también reporta: sus métricas son optimistas. La CV y sus limitaciones
se explican en [Evaluación sobre Jamaica](docs/evaluacion_jamaica.md).

## Dónde quedan los resultados

Las corridas comunes se guardan por modelo y modo en:
`Mi unidad/PDI-TP/resultados/<modelo>/<modo>/`.
Cada corrida completa contiene pesos, `config.json`, `metricas.json` y `predicciones_val.csv`
sobre las 162 imágenes de val. Los detectores también guardan las cajas de sus predicciones.

La CV usa `analisis/cv_jamaica/<modelo>/<modo>/fold_N/`, con sus pesos, configuración,
particiones y predicciones fuera de fold; queda separada de las corridas comunes.
Los resultados locales usan la misma estructura bajo `resultados_locales/`.
Las repeticiones se archivan con `__anterior_<fecha>`; no se borran archivos de Drive.

El 15 muestra las corridas disponibles, su receta y sus métricas; el 16 informa cuáles tienen
los cinco folds completos. Ese es el estado real de entrenamiento: no se deduce de que el código esté implementado.

## Subir notebooks y cambios al repositorio

Trabajar en una rama y abrir un PR hacia `main`. Los notebooks se suben sin salidas ni contadores.
Antes de commitear:

```bash
python scripts/verificar_notebooks.py
python -m unittest discover -s tests
```

Activar el hook una vez por clon con `git config core.hooksPath .githooks`.
El PR requiere el chequeo `verificar` en verde y las revisiones indicadas en [AGENTS.md](AGENTS.md).
No subir datos, pesos ni resultados al repo.

## Documentación

- [Alcance y requisitos de la consigna](docs/alcance.md).
- [Instalación, descarga del dataset y problemas comunes](INSTRUCCIONES.md).
- [Guía de entornos, guardado y recuperación](docs/guia_notebooks.md).
- [Validación del dataset](docs/validacion_dataset.md) y [revisión de anotaciones](docs/revision_anotaciones.md).
- [Receta de detección](docs/deteccion_yolov8n.md) y [evaluación de Jamaica](docs/evaluacion_jamaica.md).
- [Normas de colaboración](AGENTS.md).
