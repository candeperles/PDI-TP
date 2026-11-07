# Guía de ejecución y recuperación

El [README](../README.md#recorrido-de-notebooks) contiene el recorrido y los botones de Colab.
Esta guía reúne los detalles de entornos, guardado y recuperación de una sesión interrumpida.
Para instalación local y descarga manual del dataset, ver [INSTRUCCIONES.md](../INSTRUCCIONES.md).

## Preparar Drive y Colab

Una persona comparte la carpeta `PDI-TP` con el equipo. Cada integrante agrega un acceso directo
desde *Compartidos conmigo* a *Mi unidad*, de modo que Colab encuentre
`/content/drive/MyDrive/PDI-TP`. Montar Drive con la cuenta que tiene acceso y usar esa misma carpeta;
una carpeta duplicada con el mismo nombre puede hacer que Colab muestre otros resultados.

El setup de cada notebook prepara las dependencias y descarga `data/grupo_5/` si falta.
Los notebooks 01–04 no montan Drive. Los de entrenamiento lo montan para guardar por modelo y modo;
los de comparación y revisión de resultados lo abren en solo lectura. Ningún notebook pide nombres
de integrantes ni borra archivos de Drive.

## Elegir el entorno

Usar sesiones nuevas de Colab al cambiar de familia de modelos. No combinar los requisitos
de exploración con los de TensorFlow: sus versiones de NumPy pueden ser incompatibles.

| Notebooks | Dependencias | Uso |
|---|---|---|
| 01–04: datos y criterios | `requirements.txt` | CPU, sin Drive. |
| 05–07: CNN; 11: variables de clasificación | `requirements-training.txt` | TensorFlow, T4 al entrenar. |
| 08–09: YOLOv8n/YOLO11n; 12: control a 256 px | `requirements-detection.txt` | Ultralytics 8.3.221, T4 al entrenar. |
| 10: YOLO26n | `requirements-yolo26.txt` | Ultralytics 8.4.184, T4 al entrenar. |
| 13: efecto de marca de agua | Entorno de la familia elegida en sus opciones | Inferencia con pesos guardados, sin reentrenar. |
| 14–15: galería y comparación de val | Entorno de análisis del propio notebook | CPU; leen CSV y métricas de Drive. |
| 16: CV de Jamaica | Entorno de la arquitectura seleccionada | T4 para entrenar; análisis sin GPU. |
| 17: test de Kingston | Entorno de la familia seleccionada | Inferencia; una sesión por familia. |

**YOLO26n requiere su versión propia de Ultralytics.** La 8.3.221 puede cargar sus pesos pero producir
otras cajas. El 13 todavía no sirve para YOLO26n: usarlo con las CNN y YOLOv8n/YOLO11n hasta adaptar
ese diagnóstico. Los notebooks 16 y 17 sí separan los entornos según la arquitectura.
La comparación de YOLO26n con los otros YOLO cambia arquitectura y versión de librería; indicarlo en el informe.

## Revisar antes de repetir

Una corrida común está completa cuando tiene pesos, `config.json`, `metricas.json` y
`predicciones_val.csv` en `resultados/<modelo>/<modo>/`. Confirmar la receta en `config.json`:
`comun_v2` para comparar CNN y `imgsz=640` para los detectores de la comparación de arquitecturas.
Los controles `variables_v1` y `coco_completo_256` se analizan por separado.

`REPETIR=False` conserva modos completos. Algunos notebooks también verifican que la configuración
coincida; no asumir que un modo completo antiguo corresponde a la receta actual.
`REPETIR=True` archiva la carpeta anterior como `<modo>__anterior_<fecha>` y vuelve a entrenar.
Después de una repetición intencional, volver a `False` para conservar lo que ya terminó.

Si se abandona un nombre de modo, archivarlo manualmente con `__anterior_<fecha>` en Drive.
Los lectores de resultados excluyen esas carpetas. Las salidas de formatos anteriores al común
no participan en la comparación: hay que generar una corrida con el formato actual.

## Recuperar una sesión interrumpida

- **05, MobileNetV2, y 07, EfficientNetB0:** saltean modos completos. Un modo incompleto empieza
  desde el principio; no continúa desde la última época. EfficientNetB0 carga la cabeza completa
  de la misma receta para fine-tuning; MobileNetV2 entrena la cabeza dentro de ese modo.
- **06, ResNet50:** continúa la etapa desde `ultimo.keras`, conservando optimizador e historial.
  Usa `mejor.keras` para evaluar. Si historial y checkpoint no coinciden, frena para revisarlos.
- **08–10 y 12, YOLO:** retoman `entrenamiento/weights/last.pt` cuando conserva el optimizador.
  Si el entrenamiento terminó, recuperan `best.pt` para completar la evaluación. Una configuración
  distinta impide reanudar sobre esa carpeta.
- **11, variables de clasificación:** conserva modos completos y compatibles. Un modo incompleto
  vuelve a empezar; los checkpoints son locales a Colab. El rescate de un fallo de guardado se
  explica debajo.
- **16, CV de Jamaica:** conserva folds completos e íntegros. El progreso del fold en curso es local
  y se pierde al cortar la sesión. Seleccionar solo ese fold en `FOLDS`, usar `REPETIR=True` para
  archivarlo y repetirlo desde cero, y volver después a `False`. Mantener arquitectura, receta,
  entorno y commit entre los cinco folds del mismo modelo.

### Fallo de guardado en el 11

Si el entrenamiento terminó pero falla la escritura a Drive, conservar la sesión y su memoria.
`PENDIENTES_GUARDADO` mantiene el modelo, las predicciones y el historial hasta completar el guardado.
Recuperar el acceso y guardar la variante pendiente, por ejemplo:

```python
rutas = entorno.preparar()
guardar_pendiente("sin_aug")
```

Usar el nombre de la variante que falló. Después, el bucle continúa con `REPETIR=False` y conserva
los modos completos. El rescate se pierde al reiniciar Colab. La selección de mejores pesos usa
memoria; los checkpoints se escriben localmente y se copian a Drive al guardar la corrida.

## Analizar sin volver a entrenar

- En el **11**, usar `ENTRENAR=False`.
- En el **12**, usar `ENTRENAR=False` y `MEDIR_INFERENCIA=False`.
- En el **16**, usar `ENTRENAR=False`: abre Drive en solo lectura y muestra qué modelos tienen
  sus cinco folds completos. Las particiones y los archivos se verifican antes de reunir predicciones.
- El **14** inspecciona errores y cajas guardadas; el **15** compara las corridas sobre val y Jamaica.
  Ninguno necesita cargar pesos ni volver a entrenar.

Las métricas de val también se usaron para elegir checkpoints y son optimistas. Las diferencias
pareadas y los intervalos del 15 y del 16 permiten expresar incertidumbre; una diferencia pequeña
no demuestra superioridad. El 16 documenta además las limitaciones de su CV y del ajuste de umbral.

Fijar el modelo y el umbral antes de ejecutar el **17**, una sola vez al final. Si se hace un
reentrenamiento final como el propuesto en el issue #45, esa evaluación queda para después.
Las decisiones y los resultados reales se registran en [evaluacion_jamaica.md](evaluacion_jamaica.md).
