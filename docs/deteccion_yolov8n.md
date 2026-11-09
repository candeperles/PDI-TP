# Detección con YOLOv8n

Notebook: `notebooks/08_deteccion_yolov8n.ipynb`, para Colab con GPU T4. La misma receta vale para
`09_deteccion_yolo11n` (con `freeze=11`) y `10_deteccion_yolo26n` (ver [YOLO26n](#yolo26n)).

**Estado de las corridas:** en la evaluación del test del 6/11, las corridas de YOLOv8n en Drive todavía
eran de **256 px** y las de YOLO11n de 640 px. El notebook 15 excluye automáticamente los detectores sin
`params.receta.imgsz = 640`: para que YOLOv8n entre en la comparación, correr el 08 con la receta actual.

## Datos y variantes

Se utilizan únicamente las 900 imágenes existentes y el split fijo 738/162. La copia
derivada local convierte las imágenes a RGB y las anotaciones a una sola clase `panel`.
Se aplica la auditoría compartida: excluir solo Z01/línea 1, conservar las cajas pequeñas
y guardar la clase y confianza originales en `cajas_derivadas.csv`. Los negativos mantienen
su etiqueta vacía; N06 fue revisado por el equipo como sin paneles y N16 sigue con referencia
provisional. No se agregan imágenes ni
se modifican originales o splits.

Las dos variantes parten independientemente de `yolov8n.pt` (COCO):

| Modo | Qué se entrena |
|---|---|
| `coco_completo` | Todos los módulos entrenables (`freeze=0`) |
| `coco_backbone_congelado` | Se congelan los módulos 0–9 del backbone (`freeze=10`) |

Ultralytics mantiene fija la proyección DFL en ambas variantes. Se usa la misma receta:
640 px (las imágenes son de 256 y Ultralytics las agranda; hasta el 2/11 se usó 256), batch 16, máximo 50 épocas, paciencia 10 y AdamW con LR inicial 0.001.
Los parámetros explícitos están en `RECETA` y se registran en `config.json`; la receta
efectiva de Ultralytics queda en `entrenamiento/args.yaml`.
Es un primer experimento; no implica que estos hiperparámetros sean óptimos.

## Ejecutar y recuperar

1. Abrir el notebook en una sesión nueva de Colab con GPU T4 y ejecutar en orden.
2. Autorizar Drive mediante `entorno.preparar()`. Tener espacio para checkpoints y gráficas.
3. Mantener `REPETIR=False`: salta variantes completas y recupera una incompleta desde
   `entrenamiento/weights/last.pt`, hasta la última época guardada. Si el entrenamiento
   terminó, reutiliza `best.pt` para completar evaluación y exportación.
4. Si se cambia la receta, usar `REPETIR=True`: archiva cada variante anterior sin borrarla.
   Si falta un checkpoint, el notebook frena para evitar sobrescribir un entrenamiento.

El derivado local usa una ruta estable para permitir reanudar desde otra sesión; se verifica
su integridad antes de reutilizarlo. Si quedó incompleto, elegir otro destino local y revisar
los archivos antes de reanudar. No se cambia automáticamente la receta de una corrida.

Entorno separado de TensorFlow: `requirements-detection.txt`, con Ultralytics 8.3.221.
Las versiones efectivas de PyTorch, torchvision y del resto del entorno se registran por corrida.

## Resultados y límites

Cada variante queda en `Mi unidad/PDI-TP/resultados/yolov8n/<modo>/`:

- `modelo.pt`, copia del mejor checkpoint; `best.pt` y `last.pt` dentro de `entrenamiento/weights/`.
- `metricas_deteccion.json`: mAP50, mAP50-95, precision y recall sobre todo val y Jamaica.
- `deteccion_val/` y `deteccion_jamaica/`: curvas PR, matrices y ejemplos de cajas.
- `cajas_val.csv`: cajas normalizadas y confianza tras NMS para las imágenes con detecciones.
- Formato común: `config.json`, `metricas.json` y `predicciones_val.csv`, una fila para cada
  imagen de val, incluidos los negativos sin cajas.

El detector se usa como clasificador tomando la máxima confianza de panel por imagen y 0 si
no hay cajas. Ese puntaje **no es una probabilidad calibrada**, aunque se guarda en la columna
común `probabilidad`. La decisión inicial usa umbral 0.25, sin optimización en este notebook.
La inferencia y la evaluación de detección usan confianza mínima 0.001, NMS IoU 0.7 y máximo
300 cajas; se registran esos límites. El umbral mínimo no se confunde con el de clasificación.

Las métricas binarias, incluidas las de Jamaica, las calcula `resultados.guardar_corrida`.
La matriz de objetos/fondo de Ultralytics es distinta de la matriz binaria por imagen.
La precision/recall de detección de Ultralytics corresponde al punto elegido por su curva F1,
no necesariamente al umbral 0.25 usado para clasificar imágenes.

**Val selecciona el checkpoint y también reporta resultados: no es un test independiente.**
Jamaica tiene solo 23 imágenes de val (11 positivas, 12 negativas), de modo que diferencias
pequeñas no permiten concluir superioridad. No se comparan recetas distintas como si solo
cambiara la arquitectura. Ver también [evaluación de Jamaica](evaluacion_jamaica.md).

## YOLO26n

`notebooks/10_deteccion_yolo26n.ipynb` usa la misma receta y los mismos dos modos, con `yolo26n.pt` y `freeze=11`
(backbone: módulos 0–10, como YOLO11n). Cambia lo siguiente:

- **Entorno:** `requirements-yolo26.txt` (Ultralytics 8.4.184), porque 8.3.221 no incluye YOLO26. Frente al 08 y el 09
  cambia la arquitectura y también la versión de la librería. Un `modelo.pt` de YOLO26n se abre solo con ese entorno:
  8.3.221 lo carga sin error, pero predice otras cajas.
- **Dos cabezas:** el entrenamiento (`E2ELoss`) ajusta la one-to-many y la one-to-one. Con `nms=None`, el default de
  esta versión, validación e inferencia usan la one-to-many con NMS, como YOLOv8n y YOLO11n. Esa es la métrica
  principal, y con ella se elige `best.pt` y se escriben `cajas_val.csv` y `predicciones_val.csv`.
- **Sin NMS:** el notebook mide además `nms=False` sobre el mismo `modelo.pt`, en `extra.deteccion_sin_nms` de
  `metricas.json`, con carpetas `deteccion_val_sin_nms/` y `deteccion_jamaica_sin_nms/`. Ahí `iou` no tiene efecto. No
  entra en la tabla de `15_comparacion_validacion`.
- **Sin DFL** (`reg_max: 1`): `dfl=1.5` es la ganancia de una pérdida L1 de las cajas. Es el mismo valor de la receta,
  con otro significado.

Referencias: [YOLOv8](https://docs.ultralytics.com/models/yolov8/),
[entrenamiento](https://docs.ultralytics.com/modes/train/),
[validación](https://docs.ultralytics.com/modes/val/),
[código fijado de entrenamiento](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/engine/trainer.py).
