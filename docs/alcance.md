# Alcance del trabajo práctico

Fuente: [Consigna_Paneles_Solares.pdf](Consigna_Paneles_Solares.pdf), páginas 1 a 4.
Este documento resume los requisitos y los puntos por resolver. No implementa modelos.

## Objetivo

Usar el mismo dataset de imágenes aéreas de paneles solares para:

1. Clasificar cada imagen como **con paneles / sin paneles**.
2. Detectar dónde están los paneles mediante cajas.
3. Usar también las detecciones para clasificar: al menos una detección implica
   "con paneles". Comparar ese resultado con los clasificadores sobre las mismas
   imágenes de validación.

Las dos tareas y su comparación forman parte del trabajo.

## Requisitos de la consigna

| Área | Requisito |
|---|---|
| Datos | Imágenes PNG de 256 × 256 píxeles y un TXT de etiquetas YOLO por imagen. El zip original no trae split. |
| Exploración | Analizar cantidades, imágenes y cajas; dibujar ejemplos y registrar observaciones antes de entrenar. |
| Organización | Trabajar en Colab con GPU T4, repartir experimentos y acordar una división común de datos. Guardar resultados y pesos durante el trabajo. |
| Clasificación | Probar al menos dos redes distintas; la consigna lo expresa como "al menos dos o tres". Comparar pesos de ImageNet con pesos aleatorios y, con preentrenamiento, extractor congelado frente a ajuste de parte o de todo el extractor. |
| Detección | Probar YOLOv8n y YOLO11n. Comparar entrenamiento de toda la red frente a congelar una parte. La guía indica `imgsz=256`. |
| Métricas de clasificación | Matriz de confusión, precision, recall, F1 y accuracy. |
| Métricas de detección | mAP50, mAP50-95 y curva precisión-recall. |
| Comparación entre tareas | Calcular las mismas métricas de clasificación para el detector y los clasificadores sobre las mismas imágenes de validación. |
| Entrega | Informe con comparaciones y conclusiones que explique arquitecturas, hiperparámetros y estrategias de entrenamiento. |

La cantidad exacta de combinaciones de experimentos no está especificada. La
matriz elegida está en la tabla de abajo: 3 redes × 3 modos de clasificación y
2 versiones de YOLO × 2 modos de detección, más YOLO26n opcional.

## Opciones y orientaciones

- YOLO26n y Faster R-CNN son opcionales.
- Tamaño de entrada en clasificación, augmentation, épocas y learning rate son
  variables adicionales propuestas; no se exige explorar todas.
- MobileNetV2, EfficientNetB0, ResNet50 y VGG16 son ejemplos de clasificadores,
  no una lista de arquitecturas obligatorias.
- La consigna ofrece una guía con Keras para clasificación y Ultralytics para
  YOLO; no formula una obligación general de usar un único framework.
- Para detección se explica el uso de pesos aleatorios, pero no se exige de
  forma explícita la misma comparación con/sin preentrenamiento que para clasificación.

## Cómo cubre el repositorio cada requisito

Revisado contra el código de `main` el **9/11/2026**. La tabla dice dónde está implementado cada
requisito; si un modo está entrenado se confirma con las corridas guardadas en Drive (notebook 15), no
con esta tabla.

| Requisito | Dónde |
|---|---|
| Exploración y dibujo de cajas | `01_exploracion`; validación en `02_validacion_datos` ([resultados](validacion_dataset.md)) y casos dudosos en `03_revision_anotaciones` ([resultados](revision_anotaciones.md)) |
| Split común | `splits/train.txt` y `splits/val.txt`: 738/162 (82/18 efectivo), agrupado por zona espacial y equilibrado por fuente y presencia de paneles |
| Clasificación: tres redes, ImageNet contra aleatorio, congelado contra fine-tuning | `05`–`07` (MobileNetV2, ResNet50, EfficientNetB0) con la receta `comun_v2`, modos `imagenet_congelado`, `imagenet_finetuning` y `aleatorio` |
| Otras variables de clasificación | `11_variables_clasificacion` (augmentation, pesos de la loss, LR y early stopping); tamaño de entrada de detección en `12_deteccion_yolov8n_256` |
| Detección: YOLOv8n y YOLO11n, red completa contra congelar una parte | `08`–`09`, modos `coco_completo` y `coco_backbone_congelado`; YOLO26n opcional en `10` |
| Métricas de clasificación | `resultados.guardar_corrida`: matriz de confusión, accuracy, precision, recall y F1, más F1 macro, balanced accuracy, recall sin paneles y AUC |
| Métricas de detección | mAP50, mAP50-95 y curva precisión-recall de Ultralytics en `08`–`10` (`deteccion_val/`) |
| Detector como clasificador | `08`–`10` guardan la máxima confianza por imagen en el formato común; `15_comparacion_validacion` lo compara con las CNN sobre las mismas 162 imágenes |
| Informe | Pendiente |

Además de la consigna: diagnóstico de la marca de agua de Google Maps (`04` y `13`), galería de errores
(`14`), validación cruzada sobre Jamaica (`16`) y test propio de Kingston (`17`).

## Decisiones tomadas

1. **Clases.** En `solarpanels_hd`, 0/1/2 son niveles de confianza de la anotación: se usa una sola clase
   `panel` y la confianza original se conserva como metadato. Ver [criterios_datos.md](criterios_datos.md).
2. **Datos.** No se modifican los originales. Se excluye solo la caja Z01 de ancho cero y se conservan las
   cajas pequeñas. N06 fue revisado por el equipo como sin paneles; N16 sigue pendiente de segunda revisión.
3. **Clase positiva y umbrales.** Positiva = con paneles. Umbral 0.5 para las CNN y 0.25 sobre la máxima
   confianza de panel para los detectores; elegir otro umbral con las 23 imágenes de Jamaica de val no
   mejoró ([evaluación sobre Jamaica](evaluacion_jamaica.md)).
4. **Comparación.** Val se ordena por F1 macro y Jamaica por balanced accuracy, con AUC para comparar CNN y
   detectores sin depender del umbral. Val también eligió los checkpoints: sus métricas son optimistas.
5. **Resolución de detección.** Los YOLO se entrenan a 640 px aunque la consigna sugiere `imgsz=256`; el
   notebook `12` es el control a 256 px que tiene que justificar esa decisión en el informe.

## Pendiente para la entrega

1. **Informe** con arquitecturas, hiperparámetros, estrategias de entrenamiento, comparaciones y conclusiones.
2. **Confirmar las corridas en Drive** con el notebook 15: 9 modos de clasificación con `comun_v2` y 4 de
   detección a 640 px.
3. **Fecha, extensión y formato de la entrega:** el PDF no los fija. Confirmarlos por el canal de la materia.
4. **Faster R-CNN** (opcional) no se implementó; el `ayuda_rcnn.py` que menciona la consigna no está en el repo.
