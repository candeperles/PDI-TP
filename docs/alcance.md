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
matriz de experimentos se definirá en una etapa posterior.

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

## Qué aporta el repositorio actual

- `notebooks/01_exploracion.ipynb`: exploración reproducida sobre la copia local.
- `notebooks/02_validacion_datos.ipynb`: auditoría estructural y visual ejecutada;
  detectó una caja de ancho cero y 37 cajas pequeñas positivas para revisión.
  Ver [validacion_dataset.md](validacion_dataset.md) para resultados y límites.
- `splits/train.txt` y `splits/val.txt`: división compartida de 738 y 162 imágenes.
  El código agrupa por cercanía espacial y busca equilibrar fuente y presencia
  de paneles. La proporción efectiva es 82/18; la consigna no impone una proporción.
- `pdi_tp/`: preparación del entorno, lectura de imágenes y cajas, manejo del split
  y registro de parámetros y métricas en JSON.
- `notebooks/00_plantilla.ipynb`: base para registrar futuros experimentos.
- Los datos están descargados en esta copia y siguen fuera de Git. Cada integrante
  debe descargarlos en su entorno. Las dependencias y el procedimiento de instalación
  están documentados en el README. No hay modelos implementados, pesos ni resultados de entrenamiento.

## Puntos pendientes antes de entrenar

1. **Clases y criterios.** Se cotejó la fuente original: en `solarpanels_hd`, 0/1/2
   son niveles de confianza. La futura fusión a `panel` preservará ese metadato;
   ver [criterios_datos.md](criterios_datos.md). No tratar archivos faltantes,
   corruptos o dudas registradas como negativos confirmados.
2. **Calidad de datos y anotaciones.** Verificar correspondencias imagen/etiqueta,
   dimensiones, coordenadas, tamaños de cajas y duplicados. Revisar visualmente
   clases raras y cajas mínimas. Conservar los originales y documentar cualquier
   transformación antes de aplicarla.
3. **Validación común.** Mantener por ahora el split versionado y comprobar su
   cobertura y separación espacial. No regenerarlo individualmente.
4. **Criterios de evaluación.** Acordar la clase positiva, el promedio de las métricas
   y cómo se elegirá el umbral de confianza al convertir detecciones en clasificación.
   La consigna no especifica esos detalles; todavía no se fijan valores.
5. **Entrega y calendario.** El PDF no fija fecha, extensión del informe, formato de
   presentación ni un requisito explícito de entregar código o pesos. Sí pide
   guardar pesos y resultados durante el trabajo. Confirmar los detalles de entrega
   por el canal de la materia.
6. **Material opcional.** Se menciona `ayuda_rcnn.py`, que no está en el repositorio.
   Solo hace falta localizarlo si se decide explorar Faster R-CNN.

## Siguiente etapa acordada en la conversación

Se implementó la parte ResNet50 del ítem 2 en
`notebooks/04_clasificacion_ResNet50.ipynb`: ImageNet congelado, ajuste fino y pesos
aleatorios, con evaluación y checkpoints. La preparación binaria aplica los criterios
a un manifiesto sin modificar originales. Los entrenamientos todavía deben ejecutarse
en Colab; ver [clasificacion_resnet50.md](clasificacion_resnet50.md).
El repositorio también incluye MobileNetV2; queda ejecutar y comparar ambas arquitecturas.

La preparación de un dataset derivado de detección sigue pendiente. Dos negativos
ambiguos requieren segunda revisión; originales y split permanecen sin cambios.
