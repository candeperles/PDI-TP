# Entrega visible sin acceso a Drive — issue #54

El [informe HTML](../informe_paneles_solares.html) y las
[13 copias con salidas](../ejecutados/README.md) permiten revisar la entrega desde GitHub.
Para usar el informe interactivo, descargar el repo y abrir el HTML junto con `docs/`.
Las copias `.ipynb` muestran sus salidas directamente en GitHub.

## Alcance y procedencia

- **01–04:** ejecuciones locales completas, con los 900 pares originales y el split fijo 738/162.
  El 03 detectó seis campos `fuente` incorrectos en `revision_anotaciones.csv`: se corrigió
  `kingston` a `kingston2` según el nombre del archivo. No cambiaron observaciones, decisiones,
  etiquetas ni particiones. Las galerías del 03 se renderizan en JPEG para cumplir el límite de tamaño.
- **05–10:** consultas nuevas de las corridas guardadas del formato común. Se recuperaron pesos
  auténticos, configuraciones, métricas, predicciones e historiales; no se entrenó ni se volvió a predecir.
  La sección independiente exige los modos completos y abre los resultados en solo lectura.
  Los logs originales no se conservaban: las cabeceras distinguen esta consulta de un entrenamiento.
- **15:** ejecución completa de comparación, con nueve CNN de receta `comun_v2` y seis detectores
  a 640 px. Val eligió checkpoints y se usa para reportar; Jamaica son 23 de esas 162 imágenes.
- **18:** copias de las salidas originales de ResNet50 y YOLO26n, sin imágenes del test ni widgets.
  ResNet50 conserva `FAMILIA="clasificadores"`; solo se omitió su celda adicional de descarga de CSV.
  El publicador registra esta preparación y el SHA del archivo recibido. Las tablas quedan visibles.
  Los archivos originales no registraban fecha, commit y versiones de la sesión: no se reconstruyen
  a partir de la fecha del archivo. La cronología, los hashes de los CSV y los límites de trazabilidad
  están en [modelo_final.md](modelo_final.md#fuentes-alcance-y-cierre).

Las cabeceras de las ejecuciones locales identifican su entorno real y el SHA del notebook limpio.
El commit de referencia es la base Git; el código incorpora los cambios de este PR sin commitear
al momento de ejecutar. La ejecución local de consulta no cambia el commit original de entrenamiento
que informa cada `config.json`.

El equipo aprobó la excepción de `ejecutados/` propuesta en
[el comentario del #54](https://github.com/candeperles/PDI-TP/issues/54#issuecomment-6744649936).
La regla sin salidas sigue vigente para `notebooks/`. CI valida ambas carpetas: fuente idéntica,
procedencia, tipo de ejecución, límite de 5 MiB, ausencia de errores, widgets, formatos activos
e imágenes en las salidas del test. Un cambio en el original invalida su copia hasta regenerarla.

## Informe y selección final

Se cotejaron las matrices y el AUC de las 15 corridas y las siete variantes de MobileNetV2 con sus
CSV y las 162 etiquetas del split fijo. Se incorporaron las dos corridas de YOLOv8n a 640 px
(commit `cdf8606b88af92f0ce5f95c089301b3793646970`) y las siete variantes recuperadas completas
tras el problema de persistencia del #44. El informe incluye las seis curvas PR originales de
`deteccion_val/`, matrices de presencia, comparaciones, CV completa y tabla final de test.

**Los finalistas siguen siendo ResNet50 con fine-tuning y YOLO26n completo**, con umbrales 0,5 y 0,25.
Por acuerdo al cerrar #54, se actualiza su referencia antigua a `17_evaluacion_test` y a YOLOv8n:
el notebook de test actual es el **18** y se publican los finalistas ya fijados, sin otra evaluación.
YOLOv8n se agrega como referencia de validación; no tiene cinco folds de CV ni test final.
No se eligieron modelos después de mirar el test. La recuperación de las salidas de ResNet50
requirió repetir esa inferencia, como ya documenta el cierre de evaluación; no se afirma que hubo
una única ejecución técnica ni se oculta la evaluación preliminar del 6/11.

El control YOLOv8n a 256 px y el análisis numérico del efecto de la marca de agua siguen siendo
limitaciones explícitas del informe. No bloquean la publicación de los resultados ya obtenidos
del #54 ni se presentan como estudios ejecutados.

## Orden de Drive

- Se archivó `resultados/resnet50/imagenet_cabeza` como
  `imagenet_cabeza__anterior_20261109_234129` (hora de Argentina). Se preservaron su ID
  `1KcFhNHOLZNPwY5aOhgSP0y33Ui4SahX7`, ubicación, contenido y permisos; se verificó el nombre
  nuevo en la carpeta de ResNet50. No se borraron archivos.
- El #44 ya había archivado sus 105 carpetas duplicadas en
  `resultados/mobilenetv2/_duplicados_20261106/`. Se verificó que no hay carpetas `(N)` nuevas
  en la raíz activa y que las siete variantes tienen sus pesos y archivos comunes.
  Los lectores excluyen los modos `__anterior_…`.

## Archivos de respaldo y revisión

[fuentes_entrega.json](fuentes_entrega.json) registra IDs, enlaces originales, tamaños y SHA-256
de los archivos consultados; no contiene enlaces temporales de descarga ni credenciales.
No se versionan pesos, dataset, capturas del test ni la carpeta local de resultados.

Este cambio requiere **revisión de otra persona antes del merge** porque toca código compartido,
el verificador y notebooks de varios integrantes. Los chequeos automáticos no sustituyen esa revisión.
