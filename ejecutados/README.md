# Resultados visibles sin Drive

Estas copias se pueden abrir en GitHub sin montar Drive ni instalar dependencias.
Son archivos de consulta: el código editable está en [`notebooks/`](../notebooks/).
El [informe](../informe_paneles_solares.html) reúne las comparaciones y conclusiones;
descargar el repo y abrir ese HTML permite usar sus controles y ver las figuras sin conexión.

- Datos, ejecuciones completas: [01 · exploración](01_exploracion.ipynb),
  [02 · validación](02_validacion_datos.ipynb), [03 · revisión](03_revision_anotaciones.ipynb)
  y [04 · marca de agua](04_marca_de_agua.ipynb).
- Entrenamiento, **consulta nueva de corridas guardadas**: [05 · MobileNetV2](05_clasificacion_mobilenetv2.ipynb),
  [06 · ResNet50](06_clasificacion_resnet50.ipynb), [07 · EfficientNetB0](07_clasificacion_efficientnet.ipynb),
  [08 · YOLOv8n](08_deteccion_yolov8n.ipynb), [09 · YOLO11n](09_deteccion_yolo11n.ipynb)
  y [10 · YOLO26n](10_deteccion_yolo26n.ipynb).
- [15 · comparación en validación](15_comparacion_validacion.ipynb), ejecución completa con las 15 corridas.
- Test final, salidas originales conservadas: [18 · ResNet50](18_test_resnet50.ipynb)
  y [18 · YOLO26n](18_test_yolo26n.ipynb). Se quitaron todas las imágenes de las salidas.

En 05–10 se ejecutó únicamente la sección independiente de consulta al final del notebook:
muestra métricas, configuraciones, matrices y curvas reales guardadas. Las celdas de entrenamiento
conservan su código y sus contadores vacíos. **No son los logs originales ni un entrenamiento nuevo.**
Val eligió checkpoints y se usa para reportar; Jamaica (23 imágenes) es un subconjunto de val.

Cada cabecera registra el original, su SHA-256, las celdas ejecutadas, fecha, commit y entorno.
En las sesiones originales del test no quedaron registrados commit, versiones ni fecha de ejecución
en el archivo: se declara esa ausencia y se conserva aparte la cronología documentada.
El commit de referencia es la base Git de publicación; los SHA identifican el código con los cambios
del PR. Los originales 01–04 y 15 se ejecutaron localmente; no se presentan como sesiones de Colab/T4.

Para regenerar una copia:

```bash
python scripts/publicar_ejecutado.py copia_con_salidas.ipynb --origen notebooks/01_exploracion.ipynb
python scripts/publicar_ejecutado.py consulta.ipynb --origen notebooks/05_clasificacion_mobilenetv2.ipynb --tipo consulta_guardada
```

El publicador verifica el código contra el original, agrega la cabecera, quita widgets y HTML activo,
rechaza errores y archivos mayores a 5 MiB. Solo el test admite la variación explícita de `FAMILIA`.
La excepción acordada no permite salidas en los originales ni editar estas copias a mano.
Ver [procedencia y cierre](../docs/entrega.md) y [protocolo](../AGENTS.md#copias-con-salidas-para-la-entrega-issue-54).
