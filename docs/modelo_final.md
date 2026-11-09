# Modelo final (issue #45)

Elección hecha con [`17_modelo_final`](../notebooks/17_modelo_final.ipynb) el 9/11/2026 sobre `main` en
`585d7a9`, leyendo Drive en solo lectura y **sin cargar el test**. `18_evaluacion_test` toma estos modelos y
umbrales (`MODELOS_FINALES`) y se corre una sola vez.

## Regla

La regla es la del notebook 16, fijada antes de correr la CV:

1. Finalistas del #43 (`cv_jamaica.FINALISTAS`), elegidos en val: ResNet50 con fine-tuning (MobileNetV2 con
   fine-tuning era opcional), YOLO11n completo y YOLO26n completo.
2. Por familia, entre los finalistas con los cinco folds, gana el de mayor balanced accuracy en la CV de Jamaica,
   con el umbral actual.
3. Si la diferencia pareada entre los dos primeros no es clara, gana el de menor costo de entrenamiento medido.
4. El umbral cambia solo si la estimación por grupos mejora claramente al actual.

**Excepción, agregada el 9/11:** el 16 pedía no declarar ganador si faltaba un candidato. Sin GPU para
completar YOLO26n, se elige igual al único finalista completo y queda marcado como *sin rival en la CV*.

## Elección

| | CNN | Detector usado como clasificador |
|---|---|---|
| Modelo final | `resnet50/imagenet_finetuning` | `yolo11n/coco_completo` |
| Motivo | único finalista con los cinco folds (MobileNetV2, opcional, no se corrió) | único finalista con los cinco folds; **sin rival en la CV**: YOLO26n tiene 1 de 5 folds |
| Umbral | 0.5 (sin cambio) | 0.25 (sin cambio) |
| Balanced accuracy en la CV (172 imágenes de Jamaica) | 0.838 [0.778, 0.891] | 0.882 [0.829, 0.931] |
| F1 macro en val (162) | 0.848 [0.757, 0.921] | 0.943 [0.881, 0.988] |
| Épocas, mediana de los folds | cabeza 11, fine-tuning 10 | 48 |

- **Umbral:** con todas las imágenes de Jamaica, ResNet50 elegiría 0.35. La estimación por grupos mejora la
  balanced accuracy en +0.014 [−0.017, +0.047]. No es clara, así que el umbral queda en 0.5. En YOLO11n el
  umbral elegido es el mismo, 0.25.
- **Detector contra CNN, en la CV:** la única diferencia clara es el recall sin paneles, a favor del detector:
  +0.146 [0.069, 0.230]. En AUC (+0.008), balanced accuracy (+0.044) y F1 macro (+0.060), los intervalos
  incluyen el 0.
- **En val**, YOLO26n completo queda muy cerca de YOLO11n: F1 macro 0.930 contra 0.943, y mejor mAP50 (0.883
  contra 0.809). No se puede decir cuál es mejor sin su CV.

## Para el informe

- Val eligió los finalistas y los checkpoints, y la CV de Jamaica decidió entre ellos. En la práctica, cada
  familia tuvo un solo finalista completo, así que la CV **confirma** la elección de val, pero no la decide.
- El 6/11 se corrió una versión preliminar del test con todas las corridas vigentes. La excepción de la regla es
  posterior a esa corrida.
- Los modelos se entrenaron solo con train (738 imágenes). No hubo reentrenamiento con train + val por falta de
  GPU: es la opción "si no da el tiempo" del #45.
