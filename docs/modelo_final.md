# Modelo final (issue #45)

## Elección actualizada con la CV completa (09/11/2026)

Con los cinco folds de cada uno de los tres finalistas, se recomienda **YOLO26n con umbral 0.25**
para presencia de paneles y como detector. Se conserva **ResNet50 con umbral 0.5** como CNN
para comparar los dos enfoques de la consigna. El análisis completo y sus fuentes están en
[`evaluacion_jamaica.md`](evaluacion_jamaica.md#resultado-de-cv-y-decisión-09112026).

La comparación se recalculó leyendo los registros de Drive y los originales del train/val,
en la rama de análisis basada en `main` `eda1da1`. **El test no se cargó ni se usó para elegir.**

## Regla aplicada

1. Finalistas del #43 elegidos en val: ResNet50 con fine-tuning, YOLO11n completo y YOLO26n
   completo; MobileNetV2 con fine-tuning era opcional y no se corrió.
2. Comparar balanced accuracy de las 172 predicciones fuera de fold, a 0.5 en CNN y 0.25 en YOLO.
3. Si el intervalo pareado entre los detectores incluye cero, elegir el de menor **mediana de
   segundos por fold**, como implementa `17_modelo_final`.
4. Cambiar el umbral solo si la mejora estimada por grupos es clara frente a su referencia.

## Modelos y umbrales recomendados

| | CNN | Detector usado como clasificador |
|---|---|---|
| Modelo | `resnet50/imagenet_finetuning` | `yolo26n/coco_completo` |
| Motivo | único finalista CNN ejecutado | diferencia no concluyente frente a YOLO11n; menor mediana de tiempo observada |
| Umbral | 0.5 | 0.25 |
| Balanced accuracy CV [IC 95 %] | 0.8377 [0.7781, 0.8914] | 0.9250 [0.8855, 0.9654] |
| F1 macro CV | 0.8283 | 0.9270 |
| AUC CV | 0.9192 | 0.9724 |
| Mediana de segundos por fold | 411.22 | 699.16 |
| Mejores épocas, mediana | cabeza 11, fine-tuning 10 | 37 |

YOLO26n − YOLO11n tiene una diferencia de balanced accuracy de **+0.0435**, IC 95 %
**[−0.0058, +0.0962]**. La elección de detector aplica el desempate por tiempo observado:
699.16 s frente a 721.30 s. Esa pequeña diferencia no prueba una ventaja general de velocidad
ni sustituye una medición de latencia de inferencia.

Frente a ResNet50, YOLO26n mejora balanced accuracy en **+0.0873**, IC **[+0.0372, +0.1394]**.
La sensibilidad remuestreando zonas completas mantiene la conclusión. Son resultados de esta CV,
con finalistas elegidos en val y checkpoints seleccionados con val no jamaicano; no son resultados
del test independiente.

## Por qué no bajar el umbral a 0.15

En YOLO26n, 0.15 maximiza el resultado observado de la grilla y la estimación por grupos da
balanced accuracy **0.9394** frente a **0.9250** con 0.25. La mejora es **+0.0144**,
IC **[−0.0160, +0.0467]**: no es clara. Se conserva **0.25**, conforme a la regla previa.
Con 0.15 se pierden 3 imágenes con paneles y se generan 8 falsas alarmas; con 0.25, 7 y 5.
No se afirma haber encontrado un umbral óptimo universal.
El umbral es el de presencia de paneles sobre la máxima confianza de las cajas; no modifica
el parámetro `conf` de inferencia que conserva las cajas usadas para calcular mAP.

ResNet50 también conserva 0.5: su candidato 0.35 no produce una mejora clara. YOLO11n ya
elige 0.25. Las limitaciones de `*_cv`, que no es una CV anidada completa, están en el análisis.

## Selección anterior y pasos pendientes

La elección inicial del 09/11, hecha con `17_modelo_final` sobre `main` `585d7a9`, seleccionó
YOLO11n cuando YOLO26n solo tenía **1 de 5 folds**. Se registró la excepción de elegir al único
finalista completo, sin rival en la CV. Esa situación ya quedó superada; esta actualización
reemplaza aquella recomendación de detector con las corridas completas.

Este análisis **no reentrena modelos ni ejecuta el test** y no modifica archivos de Drive.
Para aplicar la elección, revisar el resultado actualizado de `17_modelo_final` y usar en el
notebook 18, antes de cualquier evaluación final:

```python
MODELOS_FINALES = {
    "clasificadores": "resnet50/imagenet_finetuning",
    "yolo26n": "yolo26n/coco_completo",
}
```

Para evaluar el detector seleccionado, usar **`FAMILIA = "yolo26n"`**: instala Ultralytics 8.4.184.
La opción `"detectores"` corresponde a YOLOv8n/YOLO11n y no sirve para abrir correctamente YOLO26n.

El notebook 18 conserva en el código la selección anterior de YOLO11n; se deja explícito el
cambio necesario para la revisión del equipo. No se ha cambiado ni ejecutado ese notebook aquí.
Los umbrales de las corridas comunes ya son 0.5 y 0.25, respectivamente.

Se mantiene el registro histórico: el 06/11 se hizo una corrida preliminar del test con las
corridas vigentes. Las decisiones de esta actualización salen solo de la CV. Si se concreta
la evaluación final, fijar antes modelos y umbrales, ejecutarla una sola vez y reportar el
resultado tal cual. El reentrenamiento final y la alternativa de usar modelos entrenados solo
con train pertenecen al #45; completar CV no significa haber reentrenado con train + val.
