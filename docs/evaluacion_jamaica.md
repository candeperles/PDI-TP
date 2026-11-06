# Evaluación sobre Jamaica (propuesta)

El test va a ser solo de Kingston, con capturas tipo Google Maps y mayoría de imágenes sin paneles (issue #7).
Evalúa **clasificación y detección**, con la misma entrada que nuestro dataset (256×256, mismo zoom): no hay
que cambiar el preprocesamiento. Este documento propone cómo medir eso con lo que ya tenemos. **Es una propuesta para discutir en equipo:**
no cambia el split fijo.

## Qué imágenes cuentan como Jamaica

`split.es_jamaica(nombre)`: `kingston_*`, `kingston2_*` y las `geo` con coordenadas dentro de Jamaica (hoy
son todas). Son 172 imágenes en 39 grupos espaciales:

| | Imágenes | Con paneles | Sin paneles | Grupos espaciales |
|---|---|---|---|---|
| train | 149 | 58 | 91 | 30 |
| val | 23 | 11 | 12 | 9 |

De los 39 grupos, 18 tienen al menos una imagen con paneles, 21 no tienen ninguna y 8 mezclan las dos.

## Paso 1 (ya implementado): métricas de val solo sobre Jamaica

`resultados.guardar_corrida` guarda en `metricas.json` la clave `jamaica` con las mismas métricas que el
total, calculadas solo sobre las 23 imágenes de Jamaica de val. `resultados.cargar_corridas` agrega las
columnas `jamaica.<métrica>` y las recalcula desde `predicciones_val.csv`, así que también aparecen en las
corridas guardadas antes del cambio, sin volver a entrenar.

**Limitación:** con 11 imágenes con paneles y 12 sin paneles, una sola imagen mueve el recall de cada clase
entre 8 y 9 puntos. Sirve para detectar problemas grandes, como el 59 % del issue #7, no para elegir entre
modelos parecidos.

## Paso 2: validación cruzada agrupada sobre Jamaica (issue #43)

Para medir con las 172 imágenes y no solo con 23:

1. Repartir las imágenes de Jamaica (train y val) en 5 folds con `StratifiedGroupKFold`, agrupando por zona
   espacial (`split.grupos_espaciales`) y estratificando por con/sin paneles. Así ninguna zona queda partida
   entre entrenamiento y evaluación.
2. En cada fold, entrenar con todo el train que no es de Jamaica más las imágenes de Jamaica de los otros
   4 folds, y predecir el fold que queda afuera.
3. Juntar las predicciones de los 5 folds: queda una predicción por cada una de las 172 imágenes, sin que
   ningún modelo haya visto la imagen ni sus vecinas.
4. Reportar recall con paneles, recall sin paneles y F1 macro, más el umbral elegido sobre esas predicciones
   (tarea de @Sciordia11 en el issue #7). Para esto alcanza con pasarle esas predicciones a
   `umbral.tabla_jamaica` (ver "Umbral de decisión" abajo).

**Costo:** 5 entrenamientos por configuración. Para los linear probes del notebook extra son segundos; para
las CNN en Colab conviene hacerlo solo con la receta común y las variantes finalistas, no con todas.

**Qué no cambia:** el split oficial sigue igual y las métricas de val del formato común se siguen
reportando como hasta ahora. Esto es una evaluación adicional.

### Particiones implementadas

`split.folds_jamaica(rutas, n=5, semilla=42)` audita los datos y devuelve una lista de cinco
diccionarios. Cada uno contiene `fold` (1 a 5) y tres tablas con las columnas del manifiesto:

- `train`: todo el train oficial no jamaicano más las imágenes de Jamaica de los otros cuatro folds.
- `validacion`: solo el val oficial **no jamaicano**, para elegir checkpoints y aplicar early stopping.
- `evaluacion`: el fold de Jamaica que queda afuera, para obtener las predicciones fuera de fold.

La agrupación espacial se calcula sobre todos los nombres oficiales juntos, antes de separar las
particiones, para conservar las cadenas de vecinos. La función frena si falta una clase, hay grupos
compartidos entre particiones, etiquetas pendientes en Jamaica o diferencias con los archivos del split.
No admite imágenes de un test externo ni escribe `splits/train.txt` o `splits/val.txt`.

El fold de `evaluacion` **no se pasa como validación al entrenamiento**: usarlo para elegir el mejor
checkpoint haría optimistas sus predicciones, aunque sus imágenes nunca entrenaran por gradiente.
Reservar el val no jamaicano permite entrenar con todas las imágenes de Jamaica de los otros cuatro
folds, como pide el issue, conservando los hiperparámetros y el criterio de checkpoint de cada receta.
Es una limitación del diseño: ese conjunto selecciona épocas por rendimiento fuera de Jamaica.
El val común se usó para elegir los finalistas y también para reportar; la CV adicional no elimina
por completo ese sesgo de selección.

**Estado:** particiones implementadas; entrenamiento, predicciones fuera de fold y elección final
pendientes. Todavía no hay resultados de esta validación cruzada ni un cambio de modelo o umbral.
El notebook y su guardado en `analisis/cv_jamaica/` requieren acordar el flujo con las reglas del
verificador, que actualmente exige el formato común a todo notebook que entrena.

La comprobación local del 06/11/2026 se detuvo en **N06**, una imagen de Jamaica cuya etiqueta sigue
pendiente de segunda revisión. Antes estaba solo en val; al reunir Jamaica de train y val para CV
entraría al entrenamiento de cuatro folds. No se transforma su referencia provisional en una etiqueta
confirmada: hay que resolver esa revisión o acordar y documentar otro tratamiento antes de entrenar.
N16 no es de Jamaica y permanece en la validación no jamaicana, con su referencia provisional.

## Umbral de decisión (implementado)

Train tiene 85 % con paneles y el test va a tener mayoría sin paneles: 0.5 no tiene por qué ser el mejor
umbral para Kingston. `pdi_tp/umbral.py` elige el umbral para Jamaica y mide si mejora algo:

- **Criterio: balanced accuracy** (promedio del acierto de cada clase). No depende de la proporción de clases,
  así que sirve aunque el test tenga otra proporción que val. La F1 macro sí depende: se reporta pero no se
  usa para elegir. Los candidatos van de 0.05 a 0.95 de a 0.05; si empatan varios, se toma el medio del tramo.
- **Estimación honesta:** elegir el umbral y medir sobre las mismas imágenes da un número optimista. Por eso
  se deja afuera un grupo espacial por vez, se elige el umbral con el resto y se aplica al grupo que quedó
  afuera (columnas `*_cv`). El umbral para el test (`umbral`) se elige con todas las imágenes.
- **Dónde se ve:** `umbral.tabla_jamaica(resultados.cargar_predicciones(rutas))` da una fila por modelo/modo
  de todo el equipo. La sección *Kingston* del notebook `04_clasificacion_mobilenetv2` la muestra.
- **No cambia las corridas:** el umbral guardado en `metricas.json` sigue siendo 0.5. Es una evaluación
  adicional.

Hoy usa las 23 imágenes de Jamaica de val (9 grupos espaciales), así que es ruidosa. Cuando existan las
predicciones fuera de fold del paso 2, se le pasan esas y mide con las 172.

### Resultado (27/10/2026): se deja 0.5

Balanced accuracy sobre las 23 imágenes de Jamaica de val (11 con paneles, 12 sin), con las corridas
guardadas en Drive a esa fecha. Val también se usó para elegir época y modo, así que todo es algo optimista.

| Corrida | Acierto con / sin (0.5) | Bal. acc. 0.5 | Bal. acc. umbral elegido (honesta) | Umbral según el grupo |
|---|---|---|---|---|
| resnet50 / imagenet_cabeza | 0.73 / 1.00 | 0.864 | 0.818 | 0.53 – 0.70 |
| mobilenetv2 / imagenet_fine_tuning | 0.73 / 0.92 | 0.822 | 0.689 | 0.25 – 0.80 |
| resnet50 / imagenet_finetuning | 0.73 / 0.83 | 0.780 | 0.780 | 0.30 – 0.78 |
| mobilenetv2 / imagenet_congelado | 0.73 / 0.83 | 0.780 | 0.602 | 0.23 – 0.83 |
| mobilenetv2 / imagenet_ft_completo | 0.64 / 0.83 | 0.735 | 0.644 | 0.18 – 0.75 |
| efficientnetb0 / imagenet_congelado | 0.64 / 0.83 | 0.735 | 0.481 | 0.10 – 0.78 |
| efficientnetb0 / imagenet_finetuning | 0.55 / 0.92 | 0.731 | 0.470 | 0.15 – 0.78 |
| resnet50 / aleatorio | 0.45 / 0.67 | 0.561 | 0.564 | 0.35 – 0.58 |
| baseline / siempre_con_paneles | 1.00 / 0.00 | 0.500 | 0.500 | 0.50 |
| efficientnetb0 / aleatorio | 1.00 / 0.00 | 0.500 | 0.447 | 0.35 – 0.80 |

- **En Jamaica los modelos se pierden paneles:** con 0.5 aciertan casi todas las imágenes sin paneles y entre
  el 45 % y el 73 % de las que tienen. Coincide con el diagnóstico del issue #7 ("parece Kingston → no tiene
  paneles").
- **Elegir el umbral no mejora:** la estimación honesta es igual o peor que 0.5 en todas las corridas
  (salvo resnet50/aleatorio, que es ruido), y el umbral elegido cambia mucho según el grupo que queda afuera.
  Con 23 imágenes, el umbral "óptimo" se ajusta a esas imágenes y no generaliza.
- **Ningún modelo se separa claramente:** una imagen mueve la balanced accuracy unos 0.04, que es la
  diferencia entre los dos primeros.
- **Decisión:** se usa 0.5. El problema de Kingston se ataca con el entrenamiento (balanceo por fuente y más
  capturas de Kingston), no con el umbral. Revisar el umbral cuando haya predicciones sobre las 172 imágenes
  o un test propio de Kingston, y después del balanceo por fuente, que cambia las probabilidades.
- Las corridas de EfficientNet pueden ser anteriores a los arreglos de fine-tuning, pesos de clase y
  BatchNorm: revisar la fecha en su `config.json` antes de sacar conclusiones sobre ese modelo.

## Paso 3 (cuando lleguen las capturas nuevas): test propio de Kingston

Las capturas nuevas de Kingston (tarea de @PiaAndreuccetti) dan la mejor estimación del test real si se
separan como **test propio que no se usa ni para entrenar ni para elegir umbral o modelo**, con una
proporción de clases parecida a la que esperamos en el test (mayoría sin paneles). Se evalúa una sola vez,
al final.

## Detección

Lo de arriba vale igual para el detector, con dos diferencias:

- **Pocas cajas de Kingston:** 113 en 58 imágenes de train y 21 en 11 imágenes de val. Un mAP solo sobre
  Jamaica de val es todavía más ruidoso que las métricas de clasificación.
- **Las imágenes sin paneles de Jamaica** (`kingston2_*` y la mayoría de `geo`) entran al entrenamiento como
  imágenes de fondo y ayudan a bajar los falsos positivos, que en un test con mayoría sin paneles pesan mucho.

Qué tiene que reportar el notebook de detección (todavía no existe):

1. mAP50 y mAP50-95 sobre todo val y **también solo sobre las imágenes de Jamaica de val** (por ejemplo, con
   un segundo `data.yaml` que liste solo esas imágenes, elegidas con `split.es_jamaica`).
2. El detector usado como clasificador se guarda con `resultados.guardar_corrida`, así que ya trae las
   métricas `jamaica` sin hacer nada extra.
3. Para la validación cruzada del paso 2 y el test propio del paso 3, las capturas nuevas tienen que venir
   **con cajas** en formato YOLO, no solo con la etiqueta con/sin paneles.

## Para decidir en equipo

- ¿Adoptamos el paso 2 para la comparación final de clasificadores?
- ¿Cuántas de las capturas nuevas van a test propio y cuántas a entrenamiento?
