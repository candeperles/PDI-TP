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

## Paso 2 (propuesto): validación cruzada agrupada sobre Jamaica

Para medir con las 172 imágenes y no solo con 23:

1. Repartir las imágenes de Jamaica (train y val) en 5 folds con `StratifiedGroupKFold`, agrupando por zona
   espacial (`split.grupos_espaciales`) y estratificando por con/sin paneles. Así ninguna zona queda partida
   entre entrenamiento y evaluación.
2. En cada fold, entrenar con todo el train que no es de Jamaica más las imágenes de Jamaica de los otros
   4 folds, y predecir el fold que queda afuera.
3. Juntar las predicciones de los 5 folds: queda una predicción por cada una de las 172 imágenes, sin que
   ningún modelo haya visto la imagen ni sus vecinas.
4. Reportar recall con paneles, recall sin paneles y F1 macro, más el umbral elegido sobre esas predicciones
   (tarea de @Sciordia11 en el issue #7).

**Costo:** 5 entrenamientos por configuración. Para los linear probes del notebook extra son segundos; para
las CNN en Colab conviene hacerlo solo con la receta común y las variantes finalistas, no con todas.

**Qué no cambia:** el split oficial sigue igual y las métricas de val del formato común se siguen
reportando como hasta ahora. Esto es una evaluación adicional.

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
