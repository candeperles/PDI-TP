# Ítem 2: ResNet50

Abrir `notebooks/06_clasificacion_resnet50.ipynb` en Colab con GPU T4 y ejecutar todo.
Monta Drive con `entorno.preparar` (como el resto de los notebooks que entrenan) y guarda cada variante con el
formato común `resultados/<modelo>/<modo>/` (ver README, *Dónde quedan los resultados*).
El dataset y el split son los mismos del resto del trabajo.

Se implementaron tres experimentos: ImageNet con extractor congelado, ajuste fino
de todo el extractor (BatchNorm congeladas) desde el mejor modelo congelado y entrenamiento desde pesos aleatorios.
Todos usan entrada 256×256, aumentos de entrenamiento, pesos de muestra por fuente × clase calculados
solo con train y umbral 0.5. El detalle de arquitectura, LR y épocas está en el notebook.
La implementación sigue las guías oficiales de
[ResNet](https://keras.io/api/applications/resnet/resnet_models/) y
[transfer learning](https://keras.io/guides/transfer_learning/).

La preparación audita todos los pares y aplica los criterios v1 a un manifiesto y
metadatos por caja, sin alterar originales ni generar un dataset de detección.
Se omite únicamente la caja Z01 de ancho cero. Su imagen conserva siete cajas y es positiva.
Se mantienen los IDs de origen y su confianza como metadatos.

N06/N16 conservan `y_original=0`. El 6/11 el equipo revisó N06 como sin paneles; N16 sigue
pendiente (ver [criterios_datos.md](criterios_datos.md)). Las métricas sobre las 162 imágenes y la
selección por `val_loss` usan la etiqueta original; `metricas.json` trae además métricas sin estos dos casos.
No se cambió el split ni se resolvieron sus dudas semánticas por entrenamiento.

Cada variante queda en Drive en `resultados/resnet50/<modo>/` (`imagenet_congelado`,
`imagenet_finetuning`, `aleatorio`) y el baseline en `resultados/baseline/siempre_con_paneles/`.
Dentro: `mejor.keras` (pesos evaluados), `ultimo.keras` (para reanudar), `historia.csv`, curvas,
matriz de confusión, `metricas.json` (globales, por fuente y sin N06/N16), `predicciones_val.csv`
(predicción contra el real) y `config.json` (parámetros, versiones y huella de datos).
El manifiesto y las cajas derivadas quedan en `analisis/resnet50/`. No se borra nada en Drive:
`REPETIR = True` archiva la variante anterior como `<modo>__anterior_<fecha>`.
Para cargar un modelo, usar `keras.models.load_model(ruta)` e ingresar RGB 0..255;
el preprocesamiento ResNet está incluido en el modelo.

La dependencia de entrenamiento se mantiene separada del entorno de exploración.
No instalar el lock de Python 3.14 junto con TensorFlow: contiene versiones que
pueden ser incompatibles. Usar Colab o un entorno nuevo con Python compatible con
la versión de TensorFlow elegida. El notebook registra las versiones efectivas.

Verificaciones sin TensorFlow: `python -m unittest discover -s tests -v` y validación
de estructura/sintaxis del notebook. La comparación con MobileNetV2 y EfficientNetB0 está en
`15_comparacion_validacion`, que solo incluye las corridas con `params.receta = "comun_v2"`.
Val sirve para elegir checkpoints y comparar; no proporciona una evaluación final
independiente. No inferir superioridad sin resultados reales y revisión de curvas.

En Drive puede quedar `resultados/resnet50/imagenet_cabeza/`, un modo de una receta anterior
(apareció en la evaluación del test del 6/11). No forma parte de la comparación: archivarlo con
`__anterior_<fecha>` para que no se confunda con `imagenet_congelado`.

Si una etapa se interrumpe, al volver a ejecutar el notebook (aunque sea en otra sesión de Colab)
continúa desde `ultimo.keras` y conserva el optimizador Adam, el historial y el mejor
checkpoint anterior. No se reinicia la etapa ni se repiten las épocas ya registradas.
Si historial y checkpoint no coinciden, se detiene sin sobrescribirlos. Las variantes ya completas
se reutilizan.

## Receta común del issue #15 (`comun_v2`)

Entrada 256×256, batch 32, dropout 0.3, semilla 42 y augmentation
`aumentacion.google_maps`. No se aplica `class_weight`: los pesos por fuente × clase
viajan en el dataset como pesos de muestra, calculados solo con train.

| Modo | LR inicial | Épocas máximas | Paciencia |
|---|---|---|---|
| `imagenet_congelado` | 1e-3 | 30 | 10 |
| `imagenet_finetuning` | 1e-5 | 20 adicionales | 10 |
| `aleatorio` | 1e-3 | 60 | 10, desde la época 20 |

El modo aleatorio reduce el LR por 0.5 tras 3 épocas sin mejorar `val_loss`, usa momentum 0.9 en BatchNorm
y no cuenta las primeras 20 épocas ni para la parada ni para elegir `mejor.keras`.
La reanudación conserva el LR del optimizador y reconstruye la espera del scheduler.
La parada temprana restaura los mejores pesos; también se carga explícitamente
`mejor.keras` antes de evaluar o iniciar fine-tuning.
Los params usan `img_size`, `batch_size`, `dropout`, `lr`, `epocas_max`,
`paciencia`, `umbral` y `semilla` en los tres modos.

Las corridas de recetas anteriores no son comparables con esta: si un modo de Drive no tiene
`params.receta = "comun_v2"`, volver a correrlo con `REPETIR = True`.
