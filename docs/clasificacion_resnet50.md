# Ítem 2: ResNet50

Abrir `notebooks/05_ResNet50_Colab_v2.ipynb` en Colab con GPU T4 y ejecutar todo.
Monta Drive con `entorno.preparar` (como el resto de los notebooks) y guarda cada variante con el
formato común `resultados/<modelo>/<modo>/` (ver README, *Dónde quedan los resultados*).
El dataset y el split son los mismos del resto del trabajo.

Se implementaron tres experimentos: ImageNet con extractor congelado, ajuste fino
de conv5 desde el mejor modelo congelado y entrenamiento desde pesos aleatorios.
Todos usan entrada 256×256, aumentos de entrenamiento, pesos de clase calculados
solo con train y umbral 0.5. El detalle de arquitectura, LR y épocas está en el notebook.
La implementación sigue las guías oficiales de
[ResNet](https://keras.io/api/applications/resnet/resnet_models/) y
[transfer learning](https://keras.io/guides/transfer_learning/).

La preparación audita todos los pares y aplica los criterios v1 a un manifiesto y
metadatos por caja, sin alterar originales ni generar un dataset de detección.
Se omite únicamente la caja Z01 de ancho cero. Su imagen conserva siete cajas y es positiva.
Se mantienen los IDs de origen y su confianza como metadatos.

N06/N16 conservan `y_original=0`, con `y` sin confirmar y bandera de revisión.
Las métricas sobre las 162 imágenes y la selección por `val_loss` usan la referencia
original **provisional**; el informe incluye además métricas sin estos dos casos.
No se cambió el split ni se resolvieron sus dudas semánticas por entrenamiento.

Cada variante queda en Drive en `resultados/resnet50/<modo>/` (`imagenet_cabeza`,
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
de estructura/sintaxis del notebook. Los entrenamientos deben ejecutarse en el
entorno de modelos; no hay métricas ni pesos producidos hasta ejecutarlos.
La comparación entre arquitecturas requiere ejecutar también MobileNetV2, disponible en el repositorio.
Val sirve para elegir checkpoints y comparar; no proporciona una evaluación final
independiente. No inferir superioridad sin resultados reales y revisión de curvas.

Si una etapa se interrumpe, al volver a ejecutar el notebook (aunque sea en otra sesión de Colab)
continúa desde `ultimo.keras` y conserva el optimizador Adam, el historial y el mejor
checkpoint anterior. No se reinicia la etapa ni se repiten las épocas ya registradas.
Si historial y checkpoint no coinciden, se detiene sin sobrescribirlos. Las variantes ya completas
se reutilizan.
