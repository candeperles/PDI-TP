# Ítem 2: ResNet50

Abrir `notebooks/05_ResNet50_Colab_v2.ipynb` en una sesión nueva de Colab con GPU T4.
Completar `INTEGRANTE` y `CORRIDA`; cuando se pida, pegar el enlace de una carpeta
existente de Drive. El notebook guarda primero en disco local y copia archivos por
ID a esa carpeta, sin montar Drive ni crear carpetas allí. Verifica escritura antes
de entrenar y copia checkpoints al terminar cada época. Guardar una copia del notebook.
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

Cada corrida guarda localmente en `/content/resultados_resnet50_v2/<integrante>/<corrida>/`, con
manifiesto, cajas, huella de datos, configuración, versiones y comparación CSV.
Cada variante guarda localmente `mejor.keras`, `ultimo.keras`, historial por época, curvas,
matriz de confusión, métricas globales y por fuente y predicciones por archivo.
En Drive se conservan solo cuatro archivos por corrida, con prefijo de integrante y corrida:
`metricas_y_graficos.zip`, `imagenet_cabeza__mejor.keras`,
`imagenet_finetuning__mejor.keras` y `aleatorio__mejor.keras`.
El ZIP contiene todos los archivos auxiliares excepto los modelos y se actualiza después
de cada época junto con los mejores modelos disponibles. `ultimo.keras` queda solo en Colab.
La celda final verifica las cuatro copias y manda a la papelera los auxiliares de la misma
corrida que hubiera subido la versión anterior, sin tocar otras corridas. Los registros JSON
para comparar resultados del equipo se pueden extraer del ZIP.
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

Si una etapa se interrumpe en la misma sesión de Colab, al volver a ejecutar su celda
continúa desde `ultimo.keras` y conserva el optimizador Adam, el historial y el mejor
checkpoint anterior. No se reinicia la etapa ni se repiten las épocas ya registradas.
Si historial y checkpoint no coinciden, se detiene sin sobrescribirlos. El último
checkpoint permanece solo en Colab; esto no recupera una sesión cuyo disco se perdió.
