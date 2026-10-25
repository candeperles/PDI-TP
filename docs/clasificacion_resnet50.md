# Ítem 2: ResNet50

Abrir `notebooks/04_clasificacion_ResNet50.ipynb`, completar `INTEGRANTE` y ejecutar
en orden en Colab con GPU T4. El notebook instala `requirements-training.txt` y usa
el mismo montaje de Drive y split del resto del trabajo. Guardar una copia del notebook.
El notebook nuevo y los módulos deben estar disponibles en el repositorio que clona
Colab; mientras los cambios sean locales, subirlos antes de abrirlo desde GitHub.

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

Cada corrida crea `resultados/<integrante>/resnet50_<fecha>/` (Drive en Colab) con
manifiesto, cajas, huella de datos, configuración, versiones y comparación CSV.
Cada variante guarda `mejor.keras`, `ultimo.keras`, historial por época, curvas,
matriz de confusión, métricas globales y por fuente y predicciones identificadas
por archivo. Los JSON resumidos se integran con `resultados.cargar_experimentos`.
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
