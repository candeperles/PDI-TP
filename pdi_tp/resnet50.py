"""Entrenamiento ResNet50 con TensorFlow/Keras; importar en el entorno de modelos."""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import aumentacion


def dataset(tabla, lado=256, batch=32, entrenamiento=False, semilla=42, pesos_muestra=None):
    """RGB 0..255 -> preprocess_input de ResNet; val conserva el orden del manifiesto."""
    pesos = np.ones(len(tabla), dtype="float32") if pesos_muestra is None else np.asarray(pesos_muestra, dtype="float32")
    if len(pesos) != len(tabla):
        raise ValueError("pesos_muestra debe estar alineado con tabla")
    ds = tf.data.Dataset.from_tensor_slices((
        tabla.ruta.to_list(), tabla.y_original.to_numpy(dtype="float32"), pesos
    ))
    if entrenamiento:
        ds = ds.shuffle(len(tabla), seed=semilla, reshuffle_each_iteration=True)

    def leer(ruta, etiqueta, peso):
        imagen = tf.io.decode_png(tf.io.read_file(ruta), channels=3)
        imagen = tf.image.resize(imagen, (lado, lado))
        return imagen, tf.reshape(etiqueta, (1,)), peso

    ds = ds.map(leer, num_parallel_calls=tf.data.AUTOTUNE)
    if entrenamiento:
        ds = ds.map(lambda imagen, etiqueta, peso: (aumentacion.google_maps(imagen), etiqueta, peso),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def construir(lado=256, preentrenado=True, augmentation=True, semilla=42):
    base = keras.applications.ResNet50(include_top=False,
        weights="imagenet" if preentrenado else None, input_shape=(lado, lado, 3))
    base.trainable = not preentrenado
    entrada = keras.Input((lado, lado, 3), name="rgb_0_255")
    x = entrada
    # La aumentación común se aplica en el dataset de entrenamiento.
    # ResNet espera BGR centrado en medias ImageNet. No dividir por 255.
    x = keras.applications.resnet.preprocess_input(x)
    # ImageNet: BatchNorm en inferencia incluso durante fine-tuning.
    # Desde cero: BatchNorm debe aprender sus estadísticas.
    x = base(x, training=False) if preentrenado else base(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3, seed=semilla + 2)(x)
    salida = keras.layers.Dense(1, activation="sigmoid", name="con_paneles")(x)
    return keras.Model(entrada, salida, name="clasificador_resnet50"), base


def descongelar(base):
    """Ajusta el último bloque residual conv5; mantiene BatchNormalization congelada."""
    base.trainable = True
    for capa in base.layers:
        capa.trainable = capa.name.startswith("conv5_") and not isinstance(capa, keras.layers.BatchNormalization)


def compilar(modelo, lr):
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                   loss="binary_crossentropy", metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])


def entrenar(modelo, train_ds, val_ds, salida, epocas, lr, pesos):
    """Guarda mejor modelo completo, última época y CSV de progreso en Drive/local."""
    salida.mkdir(parents=True, exist_ok=False)
    compilar(modelo, lr)
    historia = modelo.fit(train_ds, validation_data=val_ds, epochs=epocas,
        callbacks=[
            keras.callbacks.ModelCheckpoint(str(salida / "mejor.keras"), monitor="val_loss", save_best_only=True),
            keras.callbacks.ModelCheckpoint(str(salida / "ultimo.keras")),
            keras.callbacks.CSVLogger(str(salida / "historia.csv")),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ], verbose=2)
    # Se evalúa exactamente el checkpoint seleccionado, también si se alcanzó el máximo de épocas.
    modelo.load_weights(salida / "mejor.keras")
    if not all(np.isfinite(v).all() for v in historia.history.values()):
        raise RuntimeError("Entrenamiento no finito; no registrar como experimento válido.")
    return historia
