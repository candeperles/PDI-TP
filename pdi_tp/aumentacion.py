"""Aumentación común para aproximar el aspecto de imágenes de Google Maps."""

import tensorflow as tf


def google_maps(imagen):
    """Aplica perturbaciones leves de color, blur y compresión JPEG."""
    imagen = tf.cast(imagen, tf.float32)
    imagen = tf.image.rot90(
        imagen, k=tf.random.uniform([], 0, 4, dtype=tf.int32)
    )
    imagen = tf.image.random_flip_left_right(imagen)
    imagen = tf.image.random_brightness(imagen, max_delta=25.5)
    imagen = tf.image.random_contrast(imagen, 0.85, 1.15)
    imagen = tf.clip_by_value(imagen, 0.0, 255.0)
    imagen = tf.cast(imagen, tf.uint8)
    imagen = tf.image.random_jpeg_quality(imagen, 70, 100)
    imagen = tf.cast(imagen, tf.float32)
    desenfocar = lambda: tf.nn.avg_pool2d(
        imagen[None, ...], ksize=3, strides=1, padding="SAME"
    )[0]
    return tf.cond(tf.random.uniform([]) < 0.35, desenfocar, lambda: imagen)
