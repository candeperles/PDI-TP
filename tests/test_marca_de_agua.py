"""Detector de la marca de agua de Google Maps, con imágenes sintéticas (sin dataset)."""
import unittest

import numpy as np

from pdi_tp import marca_de_agua


def fondo(semilla=0):
    """Imagen aérea falsa: manchas de color con ruido."""
    rng = np.random.default_rng(semilla)
    base = rng.uniform(60, 200, size=(8, 8, 3)).repeat(32, 0).repeat(32, 1)
    return np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)


def con_barra(imagen, fila=227, desde=40):
    imagen = imagen.astype(float)
    gris = imagen[fila:, desde:].mean(2, keepdims=True) * 0.25  # overlay oscuro y sin color
    imagen[fila:, desde:] = gris
    imagen[fila + 8:fila + 20, desde + 10:250:6] = 210  # letras claras
    return imagen.astype(np.uint8)


def con_logo(imagen):
    imagen = imagen.copy()
    for x in range(10, 110, 22):  # letras blancas con contorno negro
        imagen[226:248, x:x + 16] = 20
        imagen[229:245, x + 3:x + 13] = 250
    return imagen


class MarcaDeAguaTest(unittest.TestCase):
    def test_imagen_sin_marca(self):
        for semilla in range(5):
            with self.subTest(semilla=semilla):
                self.assertFalse(marca_de_agua.detectar(fondo(semilla))["marca"])

    def test_detecta_barra(self):
        resultado = marca_de_agua.detectar(con_barra(fondo()))
        self.assertTrue(resultado["barra"])
        self.assertTrue(resultado["marca"])
        self.assertFalse(resultado["logo"])
        self.assertEqual(resultado["barra_fila"], 227)

    def test_barra_fuera_de_altura_no_cuenta(self):
        self.assertFalse(marca_de_agua.detectar(con_barra(fondo(), fila=200))["barra"])

    def test_barra_corta_no_cuenta(self):
        self.assertFalse(marca_de_agua.detectar(con_barra(fondo(), desde=240))["barra"])

    def test_detecta_logo_sin_barra(self):
        resultado = marca_de_agua.detectar(con_logo(fondo()))
        self.assertTrue(resultado["logo"])
        self.assertFalse(resultado["barra"])
        self.assertTrue(resultado["marca"])

    def test_rechaza_otro_tamano(self):
        with self.assertRaises(ValueError):
            marca_de_agua.detectar(np.zeros((128, 128, 3)))


if __name__ == "__main__":
    unittest.main()
