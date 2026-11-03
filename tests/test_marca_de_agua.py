"""Detector de la marca de agua de Google Maps, con imágenes sintéticas (sin dataset)."""
import unittest

import numpy as np
import pandas as pd

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

    def test_tapar_saca_la_marca_y_refleja_lo_de_arriba(self):
        imagen = con_logo(con_barra(fondo()))
        tapada = marca_de_agua.tapar(imagen)
        self.assertFalse(marca_de_agua.detectar(tapada)["marca"])
        desde = marca_de_agua.FILA_INICIO_MARCA
        np.testing.assert_array_equal(tapada[:desde], imagen[:desde])
        np.testing.assert_array_equal(tapada[desde], imagen[desde - 1])
        np.testing.assert_array_equal(tapada[-1], imagen[2 * desde - 256])
        self.assertTrue(marca_de_agua.detectar(imagen)["marca"])  # no modifica la original

    def test_pegar_franja_trae_la_marca_del_donante(self):
        destino, donante = fondo(1), con_barra(fondo(2))
        pegada = marca_de_agua.pegar_franja(destino, donante)
        self.assertTrue(marca_de_agua.detectar(pegada)["barra"])
        desde = marca_de_agua.FILA_INICIO_MARCA
        np.testing.assert_array_equal(pegada[:desde], destino[:desde])
        np.testing.assert_array_equal(pegada[desde:], donante[desde:])
        with self.assertRaises(ValueError):
            marca_de_agua.pegar_franja(destino, np.zeros((128, 128, 3)))


def tabla_imagenes():
    filas = [("j_marca_sin", "train", True, True, 0), ("j_marca_val", "val", True, True, 1),
             ("j_limpia_sin", "train", True, False, 0), ("j_limpia_val", "val", True, False, 1),
             ("otra_con_1", "val", False, False, 1), ("otra_con_2", "val", False, False, 1),
             ("otra_sin", "val", False, False, 0), ("otra_train", "train", False, False, 1)]
    return pd.DataFrame(filas, columns=["archivo", "split", "jamaica", "marca", "con_paneles"])


class PruebasTest(unittest.TestCase):
    def test_armar_pruebas(self):
        pruebas = marca_de_agua.armar_pruebas(tabla_imagenes())
        tapar = pruebas[pruebas["prueba"] == "tapar"].set_index("archivo")
        self.assertEqual(set(tapar.index), {"j_marca_sin", "j_marca_val", "j_limpia_sin", "j_limpia_val"})
        self.assertEqual(tapar.loc["j_marca_val", "grupo"], "con marca")
        self.assertEqual(tapar.loc["j_limpia_val", "grupo"], "control")
        pegar = pruebas[pruebas["prueba"] == "pegar"]
        # Solo destinos de val, con paneles y de otras fuentes; cada uno con un donante por grupo, sin paneles.
        self.assertEqual(sorted(pegar["archivo"]), ["otra_con_1", "otra_con_1", "otra_con_2", "otra_con_2"])
        self.assertEqual(set(pegar.loc[pegar["grupo"] == "con marca", "donante"]), {"j_marca_sin"})
        self.assertEqual(set(pegar.loc[pegar["grupo"] == "control", "donante"]), {"j_limpia_sin"})
        self.assertTrue((pegar["y_real"] == 1).all())

    def test_armar_pruebas_exige_columnas(self):
        with self.assertRaises(ValueError):
            marca_de_agua.armar_pruebas(tabla_imagenes().drop(columns="marca"))

    def test_modificar(self):
        imagenes = {"a": fondo(1), "b": con_barra(fondo(2))}
        pegar = pd.Series({"prueba": "pegar", "archivo": "a", "donante": "b"})
        self.assertTrue(marca_de_agua.detectar(marca_de_agua.modificar(pegar, imagenes.get))["barra"])
        tapar = pd.Series({"prueba": "tapar", "archivo": "b", "donante": ""})
        self.assertFalse(marca_de_agua.detectar(marca_de_agua.modificar(tapar, imagenes.get))["barra"])

    def test_efecto_detecta_un_modelo_que_usa_la_marca(self):
        filas = []
        for corrida, cae in (("usa/marca", 0.5), ("no/usa", 0.0)):
            for i in range(30):
                for grupo, cambio in (("con marca", -cae), ("control", 0.0)):
                    filas.append(dict(corrida=corrida, prueba="pegar", grupo=grupo, archivo=f"x{i}", split="val",
                                      original=0.9, modificada=0.9 + cambio + 0.01 * (i % 3), umbral=0.5))
        tabla = marca_de_agua.efecto(pd.DataFrame(filas)).set_index("corrida")
        self.assertAlmostEqual(tabla.loc["usa/marca", "efecto"], -0.5)
        self.assertLess(tabla.loc["usa/marca", "efecto_sup"], 0)
        self.assertEqual(tabla.loc["usa/marca", "cambian_con_marca"], 30)
        self.assertAlmostEqual(tabla.loc["no/usa", "efecto"], 0)
        self.assertEqual(tabla.loc["no/usa", "cambian_control"], 0)

    def test_efecto_en_tapar_compara_grupos_distintos(self):
        filas = [dict(corrida="a/b", prueba="tapar", grupo="con marca", archivo=f"m{i}", split="val",
                      original=0.2, modificada=0.6, umbral=0.5) for i in range(5)]
        filas += [dict(corrida="a/b", prueba="tapar", grupo="control", archivo=f"c{i}", split="val",
                       original=0.2, modificada=0.3, umbral=0.5) for i in range(7)]
        fila = marca_de_agua.efecto(pd.DataFrame(filas)).iloc[0]
        self.assertAlmostEqual(fila["efecto"], 0.3)
        self.assertEqual((fila["n_con_marca"], fila["n_control"]), (5, 7))
        self.assertEqual(fila["cambian_con_marca"], 5)


if __name__ == "__main__":
    unittest.main()
