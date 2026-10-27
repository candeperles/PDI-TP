"""Umbral elegido por balanced accuracy y métricas solo sobre Jamaica (issue #7)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pdi_tp import split, umbral

GEO = "2163483_unknown_17.96567005007602_-77.24333773013821_0_1.png"  # Jamaica


class UmbralTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.splits = Path(tmp.name)
        (self.splits / "train.txt").write_text("PV01_100_100.png\n")
        (self.splits / "val.txt").write_text("")
        self.addCleanup(setattr, split, "SPLITS_DIR", split.SPLITS_DIR)
        split.SPLITS_DIR = self.splits

    def test_balanced_accuracy_promedia_las_clases(self):
        self.assertEqual(umbral.balanced_accuracy([0, 0, 1, 1], [0, 1, 1, 1]), 0.75)
        self.assertEqual(umbral.balanced_accuracy([1, 1], [1, 0]), 0.5)  # falta una clase

    def test_elegir_umbral_toma_el_medio_del_tramo_empatado(self):
        # Separa perfecto cualquier umbral de 0.25 a 0.80: medio = 0.525.
        self.assertAlmostEqual(umbral.elegir_umbral([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 0.525)

    def test_elegir_umbral_sube_si_el_modelo_sobreestima_paneles(self):
        # Las sin paneles tienen p alta (como Kingston con un modelo entrenado mayormente con paneles).
        self.assertAlmostEqual(umbral.elegir_umbral([0, 0, 1], [0.6, 0.7, 0.9]), 0.825)

    def test_evaluar_por_grupos_no_usa_el_grupo_que_evalua(self):
        y, p, grupos = [0, 1, 0, 1], [0.3, 0.6, 0.7, 0.9], [0, 0, 1, 1]
        pred, umbrales = umbral.evaluar_umbral_por_grupos(y, p, grupos)
        np.testing.assert_allclose(umbrales, [0.825, 0.825, 0.475, 0.475])
        np.testing.assert_array_equal(pred, [0, 0, 1, 1])
        # Con todas juntas el umbral separa perfecto; la estimación honesta acierta la mitad.
        self.assertEqual(umbral.balanced_accuracy(y, pred), 0.5)

    def test_grupos_jamaica_une_vecinas_a_traves_de_train(self):
        archivos = ["kingston_1_0_0_0.png", "kingston_3_0_0_0.png"]
        self.assertNotEqual(*umbral.grupos_jamaica(archivos))
        (self.splits / "train.txt").write_text("kingston_2_0_0_0.png\n")  # vecina de las dos
        self.assertEqual(*umbral.grupos_jamaica(archivos))

    def test_tabla_jamaica_solo_cuenta_jamaica(self):
        archivos = ["kingston_1_0_0_0.png", "kingston_5_0_0_0.png", "kingston_9_0_0_0.png", GEO,
                    "PV01_100_100.png"]
        y = [0, 0, 1, 1, 0]
        predicciones = pd.concat([
            pd.DataFrame({"modelo": "m", "modo": "bueno", "archivo": archivos, "y_real": y,
                          "probabilidad": [0.1, 0.2, 0.8, 0.9, 0.99]}),
            pd.DataFrame({"modelo": "m", "modo": "malo", "archivo": archivos, "y_real": y,
                          "probabilidad": [0.9, 0.8, 0.2, 0.1, 0.0]}),
        ])
        tabla = umbral.tabla_jamaica(predicciones)
        self.assertEqual(tabla["modo"].tolist(), ["bueno", "malo"])
        bueno = tabla.iloc[0]
        self.assertEqual((bueno["n"], bueno["n_con"], bueno["n_sin"]), (4, 2, 2))
        self.assertEqual(bueno["balanced_accuracy_05"], 1.0)
        self.assertAlmostEqual(bueno["umbral"], 0.525)
        self.assertLessEqual(tabla.iloc[1]["balanced_accuracy_cv"], 0.5)

    def test_tabla_jamaica_vacia_sin_imagenes_de_jamaica(self):
        predicciones = pd.DataFrame({"modelo": ["m"], "modo": ["a"], "archivo": ["PV01_100_100.png"],
                                     "y_real": [1], "probabilidad": [0.9]})
        self.assertTrue(umbral.tabla_jamaica(predicciones).empty)


if __name__ == "__main__":
    unittest.main()
