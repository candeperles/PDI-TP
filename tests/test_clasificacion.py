"""Controles de referencia y métricas previos al entrenamiento."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from pdi_tp import clasificacion as clf


class ClasificacionTest(unittest.TestCase):
    def test_metricas_orden_y_clase_positiva(self):
        m = clf.evaluar([0, 0, 1, 1], [0.1, 0.6, 0.5, 0.9])
        self.assertEqual(m["matriz_confusion"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertEqual(m["recall"], 1)
        self.assertAlmostEqual(m["f1"], 0.8)
        self.assertEqual(clf.evaluar([1, 1], [1, 1])["matriz_confusion"], [[0, 0], [0, 2]])

    def test_rechaza_probabilidades_invalidas_o_desalineadas(self):
        for y, p in [([0, 1], [0.5]), ([0], [np.nan]), ([0], [1.1]), ([], []), ([2], [0.5])]:
            with self.assertRaises(ValueError):
                clf.evaluar(y, p)

    def test_pesos_balanceados(self):
        self.assertEqual(clf.pesos_clase([0, 1, 1, 1]), {0: 2, 1: 2 / 3})
        with self.assertRaises(ValueError):
            clf.pesos_clase([1, 1])

    def test_error_nuevo_impide_preparar(self):
        informe = SimpleNamespace(
            incidencias=pd.DataFrame([dict(nivel="error", archivo="otro.png", linea=1,
                                         control="dimension_caja_no_positiva")]),
            cajas=pd.DataFrame(columns=["archivo", "linea", "w", "h"]))
        with patch.object(clf.validacion, "validar", return_value=informe):
            with self.assertRaisesRegex(ValueError, "Errores nuevos"):
                clf.preparar_tabla(None)


if __name__ == "__main__":
    unittest.main()
