"""Comparación final: métricas, intervalos bootstrap y corridas vigentes, sin dataset ni TensorFlow."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from pdi_tp import comparacion

ARCHIVOS = [f"PV01_{i}_0.png" for i in range(30)] + [f"kingston_{i}_0_0_0.png" for i in range(20)]


def predicciones(corridas: dict, y: np.ndarray, umbral: float = 0.5) -> pd.DataFrame:
    tablas = []
    for corrida, prob in corridas.items():
        modelo, modo = corrida.split("/")
        pred = (prob >= umbral).astype(int)
        tablas.append(pd.DataFrame({"archivo": ARCHIVOS, "fuente": "x", "y_real": y, "probabilidad": prob,
                                    "prediccion": pred, "acierto": (pred == y).astype(int),
                                    "modelo": modelo, "modo": modo}))
    return pd.concat(tablas, ignore_index=True)


class ComparacionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.array([1] * 35 + [0] * 15)
        self.buena = np.clip(self.y * 0.7 + rng.uniform(0, 0.3, 50), 0, 1)
        self.mala = rng.uniform(0, 1, 50)
        self.pred = predicciones({"a/bueno": self.buena, "b/malo": self.mala, "c/igual": self.buena}, self.y)

    def test_puntual_coincide_con_sklearn(self):
        tabla = comparacion.intervalos(self.pred, ["b/malo"], n=50).iloc[0]
        pred = (self.mala >= 0.5).astype(int)
        self.assertAlmostEqual(tabla.f1_macro, f1_score(self.y, pred, average="macro"))
        self.assertAlmostEqual(tabla.balanced_accuracy, balanced_accuracy_score(self.y, pred))
        self.assertAlmostEqual(tabla.recall_sin_paneles, recall_score(self.y, pred, pos_label=0))
        self.assertAlmostEqual(tabla.roc_auc, roc_auc_score(self.y, self.mala))
        self.assertLessEqual(tabla.f1_macro_inf, tabla.f1_macro)
        self.assertGreaterEqual(tabla.f1_macro_sup, tabla.f1_macro)

    def test_auc_con_empates(self):
        prob = np.where(self.y == 1, 0.8, 0.2)
        prob[:5] = 0.2
        pred = predicciones({"x/empates": prob}, self.y)
        self.assertAlmostEqual(comparacion.intervalos(pred, ["x/empates"], n=10).iloc[0].roc_auc,
                               roc_auc_score(self.y, prob))

    def test_bootstrap_reproducible(self):
        uno = comparacion.intervalos(self.pred, ["a/bueno", "b/malo"], n=200, semilla=3)
        dos = comparacion.intervalos(self.pred, ["a/bueno", "b/malo"], n=200, semilla=3)
        pd.testing.assert_frame_equal(uno, dos)

    def test_diferencias_pareadas(self):
        tabla = comparacion.diferencias(self.pred, ["b/malo", "c/igual"], "a/bueno", n=500).set_index("corrida")
        self.assertEqual(tabla.loc["c/igual", "diferencia"], 0)
        self.assertEqual((tabla.loc["c/igual", "inf"], tabla.loc["c/igual", "sup"]), (0, 0))
        self.assertFalse(tabla.loc["c/igual", "clara"])
        self.assertTrue(tabla.loc["b/malo", "clara"])
        self.assertLess(tabla.loc["b/malo", "sup"], 0)

    def test_solo_jamaica(self):
        tabla = comparacion.intervalos(self.pred, ["a/bueno"], solo_jamaica=True, n=10).iloc[0]
        self.assertEqual(tabla.n, 20)
        self.assertEqual(tabla.n_sin_paneles, 15)

    def test_rechaza_corridas_que_no_estan(self):
        with self.assertRaisesRegex(ValueError, "z/no"):
            comparacion.intervalos(self.pred, ["a/bueno", "z/no"], n=10)

    def test_rechaza_imagenes_distintas(self):
        incompleta = self.pred.drop(index=self.pred.index[(self.pred.modelo == "b")][:3])
        with self.assertRaisesRegex(ValueError, "mismas imágenes"):
            comparacion.intervalos(incompleta, ["a/bueno", "b/malo"], n=10)

    def test_vigentes_segun_receta_y_familia(self):
        corridas = pd.DataFrame({
            "modelo": ["mobilenetv2", "mobilenetv2", "yolov8n", "baseline"],
            "modo": ["imagenet_congelado", "imagenet_fine_tuning", "coco_completo", "siempre_con_paneles"],
            "completa": [True, True, True, True], "fecha": None, "umbral": [0.5, 0.5, 0.25, 0.5],
            "params.receta": ["comun_v2", None, None, None],
            **{m: 0.5 for m in comparacion.METRICAS}})
        tabla = comparacion.corridas_vigentes(corridas).set_index("corrida")
        self.assertEqual(tabla.vigente.to_dict(), {"mobilenetv2/imagenet_congelado": True,
                                                   "mobilenetv2/imagenet_fine_tuning": False,
                                                   "yolov8n/coco_completo": True,
                                                   "baseline/siempre_con_paneles": True})
        self.assertEqual(tabla.loc["yolov8n/coco_completo", "familia"], "detector")

    def test_tabla_deteccion_lee_extra(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            for corrida, extra in [("yolov8n/coco_completo",
                                    {"deteccion": {"val": {"map50": 0.76, "map50_95": 0.45, "precision": 0.8,
                                                           "recall": 0.7},
                                                   "jamaica": {"map50": 0.74, "map50_95": 0.5, "precision": 0.7,
                                                               "recall": 0.6}}}),
                                   ("mobilenetv2/imagenet_congelado", {"epocas_por_etapa": {}})]:
                (raiz / corrida).mkdir(parents=True)
                (raiz / corrida / "metricas.json").write_text(json.dumps({"extra": extra}))
            tabla = comparacion.tabla_deteccion(raiz, ["yolov8n/coco_completo", "mobilenetv2/imagenet_congelado"])
        self.assertEqual(tabla.conjunto.tolist(), ["val", "jamaica"])
        self.assertEqual(tabla.map50.tolist(), [0.76, 0.74])


if __name__ == "__main__":
    unittest.main()
