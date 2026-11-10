"""La evaluación final usa la elección previa y controla la carga antes de tocar el test."""
import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((RAIZ / "notebooks/18_evaluacion_test.ipynb").read_text(encoding="utf-8"))
CELDAS = {c["id"]: "".join(c["source"]) for c in NOTEBOOK["cells"]}


class EvaluacionFinalTest(unittest.TestCase):
    def entorno(self, umbral=0.5):
        corrida = {"corrida": "resnet50/imagenet_finetuning", "familia": "clasificador", "vigente": True,
                   "modelo": "resnet50", "modo": "imagenet_finetuning", "fecha": "2026-11-03", "umbral": umbral}
        return {"np": np, "pd": pd, "FAMILIA": "clasificadores", "SOLO_FINALES": True,
                "MODELOS_FINALES": {"clasificadores": corrida["corrida"]},
                "UMBRALES_FINALES": {"clasificadores": 0.5}, "corridas": pd.DataFrame([corrida]),
                "resultados": SimpleNamespace(cargar_corridas=Mock(return_value=pd.DataFrame([corrida])),
                                              ruta_corrida=Mock(return_value=Path("pesos"))),
                "comparacion": SimpleNamespace(corridas_vigentes=lambda tabla: tabla),
                "split": SimpleNamespace(cargar_split=Mock(return_value={"val": [Path("a.png"), Path("b.png")]})),
                "datos": SimpleNamespace(leer_imagen=Mock(return_value=np.zeros((2, 2, 3)))),
                "rutas": SimpleNamespace(imagenes=Path("imagenes")),
                "archivos_val": ["a.png", "b.png"], "imagenes_val": np.zeros((2, 2, 2, 3)),
                "imagenes": np.ones((1, 2, 2, 3)), "etiquetas": pd.DataFrame({"archivo": ["test.png"]}),
                "y_test": np.array([1])}

    def ejecutar_inferencia(self, entorno, p_val):
        entorno["predecir"] = Mock(side_effect=[(np.asarray(p_val), None), (np.array([0.7]), None)])
        guardadas = pd.DataFrame({"archivo": ["a.png", "b.png"], "probabilidad": [0.1, 0.9]})
        codigo = "TOLERANCIA =" + CELDAS["9e280afe"].split("TOLERANCIA =", 1)[1]
        arbol = ast.parse(codigo)
        # El último enunciado solo presenta el Styler; no forma parte de la inferencia.
        arbol.body = arbol.body[:-1]
        with patch.object(pd, "read_csv", return_value=guardadas):
            exec(compile(arbol, "inferencia_final", "exec"), entorno)

    def test_carga_correcta_predice_test_con_umbral_fijado(self):
        entorno = self.entorno()
        self.ejecutar_inferencia(entorno, [0.1, 0.9])
        self.assertEqual(entorno["predecir"].call_count, 2)
        self.assertIs(entorno["predecir"].call_args_list[0].args[1], entorno["imagenes_val"])
        self.assertIs(entorno["predecir"].call_args_list[1].args[1], entorno["imagenes"])
        self.assertEqual(entorno["predicciones"]["prediccion"].tolist(), [1])
        self.assertTrue(entorno["chequeo"]["valida"].all())

    def test_carga_distinta_frena_antes_de_predecir_test(self):
        entorno = self.entorno()
        with self.assertRaisesRegex(RuntimeError, "antes del test"):
            self.ejecutar_inferencia(entorno, [0.1, 0.5])
        self.assertEqual(entorno["predecir"].call_count, 1)

    def test_prediccion_no_finita_frena_antes_de_predecir_test(self):
        entorno = self.entorno()
        with self.assertRaisesRegex(RuntimeError, "antes del test"):
            self.ejecutar_inferencia(entorno, [float("nan"), 0.9])
        self.assertEqual(entorno["predecir"].call_count, 1)

    def test_seleccion_solo_carga_el_finalista(self):
        entorno = self.entorno()
        otra = entorno["corridas"].iloc[0].to_dict()
        otra.update(corrida="mobilenetv2/imagenet_finetuning", modelo="mobilenetv2")
        entorno["resultados"].cargar_corridas.return_value = pd.concat(
            [entorno["corridas"], pd.DataFrame([otra])], ignore_index=True)
        exec(compile(CELDAS["6fa0c275"], "seleccion_final", "exec"), entorno)
        self.assertEqual(entorno["corridas"]["corrida"].tolist(), ["resnet50/imagenet_finetuning"])

    def test_umbral_cambiado_frena_antes_de_cargar_imagenes(self):
        entorno = self.entorno(umbral=0.35)
        with self.assertRaisesRegex(RuntimeError, "umbral guardado"):
            exec(compile(CELDAS["6fa0c275"], "seleccion_final", "exec"), entorno)
        entorno["datos"].leer_imagen.assert_not_called()

    def test_finalista_ausente_no_se_reemplaza_por_otro_modelo(self):
        entorno = self.entorno()
        entorno["resultados"].cargar_corridas.return_value["corrida"] = "resnet50/imagenet_congelado"
        with self.assertRaisesRegex(RuntimeError, "Falta la corrida final"):
            exec(compile(CELDAS["6fa0c275"], "seleccion_final", "exec"), entorno)
        entorno["datos"].leer_imagen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
