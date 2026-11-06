"""Estudios controlados: aislamiento de variables y recall pareado por imagen."""
import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pdi_tp import clasificacion, estudios

RAIZ = Path(__file__).resolve().parents[1]


def celdas(nombre):
    nb = json.loads((RAIZ / "notebooks" / f"{nombre}.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


class EstudiosTest(unittest.TestCase):
    def test_control_rechaza_cambios_ocultos_o_campos_ausentes(self):
        referencia = dict(imgsz=640, seed=42, augmentation=True)
        self.assertEqual(estudios.validar_control(referencia, {**referencia, "imgsz": 256}, ["imgsz"]),
                         {"imgsz"})
        for variante in ({**referencia, "seed": 1}, dict(imgsz=256, seed=42),
                         {**referencia, "lr": 0.1}):
            with self.assertRaisesRegex(ValueError, "no autorizados"):
                estudios.validar_control(referencia, variante, ["imgsz"])

    def test_tiempos_son_diferencias_y_rechazan_reinicio(self):
        np.testing.assert_allclose(estudios.tiempos_por_epoca(pd.DataFrame({"time": [3, 7, 12]})), [3, 4, 5])
        for tiempos in ([], [3, 1], [np.nan], [0]):
            with self.assertRaises(ValueError):
                estudios.tiempos_por_epoca(pd.DataFrame({"time": tiempos}))

    def test_cajas_se_emparejan_antes_de_dividir_tamanos(self):
        tabla = pd.DataFrame([dict(archivo="PV01_0_0.png", split="val", ancho=256, alto=256)])
        # Dos referencias concéntricas con IoU>0.5, a lados distintos del corte 16.
        anotadas = pd.DataFrame([dict(archivo="PV01_0_0.png", incluida=True, xc=0.5, yc=0.5,
                                     w=lado/256, h=lado/256) for lado in (15, 17)])
        pred = pd.DataFrame([dict(archivo="PV01_0_0.png", confianza=0.9, xc=0.5, yc=0.5, w=15/256, h=15/256)])
        aciertos = estudios.aciertos_por_caja(tabla, anotadas, {"a": pred})
        self.assertEqual(aciertos.tamano.tolist(), ["<16", "16–32"])
        self.assertEqual(aciertos.a.tolist(), [True, False])
        # Un duplicado no aumenta el recall de una referencia ya asignada.
        una = estudios.aciertos_por_caja(tabla, anotadas.iloc[:1], {"a": pd.concat([pred, pred])})
        self.assertEqual(int(una.a.sum()), 1)

    def test_recall_pareado_remuestrea_imagenes_y_preserva_bins_vacios(self):
        nombres = ["kingston_1_0_0.png", "PV01_0_0.png", "PV01_1_0.png"]
        tabla = pd.DataFrame(dict(archivo=nombres, split="val"))
        aciertos = pd.DataFrame(dict(archivo=[nombres[0], nombres[0], nombres[1]],
            tamano=["<16"] * 3, a=[True, False, True], b=[False, False, True], igual=[True, False, True]))
        resultado = estudios.recall_por_tamano(tabla, aciertos, ["a", "b", "igual"], "a", n=100, semilla=5)
        val = resultado.query("conjunto == 'val' and tamano == '<16'").set_index("corrida")
        self.assertAlmostEqual(val.loc["b", "recall"], 1/3)
        self.assertAlmostEqual(val.loc["b", "diferencia"], -1/3)
        self.assertEqual((val.loc["igual", "inf"], val.loc["igual", "sup"]), (0, 0))
        # Dos cajas de la misma imagen permanecen juntas: Jamaica tiene solo esa imagen.
        jamaica = resultado.query("conjunto == 'jamaica' and tamano == '<16'").set_index("corrida")
        self.assertEqual((jamaica.loc["b", "inf"], jamaica.loc["b", "sup"]), (-0.5, -0.5))
        vacios = resultado.query("tamano == '≥64'")
        self.assertTrue(vacios.recall.isna().all())
        self.assertTrue((vacios.remuestreos_validos == 0).all())
        pd.testing.assert_frame_equal(resultado, estudios.recall_por_tamano(
            tabla, aciertos, ["a", "b", "igual"], "a", n=100, semilla=5))

    def test_estudios_conservan_recetas_originales_salvo_variable(self):
        espacio = {"estudios": estudios}
        codigo = next(c for c in celdas("14_variables_clasificacion") if c.startswith("MODELO ="))
        exec(codigo.split("# No se puede")[0], espacio)
        original = {}
        codigo_original = next(c for c in celdas("04_clasificacion_mobilenetv2") if c.startswith("REPETIR ="))
        exec(codigo_original.split("a_correr =")[0], original)
        self.assertEqual(espacio["BASE"], {**original["BASE"], "early_stopping": True})
        self.assertEqual(len(espacio["CONFIGURACIONES"]), 7)
        for config in espacio["CONFIGURACIONES"].values():
            efectiva = {k: v for k, v in config.items() if k != "variable"}
            cambios = estudios.validar_control(espacio["BASE"], efectiva, [config["variable"], "receta"])
            self.assertEqual(cambios, {config["variable"], "receta"})
            self.assertEqual(config["receta"], "variables_v1")
        # La receta YOLO ejecutada es literalmente la del 08 salvo imgsz.
        recetas = []
        for nombre in ("08_deteccion_yolov8n", "15_deteccion_yolov8n_256"):
            espacio_yolo = {}
            opciones = next(c for c in celdas(nombre) if "MODOS =" in c)
            exec(opciones, espacio_yolo)
            preparacion = next(c for c in celdas(nombre) if "RECETA = dict" in c)
            arbol = ast.parse(preparacion)
            receta = next(n for n in arbol.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "RECETA" for t in n.targets))
            exec(compile(ast.Module(body=[receta], type_ignores=[]), nombre, "exec"), espacio_yolo)
            recetas.append(espacio_yolo["RECETA"])
            if nombre.startswith("15"):
                self.assertEqual(espacio_yolo["MODOS"], {"coco_completo_256": 0})
        self.assertEqual(estudios.validar_control(recetas[0], recetas[1], ["imgsz"]), {"imgsz"})
        self.assertEqual(recetas[1]["imgsz"], 256)

    def test_entrenamiento_aplica_pesos_y_sin_early_stopping_selecciona_checkpoint(self):
        espacio = {"pd": pd, "np": np}
        for codigo in celdas("14_variables_clasificacion"):
            for nodo in ast.parse(codigo).body:
                if isinstance(nodo, ast.FunctionDef):
                    exec(compile(ast.Module(body=[nodo], type_ignores=[]), "notebook14", "exec"), espacio)
        modelo = Mock()
        modelo.fit.return_value.history = {"loss": [0.4, 0.3], "val_loss": [0.2, 0.3]}
        mejor = Mock()
        mejor.get_weights.return_value = ["pesos_mejores"]
        keras = SimpleNamespace(optimizers=SimpleNamespace(Adam=Mock()),
            metrics=SimpleNamespace(Precision=Mock(), Recall=Mock(), AUC=Mock()),
            callbacks=SimpleNamespace(ModelCheckpoint=Mock(), EarlyStopping=Mock()),
            models=SimpleNamespace(load_model=Mock(return_value=mejor)))
        espacio["keras"] = keras
        config = dict(early_stopping=False, paciencia=10, inicio_early_stopping=0)
        historia = espacio["entrenar_etapa"](modelo, "cabeza", 1e-3, 30, "train", "val", config, Path("corrida"))
        self.assertEqual(modelo.fit.call_args.kwargs["epochs"], 30)
        keras.callbacks.EarlyStopping.assert_not_called()
        modelo.set_weights.assert_called_once_with(["pesos_mejores"])
        self.assertEqual(historia.etapa.tolist(), ["cabeza", "cabeza"])
        # Se comprueba la función que corre una variante, sin dataset ni TensorFlow.
        espacio.update(clasificacion=clasificacion, fuentes_train=["a", "a", "a", "b"],
            y_train=np.array([1, 1, 1, 0]), X_train="imagenes_train", X_val="imagenes_val", y_val="y_val",
            PESOS_VAL="pesos_val", armar_ds=Mock(return_value="ds"), armar_modelo=Mock(side_effect=RuntimeError("fin")),
            resultados=SimpleNamespace(carpeta_corrida=Mock()), rutas=None, MODELO="mobilenetv2", REPETIR=False)
        keras.backend = SimpleNamespace(clear_session=Mock())
        keras.utils = SimpleNamespace(set_random_seed=Mock())
        with self.assertRaisesRegex(RuntimeError, "fin"):
            espacio["correr_experimento"]("pesos_clase", dict(semilla=42, sample_weight="clase"))
        pesos_reales = espacio["armar_ds"].call_args_list[0].args[2]
        np.testing.assert_allclose(pesos_reales, [2/3, 2/3, 2/3, 2])


if __name__ == "__main__":
    unittest.main()
