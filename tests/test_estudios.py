"""Estudios controlados: aislamiento de variables y recall pareado por imagen."""
import ast
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pdi_tp import clasificacion, estudios, resultados

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

    def test_entrenamiento_aplica_pesos_y_restauracion_no_depende_de_leer_checkpoints(self):
        espacio = {"pd": pd, "np": np, "PENDIENTES_GUARDADO": {}}
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
        keras.callbacks.EarlyStopping.assert_called_once_with(monitor="val_loss", patience=31,
            restore_best_weights=True, start_from_epoch=0)
        keras.models.load_model.assert_not_called()
        self.assertIn(keras.callbacks.EarlyStopping.return_value, modelo.fit.call_args.kwargs["callbacks"])
        self.assertEqual(historia.etapa.tolist(), ["cabeza", "cabeza"])
        # Con parada temprana conserva paciencia 10; en fine-tuning sin parada, usa 21.
        for activa, epocas, paciencia in ((True, 30, 10), (False, 20, 21)):
            keras.callbacks.EarlyStopping.reset_mock()
            espacio["entrenar_etapa"](modelo, "fine_tuning", 1e-5, epocas, "train", "val",
                                      {**config, "early_stopping": activa}, Path("corrida"))
            self.assertEqual(modelo.fit.call_args.kwargs["epochs"], epocas)
            keras.callbacks.EarlyStopping.assert_called_once_with(monitor="val_loss", patience=paciencia,
                restore_best_weights=True, start_from_epoch=0)
        keras.models.load_model.assert_not_called()
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

    def test_guardado_recrea_destino_y_conserva_modelo_si_drive_falla(self):
        espacio = {"pd": pd, "np": np, "estudios": estudios, "MODELO": "mobilenetv2",
                   "REFERENCIA": "mobilenetv2/imagenet_finetuning"}
        codigo = next(c for c in celdas("14_variables_clasificacion") if "def guardar_pendiente" in c)
        arbol = ast.parse(codigo)
        funciones = [n for n in arbol.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=funciones, type_ignores=[]), "notebook14", "exec"), espacio)
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal) / "resultados"
            referencia = raiz / espacio["REFERENCIA"] / "config.json"
            referencia.parent.mkdir(parents=True)
            referencia.write_text("{}")
            modelo = Mock()
            modelo.save.side_effect = FileNotFoundError("Carpeta dejó de estar disponible")
            pendiente = dict(modelo=modelo, config={"umbral": 0.5}, archivos=["a.png"],
                probabilidad=np.array([0.7]), historia=pd.DataFrame({"loss": [0.3]}), extra={})
            espacio.update(rutas=SimpleNamespace(resultados_equipo=raiz),
                resultados=SimpleNamespace(carpeta_corrida=resultados.carpeta_corrida, guardar_corrida=Mock()),
                PENDIENTES_GUARDADO={"sin_aug": pendiente})
            with self.assertRaises(FileNotFoundError):
                espacio["guardar_pendiente"]("sin_aug")
            self.assertIs(espacio["PENDIENTES_GUARDADO"]["sin_aug"]["modelo"], modelo)
            espacio["resultados"].guardar_corrida.assert_not_called()
            # Reintentar correr el modo solo guarda el mismo modelo; no llega a Keras ni a fit.
            modelo.save.side_effect = lambda destino: destino.write_bytes(b"pesos de prueba")
            espacio["correr_experimento"]("sin_aug", pendiente["config"])
            carpeta = raiz / "mobilenetv2" / "sin_aug"
            self.assertTrue((carpeta / "modelo.keras").is_file())
            self.assertTrue((carpeta / "historia.csv").is_file())
            self.assertNotIn("sin_aug", espacio["PENDIENTES_GUARDADO"])
            self.assertEqual(list(carpeta.parent.glob("*__anterior_*")), [])
            espacio["resultados"].guardar_corrida.assert_called_once_with(
                espacio["rutas"], carpeta, archivos=pendiente["archivos"], probabilidad=pendiente["probabilidad"],
                params=pendiente["config"], umbral=0.5, metricas_extra={})

    def test_etapas_usan_checkpoints_locales_y_se_publican_junto_al_modelo(self):
        espacio = {"pd": pd, "np": np, "time": time, "estudios": estudios, "clasificacion": clasificacion,
                   "MODELO": "mobilenetv2", "REFERENCIA": "mobilenetv2/imagenet_finetuning",
                   "PENDIENTES_GUARDADO": {}, "REPETIR": False}
        codigo = next(c for c in celdas("14_variables_clasificacion") if "def guardar_pendiente" in c)
        funciones = [n for n in ast.parse(codigo).body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=funciones, type_ignores=[]), "notebook14", "exec"), espacio)
        with tempfile.TemporaryDirectory() as temporal:
            repo = Path(temporal) / "repo"
            raiz = Path(temporal) / "drive" / "resultados"
            referencia = raiz / espacio["REFERENCIA"] / "config.json"
            referencia.parent.mkdir(parents=True)
            referencia.write_text("{}")
            modelo = Mock()
            modelo.predict.return_value = np.array([[0.8]])
            modelo.save.side_effect = lambda destino: destino.write_bytes(b"modelo final de prueba")
            extractor = SimpleNamespace(trainable=True, layers=[])
            carpetas = []

            def etapa_local(modelo, etapa, lr, epocas, train_ds, val_ds, config, carpeta):
                carpetas.append(carpeta)
                self.assertTrue(carpeta.is_relative_to(repo / "resultados_locales"))
                self.assertFalse(carpeta.is_relative_to(raiz))
                (carpeta / f"checkpoint_{etapa}.keras").write_bytes(etapa.encode())
                return pd.DataFrame({"loss": [0.2], "val_loss": [0.3], "etapa": [etapa], "epoca": [1]})

            espacio.update(repo=repo, rutas=SimpleNamespace(resultados_equipo=raiz),
                resultados=SimpleNamespace(carpeta_corrida=resultados.carpeta_corrida, guardar_corrida=Mock()),
                keras=SimpleNamespace(backend=SimpleNamespace(clear_session=Mock()),
                    utils=SimpleNamespace(set_random_seed=Mock())),
                armar_modelo=Mock(return_value=(modelo, extractor)), armar_ds=Mock(),
                entrenar_etapa=etapa_local, archivos={"val": ["a.png"]},
                fuentes_train=["a", "a"], y_train=np.array([0, 1]),
                X_train="train", X_val="val", y_val=np.array([1]), PESOS_VAL=np.ones(1))
            config = dict(semilla=42, sample_weight=None, umbral=0.5,
                          lr_cabeza=1e-3, epocas_cabeza=30, lr_fine_tuning=1e-5, epocas_fine_tuning=20)
            espacio["correr_experimento"]("sin_early_stopping", config)
            destino = raiz / "mobilenetv2" / "sin_early_stopping"
            self.assertEqual(len(carpetas), 2)
            for etapa in ("cabeza", "fine_tuning"):
                self.assertEqual((destino / f"checkpoint_{etapa}.keras").read_bytes(), etapa.encode())
            self.assertEqual((destino / "modelo.keras").read_bytes(), b"modelo final de prueba")
            espacio["resultados"].guardar_corrida.assert_called_once()
            self.assertFalse(espacio["PENDIENTES_GUARDADO"])

    def test_guardado_no_crea_carpeta_si_se_perdio_acceso_a_drive(self):
        espacio = {"pd": pd, "np": np, "MODELO": "mobilenetv2",
                   "REFERENCIA": "mobilenetv2/imagenet_finetuning"}
        codigo = next(c for c in celdas("14_variables_clasificacion") if "def guardar_pendiente" in c)
        funciones = [n for n in ast.parse(codigo).body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=funciones, type_ignores=[]), "notebook14", "exec"), espacio)
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal) / "drive_inaccesible" / "resultados"
            modelo = Mock()
            espacio.update(rutas=SimpleNamespace(resultados_equipo=raiz), resultados=Mock(),
                           PENDIENTES_GUARDADO={"sin_aug": {"modelo": modelo}})
            with self.assertRaisesRegex(FileNotFoundError, "Conservar esta sesión"):
                espacio["guardar_pendiente"]("sin_aug")
            self.assertFalse(raiz.exists())
            espacio["resultados"].carpeta_corrida.assert_not_called()
            modelo.save.assert_not_called()
            self.assertIn("sin_aug", espacio["PENDIENTES_GUARDADO"])


if __name__ == "__main__":
    unittest.main()
