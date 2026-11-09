"""Controles del notebook 16 y sus etapas, con dobles de Keras en CPU."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
NB = json.loads((RAIZ / "notebooks/16_validacion_cruzada_jamaica.ipynb").read_text(encoding="utf-8"))
CODIGO = "\n\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code")


def funciones(nombres, entorno):
    arbol = ast.parse(CODIGO)
    arbol.body = [n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name in nombres]
    exec(compile(arbol, "cv_notebook", "exec"), entorno)
    return entorno


class CvNotebookTest(unittest.TestCase):
    def test_notebook_limpio_valido_y_entornos_separados(self):
        self.assertEqual(NB["metadata"]["pdi_tp"]["flujo"], "cv_jamaica_v1")
        ast.parse(CODIGO)
        for c in NB["cells"]:
            if c["cell_type"] == "code":
                self.assertIsNone(c["execution_count"])
                self.assertEqual(c["outputs"], [])
        self.assertIn('requirements-yolo26.txt', CODIGO)
        self.assertIn('requirements-detection.txt', CODIGO)
        # El análisis no puede importar módulos que requieran GPU/TensorFlow.
        arbol = ast.parse(CODIGO)
        importaciones_directas = [ast.unparse(n) for n in arbol.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("tensorflow" in n or "aumentacion" in n for n in importaciones_directas))

    def test_fine_tuning_parte_de_su_cabeza_y_no_valida_sobre_el_fold_evaluado(self):
        class BatchNorm:
            trainable = True
        bn, convolucion = BatchNorm(), SimpleNamespace(trainable=False)
        base = SimpleNamespace(trainable=False, layers=[bn, convolucion])
        modelo = Mock()
        modelo.predict.return_value = np.array([[0.3], [0.9]])
        crear_ds = Mock(side_effect=lambda t, r, entrenamiento=False: t)
        etapas = []
        def entrenar_etapa(k, rutas, carpeta, local, red, train_ds, val_ds, receta, etapa, **kw):
            etapas.append(etapa)
            self.assertIs(red, modelo)
            self.assertEqual(train_ds, "train_del_fold")
            self.assertEqual(val_ds, "val_no_jamaica")
            if etapa == "fine_tuning":
                self.assertEqual(etapas, ["cabeza", "fine_tuning"])
                self.assertTrue(base.trainable)
                self.assertTrue(convolucion.trainable)
                self.assertFalse(bn.trainable)
            return red, dict(epocas=1, mejor_epoca=1, segundos=1.0), None
        keras = SimpleNamespace(backend=SimpleNamespace(clear_session=Mock()),
            utils=SimpleNamespace(set_random_seed=Mock()), layers=SimpleNamespace(BatchNormalization=BatchNorm))
        entorno = funciones({"entrenar_cnn"}, dict(keras=keras, MODELO="resnet50", dataset_cnn=crear_ds,
            construir_cnn=Mock(return_value=(modelo, base)), rutas="rutas",
            reanudacion_cnn=SimpleNamespace(cargar=Mock(return_value=None), entrenar_etapa=entrenar_etapa)))
        fold = dict(train="train_del_fold", validacion="val_no_jamaica", evaluacion="jamaica_afuera")
        with tempfile.TemporaryDirectory() as tmp:
            prob, cajas, entrenamiento = entorno["entrenar_cnn"](fold, {"semilla": 42}, Path(tmp), Path(tmp))
        modelo.predict.assert_called_once_with("jamaica_afuera", verbose=0)
        self.assertIsNone(cajas)
        self.assertEqual(set(entrenamiento["etapas"]), {"cabeza", "fine_tuning"})

    def test_retomar_fine_tuning_no_repite_cabeza_ni_descongela_batchnorm(self):
        class ModeloBase:
            pass
        base = ModeloBase()
        modelo = Mock()
        modelo.layers = [base]
        modelo.predict.return_value = np.array([[0.8]])
        cabeza = dict(epocas=3, mejor_epoca=2, segundos=2.0)
        estado = dict(etapa="fine_tuning", etapas_anteriores={"cabeza": cabeza})
        reanudar = SimpleNamespace(cargar=Mock(return_value=estado),
            entrenar_etapa=Mock(return_value=(modelo, dict(segundos=4.0), None)))
        keras = SimpleNamespace(backend=SimpleNamespace(clear_session=Mock()), Model=ModeloBase,
            utils=SimpleNamespace(set_random_seed=Mock()), models=SimpleNamespace(load_model=Mock(return_value=modelo)))
        crear = Mock()
        entorno = funciones({"entrenar_cnn"}, dict(keras=keras, MODELO="resnet50", rutas="rutas",
            dataset_cnn=lambda tabla, receta, **kw: tabla, construir_cnn=crear, reanudacion_cnn=reanudar))
        with tempfile.TemporaryDirectory() as tmp:
            _, _, resumen = entorno["entrenar_cnn"](
                dict(train="train", validacion="val", evaluacion="fuera"), {"semilla": 42}, Path(tmp), Path(tmp))
        crear.assert_not_called()
        reanudar.entrenar_etapa.assert_called_once()
        self.assertEqual(reanudar.entrenar_etapa.call_args.args[-1], "fine_tuning")
        self.assertIs(reanudar.entrenar_etapa.call_args.kwargs["anterior"], estado)
        self.assertEqual(resumen["segundos"], 6.0)

    def test_bucle_salta_completos_y_entrena_solo_el_primer_pendiente(self):
        arbol = ast.parse(CODIGO)
        bucle = next(n for n in arbol.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                     and n.test.id == "ENTRENAR"
                     and any(isinstance(h, ast.Name) and h.id == "FOLDS" for h in ast.walk(n)))
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            (carpeta / "modelo.keras").write_bytes(b"modelo")
            (carpeta / "ultimo.keras").write_bytes(b"checkpoint local")
            destino = carpeta / "drive"
            destino.mkdir()
            cv = SimpleNamespace(VERSIONES_YOLO={}, FINALISTAS={"resnet50": "imagenet_finetuning"},
                referencia=Mock(return_value={"receta": "receta"}),
                fold_completo=Mock(side_effect=lambda r, m, mo, f, p, **kw: f["fold"] < 3),
                preparar_fold=Mock(return_value=destino), guardar_fold=Mock())
            entrenar = Mock(return_value=([0.8], None, dict(segundos=2.0)))
            entorno = dict(ENTRENAR=True, FOLDS=[1, 2, 3, 4, 5], REPETIR=False, MODELO="resnet50",
                cv_jamaica=cv, split=SimpleNamespace(ESTRATEGIA_FOLDS_JAMAICA="estrategia"),
                HUELLA_DATOS="huella", gpus=["T4"], rutas="rutas", repo=carpeta,
                folds=[dict(fold=i) for i in range(1, 6)], platform=SimpleNamespace(python_version=lambda: "3"),
                version=lambda p: "version", Path=Path, tempfile=SimpleNamespace(mkdtemp=lambda **kw: str(carpeta)),
                time=SimpleNamespace(perf_counter=lambda: 1.0), entrenar_cnn=entrenar,
                shutil=__import__("shutil"), gc=SimpleNamespace(collect=lambda: None))
            exec(compile(ast.Module(body=[bucle], type_ignores=[]), "bucle", "exec"), entorno)
            self.assertEqual([c.args[3]["fold"] for c in cv.fold_completo.call_args_list], [1, 2, 3])
            entrenar.assert_called_once()
            self.assertEqual(entrenar.call_args.args[0]["fold"], 3)
            self.assertTrue((destino / "modelo.keras").is_file())
            self.assertFalse((destino / "ultimo.keras").exists())

    def test_detector_entrena_con_validacion_separada_y_guarda_la_epoca_de_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp)
            modelo, final = Mock(), Mock()
            estado = {}
            modelo.add_callback.side_effect = lambda evento, callback: estado.update(callback=callback)
            def entrenar(**kw):
                yaml = json.loads(Path(kw["data"]).read_text())
                self.assertEqual((local / yaml["val"]).read_text().strip(), "/derivado/images/val/PV01_300_100.png")
                self.assertNotIn("kingston", (local / yaml["val"]).read_text())
                pesos = local / "entrenamiento" / "weights"
                pesos.mkdir(parents=True)
                (pesos / "best.pt").write_bytes(b"checkpoint ficticio")
                pd.DataFrame(dict(epoch=[1, 2, 3], loss=[1, 0.3, 0.4])).to_csv(pesos.parent / "results.csv", index=False)
                (pesos.parent / "args.yaml").write_text("imgsz: 640\n")
                for i, valor in enumerate([0.2, 0.5, 0.3]):
                    estado["callback"](SimpleNamespace(epoch=i, fitness=valor, best_fitness=max(0.5 if i else 0.2, valor)))
            modelo.train.side_effect = entrenar
            yolo = Mock(side_effect=[modelo, final])
            resumir = Mock(return_value=(np.array([0.8]), []))
            entorno = funciones({"entrenar_detector"}, dict(YOLO=yolo, MODELO="yolo26n", json=json,
                pd=pd, np=np, shutil=__import__("shutil"), deteccion=SimpleNamespace(resumir_predicciones=resumir)))
            fold = dict(train=pd.DataFrame([dict(split="train", archivo="kingston_1_0_0_0.png")]),
                validacion=pd.DataFrame([dict(split="val", archivo="PV01_300_100.png")]),
                evaluacion=pd.DataFrame([dict(split="val", archivo="kingston_30_0_0_0.png")]))
            receta = dict(imgsz=640, batch=16, conf=0.001, iou=0.7, max_det=300)
            # Las rutas POSIX del notebook se comprueban también al correr CI en Windows.
            derivado = __import__("pathlib").PurePosixPath("/derivado")
            _, _, entrenamiento = entorno["entrenar_detector"](fold, receta, local, derivado)
            self.assertEqual(entrenamiento["etapas"]["deteccion"]["mejor_epoca"], 2)
            self.assertEqual(yolo.call_args_list[0].args, ("yolo26n.pt",))
            self.assertIsNone(final.predict.call_args.kwargs["nms"])


if __name__ == "__main__":
    unittest.main()
