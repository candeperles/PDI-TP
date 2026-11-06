"""Controles del notebook 18 y sus etapas, con dobles de Keras en CPU."""
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
NB = json.loads((RAIZ / "notebooks/18_validacion_cruzada_jamaica.ipynb").read_text(encoding="utf-8"))
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

    def test_restaurar_mejor_epoca_aun_si_se_agota_el_presupuesto(self):
        class Callback:
            pass
        modelo = Mock()
        estado = {"epoca": 0}
        modelo.get_weights.side_effect = lambda: [np.array([estado["epoca"]])]
        def entrenar(*args, **kw):
            mejor = kw["callbacks"][0]
            mejor.model = modelo
            for epoca, perdida in enumerate([0.3, 0.1, 0.2]):
                estado["epoca"] = epoca
                mejor.on_epoch_end(epoca, {"val_loss": perdida})
            return SimpleNamespace(history=dict(loss=[0.5, 0.3, 0.2], val_loss=[0.3, 0.1, 0.2]))
        modelo.fit.side_effect = entrenar
        keras = SimpleNamespace(callbacks=SimpleNamespace(Callback=Callback, EarlyStopping=Mock(), TerminateOnNaN=Mock()),
            optimizers=SimpleNamespace(Adam=Mock()), metrics=SimpleNamespace(BinaryAccuracy=Mock()))
        entorno = funciones({"entrenar_etapa_cnn"}, dict(keras=keras, np=np, pd=pd))
        receta = dict(lr_cabeza=1e-3, epocas_cabeza=30, paciencia=10, loss="binary_crossentropy")
        with tempfile.TemporaryDirectory() as tmp:
            resultado = entorno["entrenar_etapa_cnn"](modelo, "train", "val_no_jamaica", receta, "cabeza", Path(tmp))
        self.assertEqual(resultado["mejor_epoca"], 2)
        self.assertEqual(resultado["epocas"], 3)
        np.testing.assert_array_equal(modelo.set_weights.call_args.args[0][0], [1])
        self.assertEqual(modelo.fit.call_args.kwargs["validation_data"], "val_no_jamaica")
        keras.callbacks.EarlyStopping.assert_called_once_with(monitor="val_loss", patience=10, restore_best_weights=True)

    def test_fine_tuning_parte_de_su_cabeza_y_no_valida_sobre_el_fold_evaluado(self):
        class BatchNorm:
            trainable = True
        bn, convolucion = BatchNorm(), SimpleNamespace(trainable=False)
        base = SimpleNamespace(trainable=False, layers=[bn, convolucion])
        modelo = Mock()
        modelo.predict.return_value = np.array([[0.3], [0.9]])
        crear_ds = Mock(side_effect=lambda t, r, entrenamiento=False: t)
        etapas = []
        def entrenar_etapa(red, train_ds, val_ds, receta, etapa, local):
            etapas.append(etapa)
            self.assertIs(red, modelo)
            self.assertEqual(train_ds, "train_del_fold")
            self.assertEqual(val_ds, "val_no_jamaica")
            if etapa == "fine_tuning":
                self.assertEqual(etapas, ["cabeza", "fine_tuning"])
                self.assertTrue(base.trainable)
                self.assertTrue(convolucion.trainable)
                self.assertFalse(bn.trainable)
            return dict(epocas=1, mejor_epoca=1)
        keras = SimpleNamespace(backend=SimpleNamespace(clear_session=Mock()),
            utils=SimpleNamespace(set_random_seed=Mock()), layers=SimpleNamespace(BatchNormalization=BatchNorm))
        entorno = funciones({"entrenar_cnn"}, dict(keras=keras, MODELO="resnet50", dataset_cnn=crear_ds,
            construir_cnn=Mock(return_value=(modelo, base)), entrenar_etapa_cnn=entrenar_etapa))
        fold = dict(train="train_del_fold", validacion="val_no_jamaica", evaluacion="jamaica_afuera")
        with tempfile.TemporaryDirectory() as tmp:
            prob, cajas, entrenamiento = entorno["entrenar_cnn"](fold, {"semilla": 42}, Path(tmp))
        modelo.predict.assert_called_once_with("jamaica_afuera", verbose=0)
        self.assertIsNone(cajas)
        self.assertEqual(set(entrenamiento["etapas"]), {"cabeza", "fine_tuning"})

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
