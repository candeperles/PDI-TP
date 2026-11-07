"""El verificador de notebooks rechaza lo que no cumple las normas del equipo."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verificar", RAIZ / "scripts" / "verificar_notebooks.py")
verificar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verificar)

ENTRENAMIENTO_OK = (
    "rutas = entorno.preparar()\n"
    "modelo = keras.applications.MobileNetV2()\n"
    "resultados.guardar_corrida(rutas, carpeta, archivos, prob, params)\n"
)


class VerificadorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=RAIZ / "notebooks")
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def notebook(self, codigo, salidas=False, metadata=None):
        ruta = self.dir / "prueba.ipynb"
        celda = {"cell_type": "code", "metadata": {}, "source": [codigo],
                 "outputs": [{"output_type": "stream", "name": "stdout", "text": ["x"]}] if salidas else [],
                 "execution_count": 1 if salidas else None}
        ruta.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": metadata or {}, "cells": [celda]}))
        relativa = ruta.relative_to(RAIZ).as_posix()
        return verificar.verificar(ruta, readme=verificar.COLAB + relativa)

    def test_cumple(self):
        self.assertEqual(self.notebook(ENTRENAMIENTO_OK), [])

    def test_rechaza_salidas_integrante_y_guardado_viejo(self):
        problemas = "\n".join(self.notebook(
            'INTEGRANTE = "nombre"\nmodel.fit(x)\nresultados.guardar_experimento(rutas, "a", {}, {})\n', salidas=True))
        for esperado in ("salidas", "INTEGRANTE", "guardar_experimento", "guardar_corrida"):
            self.assertIn(esperado, problemas)

    def test_rechaza_api_de_drive_y_borrados(self):
        problemas = "\n".join(self.notebook(ENTRENAMIENTO_OK + "from googleapiclient.discovery import build\n"
                                            "shutil.rmtree(carpeta)\ndrive.mount('/content/drive')\n"))
        for esperado in ("googleapiclient", "rmtree", "drive.mount"):
            self.assertIn(esperado, problemas)

    def test_notebook_sin_modelo_no_monta_drive(self):
        self.assertIn("usar_drive=False", "\n".join(self.notebook("rutas = entorno.preparar()\nprint(1)\n")))
        self.assertEqual(self.notebook("rutas = entorno.preparar(usar_drive=False)\nprint(1)\n"), [])

    def test_solo_lectura_monta_drive_pero_no_escribe_ni_entrena(self):
        lectura = "rutas = entorno.preparar(solo_lectura=True)\ntabla = resultados.cargar_corridas(rutas)\n"
        self.assertEqual(self.notebook(lectura), [])
        for agregado, esperado in [("tabla.to_csv('x.csv')\n", ".to_csv("), ("fig.savefig('x.png')\n", ".savefig("),
                                   ("resultados.carpeta_corrida(rutas, 'a', 'b')\n", "carpeta_corrida("),
                                   ("modelo.fit(x)\n", "no entrenan"),
                                   ("modelo = YOLO('a.pt')\nmodelo.train(data='d.yaml')\n", "no entrenan")]:
            with self.subTest(agregado=agregado):
                self.assertIn(esperado, "\n".join(self.notebook(lectura + agregado)))

    def test_solo_lectura_puede_cargar_modelos_para_predecir(self):
        lectura = ("rutas = entorno.preparar(solo_lectura=True)\nfrom ultralytics import YOLO\n"
                   "detector = YOLO('modelo.pt')\nsalidas = detector.predict(imagenes)\n"
                   "red = keras.models.load_model('modelo.keras', compile=False)\nprob = red.predict(imagenes)\n")
        self.assertEqual(self.notebook(lectura), [])

    def test_exige_enlace_en_readme(self):
        ruta = self.dir / "prueba.ipynb"
        self.notebook(ENTRENAMIENTO_OK)
        self.assertIn("README", "\n".join(verificar.verificar(ruta, readme="")))

    def test_flujo_cv_acordado_exige_protocolo_completo_y_no_strings(self):
        codigo = ("rutas = entorno.preparar(solo_lectura=not ENTRENAR)\n"
                  "folds = split.folds_jamaica(rutas)\n"
                  "cv_jamaica.fold_completo(rutas, modelo, modo, fold, params)\n"
                  "carpeta = cv_jamaica.preparar_fold(rutas, modelo, modo, fold, params)\n"
                  "modelo.fit(train, validation_data=validacion)\n"
                  "cv_jamaica.guardar_fold(rutas, carpeta, modelo, modo, fold, params, prob, entrenamiento)\n"
                  "cv_jamaica.cargar_predicciones(rutas, folds)\n")
        metadata = {"pdi_tp": {"flujo": "cv_jamaica_v1"}}
        relativa = (self.dir / "prueba.ipynb").relative_to(RAIZ).as_posix()
        original = verificar.FLUJOS_CV
        self.addCleanup(setattr, verificar, "FLUJOS_CV", original)
        verificar.FLUJOS_CV = {relativa: "cv_jamaica_v1"}
        self.assertEqual(self.notebook(codigo, metadata=metadata), [])
        for llamada in ("cv_jamaica.guardar_fold", "cv_jamaica.preparar_fold", "split.folds_jamaica"):
            alterado = "\n".join(repr(linea) if llamada in linea else linea for linea in codigo.splitlines())
            self.assertIn(llamada, "\n".join(self.notebook(alterado, metadata=metadata)))
        self.assertIn("CV no puede escribir corridas comunes", "\n".join(self.notebook(
            codigo + "resultados.guardar_corrida(rutas, carpeta)\n", metadata=metadata)))
        self.assertIn("solo_lectura=not ENTRENAR", "\n".join(self.notebook(
            codigo.replace("solo_lectura=not ENTRENAR", ""), metadata=metadata)))
        self.assertIn("salidas", "\n".join(self.notebook(codigo, salidas=True, metadata=metadata)))
        self.assertIn("googleapiclient", "\n".join(self.notebook(codigo + "import googleapiclient\n", metadata=metadata)))

    def test_cv_no_habilita_otros_notebooks_ni_modifica_entrenamiento_comun(self):
        problemas = self.notebook("modelo.fit(x)\n", metadata={"pdi_tp": {"flujo": "cv_jamaica_v1"}})
        self.assertIn("flujo de CV no autorizado", "\n".join(problemas))
        self.assertIn("guardar_corrida", "\n".join(problemas))
        self.assertEqual(self.notebook(ENTRENAMIENTO_OK), [])


if __name__ == "__main__":
    unittest.main()
