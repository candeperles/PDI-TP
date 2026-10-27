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

    def notebook(self, codigo, salidas=False):
        ruta = self.dir / "prueba.ipynb"
        celda = {"cell_type": "code", "metadata": {}, "source": [codigo],
                 "outputs": [{"output_type": "stream", "name": "stdout", "text": ["x"]}] if salidas else [],
                 "execution_count": 1 if salidas else None}
        ruta.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [celda]}))
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

    def test_excepcion_solo_libera_la_regla_de_drive(self):
        ruta = RAIZ / next(iter(verificar.EXCEPCIONES))
        readme = (RAIZ / "README.md").read_text(encoding="utf-8")
        self.assertEqual(verificar.verificar(ruta, readme), [])
        self.assertIn("README", "\n".join(verificar.verificar(ruta, readme="")))

    def test_exige_enlace_en_readme(self):
        ruta = self.dir / "prueba.ipynb"
        self.notebook(ENTRENAMIENTO_OK)
        self.assertIn("README", "\n".join(verificar.verificar(ruta, readme="")))


if __name__ == "__main__":
    unittest.main()
