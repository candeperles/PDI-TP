"""Controles del derivado y las salidas, sin dataset ni PyTorch en CI."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from pdi_tp import deteccion


class TensorFalso:
    def __init__(self, valores):
        self.valores = np.asarray(valores)

    def cpu(self):
        return self.valores


def prediccion(nombre, conf=(), clases=None):
    return SimpleNamespace(path=f"/images/val/{nombre}", boxes=SimpleNamespace(
        conf=TensorFalso(conf), cls=TensorFalso([0] * len(conf) if clases is None else clases),
        xywhn=TensorFalso([[0.5, 0.5, 0.1, 0.1]] * len(conf))))


class DeteccionTest(unittest.TestCase):
    def test_alinea_por_nombre_y_conserva_negativos_y_cajas_de_confianza_baja(self):
        puntajes, cajas = deteccion.resumir_predicciones([
            prediccion("b.png"), prediccion("a.png", [0.02, 0.8])], ["a.png", "b.png"])
        np.testing.assert_allclose(puntajes, [0.8, 0.0])
        self.assertEqual(len(cajas), 2)
        self.assertEqual(cajas[0]["confianza"], 0.02)

    def test_no_convierte_salida_faltante_en_negativo(self):
        casos = [[], [prediccion("otro.png")],
                 [prediccion("a.png"), prediccion("a.png")],
                 [prediccion("a.png", [0.5], [1])],
                 [prediccion("a.png", [np.nan])],
                 [prediccion("a.png", [1.1])],
                 [SimpleNamespace(path="a.png", boxes=None)]]
        for caso in casos:
            with self.subTest(caso=caso), self.assertRaises(ValueError):
                deteccion.resumir_predicciones(caso, ["a.png"])

    def test_derivado_rgb_mapea_confianzas_conserva_caja_pequena_y_negativo(self):
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            datos = base / "original"
            imagenes = datos / "images"
            imagenes.mkdir(parents=True)
            nombres = ["solarpanels_hd_1.png", "kingston2_1_33_12.png"]
            for n in nombres:
                Image.new("P", (256, 256)).save(imagenes / n)
            originales = {n: hashlib.sha256((imagenes / n).read_bytes()).hexdigest() for n in nombres}
            tabla = pd.DataFrame(dict(archivo=nombres, split=["train", "val"],
                sha256=list(originales.values()), sha256_etiqueta=["a", "b"],
                y_original=[1, 0], pendiente_revision=[False, False]))
            cajas = pd.DataFrame([
                dict(archivo=nombres[0], xc=0.5, yc=0.5, w=0.01, h=0.01,
                     clase_original=2, confianza_original="baja", incluida=True),
                dict(archivo=nombres[0], xc=0.5, yc=0.5, w=0.0, h=0.2,
                     clase_original=0, confianza_original="alta", incluida=False)])
            rutas = SimpleNamespace(datos=datos, imagenes=imagenes)
            destino = base / "derivado"
            with patch.object(deteccion.clasificacion, "preparar_tabla", return_value=(tabla, cajas)):
                deteccion.preparar_dataset(rutas, destino)
                # Reejecutar la preparación no duplica ni invalida las rutas de reanudación.
                deteccion.preparar_dataset(rutas, destino)
                positivo = (destino / "labels/train" / Path(nombres[0]).with_suffix(".txt"))
                linea = positivo.read_text().split()
                self.assertEqual(linea[0], "0")
                self.assertEqual(len(linea), 5)
                self.assertAlmostEqual(float(linea[3]), 0.01)
                self.assertEqual((destino / "labels/val" / Path(nombres[1]).with_suffix(".txt")).read_text(), "")
                with Image.open(destino / "images/train" / nombres[0]) as img:
                    self.assertEqual(img.mode, "RGB")
                self.assertEqual(len((destino / "jamaica.txt").read_text().splitlines()), 1)
                yaml = json.loads((destino / "datos_jamaica.yaml").read_text())
                self.assertEqual(yaml["val"], "jamaica.txt")
                self.assertEqual(yaml["names"], {"0": "panel"})
                self.assertEqual(pd.read_csv(destino / "cajas_derivadas.csv").iloc[0].confianza_original, "baja")
                self.assertEqual({n: hashlib.sha256((imagenes / n).read_bytes()).hexdigest() for n in nombres}, originales)
                positivo.write_text("")
                with self.assertRaisesRegex(ValueError, "modificado"):
                    deteccion.preparar_dataset(rutas, destino)
            with self.assertRaisesRegex(ValueError, "original"):
                deteccion.preparar_dataset(rutas, datos / "derivado")

    def test_notebook_codigo_valido_y_dos_corridas_independientes(self):
        raiz = Path(__file__).resolve().parent.parent
        nb = json.loads((raiz / "notebooks/08_deteccion_yolov8n.ipynb").read_text(encoding="utf-8"))
        celdas = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, codigo in enumerate(celdas):
            compile(codigo, f"celda_{i}", "exec")
        opciones = {}
        exec(celdas[0], opciones)
        self.assertEqual(opciones["MODOS"], {"coco_completo": 0, "coco_backbone_congelado": 10})
        self.assertLess(opciones["CONF_MIN"], opciones["UMBRAL"])


if __name__ == "__main__":
    unittest.main()
