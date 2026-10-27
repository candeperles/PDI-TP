"""Formato único de resultados: carpeta por modelo/modo con config, métricas, predicciones y pesos."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pdi_tp import resultados
from pdi_tp.entorno import Rutas

VAL = ["PV01_100_100.png", "kingston_1_0_0_0.png", "solarpanels_hd_1__x0_0_y0_0_dxdy_832_0_0.png"]
CON_PANELES = {"PV01_100_100", "solarpanels_hd_1__x0_0_y0_0_dxdy_832_0_0"}


class ResultadosTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raiz = Path(tmp.name)
        (raiz / "splits").mkdir()
        (raiz / "splits" / "val.txt").write_text("\n".join(VAL) + "\n")
        (raiz / "splits" / "train.txt").write_text("otra.png\n")
        labels = raiz / "labels"
        labels.mkdir()
        for nombre in VAL:
            stem = Path(nombre).stem
            (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n" if stem in CON_PANELES else "")
        self.rutas = Rutas(raiz, raiz, raiz / "images", labels, raiz / "resultados", raiz / "resultados")
        resultados.split.SPLITS_DIR = raiz / "splits"
        self.addCleanup(setattr, resultados.split, "SPLITS_DIR", resultados.split.SPLITS_DIR)

    def corrida(self, modo="modo_a", pesos=True):
        carpeta = resultados.carpeta_corrida(self.rutas, "modelo-x", modo)
        if pesos:
            (carpeta / "modelo.keras").write_bytes(b"pesos")
        return carpeta

    def test_guarda_formato_completo_con_y_real_del_dataset(self):
        carpeta = self.corrida()
        metricas = resultados.guardar_corrida(self.rutas, carpeta, VAL[::-1], [0.9, 0.2, 0.8], {"lr": 1e-3},
                                              metricas_extra={"epocas": 3})
        self.assertEqual(resultados.verificar_corrida(carpeta), [])
        pred = pd.read_csv(carpeta / "predicciones_val.csv")
        self.assertEqual(list(pred.columns), list(resultados.COLUMNAS_PREDICCIONES))
        self.assertEqual(dict(zip(pred.archivo, pred.y_real)), {VAL[0]: 1, VAL[1]: 0, VAL[2]: 1})
        self.assertEqual(metricas["matriz_confusion"], [[1, 0], [0, 2]])
        config = json.loads((carpeta / "config.json").read_text())
        self.assertEqual((config["modelo"], config["modo"]), ("modelo-x", "modo_a"))
        self.assertNotIn("integrante", config)
        self.assertEqual(json.loads((carpeta / "metricas.json").read_text())["extra"], {"epocas": 3})

    def test_sin_pesos_frena_salvo_que_se_declare(self):
        carpeta = self.corrida(pesos=False)
        with self.assertRaisesRegex(RuntimeError, "pesos"):
            resultados.guardar_corrida(self.rutas, carpeta, VAL, [0.5] * 3, {})
        resultados.guardar_corrida(self.rutas, carpeta, VAL, [1.0] * 3, {}, sin_pesos=True)
        self.assertTrue(resultados.corrida_completa(carpeta))

    def test_exige_exactamente_val_y_probabilidades_validas(self):
        carpeta = self.corrida()
        with self.assertRaisesRegex(ValueError, "val"):
            resultados.guardar_corrida(self.rutas, carpeta, VAL[:2], [0.5, 0.5], {})
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            resultados.guardar_corrida(self.rutas, carpeta, VAL, [0.5, 1.5, 0.1], {})
        with self.assertRaises(ValueError):
            resultados.carpeta_corrida(self.rutas, "Modelo Con Espacios", "modo")

    def test_reiniciar_archiva_sin_borrar_y_cargar_ignora_archivadas(self):
        carpeta = self.corrida()
        resultados.guardar_corrida(self.rutas, carpeta, VAL, [0.9, 0.1, 0.9], {})
        nueva = resultados.carpeta_corrida(self.rutas, "modelo-x", "modo_a", reiniciar=True)
        self.assertEqual(list(nueva.iterdir()), [])
        archivadas = [p for p in nueva.parent.iterdir() if resultados.ARCHIVADA in p.name]
        self.assertEqual(len(archivadas), 1)
        self.assertTrue((archivadas[0] / "modelo.keras").is_file())
        self.assertTrue(resultados.cargar_corridas(self.rutas).empty)

    def test_cargar_corridas_y_predicciones(self):
        for modo in ("modo_a", "modo_b"):
            resultados.guardar_corrida(self.rutas, self.corrida(modo), VAL, np.array([0.9, 0.1, 0.9]), {"m": modo})
        tabla = resultados.cargar_corridas(self.rutas)
        self.assertEqual(list(tabla.modo), ["modo_a", "modo_b"])
        self.assertEqual(tabla.f1_macro.tolist(), [1.0, 1.0])
        self.assertEqual(len(resultados.cargar_predicciones(self.rutas)), 6)

    def test_es_jamaica_por_nombre_y_coordenadas(self):
        es_jamaica = resultados.split.es_jamaica
        self.assertTrue(es_jamaica("kingston_139_0_1_23.png"))
        self.assertTrue(es_jamaica("kingston2_71_59_10"))
        self.assertTrue(es_jamaica("2163483_unknown_17.96567005007602_-77.24333773013821_0_1.png"))
        self.assertFalse(es_jamaica("2163483_unknown_-34.6_-58.4_0_1.png"))
        self.assertFalse(es_jamaica("PV01_325222_1204189.png"))
        self.assertFalse(es_jamaica("solarpanels_hd_1__x0_0_y0_0_dxdy_832_0_0.png"))

    def test_metricas_jamaica_solo_con_imagenes_de_jamaica(self):
        # Solo kingston_1_0_0_0 es de Jamaica (sin paneles): con p=0.2 acierta; el resto no cuenta.
        metricas = resultados.guardar_corrida(self.rutas, self.corrida(), VAL, [0.1, 0.2, 0.1], {})
        self.assertEqual(metricas["jamaica"]["n"], 1)
        self.assertEqual(metricas["jamaica"]["matriz_confusion"], [[1, 0], [0, 0]])
        self.assertEqual(metricas["jamaica"]["recall_sin_paneles"], 1.0)
        self.assertAlmostEqual(metricas["accuracy"], 1 / 3)

    def test_cargar_corridas_recalcula_jamaica_en_corridas_viejas(self):
        carpeta = self.corrida()
        resultados.guardar_corrida(self.rutas, carpeta, VAL, [0.1, 0.9, 0.1], {})
        archivo = carpeta / "metricas.json"
        metricas = json.loads(archivo.read_text())
        del metricas["jamaica"]  # como una corrida guardada antes de este cambio
        archivo.write_text(json.dumps(metricas))
        fila = resultados.cargar_corridas(self.rutas).iloc[0]
        self.assertEqual(fila["jamaica.n"], 1)
        self.assertEqual(fila["jamaica.recall_sin_paneles"], 0.0)
        self.assertEqual(fila["jamaica.matriz_confusion"], [[0, 1], [0, 0]])


if __name__ == "__main__":
    unittest.main()
