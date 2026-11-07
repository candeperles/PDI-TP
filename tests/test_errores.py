"""Galería: correspondencia de cajas, integridad de exportaciones y escala visual."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pdi_tp import errores


def propuestas(filas=()):
    return pd.DataFrame(filas, columns=["confianza", *errores.COORDENADAS])


class EmparejamientoTest(unittest.TestCase):
    def test_iou_exacta_parcial_disjunta_y_vacios(self):
        real = [[0.5, 0.5, 0.4, 0.4]]
        pred = [[0.5, 0.5, 0.4, 0.4], [0.7, 0.5, 0.4, 0.4], [0.05, 0.05, 0.05, 0.05]]
        np.testing.assert_allclose(errores.iou_cajas(real, pred), [[1, 1/3, 0]])
        self.assertEqual(errores.iou_cajas([], pred).shape, (0, 3))
        self.assertEqual(errores.iou_cajas(real, []).shape, (1, 0))

    def test_una_referencia_no_admite_dos_aciertos_y_prioriza_confianza(self):
        real = [[0.5, 0.5, 0.4, 0.4]]
        # La de mayor IoU tiene menor confianza: se empareja primero la otra.
        pred = propuestas([[0.6, 0.5, 0.5, 0.4, 0.4], [0.9, 0.55, 0.5, 0.4, 0.4]])
        copia = pred.copy(deep=True)
        d = errores.diagnosticar_cajas(real, pred)
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (1, 1, 0))
        self.assertEqual(d["coincidencias"][0][:2], (0, 0))
        self.assertEqual(d["falsas"], [1])
        self.assertEqual(len(d["cercanas"]), 0)
        pd.testing.assert_frame_equal(pred, copia)

    def test_caja_cercana_sigue_siendo_fp_y_fn_al_umbral(self):
        d = errores.diagnosticar_cajas([[0.5, 0.5, 0.4, 0.4]],
            propuestas([[0.8, 0.7, 0.5, 0.4, 0.4]]))
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (0, 1, 1))
        self.assertAlmostEqual(d["cercanas"][0][2], 1/3)
        self.assertEqual(d["omitidas"], [])
        self.assertEqual(d["falsas"], [])
        zonas = errores.zonas_error({"yolo/modo": d})
        self.assertIn("P1/R1", zonas[0][0])
        np.testing.assert_allclose(zonas[0][1], [0.3, 0.3, 0.9, 0.7])

    def test_sin_solapamiento_distingue_omision_y_falsa_alarma(self):
        d = errores.diagnosticar_cajas([[0.2, 0.2, 0.1, 0.1]],
            propuestas([[0.9, 0.8, 0.8, 0.1, 0.1]]))
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (0, 1, 1))
        self.assertEqual((d["omitidas"], d["falsas"]), ([0], [0]))
        self.assertEqual(len(errores.zonas_error({"yolo/modo": d})), 2)

    def test_umbral_incluye_igualdad_y_no_cambia_predicciones_originales(self):
        pred = propuestas([[0.249, 0.5, 0.5, 0.2, 0.2], [0.25, 0.5, 0.5, 0.2, 0.2]])
        d = errores.diagnosticar_cajas([], pred)
        self.assertEqual((len(d["predichas"]), d["fp"], d["fn"]), (1, 1, 0))
        d = errores.diagnosticar_cajas([[0.5, 0.5, 0.2, 0.2]], propuestas())
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (0, 0, 1))
        d = errores.diagnosticar_cajas([], propuestas())
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (0, 0, 0))

    def test_rechaza_nan_geometria_degenerada_y_umbrales_incompatibles(self):
        for caja in ([0.5, 0.5, 0, 0.1], [0.5, np.nan, 0.1, 0.1], [2, 0.5, 0.1, 0.1]):
            with self.subTest(caja=caja), self.assertRaises(ValueError):
                errores.diagnosticar_cajas([caja], propuestas())
        with self.assertRaises(ValueError):
            errores.diagnosticar_cajas([], propuestas([[np.inf, 0.5, 0.5, 0.1, 0.1]]))
        for opciones in (dict(iou_min=0.1), dict(confianza=-1), dict(iou_min=1.1)):
            with self.subTest(opciones=opciones), self.assertRaises(ValueError):
                errores.diagnosticar_cajas([], propuestas(), **opciones)


class LecturaTest(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.raiz = Path(temporal.name)
        self.rutas = SimpleNamespace(resultados_equipo=self.raiz, imagenes=self.raiz)
        self.nombres = ["kingston_1_0_0_0.png", "kingston2_1_0_0.png"]
        self.tabla = pd.DataFrame(dict(archivo=self.nombres, split="val", fuente=["kingston", "kingston2"],
            y_original=[1, 0], pendiente_revision=[False, True]))
        self.corridas = pd.DataFrame([dict(modelo="yolov8n", modo="coco_completo", completa=True,
            umbral=0.25, fecha="2026-11-03", **{"params.receta": dict(imgsz=640, conf=0.001)})])
        self.pred = pd.DataFrame(dict(archivo=self.nombres, fuente=["kingston", "kingston2"],
            y_real=[1, 0], probabilidad=[0.8, 0], prediccion=[1, 0], acierto=[1, 1],
            modelo="yolov8n", modo="coco_completo"))
        self.propuestas = pd.DataFrame([[self.nombres[0], 0, 0.8, 0.5, 0.5, 0.2, 0.2]],
                                       columns=errores.COLUMNAS_CAJAS)
        self.carpeta = self.raiz / "yolov8n/coco_completo"
        self.carpeta.mkdir(parents=True)

    def cargar(self, **opciones):
        with patch.object(errores.resultados, "cargar_corridas", return_value=self.corridas), \
                patch.object(errores.resultados, "cargar_predicciones", return_value=self.pred):
            return errores.cargar_revision(self.rutas, self.tabla, **opciones)

    def exportar(self):
        self.propuestas.to_csv(self.carpeta / "cajas_val.csv", index=False)

    def test_lee_alinea_y_no_escribe_ni_modifica_archivos(self):
        self.exportar()
        self.pred = self.pred.iloc[::-1]
        archivo = self.carpeta / "cajas_val.csv"
        huella = hashlib.sha256(archivo.read_bytes()).hexdigest()
        inventario, pred, cajas = self.cargar()
        self.assertTrue(inventario.iloc[0].disponible)
        self.assertEqual(set(pred.archivo), set(self.nombres))
        self.assertEqual(list(cajas), ["yolov8n/coco_completo"])
        self.assertEqual(hashlib.sha256(archivo.read_bytes()).hexdigest(), huella)
        self.assertEqual([p.name for p in self.carpeta.iterdir()], ["cajas_val.csv"])

    def test_csv_faltante_no_se_trata_como_cero_detecciones(self):
        inventario, pred, cajas = self.cargar()
        self.assertFalse(inventario.iloc[0].disponible)
        self.assertIn("Falta cajas_val.csv", inventario.iloc[0].motivo)
        self.assertTrue(pred.empty)
        self.assertEqual(cajas, {})

    def test_exportacion_vacia_es_valida_solo_con_puntajes_cero(self):
        self.propuestas = self.propuestas.iloc[:0]
        self.exportar()
        inventario, _, _ = self.cargar()
        self.assertFalse(inventario.iloc[0].disponible)
        self.pred["probabilidad"] = 0.0
        self.pred["prediccion"] = 0
        self.pred["acierto"] = [0, 1]
        inventario, _, cajas = self.cargar()
        self.assertTrue(inventario.iloc[0].disponible)
        self.assertTrue(cajas["yolov8n/coco_completo"].empty)

    def test_rechaza_cajas_de_otra_corrida_o_fuera_de_val(self):
        for cambio, valor in (("confianza", 0.7), ("archivo", "otra.png"), ("clase", 2), ("w", 0)):
            with self.subTest(cambio=cambio):
                previa = self.propuestas.copy()
                self.propuestas[cambio] = valor
                self.exportar()
                inventario, _, cajas = self.cargar()
                self.assertFalse(inventario.iloc[0].disponible)
                self.assertEqual(cajas, {})
                self.propuestas = previa

    def test_rechaza_predicciones_incompletas_o_etiqueta_distinta(self):
        self.exportar()
        self.pred.loc[0, "y_real"] = 0
        inventario, _, _ = self.cargar()
        self.assertIn("referencia actual", inventario.iloc[0].motivo)
        self.pred = self.pred.iloc[:1]
        inventario, _, _ = self.cargar()
        self.assertIn("exactamente", inventario.iloc[0].motivo)

    def test_recetas_anteriores_no_entran_sin_eleccion_explicita(self):
        self.exportar()
        self.corridas.at[0, "params.receta"] = dict(imgsz=256)
        inventario, _, _ = self.cargar()
        self.assertFalse(inventario.iloc[0].disponible)
        inventario, _, _ = self.cargar(incluir_detectores_anteriores=True)
        self.assertTrue(inventario.iloc[0].disponible)
        self.corridas.loc[0, "modelo"] = "efficientnetb0"
        self.corridas.at[0, "params.receta"] = "comun"
        inventario, _, _ = self.cargar(incluir_detectores_anteriores=True)
        self.assertIn("CNN con receta anterior", inventario.iloc[0].motivo)

    def test_cnn_actual_no_requiere_exportacion_de_cajas(self):
        self.corridas.loc[0, "modelo"] = "efficientnetb0"
        self.corridas.loc[0, "modo"] = "imagenet_finetuning"
        self.corridas.at[0, "params.receta"] = "comun_v2"
        self.pred["modelo"] = "efficientnetb0"
        self.pred["modo"] = "imagenet_finetuning"
        inventario, pred, cajas = self.cargar()
        self.assertTrue(inventario.iloc[0].disponible)
        self.assertEqual(pred.corrida.unique().tolist(), ["efficientnetb0/imagenet_finetuning"])
        self.assertEqual(cajas, {})

    def test_indice_conserva_negativos_pendientes_y_errores_de_cnn(self):
        anotaciones = pd.DataFrame([[self.nombres[0], 0.5, 0.5, 0.2, 0.2, True]],
                                  columns=["archivo", *errores.COORDENADAS, "incluida"])
        # Omitir la imagen positiva en el detector; falso positivo en la negativa.
        cajas = self.propuestas.copy()
        cajas["archivo"] = self.nombres[1]
        self.pred["prediccion"] = [0, 0]
        indice, diagnosticos = errores.revisar_imagenes(self.tabla, anotaciones, {"yolo/modo": cajas}, self.pred)
        self.assertEqual(indice.omitidas.tolist(), [1, 0])
        self.assertEqual(indice.falsas.tolist(), [0, 1])
        self.assertEqual(indice.errores_clasificacion.tolist(), [1, 0])
        self.assertEqual(indice.pendiente_revision.tolist(), [False, True])
        self.assertEqual(diagnosticos[self.nombres[1]]["yolo/modo"]["fn"], 0)


class VistaTest(unittest.TestCase):
    def test_rectangulos_se_escalan_a_imagen_original_y_zoom_compartido(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            Image.new("RGB", (256, 256), "gray").save(raiz / "a.png")
            reales = np.array([[0.5, 0.5, 0.4, 0.4]])
            d = errores.diagnosticar_cajas(reales, propuestas([[0.8, 0.7, 0.5, 0.4, 0.4]]))
            zoom = errores.zonas_error({"yolo/modo": d})[0][1]
            figura = errores.mostrar_imagen(SimpleNamespace(imagenes=raiz), "a.png", reales,
                {"yolo/modo": d}, zoom=zoom, mostrar_aciertos=True)
            self.addCleanup(plt.close, figura)
            self.assertEqual(len(figura.axes), 4)
            figura.canvas.draw()
            self.assertGreater(figura.axes[0].get_position().width, 0.25)
            self.assertAlmostEqual(figura.axes[0].patches[0].get_width(), 102.4)
            self.assertAlmostEqual(figura.axes[0].patches[0].get_x(), 76.8)
            self.assertEqual(figura.axes[2].get_xlim(), figura.axes[3].get_xlim())
            self.assertEqual(figura.axes[2].get_ylim(), figura.axes[3].get_ylim())
            self.assertTrue(any("P1" in t.get_text() and "0.80" in t.get_text() for t in figura.axes[1].texts))
            self.assertTrue(any("IoU=" in t.get_text() for t in figura.axes[1].texts))

    def test_zoom_en_borde_no_sale_de_la_imagen(self):
        x0, y0, x1, y1 = errores.limites_zoom([0, 0, 0.01, 0.01], 256, 256)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 256)
        self.assertLessEqual(y1, 256)
        self.assertGreater(x1 - x0, 20)

    def test_notebook_limpio_codigo_valido_y_lectura_sin_inferencia(self):
        raiz = Path(__file__).resolve().parent.parent
        nb = json.loads((raiz / "notebooks/14_galeria_errores.ipynb").read_text(encoding="utf-8"))
        codigo = []
        for i, celda in enumerate(nb["cells"]):
            if celda["cell_type"] == "code":
                fuente = "".join(celda["source"])
                compile(fuente, f"celda_{i}", "exec")
                self.assertEqual(celda["outputs"], [])
                self.assertIsNone(celda["execution_count"])
                codigo.append(fuente)
        self.assertIn("entorno.preparar(solo_lectura=True)", "\n".join(codigo))
        self.assertNotIn("ultralytics", "\n".join(codigo))


if __name__ == "__main__":
    unittest.main()
