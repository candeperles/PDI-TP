"""Casos sintéticos: la auditoría debe informar problemas sin detener el análisis."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pdi_tp.entorno import Rutas
from pdi_tp.validacion import validar


class ValidacionTest(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raiz = Path(self.temporal.name)
        self.imagenes = self.raiz / "images"
        self.labels = self.raiz / "labels"
        self.splits = self.raiz / "splits"
        for p in (self.imagenes, self.labels, self.splits):
            p.mkdir()
        self.rutas = Rutas(self.raiz, self.raiz, self.imagenes, self.labels,
                           self.raiz / "resultados", self.raiz / "resultados" / "test")
        self.crear("PV01_0_0", "0 0.5 0.5 0.25 0.25\n", "red")
        self.crear("PV01_100_100", "", "blue")
        self.crear("PV01_200_200", "0 0.5 0.5 0.5 0.5\n", "green")
        self.dividir(["PV01_0_0.png", "PV01_100_100.png"], ["PV01_200_200.png"])

    def crear(self, nombre, etiqueta, color):
        Image.new("RGB", (256, 256), color).save(self.imagenes / f"{nombre}.png")
        (self.labels / f"{nombre}.txt").write_text(etiqueta, encoding="utf-8")

    def dividir(self, train, val):
        for nombre, entradas in (("train", train), ("val", val)):
            (self.splits / f"{nombre}.txt").write_text("\n".join(entradas) + "\n", encoding="utf-8")

    def auditar(self):
        return validar(self.rutas, self.splits)

    def test_dataset_sano_y_etiqueta_vacia(self):
        informe = self.auditar()
        self.assertEqual(informe.resumen["errores"], 0)
        self.assertEqual(informe.resumen["advertencias"], 0)
        self.assertEqual(informe.resumen["imagenes_sin_paneles"], 1)
        self.assertEqual(informe.resumen["cajas_validas"], 2)
        salida = self.raiz / "exportado"
        informe.guardar(salida)
        self.assertTrue((salida / "incidencias.csv").is_file())
        self.assertTrue((salida / "resumen.json").is_file())

    def test_faltante_y_formato_invalido_no_son_negativos(self):
        (self.labels / "PV01_100_100.txt").unlink()
        (self.labels / "PV01_0_0.txt").write_text(
            "0 0.5 0.5 0.2\n0 nan 0.5 0.2 0.2\n0 hola 0.5 0.2 0.2\n", encoding="utf-8")
        informe = self.auditar()
        self.assertEqual(informe.resumen["imagenes_sin_paneles"], 0)
        self.assertEqual(informe.resumen["imagenes_etiqueta_desconocida"], 2)
        self.assertTrue({"etiqueta_faltante", "columnas_yolo", "valor_no_finito", "valor_no_numerico"}
                        <= set(informe.incidencias["control"]))

    def test_geometria_clases_y_tolerancia(self):
        (self.labels / "PV01_0_0.txt").write_text(
            "0 0.05 0.5 0.2 0.2\n0 0.5 0.5 0 0.2\n"
            "1.5 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n"
            "0 1.1 0.5 1.2 0.2\n0 0.5 0.5 -0.1 0.2\n"
            "0 0.1 0.5 0.200001 0.2\n0 0.5 0.5 0.001 0.2\n", encoding="utf-8")
        informe = self.auditar()
        self.assertTrue({"caja_fuera_imagen", "dimension_caja_no_positiva", "clase_invalida",
                         "clase_no_admitida", "centro_fuera_rango", "dimension_caja_fuera_rango", "caja_pequena"}
                        <= set(informe.incidencias["control"]))
        caja = informe.cajas[(informe.cajas["archivo"] == "PV01_0_0.png") & (informe.cajas["linea"] == 7)]
        self.assertTrue(bool(caja.iloc[0]["valida"]))

    def test_duplicado_visual_entre_splits_con_etiquetas_distintas(self):
        # Mismos píxeles, distinto modo y bytes: no basta comparar archivos.
        Image.new("RGB", (256, 256), "red").convert("P").save(self.imagenes / "PV01_200_200.png")
        informe = self.auditar()
        self.assertEqual(informe.resumen["grupos_duplicados_rgb_entre_splits"], 1)
        self.assertEqual(informe.resumen["grupos_duplicados_bytes"], 0)
        self.assertIn("duplicado_anotaciones_distintas", set(informe.incidencias["control"]))

    def test_split_repeticiones_cobertura_y_vecinos(self):
        self.crear("PV01_1_1", "", "yellow")
        self.dividir(["PV01_0_0.png", "PV01_0_0.png", "PV01_100_100.png", "inexistente.png"],
                     ["PV01_0_0.png", "PV01_1_1.png"])
        informe = self.auditar()
        self.assertTrue({"split_entrada_repetida", "split_imagen_inexistente", "imagen_en_ambos_splits",
                         "imagen_sin_split", "grupo_espacial_entre_splits"} <= set(informe.incidencias["control"]))

    def test_imagen_corrupta_dimension_y_etiqueta_huerfana(self):
        (self.imagenes / "PV01_0_0.png").write_bytes(b"no es una imagen")
        Image.new("RGB", (128, 64), "blue").save(self.imagenes / "PV01_100_100.png")
        (self.labels / "huerfana.txt").write_text("", encoding="utf-8")
        informe = self.auditar()
        self.assertTrue({"imagen_no_legible", "dimension_imagen", "etiqueta_sin_imagen"}
                        <= set(informe.incidencias["control"]))
        self.assertEqual(informe.resumen["imagenes"], 3)


if __name__ == "__main__":
    unittest.main()
