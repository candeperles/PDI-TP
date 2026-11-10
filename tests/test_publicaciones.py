"""El acuerdo de copias ejecutadas no permite alterar código ni ocultar errores."""
import copy
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from pdi_tp import lectura_entrega, publicaciones, split


class PublicacionesTest(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.raiz = Path(temporal.name)
        (self.raiz / "notebooks").mkdir()
        self.origen = "notebooks/05_prueba.ipynb"
        self.nb = dict(nbformat=4, nbformat_minor=5, metadata={}, cells=[
            dict(cell_type="code", metadata={}, source=["print('entrenar')\n"], execution_count=None, outputs=[]),
            dict(cell_type="code", metadata={"pdi_tp": {"consulta_entrega": True}}, source=["print('consulta')\n"], execution_count=None, outputs=[])])
        self.guardar_original()
        self.nb["cells"][1].update(execution_count=1, outputs=[dict(output_type="stream", name="stdout", text=["consulta\n"])])

    def guardar_original(self):
        (self.raiz / self.origen).write_text(json.dumps(self.nb), encoding="utf-8")

    def publicar(self, **opciones):
        return publicaciones.preparar(self.nb, raiz=self.raiz, origen=self.origen, commit="a" * 40,
                                      tipo=opciones.pop("tipo", "consulta_guardada"), **opciones)

    def test_consulta_identica_con_procedencia_y_sin_widgets(self):
        self.nb["metadata"]["widgets"] = {"estado": "privado"}
        copia = self.publicar()
        self.assertEqual(publicaciones.validar(copia, raiz=self.raiz), [])
        self.assertNotIn("widgets", copia["metadata"])
        self.assertIn("no registrado", publicaciones.texto(copia["cells"][0]["source"]))
        self.assertEqual(copia["metadata"]["publicacion"]["celdas_ejecutadas"], [1])
        self.assertIsNone(copia["cells"][1]["execution_count"])

    def test_no_publica_codigo_alterado_ni_ejecucion_incompleta(self):
        with self.assertRaisesRegex(ValueError, "incompleta"):
            self.publicar(tipo="ejecucion_completa")
        self.nb["cells"][0]["source"] = ["print('otra receta')\n"]
        with self.assertRaisesRegex(ValueError, "fuente distinta"):
            self.publicar()

    def test_rechaza_errores_y_salidas_en_celdas_no_ejecutadas(self):
        self.nb["cells"][1]["outputs"] = [dict(output_type="error", ename="ValueError", evalue="falló", traceback=[])]
        with self.assertRaisesRegex(ValueError, "errores"):
            self.publicar()
        self.nb["cells"][1]["outputs"] = []
        self.nb["cells"][0]["outputs"] = [dict(output_type="stream", name="stdout", text="salida inventada")]
        with self.assertRaisesRegex(ValueError, "sin ejecución"):
            self.publicar()

    def test_ci_detecta_original_o_cabecera_alterados(self):
        copia = self.publicar()
        corrupta = copy.deepcopy(copia)
        corrupta["cells"][0]["source"] = ["sin procedencia"]
        self.assertIn("Cabecera", " ".join(publicaciones.validar(corrupta, raiz=self.raiz)))
        (self.raiz / self.origen).write_text("{}", encoding="utf-8")
        self.assertIn("original", " ".join(publicaciones.validar(copia, raiz=self.raiz)))

    def test_huella_portable_entre_windows_y_linux(self):
        copia = self.publicar()
        ruta = self.raiz / self.origen
        ruta.write_bytes(ruta.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertEqual(publicaciones.validar(copia, raiz=self.raiz), [])

    def test_rutas_y_tamano_acotados(self):
        with self.assertRaisesRegex(ValueError, "notebooks"):
            publicaciones.preparar(self.nb, raiz=self.raiz, origen="../fuera.ipynb", commit="a" * 40)
        self.nb["cells"][1]["outputs"][0]["text"] = "x" * publicaciones.LIMITE_BYTES
        with self.assertRaisesRegex(ValueError, "5 MiB"):
            self.publicar()

    def test_test_no_conserva_imagenes_ni_html_activo_y_registra_familia(self):
        self.origen = "notebooks/18_evaluacion_test.ipynb"
        self.nb["cells"] = [dict(cell_type="code", metadata={}, source=['FAMILIA = "yolo26n"\n'], execution_count=None, outputs=[])]
        self.guardar_original()
        self.nb["cells"][0].update(source=['FAMILIA = "clasificadores"\n'], execution_count=1, outputs=[
            dict(output_type="display_data", metadata={}, data={"image/png": "imagen", "image/svg+xml": "<svg/>",
                "text/html": '<script>privado()</script><style>body{}</style><table onclick="x()"><tr><td>0.8</td></tr></table><img src="mapa.png">'})])
        copia = self.publicar(tipo="ejecucion_completa", familia="clasificadores")
        data = copia["cells"][1]["outputs"][0]["data"]
        self.assertEqual(set(data), {"text/html"})
        self.assertEqual(data["text/html"], "<table><tr><td>0.8</td></tr></table>")
        data["image/png"] = "oculta"
        self.assertIn("imagen de test", " ".join(publicaciones.validar(copia, raiz=self.raiz)))


class RegistroRevisionTest(unittest.TestCase):
    def test_fuentes_del_registro_coinciden_con_archivos(self):
        ruta = Path(__file__).resolve().parents[1] / "docs/revision_anotaciones.csv"
        with ruta.open(encoding="utf-8") as archivo:
            for fila in csv.DictReader(archivo):
                self.assertEqual(fila["fuente"], split.fuente(Path(fila["archivo"]).stem), fila["id_caso"] if "id_caso" in fila else fila["archivo"])


class LecturaEntregaTest(unittest.TestCase):
    def test_consulta_exige_modos_completos_y_abre_sin_permitir_escritura(self):
        import pandas as pd
        tabla = pd.DataFrame([dict(modelo="resnet50", modo=m, completa=(m != "aleatorio"))
                              for m in ["imagenet_congelado", "imagenet_finetuning", "aleatorio"]])
        with patch.object(lectura_entrega.entorno, "preparar") as preparar, \
                patch.object(lectura_entrega.resultados, "cargar_corridas", return_value=tabla):
            with self.assertRaisesRegex(ValueError, "Faltan corridas completas"):
                lectura_entrega.mostrar_modelo("resnet50")
            preparar.assert_called_once_with(solo_lectura=True)
