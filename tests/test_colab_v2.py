import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/05_ResNet50_Colab_v2.ipynb'
nb = nbformat.read(NOTEBOOK, as_version=4)
drive = types.ModuleType('drive_v2')
exec(next(c.source for c in nb.cells if c.cell_type == 'code' and 'class RespaldoDrive:' in c.source), drive.__dict__)

class HttpError(Exception):
    def __init__(self, status):
        self.resp = types.SimpleNamespace(status=status)

errors = types.ModuleType('googleapiclient.errors')
errors.HttpError = HttpError

class Request:
    def __init__(self, fn): self.fn = fn
    def execute(self, **kw): return self.fn()
    def next_chunk(self, **kw): return None, self.fn()

class FakeDrive:
    def __init__(self):
        self.items = {'folder123456': dict(id='folder123456', name='pia',
            mimeType='application/vnd.google-apps.folder', capabilities={'canAddChildren': True})}
        self.creates = 0
        self.updates = 0
        self.counter = 0
        self.lose_response = False
    def files(self): return self
    def get(self, fileId, **kw):
        def get():
            if fileId not in self.items: raise HttpError(404)
            return self.items[fileId].copy()
        return Request(get)
    def list(self, q, **kw):
        name = q.split("name = '", 1)[1][:-1]
        return Request(lambda: {'files': [v.copy() for v in self.items.values() if v.get('name') == name and v.get('parents') == ['folder123456'] and not v.get('trashed')]})
    def generateIds(self, **kw):
        self.counter += 1
        return Request(lambda: {'ids': [f'generated{self.counter}']})
    def create(self, body, media_body, **kw):
        assert body.get('mimeType') != 'application/vnd.google-apps.folder', 'No crear carpetas'
        def create():
            self.creates += 1
            assert body['id'] not in self.items, 'ID repetido'
            self.items[body['id']] = dict(body, md5Checksum=hashlib.md5(Path(media_body).read_bytes()).hexdigest())
            if self.lose_response:
                self.lose_response = False
                raise IOError('Respuesta perdida después de subir')
            return self.items[body['id']].copy()
        return Request(create)
    def update(self, fileId, media_body=None, body=None, **kw):
        def update():
            self.updates += 1
            if body:
                self.items[fileId].update(body)
            if media_body:
                self.items[fileId]['md5Checksum'] = hashlib.md5(Path(media_body).read_bytes()).hexdigest()
            return self.items[fileId].copy()
        return Request(update)

class GuardadoTest(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        mock_modules = patch.dict(sys.modules, {'googleapiclient.errors': errors})
        mock_modules.start()
        self.addCleanup(mock_modules.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.api = FakeDrive()
        self.r = self.nuevo()
        self.archivo = self.root / 'datos.csv'
        self.archivo.write_text('original')
    def nuevo(self):
        return drive.RespaldoDrive(self.api, 'folder123456', 'pia__resnet50_01', self.root / 'estado.json', lambda p, **kw: p)
    def preparar_corrida(self):
        (self.root / 'metricas.json').write_text('{"accuracy": 0.9}')
        for variante in self.r.VARIANTES:
            carpeta = self.root / variante
            carpeta.mkdir()
            (carpeta / 'mejor.keras').write_bytes(b'modelo')
            (carpeta / 'ultimo.keras').write_bytes(b'ultimo')
            (carpeta / 'historia.csv').write_text('epoch,loss\n0,0.1')
    def test_sincronizar_solo_cuatro_y_zip_actualizado(self):
        import zipfile
        self.preparar_corrida()
        self.r.sincronizar(self.root)
        archivos = [v for v in self.api.items.values() if v.get('parents')]
        self.assertEqual(len(archivos), 4)
        self.assertFalse(any('ultimo' in a['name'] for a in archivos))
        with zipfile.ZipFile(self.root / 'metricas_y_graficos.zip') as z:
            self.assertIn('metricas.json', z.namelist())
            self.assertIn('aleatorio/historia.csv', z.namelist())
            self.assertFalse(any(n.endswith('.keras') for n in z.namelist()))
        (self.root / 'metricas.json').write_text('{"accuracy": 1.0}')
        self.r.sincronizar(self.root)
        self.assertEqual(self.api.creates, 4)
        with zipfile.ZipFile(self.root / 'metricas_y_graficos.zip') as z:
            self.assertIn(b'1.0', z.read('metricas.json'))
    def test_finalizar_limpia_solo_auxiliares_propios_despues_de_verificar(self):
        self.preparar_corrida()
        self.r.subir(self.root / 'metricas.json', 'metricas.json')
        self.r.subir(self.root / 'aleatorio/ultimo.keras', 'aleatorio/ultimo.keras')
        self.api.items['ajeno'] = dict(id='ajeno', name='otra__corrida__metricas.json', parents=['folder123456'])
        self.r.finalizar(self.root)
        activos = [v for v in self.api.items.values() if v.get('parents') and not v.get('trashed')]
        self.assertEqual(len(activos), 5)
        self.assertFalse(self.api.items['ajeno'].get('trashed'))
        self.assertTrue(self.api.items['generated1']['trashed'])
        self.assertTrue(self.api.items['generated2']['trashed'])
    def test_no_limpia_una_corrida_incompleta(self):
        self.r.subir(self.archivo, 'datos.csv')
        with self.assertRaisesRegex(RuntimeError, 'Falta'):
            self.r.finalizar(self.root)
        self.assertFalse(self.api.items['generated1'].get('trashed'))
    def test_no_limpia_si_falla_verificacion_remota(self):
        self.preparar_corrida()
        self.r.subir(self.root / 'metricas.json', 'metricas.json')
        self.r.sincronizar(self.root)
        modelo = self.r.datos['archivos']['pia__resnet50_01__aleatorio__mejor.keras']
        self.api.items[modelo['id']]['md5Checksum'] = 'corrupto'
        with self.assertRaisesRegex(IOError, 'cuatro archivos'):
            self.r.finalizar(self.root)
        self.assertFalse(self.api.items['generated1'].get('trashed'))
    def test_repetir_y_cambiar_contenido_actualiza_mismo_id(self):
        self.assertTrue(self.r.subir(self.archivo, 'etapa/datos.csv'))
        self.assertFalse(self.nuevo().subir(self.archivo, 'etapa/datos.csv'))
        self.archivo.write_text('nuevo')
        self.assertTrue(self.nuevo().subir(self.archivo, 'etapa/datos.csv'))
        self.assertEqual((self.api.creates, self.api.updates), (1, 1))
    def test_respuesta_perdida_no_duplica_archivo(self):
        self.api.lose_response = True
        with self.assertRaises(IOError): self.r.subir(self.archivo, 'datos.csv')
        self.assertFalse(self.nuevo().subir(self.archivo, 'datos.csv'))
        self.assertEqual(self.api.creates, 1)
    def test_duplicados_se_detectan_sin_crear_otro(self):
        for n in [1, 2]:
            self.api.items[str(n)] = dict(id=str(n), name='pia__resnet50_01__datos.csv', parents=['folder123456'])
        with self.assertRaisesRegex(RuntimeError, 'duplicados'): self.r.subir(self.archivo, 'datos.csv')
        self.assertEqual(self.api.creates, 0)
    def test_carpeta_movida_frena_guardado(self):
        self.r.subir(self.archivo, 'datos.csv')
        self.api.items['generated1']['parents'] = ['otra']
        self.archivo.write_text('cambio')
        with self.assertRaisesRegex(RuntimeError, 'movió'): self.r.subir(self.archivo, 'datos.csv')
    def test_url_y_nombre_duplicado_no_intervienen(self):
        self.assertEqual(drive.id_carpeta('https://drive.google.com/drive/folders/folder123456?usp=sharing'), 'folder123456')
        with self.assertRaises(ValueError): drive.id_carpeta('pia')

    def test_limpieza_anterior_conserva_zip_sin_reempaquetar(self):
        self.preparar_corrida()
        self.r.sincronizar(self.root)
        zip_original = (self.root / 'metricas_y_graficos.zip').read_bytes()
        self.r.subir(self.root / 'metricas.json', 'metricas.json')
        self.r.finalizar(self.root, actualizar=False)
        self.assertEqual((self.root / 'metricas_y_graficos.zip').read_bytes(), zip_original)

class NotebookTest(unittest.TestCase):
    def test_formato_sintaxis_sin_salidas_ni_mount(self):
        limpieza = nbformat.read(NOTEBOOK.with_name('06_ordenar_resultados_resnet50.ipynb'), as_version=4)
        nbformat.validate(limpieza)
        for cell in limpieza.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
        self.assertEqual(limpieza.cells[1].source, drive_source := next(c.source for c in nb.cells if c.cell_type == 'code' and 'class RespaldoDrive:' in c.source))
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                ast.parse(cell.source, filename=f'celda_{i}')
                self.assertFalse(cell.outputs)
                self.assertIsNone(cell.execution_count)
                self.assertNotIn('drive.mount(', cell.source)

if __name__ == '__main__':
    unittest.main(verbosity=2)
