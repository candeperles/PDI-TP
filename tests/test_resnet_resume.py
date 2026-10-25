import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / 'notebooks/05_ResNet50_Colab_v2.ipynb').read_text(encoding='utf-8'))
SOURCE = next(''.join(c['source']) for c in NB['cells']
              if c['cell_type'] == 'code' and 'def entrenar(' in ''.join(c['source']))


class EarlyStopping:
    def __init__(self, **kw): pass
    def on_train_begin(self, logs=None):
        self.best = None
        self.wait = 0


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.callbacks = SimpleNamespace(EarlyStopping=EarlyStopping,
            TerminateOnNaN=Mock(), ModelCheckpoint=Mock(), CSVLogger=Mock())
        self.original = Mock()
        self.loaded = Mock()
        self.loaded.optimizer.iterations.numpy.return_value = 6
        self.keras = SimpleNamespace(callbacks=self.callbacks, models=SimpleNamespace(load_model=Mock(return_value=self.loaded)))
        self.env = dict(np=np, pd=pd, json=json, keras=self.keras, compilar=Mock(),
            salida=self.folder)
        tree = ast.parse(SOURCE)
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('entrenar', 'ParadaReanudada')]
        exec(compile(tree, 'resume', 'exec'), self.env)
        self.entrenar = self.env['entrenar']
    def anterior(self):
        pd.DataFrame(dict(epoch=[0, 1], val_loss=[0.2, 0.3])).to_csv(self.folder / 'historia.csv', index=False)
        for name in ('ultimo.keras', 'mejor.keras'):
            (self.folder / name).write_bytes(b'checkpoint')
    def ejecutar(self, epocas=3):
        return self.entrenar(self.original, [1, 2, 3], [], self.folder, epocas, 1e-5, {})
    def test_reanuda_modelo_completo_sin_recompilar_y_anexa_historial(self):
        self.anterior()
        def fit(*args, **kw):
            pd.DataFrame(dict(epoch=[2], val_loss=[0.4])).to_csv(self.folder / 'historia.csv', mode='a', header=False, index=False)
            return SimpleNamespace(history={'loss': [0.4], 'val_loss': [0.4]})
        self.loaded.fit.side_effect = fit
        self.assertIs(self.ejecutar(), self.loaded)
        self.env['compilar'].assert_not_called()
        self.keras.models.load_model.assert_called_once_with(self.folder / 'ultimo.keras')
        self.assertEqual(self.loaded.fit.call_args.kwargs['initial_epoch'], 2)
        self.callbacks.CSVLogger.assert_called_once_with(str(self.folder / 'historia.csv'), append=True)
        self.assertEqual(self.callbacks.ModelCheckpoint.call_args_list[0].kwargs['initial_value_threshold'], 0.2)
        self.assertEqual(json.loads((self.folder / 'completado.json').read_text())['epocas'], 3)
        self.loaded.load_weights.assert_called_once_with(self.folder / 'mejor.keras')
    def test_finaliza_sin_repetir_si_alcanzo_epocas(self):
        self.anterior()
        self.ejecutar(epocas=2)
        self.loaded.fit.assert_not_called()
        self.assertTrue((self.folder / 'completado.json').exists())
    def test_checkpoint_desalineado_no_entrena_ni_marca_completado(self):
        self.anterior()
        self.loaded.optimizer.iterations.numpy.return_value = 9
        with self.assertRaisesRegex(RuntimeError, 'no coinciden'):
            self.ejecutar()
        self.loaded.fit.assert_not_called()
        self.assertFalse((self.folder / 'completado.json').exists())
    def test_etapa_completada_no_entrena(self):
        (self.folder / 'completado.json').write_text('{}')
        self.assertIs(self.ejecutar(), self.original)
        self.original.fit.assert_not_called()
    def test_checkpoint_faltante_no_sobrescribe_historial(self):
        self.anterior()
        (self.folder / 'ultimo.keras').unlink()
        original = (self.folder / 'historia.csv').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'falta un checkpoint'):
            self.ejecutar()
        self.assertEqual((self.folder / 'historia.csv').read_bytes(), original)
    def test_parada_conserva_mejor_valor_y_paciencia(self):
        callback = self.env['ParadaReanudada']([0.5, 0.2, 0.3, 0.4])
        callback.on_train_begin()
        self.assertEqual(callback.best, 0.2)
        self.assertEqual(callback.wait, 2)


if __name__ == '__main__':
    unittest.main()
