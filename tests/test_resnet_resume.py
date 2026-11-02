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
NB = json.loads((ROOT / 'notebooks/05_clasificacion_resnet50.ipynb').read_text(encoding='utf-8'))
SOURCE = next(''.join(c['source']) for c in NB['cells']
              if c['cell_type'] == 'code' and 'def entrenar(' in ''.join(c['source']))


class EarlyStopping:
    def __init__(self, **kw): self.__dict__.update(kw)
    def on_train_begin(self, logs=None):
        self.best = None
        self.wait = 0


class ReduceLROnPlateau:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.min_delta = 1e-4
    def on_train_begin(self, logs=None):
        self.best = None
        self.wait = 0


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.callbacks = SimpleNamespace(EarlyStopping=EarlyStopping, ReduceLROnPlateau=ReduceLROnPlateau,
            TerminateOnNaN=Mock(), ModelCheckpoint=Mock(), CSVLogger=Mock())
        self.original = Mock()
        self.loaded = Mock()
        self.loaded.optimizer.iterations.numpy.return_value = 6
        self.keras = SimpleNamespace(callbacks=self.callbacks, models=SimpleNamespace(load_model=Mock(return_value=self.loaded)))
        self.env = dict(np=np, pd=pd, json=json, keras=self.keras, compilar=Mock(),
            salida=self.folder)
        tree = ast.parse(SOURCE)
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('entrenar', 'ParadaReanudada', 'ReduccionReanudada')]
        exec(compile(tree, 'resume', 'exec'), self.env)
        self.entrenar = self.env['entrenar']
    def anterior(self):
        pd.DataFrame(dict(epoch=[0, 1], val_loss=[0.2, 0.3])).to_csv(self.folder / 'historia.csv', index=False)
        for name in ('ultimo.keras', 'mejor.keras'):
            (self.folder / name).write_bytes(b'checkpoint')
    def ejecutar(self, epocas=3, paciencia=5, reducir_lr=False):
        return self.entrenar(self.original, [1, 2, 3], [], self.folder, epocas, 1e-5, paciencia=paciencia, reducir_lr=reducir_lr)
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

    def test_paciencia_aleatorio_no_corta_con_cinco_epocas_sin_mejora(self):
        valores = [0.2] + [0.3] * 5
        pd.DataFrame(dict(epoch=range(6), val_loss=valores)).to_csv(self.folder / 'historia.csv', index=False)
        for nombre in ('ultimo.keras', 'mejor.keras'):
            (self.folder / nombre).write_bytes(b'checkpoint')
        self.loaded.optimizer.iterations.numpy.return_value = 18
        self.loaded.fit.return_value = SimpleNamespace(history={'loss': [0.3], 'val_loss': [0.3]})
        self.ejecutar(epocas=60, paciencia=10, reducir_lr=True)
        callbacks = self.loaded.fit.call_args.kwargs['callbacks']
        parada = next(c for c in callbacks if isinstance(c, EarlyStopping))
        self.assertEqual(parada.patience, 10)
        self.assertTrue(parada.restore_best_weights)
        reduccion = next(c for c in callbacks if isinstance(c, ReduceLROnPlateau))
        self.assertEqual(reduccion.factor, 0.5)
        # El último checkpoint se guarda después de actualizar el LR.
        self.assertIs(callbacks[-1], self.callbacks.ModelCheckpoint.return_value)
        reduccion.on_train_begin()
        self.assertEqual(reduccion.wait, 2)
        self.assertEqual(reduccion.best, 0.2)

    def test_parada_no_repite_epocas_si_agoto_paciencia(self):
        pd.DataFrame(dict(epoch=range(6), val_loss=[0.2] + [0.3] * 5)).to_csv(self.folder / 'historia.csv', index=False)
        for nombre in ('ultimo.keras', 'mejor.keras'):
            (self.folder / nombre).write_bytes(b'checkpoint')
        self.loaded.optimizer.iterations.numpy.return_value = 18
        self.ejecutar(epocas=15, paciencia=5)
        self.loaded.fit.assert_not_called()
        self.loaded.load_weights.assert_called_once_with(self.folder / 'mejor.keras')

    def test_reduccion_reanuda_ciclos_y_reinicia_espera_al_mejorar(self):
        for valores, mejor, espera in [([], float('inf'), 0),
                ([0.2] + [0.3] * 4, 0.2, 1),
                ([0.2] + [0.3] * 4 + [0.1], 0.1, 0),
                ([0.2, 0.19995], 0.2, 1)]:
            with self.subTest(valores=valores):
                callback = self.env['ReduccionReanudada'](valores)
                callback.on_train_begin()
                self.assertEqual(callback.best, mejor)
                self.assertEqual(callback.wait, espera)

    def test_finetuning_descongela_todo_excepto_batchnorm(self):
        class BatchNorm:
            trainable = True
        for fuente in [SOURCE, (ROOT / 'pdi_tp/resnet50.py').read_text(encoding='utf-8')]:
            with self.subTest(modulo=fuente is not SOURCE):
                arbol = ast.parse(fuente)
                arbol.body = [n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name == 'descongelar']
                entorno = dict(keras=SimpleNamespace(layers=SimpleNamespace(BatchNormalization=BatchNorm)))
                exec(compile(arbol, 'descongelar', 'exec'), entorno)
                capas = [SimpleNamespace(name=nombre, trainable=False)
                         for nombre in ('conv1_conv', 'conv2_block1', 'conv5_block3')]
                bn = BatchNorm()
                base = SimpleNamespace(layers=capas + [bn], trainable=False)
                entorno['descongelar'](base)
                self.assertTrue(base.trainable)
                self.assertTrue(all(c.trainable for c in capas))
                self.assertFalse(bn.trainable)


if __name__ == '__main__':
    unittest.main()
