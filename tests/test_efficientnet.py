"""Regresiones del flujo de EfficientNet, sin TensorFlow ni dataset."""
import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / '06_clasificacion_efficientnet.ipynb'


def cargar_funciones():
    """Ejecuta las funciones reales del notebook, sin su setup ni entrenamiento."""
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    espacio = {'SEMILLA': 42, 'json': json, 'pd': pd}
    for celda in notebook['cells']:
        if celda['cell_type'] != 'code':
            continue
        codigo = ''.join(celda['source'])
        if codigo.startswith('# Receta común'):
            exec(compile(codigo, str(NOTEBOOK), 'exec'), espacio)
        arbol = ast.parse(codigo)
        funciones = [nodo for nodo in arbol.body if isinstance(nodo, ast.FunctionDef)]
        exec(compile(ast.Module(body=funciones, type_ignores=[]), str(NOTEBOOK), 'exec'), espacio)
    return espacio


class BatchNormSimulada:
    trainable = True


class EfficientNetTest(unittest.TestCase):
    def setUp(self):
        self.espacio = cargar_funciones()
        self.modelo = Mock()
        self.modelo.fit.return_value = SimpleNamespace(history={'loss': [0.4], 'val_accuracy': [0.5]})
        self.modelo.predict.return_value = np.array([[0.2], [0.8]])
        self.keras = SimpleNamespace(
            models=SimpleNamespace(load_model=Mock(return_value=self.modelo)),
            optimizers=SimpleNamespace(Adam=Mock()),
            metrics=SimpleNamespace(BinaryAccuracy=Mock()),
            callbacks=SimpleNamespace(EarlyStopping=Mock(return_value=sentinel.parada),
                                      ReduceLROnPlateau=Mock(return_value=sentinel.reduccion)),
            backend=SimpleNamespace(clear_session=Mock()),
            utils=SimpleNamespace(set_random_seed=Mock()),
        )
        self.espacio.update(keras=self.keras, layers=SimpleNamespace(BatchNormalization=BatchNormSimulada),
                            ds_entrenamiento=sentinel.train, ds_validacion=sentinel.val)

    def test_epocas_y_callbacks_segun_modo(self):
        self.espacio['construir_modelo'] = Mock(return_value=self.modelo)
        for nombre, epocas, paciencia in [('imagenet_congelado', 15, 5), ('aleatorio', 60, 10)]:
            with self.subTest(modo=nombre):
                config = self.espacio['EXPERIMENTOS'][nombre]
                self.espacio['entrenar'](config)
                self.assertEqual(self.modelo.fit.call_args.kwargs['epochs'], epocas)
                self.keras.callbacks.EarlyStopping.assert_called_with(
                    monitor='val_loss', patience=paciencia, restore_best_weights=True)
                self.keras.optimizers.Adam.assert_called_with(1e-3)
                if nombre == 'aleatorio':
                    self.keras.callbacks.ReduceLROnPlateau.assert_called_once_with(
                        monitor='val_loss', factor=0.5, patience=3)
                    self.assertEqual(self.modelo.fit.call_args.kwargs['callbacks'],
                                     [sentinel.parada, sentinel.reduccion])
                else:
                    self.keras.callbacks.ReduceLROnPlateau.assert_not_called()

    def test_finetuning_carga_cabeza_y_congela_batchnorm_antes_de_compilar(self):
        normalizacion = BatchNormSimulada()
        convolucion = SimpleNamespace(trainable=True)
        extractor = SimpleNamespace(layers=[normalizacion, convolucion], trainable=False)
        self.modelo.get_layer.return_value = extractor
        self.espacio['construir_modelo'] = Mock()

        def verificar_congelamiento(**opciones):
            self.assertTrue(extractor.trainable)
            self.assertFalse(normalizacion.trainable)
            self.assertTrue(convolucion.trainable)

        self.modelo.compile.side_effect = verificar_congelamiento
        carpeta = Path('corrida_simulada')
        self.espacio['entrenar'](self.espacio['EXPERIMENTOS']['imagenet_finetuning'], carpeta)
        self.keras.models.load_model.assert_called_once_with(carpeta / 'modelo.keras', compile=False)
        self.espacio['construir_modelo'].assert_not_called()
        self.modelo.fit.assert_called_once()
        self.assertEqual(self.modelo.fit.call_args.kwargs['epochs'], 20)
        self.keras.optimizers.Adam.assert_called_once_with(1e-5)
        self.keras.callbacks.ReduceLROnPlateau.assert_not_called()

    def test_batchnorm_entrena_desde_cero_y_usa_inferencia_con_imagenet(self):
        extractor = Mock()
        self.keras.applications = SimpleNamespace(EfficientNetB0=Mock(return_value=extractor))
        self.keras.Input = Mock()
        self.keras.Model = Mock()
        self.espacio['layers'] = SimpleNamespace(GlobalAveragePooling2D=Mock(), Dropout=Mock(), Dense=Mock())
        for pesos in ['imagenet', None]:
            with self.subTest(pesos=pesos):
                extractor.reset_mock()
                self.espacio['construir_modelo'](pesos)
                self.assertEqual(extractor.trainable, pesos is None)
                self.assertEqual(extractor.call_args.kwargs, {'training': False} if pesos else {})

    def preparar_guardado_simulado(self, raiz):
        """Imita el contrato de guardado con archivos temporales, sin calcular métricas."""
        def ruta_corrida(rutas, modelo, modo):
            return raiz / modelo / modo

        def carpeta_corrida(rutas, modelo, modo, reiniciar=False):
            carpeta = ruta_corrida(rutas, modelo, modo)
            carpeta.mkdir(parents=True, exist_ok=True)
            return carpeta

        def guardar_corrida(rutas, carpeta, **opciones):
            self.assertTrue((carpeta / 'modelo.keras').is_file())
            self.assertTrue((carpeta / 'historia.csv').is_file())
            self.assertEqual(opciones['archivos'], ['simulada_0.png', 'simulada_1.png'])
            (carpeta / 'config.json').write_text(json.dumps({'params': opciones['params']}), encoding='utf-8')
            return {}

        self.modelo.save.side_effect = lambda ruta: ruta.write_bytes(b'pesos simulados')
        self.espacio.update(
            rutas=sentinel.rutas, archivos_validacion=['simulada_0.png', 'simulada_1.png'],
            resultados=SimpleNamespace(
                ruta_corrida=ruta_corrida, carpeta_corrida=Mock(side_effect=carpeta_corrida),
                corrida_completa=lambda carpeta: (carpeta / 'config.json').is_file(),
                guardar_corrida=Mock(side_effect=guardar_corrida)),
        )
        return self.espacio['resultados']

    def test_interrupcion_posterior_conserva_los_modos_completos_y_sus_params(self):
        with tempfile.TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            raiz = Path(temporal)
            resultados = self.preparar_guardado_simulado(raiz)

            def entrenar(config, carpeta_inicial):
                if config['pesos'] is None:
                    raise RuntimeError('Interrupción simulada')
                if config['fine_tuning']:
                    self.assertTrue((carpeta_inicial / 'config.json').is_file())
                return self.modelo, self.modelo.fit.return_value

            self.espacio['entrenar'] = entrenar
            with self.assertRaisesRegex(RuntimeError, 'Interrupción simulada'):
                self.espacio['ejecutar_experimentos']()
            self.assertEqual(resultados.guardar_corrida.call_count, 2)
            for nombre, lr, epocas in [('imagenet_congelado', 1e-3, 15), ('imagenet_finetuning', 1e-5, 20)]:
                params = json.loads((raiz / 'efficientnetb0' / nombre / 'config.json').read_text())['params']
                self.assertEqual([params[clave] for clave in ['img_size', 'batch_size', 'dropout',
                                                             'lr', 'epocas_max', 'paciencia', 'umbral', 'semilla']],
                                 [256, 32, 0.3, lr, epocas, 5, 0.5, 42])
                self.assertEqual(params['epocas_ejecutadas'], 1)
                self.assertNotIn('class_weight', params)
            self.assertFalse((raiz / 'efficientnetb0' / 'aleatorio' / 'config.json').exists())

    def test_no_reentrena_ni_reescribe_modos_completos(self):
        with tempfile.TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            resultados = self.preparar_guardado_simulado(Path(temporal))
            entrenar = Mock(return_value=(self.modelo, self.modelo.fit.return_value))
            self.espacio['entrenar'] = entrenar
            self.espacio['ejecutar_experimentos']()
            entrenar.reset_mock()
            resultados.carpeta_corrida.reset_mock()
            resultados.guardar_corrida.reset_mock()
            self.espacio['ejecutar_experimentos']()
            entrenar.assert_not_called()
            resultados.carpeta_corrida.assert_not_called()
            resultados.guardar_corrida.assert_not_called()

    def test_finetuning_rechaza_cabeza_de_receta_anterior(self):
        with tempfile.TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            raiz = Path(temporal)
            resultados = self.preparar_guardado_simulado(raiz)
            carpeta = raiz / 'efficientnetb0' / 'imagenet_congelado'
            carpeta.mkdir(parents=True)
            (carpeta / 'config.json').write_text(json.dumps({'params': {'epocas': [10, 10]}}))
            self.espacio['entrenar'] = Mock()
            with self.assertRaisesRegex(ValueError, 'REPETIR = True'):
                self.espacio['ejecutar_experimentos']()
            self.espacio['entrenar'].assert_not_called()
            resultados.carpeta_corrida.assert_not_called()


if __name__ == '__main__':
    unittest.main()
