"""Cortes reales del flujo de callbacks con un modelo serializable sin TensorFlow."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from pdi_tp import reanudacion_cnn as reanudar


class Callback:
    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class Parada(Callback):
    def __init__(self, **opciones):
        self.__dict__.update(opciones)

    def on_train_begin(self, logs=None):
        self.best, self.wait, self.best_weights = float("inf"), 0, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best:
            self.best, self.wait = logs["val_loss"], 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch > 0:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights:
            self.model.set_weights(self.best_weights)


class Modelo:
    def __init__(self, perdidas, corte=None):
        self.perdidas, self.corte = perdidas, corte
        self.peso, self.iteraciones, self.compilaciones, self.epocas = -1, 0, 0, []
        self.optimizer = SimpleNamespace(iterations=SimpleNamespace(numpy=lambda: self.iteraciones))

    def compile(self, **opciones):
        self.compilaciones += 1
        self.iteraciones = 0

    def get_weights(self):
        return [np.array([self.peso])]

    def set_weights(self, pesos):
        self.peso = int(pesos[0][0])

    def save(self, archivo):
        Path(archivo).write_text(json.dumps(dict(peso=self.peso, iteraciones=self.iteraciones)))

    def fit(self, datos, *, validation_data, initial_epoch, epochs, callbacks, **opciones):
        self.stop_training = False
        self.validacion = validation_data
        for callback in callbacks:
            callback.model = self
            callback.on_train_begin()
        for epoca in range(initial_epoch, epochs):
            self.epocas.append(epoca)
            self.peso, self.iteraciones = epoca, self.iteraciones + len(datos)
            for callback in callbacks:
                callback.on_epoch_end(epoca, dict(loss=0.3, val_loss=self.perdidas[epoca]))
            if epoca == self.corte:
                raise ConnectionError("Simular fin de runtime después del checkpoint")
            if self.stop_training:
                break
        for callback in callbacks:
            callback.on_train_end()


class ReanudacionTest(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.raiz = Path(temporal.name)
        self.carpeta, self.local = self.raiz / "fold_1", self.raiz / "local"
        self.carpeta.mkdir()
        self.local.mkdir()
        (self.carpeta / "config.json").write_text('{"fold":1}')
        self.rutas = SimpleNamespace(solo_lectura=False)
        self.keras = SimpleNamespace(callbacks=SimpleNamespace(Callback=Callback, EarlyStopping=Parada,
            TerminateOnNaN=Callback), optimizers=SimpleNamespace(Adam=lambda **kw: kw),
            metrics=SimpleNamespace(BinaryAccuracy=lambda **kw: kw))
        self.receta = dict(epocas_cabeza=6, epocas_fine_tuning=6, paciencia=2,
            lr_cabeza=1e-3, lr_fine_tuning=1e-5, loss="binary_crossentropy")
        parche = patch.object(reanudar.resultados, "_commit", return_value="c" * 40)
        parche.start()
        self.addCleanup(parche.stop)

    def entrenar(self, modelo, anterior=None, etapa="cabeza", **opciones):
        return reanudar.entrenar_etapa(self.keras, self.rutas, self.carpeta, self.local,
            modelo, [1, 2, 3], "val_no_jamaica", self.receta, etapa, anterior=anterior, **opciones)

    def recuperar(self, perdidas):
        estado = reanudar.cargar(self.rutas, self.carpeta, self.local)
        valores = json.loads((self.local / "ultimo.keras").read_text())
        modelo = Modelo(perdidas)
        modelo.peso, modelo.iteraciones = valores["peso"], valores["iteraciones"]
        return modelo, estado

    def test_corte_conserva_adam_historial_paciencia_y_mejor_epoca(self):
        perdidas = [0.4, 0.1, 0.3, 0.2, 0.5, 0.6]
        modelo = Modelo(perdidas, corte=2)
        with self.assertRaises(ConnectionError):
            self.entrenar(modelo)
        recuperado, estado = self.recuperar(perdidas)
        with patch.object(reanudar.resultados, "_commit", return_value="d" * 40):
            final, resumen, estado_final = self.entrenar(recuperado, anterior=estado)
        self.assertEqual(recuperado.compilaciones, 0)
        self.assertEqual(recuperado.epocas, [3])
        self.assertEqual(final.iteraciones, 12)
        self.assertEqual(final.peso, 1)
        self.assertEqual(resumen["epocas"], 4)
        self.assertEqual(resumen["mejor_epoca"], 2)
        self.assertEqual(estado_final["espera"], 2)
        self.assertTrue(estado_final["terminada"])
        self.assertEqual(resumen["commits_ejecucion"], ["c" * 40, "d" * 40])
        historia = pd.read_csv(self.local / "historia_cabeza.csv")
        np.testing.assert_allclose(historia.val_loss, perdidas[:4])
        self.assertEqual(recuperado.validacion, "val_no_jamaica")

    def test_maximo_de_epocas_restaura_mejor_y_no_repite_etapa_terminada(self):
        self.receta["epocas_cabeza"] = 3
        self.receta["paciencia"] = 10
        modelo, resumen, _ = self.entrenar(Modelo([0.2, 0.1, 0.3]))
        self.assertEqual(modelo.peso, 1)
        self.assertEqual(resumen["epocas"], 3)
        recuperado, estado = self.recuperar([0.2, 0.1, 0.3])
        recuperado, _, _ = self.entrenar(recuperado, anterior=estado)
        self.assertEqual(recuperado.epocas, [])
        self.assertEqual(recuperado.peso, 1)

    def test_checkpoint_incompleto_recupera_copia_anterior(self):
        with self.assertRaises(ConnectionError):
            self.entrenar(Modelo([0.3, 0.2, 0.1], corte=1))
        ultima = self.carpeta / "reanudacion" / "copia_0"
        (ultima / "ultimo.keras").write_bytes(b"copia interrumpida")
        recuperado, estado = self.recuperar([0.3, 0.2, 0.1])
        self.assertEqual(len(estado["historia"]), 1)
        self.assertEqual(recuperado.iteraciones, 3)

    def test_corte_al_escribir_no_destruye_checkpoint_anterior(self):
        with self.assertRaises(ConnectionError):
            self.entrenar(Modelo([0.3, 0.2, 0.1], corte=0))
        modelo, estado = self.recuperar([0.3, 0.2, 0.1])
        copiar = reanudar.shutil.copy2
        def cortar(origen, destino):
            if "copia_0" in str(destino):
                Path(destino).write_bytes(b"truncado")
                raise OSError("Se cortó Drive")
            return copiar(origen, destino)
        with patch.object(reanudar.shutil, "copy2", side_effect=cortar), self.assertRaises(OSError):
            self.entrenar(modelo, anterior=estado)
        _, guardado = self.recuperar([0.3, 0.2, 0.1])
        self.assertEqual(len(guardado["historia"]), 1)

    def test_checkpoint_invalido_o_ajeno_nunca_reinicia_automaticamente(self):
        with self.assertRaises(ConnectionError):
            self.entrenar(Modelo([0.3, 0.2], corte=0))
        (self.carpeta / "config.json").write_text('{"fold":2}')
        with self.assertRaisesRegex(RuntimeError, "No hay checkpoint íntegro"):
            self.recuperar([0.3, 0.2])

    def test_adam_desalineado_y_estado_de_fin_adulterado_frenan(self):
        with self.assertRaises(ConnectionError):
            self.entrenar(Modelo([0.3, 0.2], corte=0))
        modelo, estado = self.recuperar([0.3, 0.2])
        modelo.iteraciones = 4
        with self.assertRaisesRegex(RuntimeError, "Adam"):
            self.entrenar(modelo, anterior=estado)
        modelo.iteraciones = 3
        with self.assertRaisesRegex(ValueError, "fin de etapa"):
            self.entrenar(modelo, anterior=dict(estado, terminada=True))

    def test_no_guardar_en_solo_lectura_ni_modificar_fold_completo(self):
        modelo = Modelo([0.3, 0.2])
        self.rutas.solo_lectura = True
        with self.assertRaises(PermissionError):
            self.entrenar(modelo)
        self.assertFalse((self.carpeta / "reanudacion").exists())
        self.rutas.solo_lectura = False
        (self.carpeta / "completo.json").write_text('{}')
        with self.assertRaises(FileExistsError):
            self.entrenar(Modelo([0.3, 0.2]))

    def test_val_loss_no_finito_no_sella_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "no finito"):
            self.entrenar(Modelo([float("nan")]))
        self.assertFalse((self.carpeta / "reanudacion").exists())


if __name__ == "__main__":
    unittest.main()
