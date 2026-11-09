"""Checkpoints de CV: dos copias verificadas, sin borrar archivos de Drive.

Keras se recibe como argumento para poder verificar el protocolo en CI sin TF.
El modelo completo conserva Adam; los mejores pesos y la paciencia se recuperan
por separado. No promete reproducir bit a bit el generador del dataset al cortar.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv_jamaica as cv, resultados


def cargar(rutas, carpeta, local):
    """Elige el último checkpoint íntegro; una copia cortada deja la otra intacta."""
    carpeta, local = Path(carpeta), Path(local)
    raiz = carpeta / "reanudacion"
    if not raiz.exists():
        return None
    validos = []
    for numero in range(2):
        copia = raiz / f"copia_{numero}"
        try:
            estado = json.loads((copia / "estado.json").read_text(encoding="utf-8"))
            if (estado["config_sha256"] != cv._sha(carpeta / "config.json")
                    or estado["etapa"] not in ("cabeza", "fine_tuning")
                    or set(estado["archivos"]) != {"ultimo.keras", "mejor.npz"}
                    or any(cv._sha(copia / n) != h for n, h in estado["archivos"].items())):
                continue
            _validar_historia(estado)
            validos.append((estado["paso"], copia, estado))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not validos:
        raise RuntimeError("No hay checkpoint íntegro. Conservar la carpeta y revisar; no se reinicia automáticamente.")
    _, copia, estado = max(validos, key=lambda v: v[0])
    # Nunca entrenar leyendo pesos directamente del montaje de Drive.
    for nombre in estado["archivos"]:
        destino = local / nombre
        shutil.copy2(copia / nombre, destino)
        if cv._sha(destino) != estado["archivos"][nombre]:
            raise OSError("La copia local del checkpoint no coincide con su sello.")
    print(f"Retomar {estado['etapa']} después de {len(estado['historia'])} épocas guardadas.")
    return estado


def _validar_historia(estado):
    historia = pd.DataFrame(estado["historia"])
    if (historia.empty or not {"loss", "val_loss"}.issubset(historia)
            or not np.isfinite(historia.to_numpy(float)).all()
            or estado["paso"] != len(historia) + (100 if estado["etapa"] == "fine_tuning" else 0)
            or estado["mejor_epoca"] != int(historia.val_loss.to_numpy().argmin()) + 1
            or estado["mejor_val_loss"] != float(historia.val_loss.min())
            or estado["espera"] != len(historia) - estado["mejor_epoca"]
            or not np.isfinite(estado["segundos"]) or estado["segundos"] < 0):
        raise ValueError("Historial del checkpoint inválido.")


def guardar(rutas, carpeta, local, modelo, mejores_pesos, estado):
    """Escribe el sello al final en una copia alterna; no necesita rename ni borrado."""
    resultados._exigir_escritura(rutas)
    carpeta, local = Path(carpeta), Path(local)
    if (carpeta / "completo.json").exists():
        raise FileExistsError("No modificar checkpoints de un fold completo.")
    _validar_historia(estado)
    modelo.save(local / "ultimo.keras")
    np.savez(local / "mejor.npz", *mejores_pesos)
    estado = dict(estado, config_sha256=cv._sha(carpeta / "config.json"),
        archivos={n: cv._sha(local / n) for n in ("ultimo.keras", "mejor.npz")})
    copia = carpeta / "reanudacion" / f"copia_{estado['paso'] % 2}"
    copia.mkdir(parents=True, exist_ok=True)
    for nombre, digest in estado["archivos"].items():
        shutil.copy2(local / nombre, copia / nombre)
        if cv._sha(copia / nombre) != digest:
            raise OSError("Drive no conservó íntegro el checkpoint; se mantiene la copia anterior.")
    (copia / "estado.json").write_text(cv._json(estado), encoding="utf-8")


def entrenar_etapa(keras, rutas, carpeta, local, modelo, train_ds, validacion_ds,
                   receta, etapa, *, anterior=None, etapas_anteriores=None):
    """Entrena o retoma una etapa con el mismo presupuesto y early stopping.

    Devuelve (modelo con mejores pesos, resumen, estado recuperable). El modelo
    recibido al retomar se carga con optimizer y no se vuelve a compilar.
    """
    resultados._exigir_escritura(rutas)
    if (Path(carpeta) / "completo.json").exists():
        raise FileExistsError("No entrenar sobre un fold completo.")
    historial = [] if anterior is None else list(anterior["historia"])
    if anterior is not None and anterior["etapa"] != etapa:
        raise ValueError("Checkpoint de otra etapa.")
    inicio = len(historial)
    limite = receta[f"epocas_{etapa}"]
    if inicio > limite:
        raise ValueError("El checkpoint excede las épocas de la receta.")
    mejores_pesos = None
    if anterior is not None:
        terminada = inicio >= limite or (anterior["espera"] >= receta["paciencia"] and inicio > 1)
        if anterior["terminada"] != terminada:
            raise ValueError("El fin de etapa no coincide con el presupuesto y la paciencia.")
        esperadas = inicio * len(train_ds)
        if (anterior["iteraciones"] != esperadas
                or int(modelo.optimizer.iterations.numpy()) != esperadas):
            raise RuntimeError("Adam y el historial no coinciden; no reanudar ni registrar el fold.")
        with np.load(local / "mejor.npz", allow_pickle=False) as entrada:
            mejores_pesos = [entrada[f"arr_{i}"].copy() for i in range(len(entrada.files))]
    else:
        modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=receta[f"lr_{etapa}"]),
            loss=receta["loss"], metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])

    class MejorEnMemoria(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.mejor = float("inf") if anterior is None else anterior["mejor_val_loss"]
            self.epoca = None if anterior is None else anterior["mejor_epoca"]
            self.pesos = mejores_pesos

        def on_epoch_end(self, epoch, logs=None):
            valor = (logs or {}).get("val_loss", float("nan"))
            if np.isfinite(valor) and valor < self.mejor:
                self.mejor, self.epoca = float(valor), epoch + 1
                self.pesos = [np.array(p, copy=True) for p in self.model.get_weights()]

    mejor = MejorEnMemoria()

    class ParadaReanudada(keras.callbacks.EarlyStopping):
        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            if anterior is not None:
                self.best = mejor.mejor
                self.best_epoch = mejor.epoca - 1
                self.best_weights = mejor.pesos
                self.wait = anterior["espera"]

    parada = ParadaReanudada(monitor="val_loss", mode="min", patience=receta["paciencia"],
                            restore_best_weights=True)
    acumulado = 0.0 if anterior is None else anterior["segundos"]
    reloj = time.perf_counter()
    commits = [] if anterior is None else list(anterior["commits_ejecucion"])
    commit = resultados._commit(rutas)
    if not commit:
        raise ValueError("No se pudo registrar el commit de ejecución.")
    if commit not in commits:
        commits.append(commit)
    estado = anterior

    class GuardarEpoca(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            nonlocal estado
            fila = {k: float(v) for k, v in (logs or {}).items()}
            if (epoch != len(historial) or not {"loss", "val_loss"}.issubset(fila)
                    or not all(np.isfinite(v) for v in fila.values()) or mejor.pesos is None):
                raise RuntimeError("Entrenamiento no finito o épocas desalineadas; no registrar el fold.")
            historial.append(fila)
            espera = len(historial) - mejor.epoca
            estado = dict(etapa=etapa, historia=list(historial), mejor_epoca=mejor.epoca,
                mejor_val_loss=mejor.mejor, espera=espera,
                paso=len(historial) + (100 if etapa == "fine_tuning" else 0),
                terminada=(len(historial) >= limite or (espera >= receta["paciencia"] and epoch > 0)),
                iteraciones=int(self.model.optimizer.iterations.numpy()),
                segundos=acumulado + time.perf_counter() - reloj,
                etapas_anteriores=etapas_anteriores or {}, commits_ejecucion=commits)
            guardar(rutas, carpeta, local, self.model, mejor.pesos, estado)
            print(f"Checkpoint en Drive: {etapa}, época {len(historial)}/{limite}.")

    if anterior is None or not anterior["terminada"]:
        modelo.fit(train_ds, validation_data=validacion_ds, initial_epoch=inicio,
            epochs=limite, callbacks=[mejor, parada, keras.callbacks.TerminateOnNaN(), GuardarEpoca()], verbose=2)
    if mejor.pesos is None or estado is None or not estado["terminada"]:
        raise RuntimeError("La etapa no terminó con un checkpoint válido; no registrar el fold.")
    modelo.set_weights(mejor.pesos)
    archivo_historia = local / f"historia_{etapa}.csv"
    pd.DataFrame(historial).to_csv(archivo_historia, index_label="epoca_desde_0")
    resultados._exigir_escritura(rutas)
    shutil.copy2(archivo_historia, carpeta / archivo_historia.name)
    resumen = dict(epocas=len(historial), mejor_epoca=mejor.epoca, mejor_val_loss=mejor.mejor,
                   monitor="val_loss_no_jamaica", segundos=estado["segundos"], commits_ejecucion=commits)
    return modelo, resumen, estado
