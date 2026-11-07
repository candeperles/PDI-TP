"""Validación cruzada adicional del #43; no escribe corridas del formato común.

Cada fold se verifica contra el split oficial y se guarda en analisis/cv_jamaica.
No importa TensorFlow ni PyTorch: el entrenamiento vive en el notebook 18.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import comparacion, resultados, split, umbral

FINALISTAS = {
    "resnet50": "imagenet_finetuning",
    "mobilenetv2": "imagenet_finetuning",  # opcional
    "yolo11n": "coco_completo",
    "yolo26n": "coco_completo",
}
VERSIONES_YOLO = {"yolo11n": "8.3.221", "yolo26n": "8.4.184"}
COLUMNAS_CAJAS = ["archivo", "clase", "confianza", "xc", "yc", "w", "h"]
CONJUNTOS = ("train", "validacion", "evaluacion")
PROTOCOLO = "cv_jamaica_v1"
# Cambiar esta versión si cambia el entrenamiento, no por ajustes de persistencia.
VERSION_CNN = "jamaica_cnn_comun_v2_v1"
COMMIT_CNN_ANTERIOR = "c1ffe523d2b6233a707cd7fbcd34f1e0c21ddbc4"


def _firma_config(config):
    """Conserva la procedencia; permite el cambio de guardado de las CNN del #43.

    Solo el commit anterior conocido comparte esta versión. Los demás formatos
    antiguos y los detectores siguen requiriendo el mismo commit exacto.
    Receta, referencias, entorno y datos siempre deben coincidir íntegramente.
    """
    version = config.get("version_entrenamiento")
    if config["modelo"] not in VERSIONES_YOLO and (
            version == VERSION_CNN or (version is None and config["commit"] == COMMIT_CNN_ANTERIOR)):
        return dict(version_entrenamiento=VERSION_CNN, params=config["params"])
    return dict(commit=config["commit"], version_entrenamiento=version, params=config["params"])


def _compatible(guardada, solicitada):
    identidad = ("protocolo", "modelo", "modo", "fold", "huella_fold")
    return (all(guardada.get(k) == solicitada.get(k) for k in identidad)
            and _firma_config(guardada) == _firma_config(solicitada))


def _json(valor):
    return json.dumps(valor, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _sha(archivo):
    resumen = hashlib.sha256()
    with Path(archivo).open("rb") as entrada:
        for bloque in iter(lambda: entrada.read(1024 * 1024), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def referencia(rutas, modelo):
    """Lee la receta guardada, nunca los pesos entrenados con Jamaica.

    CNN: normaliza los nombres del 04 y del 05 y exige comun_v2 completa.
    YOLO: conserva íntegra la RECETA guardada de la corrida coco_completo.
    """
    if modelo not in FINALISTAS:
        raise ValueError(f"Finalista desconocido: {modelo}")
    modo = FINALISTAS[modelo]

    def leer(modo_ref):
        carpeta = resultados.ruta_corrida(rutas, modelo, modo_ref)
        if not resultados.corrida_completa(carpeta):
            raise ValueError(f"Falta una referencia completa: {modelo}/{modo_ref}")
        return json.loads((carpeta / "config.json").read_text(encoding="utf-8"))

    config = leer(modo)
    p = config["params"]
    referencias = {modo: config}
    if modelo in VERSIONES_YOLO:
        receta = dict(p["receta"])
        if receta.get("imgsz") != 640 or p.get("freeze") != 0:
            raise ValueError("La referencia debe ser coco_completo a 640, sin congelar.")
        paquetes = p.get("entorno", {}).get("paquetes", {})
        if paquetes.get("ultralytics") != VERSIONES_YOLO[modelo]:
            raise ValueError("La versión de Ultralytics de la referencia no coincide con el finalista.")
        umbral_base = p["umbral_clasificacion"]
    else:
        cabeza = leer("imagenet_congelado")
        referencias["imagenet_congelado"] = cabeza
        c = cabeza["params"]
        if p.get("receta") != "comun_v2" or c.get("receta") != "comun_v2":
            raise ValueError("Ambas etapas de la referencia deben usar comun_v2.")
        comunes = dict(img_size=256, batch_size=32, dropout=0.3, paciencia=10, semilla=42, umbral=0.5)
        for clave, valor in comunes.items():
            if p.get(clave) != valor or c.get(clave) != valor:
                raise ValueError(f"La referencia no coincide con comun_v2: {clave}")
        for etapa in (p, c):
            if (etapa.get("sample_weight", etapa.get("pesos_muestra")) != "igual_masa_por_fuente_x_clase"
                    or etapa.get("augmentation") not in (True, "aumentacion.google_maps")):
                raise ValueError("La referencia requiere aumentación y pesos por fuente × clase.")
        receta = dict(comunes, receta="comun_v2", augmentation="aumentacion.google_maps",
            sample_weight="igual_masa_por_fuente_x_clase", optimizador="Adam", loss="binary_crossentropy",
            epocas_cabeza=c.get("epocas_cabeza", c.get("epocas_max")),
            lr_cabeza=c.get("lr_cabeza", c.get("lr")),
            epocas_fine_tuning=p.get("epocas_fine_tuning", p.get("epocas_max")),
            lr_fine_tuning=p.get("lr_fine_tuning", p.get("lr")),
            monitor="val_loss", restaurar_mejores_pesos=True, batchnorm_congelada=True)
        if [receta[k] for k in ("epocas_cabeza", "lr_cabeza", "epocas_fine_tuning", "lr_fine_tuning")] != [30, 1e-3, 20, 1e-5]:
            raise ValueError("Épocas o LR de la referencia no coinciden con comun_v2.")
        umbral_base = p["umbral"]
    if umbral_base != (0.25 if modelo in VERSIONES_YOLO else 0.5):
        raise ValueError("La referencia no tiene el umbral inicial esperado.")
    return dict(receta=receta, umbral=umbral_base, referencias=referencias)


def _manifiesto(rutas, fold):
    """Comprueba roles, etiquetas y zonas antes de escribir o reutilizar un fold."""
    numero = fold["fold"]
    if isinstance(numero, bool) or not isinstance(numero, int) or not 1 <= numero <= 5:
        raise ValueError("Número de fold inválido (1 a 5).")
    tablas = [fold[c].assign(conjunto=c) for c in CONJUNTOS]
    tabla = pd.concat(tablas, ignore_index=True).sort_values("archivo").reset_index(drop=True)
    oficiales = split.cargar_split(rutas)
    nombres = {p.name: c for c, archivos in oficiales.items() for p in archivos}
    if (len(nombres) != sum(map(len, oficiales.values())) or tabla.archivo.duplicated().any()
            or set(tabla.archivo) != set(nombres)
            or not tabla.split.eq(tabla.archivo.map(nombres)).all()):
        raise ValueError("El fold debe contener exactamente el split oficial; nunca el test.")
    es_jamaica = tabla.archivo.map(split.es_jamaica)
    validacion = ~es_jamaica & tabla.split.eq("val")
    evaluacion = tabla.conjunto.eq("evaluacion")
    if (not tabla.conjunto.eq("validacion").equals(validacion)
            or not es_jamaica[evaluacion].all() or not evaluacion.any()):
        raise ValueError("Roles inválidos: checkpoints solo en val no jamaicano y evaluación solo en Jamaica.")
    grupos = split.grupos_espaciales([Path(a).stem for a in tabla.archivo])
    tabla["grupo"] = grupos
    if tabla.groupby("grupo").conjunto.nunique().gt(1).any():
        raise ValueError("Grupos espaciales compartidos entre particiones.")
    if not tabla.y_original.isin([0, 1]).all():
        raise ValueError("Etiquetas binarias requeridas.")
    if tabla.loc[tabla.conjunto.ne("validacion"), "pendiente_revision"].any():
        raise ValueError("Hay etiquetas pendientes en entrenamiento o evaluación.")
    if any(set(t.y_original) != {0, 1} for t in tablas):
        raise ValueError("Cada partición debe contener ambas clases.")
    columnas = ["archivo", "split", "conjunto", "grupo", "y_original", "pendiente_revision"]
    columnas += [c for c in ("sha256", "sha256_etiqueta") if c in tabla]
    # Orden e ids estables aunque cambie el rol de una imagen entre folds.
    return tabla[columnas]


def ruta_fold(rutas, modelo, modo, numero):
    resultados._validar_nombre("modelo", modelo)
    resultados._validar_nombre("modo", modo)
    if FINALISTAS.get(modelo) != modo or numero not in range(1, 6):
        raise ValueError("Modelo, modo o fold fuera del protocolo.")
    return rutas.analisis / "cv_jamaica" / modelo / modo / f"fold_{numero}"


def _solicitud(rutas, modelo, modo, fold, params):
    manifiesto = _manifiesto(rutas, fold)
    if (not params.get("referencias") or not params.get("receta")
            or not params.get("entorno") or not params.get("huella_datos")
            or params.get("semilla_folds") != 42
            or params.get("estrategia_folds") != split.ESTRATEGIA_FOLDS_JAMAICA):
        raise ValueError("Faltan receta, referencias, entorno, huella de datos, semilla o estrategia de folds.")
    base = referencia(rutas, modelo)
    if any(params.get(k) != base[k] for k in ("receta", "umbral", "referencias")):
        raise ValueError("La configuración de CV debe conservar exactamente la receta de la referencia.")
    commit = resultados._commit(rutas)
    if not commit:
        raise ValueError("No se pudo registrar el commit del código.")
    config = dict(protocolo=PROTOCOLO, modelo=modelo, modo=modo, fold=fold["fold"],
        commit=commit, params=params,
        huella_fold=hashlib.sha256(manifiesto.to_csv(index=False).encode("utf-8")).hexdigest())
    if modelo not in VERSIONES_YOLO:
        config["version_entrenamiento"] = VERSION_CNN
    # Fuerza un JSON finito antes de crear carpetas; no serializa silenciosamente objetos como texto.
    return json.loads(_json(config)), manifiesto


def preparar_fold(rutas, modelo, modo, fold, params, *, reiniciar=False, reanudar=False):
    """Crea solo el fold que se va a entrenar; archiva al repetir, sin borrados.

    Las CNN pueden retomar checkpoints compatibles. Un parcial anterior a esa
    función se archiva automáticamente, conservando todos sus archivos.
    """
    resultados._exigir_escritura(rutas)
    config, manifiesto = _solicitud(rutas, modelo, modo, fold, params)
    carpeta = ruta_fold(rutas, modelo, modo, fold["fold"])
    if carpeta.exists() and any(carpeta.iterdir()):
        if reanudar and not reiniciar and modelo not in VERSIONES_YOLO:
            guardada = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
            if not _compatible(guardada, config):
                raise ValueError("Cambió receta, entorno, datos o versión de entrenamiento: no reanudar ese fold.")
            if (carpeta / "completo.json").exists():
                raise FileExistsError("El fold ya está completo; comprobar fold_completo antes de preparar.")
            if (carpeta / "reanudacion").exists():
                return carpeta
            print(f"Fold {fold['fold']} sin checkpoint recuperable: se archiva el parcial y empieza de nuevo.")
            reiniciar = True
        if not reiniciar:
            raise FileExistsError("Fold con contenido: comprobar fold_completo o archivar con REPETIR=True.")
        carpeta.rename(carpeta.with_name(f"{carpeta.name}__anterior_{datetime.now():%Y%m%d_%H%M%S_%f}"))
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(_json(config), encoding="utf-8")
    manifiesto.to_csv(carpeta / "particiones.csv", index=False)
    return carpeta


def _leer_fold(rutas, modelo, modo, fold, params=None):
    carpeta = ruta_fold(rutas, modelo, modo, fold["fold"])
    config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    manifiesto = _manifiesto(rutas, fold)
    huella = hashlib.sha256(manifiesto.to_csv(index=False).encode("utf-8")).hexdigest()
    if (config.get("protocolo"), config.get("modelo"), config.get("modo"), config.get("fold"), config.get("huella_fold")) != (
            PROTOCOLO, modelo, modo, fold["fold"], huella):
        raise ValueError("La configuración no corresponde al fold actual.")
    if params is not None and not _compatible(config, _solicitud(rutas, modelo, modo, fold, params)[0]):
        raise ValueError("Cambió receta, entorno, datos o versión de entrenamiento: no reutilizar ese fold.")
    control = json.loads((carpeta / "completo.json").read_text(encoding="utf-8"))
    pesos = "modelo.pt" if modelo in VERSIONES_YOLO else "modelo.keras"
    obligatorios = {"config.json", "particiones.csv", "predicciones_oof.csv", "entrenamiento.json", pesos}
    if modelo in VERSIONES_YOLO:
        obligatorios.add("cajas_oof.csv")
    if not obligatorios.issubset(control["archivos"]):
        raise ValueError("Faltan archivos obligatorios del fold.")
    for nombre, digest in control["archivos"].items():
        if Path(nombre).name != nombre or _sha(carpeta / nombre) != digest:
            raise ValueError(f"Archivo modificado o incompleto: {nombre}")
    pred = pd.read_csv(carpeta / "predicciones_oof.csv")
    reales = fold["evaluacion"].set_index("archivo").y_original
    if (pred.archivo.duplicated().any() or set(pred.archivo) != set(reales.index)
            or not pred.y_real.eq(pred.archivo.map(reales)).all()):
        raise ValueError("Las predicciones no corresponden a la evaluación de este fold.")
    p = pred.probabilidad.to_numpy(float)
    u = config["params"]["umbral"]
    if (not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all()
            or not pred.prediccion.eq((p >= u).astype(int)).all()):
        raise ValueError("Puntajes o predicciones inválidos.")
    return pred.assign(modelo=modelo, modo=modo, fold=fold["fold"], umbral=u), config


def fold_completo(rutas, modelo, modo, fold, params, *, reanudar=False):
    carpeta = ruta_fold(rutas, modelo, modo, fold["fold"])
    if not carpeta.exists() or not any(carpeta.iterdir()):
        return False
    if not (carpeta / "completo.json").is_file():
        if reanudar and modelo not in VERSIONES_YOLO:
            config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
            if not _compatible(config, _solicitud(rutas, modelo, modo, fold, params)[0]):
                raise ValueError("El fold incompleto tiene otra receta, entorno, datos o versión de entrenamiento.")
            return False
        raise FileExistsError(f"Fold incompleto: seleccionar solo {fold['fold']} y usar REPETIR=True para archivarlo.")
    _leer_fold(rutas, modelo, modo, fold, params)
    return True


def guardar_fold(rutas, carpeta, modelo, modo, fold, params, probabilidad, entrenamiento, *, cajas=None):
    """Verifica pesos y predicciones fuera de fold; escribe el sello de completitud al final."""
    resultados._exigir_escritura(rutas)
    carpeta = Path(carpeta)
    if carpeta.resolve() != ruta_fold(rutas, modelo, modo, fold["fold"]).resolve():
        raise ValueError("Guardar solo dentro de analisis/cv_jamaica/<modelo>/<modo>/fold_N.")
    config, _ = _solicitud(rutas, modelo, modo, fold, params)
    if not _compatible(json.loads((carpeta / "config.json").read_text(encoding="utf-8")), config):
        raise ValueError("La solicitud cambió durante el entrenamiento.")
    if (carpeta / "completo.json").exists():
        raise FileExistsError("El fold ya está completo; no se sobrescribe.")
    p = np.asarray(probabilidad, dtype=float).reshape(-1)
    evaluacion = fold["evaluacion"]
    if len(p) != len(evaluacion) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Se requiere un puntaje en [0, 1] por imagen evaluada, en el orden del fold.")
    pesos = carpeta / ("modelo.pt" if modelo in VERSIONES_YOLO else "modelo.keras")
    if not pesos.is_file() or pesos.stat().st_size == 0:
        raise ValueError("Faltan los pesos del modelo del fold.")
    etapas = entrenamiento.get("etapas", {})
    esperadas = {"deteccion"} if modelo in VERSIONES_YOLO else {"cabeza", "fine_tuning"}
    if (set(etapas) != esperadas or any(
            not isinstance(e.get("mejor_epoca"), int) or not isinstance(e.get("epocas"), int)
            or not 1 <= e["mejor_epoca"] <= e["epocas"] for e in etapas.values())):
        raise ValueError("Registrar las mejores épocas de cada etapa, elegidas solo con validacion.")
    for etapa, datos_etapa in etapas.items():
        clave = "epochs" if etapa == "deteccion" else f"epocas_{etapa}"
        if datos_etapa["epocas"] > params["receta"][clave]:
            raise ValueError("Se excedió el presupuesto de épocas de la receta.")
    segundos = entrenamiento.get("segundos")
    if isinstance(segundos, bool) or not isinstance(segundos, (int, float)) or not np.isfinite(segundos) or segundos <= 0:
        raise ValueError("Registrar el tiempo de entrenamiento e inferencia en segundos positivos y finitos.")
    y = evaluacion.y_original.to_numpy(int)
    pred = pd.DataFrame(dict(archivo=evaluacion.archivo.to_list(), y_real=y, probabilidad=p,
        prediccion=(p >= params["umbral"]).astype(int)))
    if modelo in VERSIONES_YOLO:
        if cajas is None:
            raise ValueError("Los detectores deben guardar sus cajas fuera de fold, incluso si están vacías.")
        cajas = pd.DataFrame(cajas, columns=COLUMNAS_CAJAS)
        if (not cajas.archivo.isin(pred.archivo).all() or not cajas.clase.eq(0).all()
                or not np.isfinite(cajas[COLUMNAS_CAJAS[2:]].to_numpy(float)).all()
                or not cajas.confianza.between(0, 1).all()
                or not cajas[["xc", "yc", "w", "h"]].ge(0).all().all()
                or not cajas[["xc", "yc", "w", "h"]].le(1).all().all()
                or not cajas[["w", "h"]].gt(0).all().all()):
            raise ValueError("Cajas fuera del fold, no finitas o de otra clase.")
        maximos = cajas.groupby("archivo").confianza.max().reindex(pred.archivo, fill_value=0).to_numpy(float)
        if not np.allclose(maximos, p, rtol=0, atol=1e-7):
            raise ValueError("El puntaje del detector debe ser la máxima confianza de sus cajas (0 sin cajas).")
        cajas.to_csv(carpeta / "cajas_oof.csv", index=False)
    pred.to_csv(carpeta / "predicciones_oof.csv", index=False)
    (carpeta / "entrenamiento.json").write_text(_json(entrenamiento), encoding="utf-8")
    archivos = {p.name: _sha(p) for p in carpeta.iterdir() if p.is_file() and p.name != "completo.json"}
    (carpeta / "completo.json").write_text(_json(dict(archivos=archivos)), encoding="utf-8")
    _leer_fold(rutas, modelo, modo, fold, params)


def cargar_predicciones(rutas, folds):
    """Solo modelos con los cinco folds completos; informa ausencias sin mezclar parciales.

    Devuelve (predicciones, estado). Un sello corrupto da error, no se trata como ausencia.
    Lectura compatible con entorno.preparar(solo_lectura=True).
    """
    if sorted(f["fold"] for f in folds) != [1, 2, 3, 4, 5]:
        raise ValueError("Se requieren los cinco folds del protocolo.")
    todas, estado = [], []
    for modelo, modo in FINALISTAS.items():
        partes, configuraciones = [], []
        for fold in folds:
            carpeta = ruta_fold(rutas, modelo, modo, fold["fold"])
            completa = (carpeta / "completo.json").is_file()
            estado.append(dict(modelo=modelo, modo=modo, fold=fold["fold"], completa=completa))
            if completa:
                pred, config = _leer_fold(rutas, modelo, modo, fold)
                partes.append(pred)
                configuraciones.append(config)
        if len(partes) != 5:
            continue
        firmas = {_json(_firma_config(c)) for c in configuraciones}
        if len(firmas) != 1:
            raise ValueError(f"No mezclar folds de distinta receta, datos, entorno o commit: {modelo}")
        pred = pd.concat(partes, ignore_index=True)
        esperadas = {p.name for archivos in split.cargar_split(rutas).values() for p in archivos if split.es_jamaica(p.name)}
        if pred.archivo.duplicated().any() or set(pred.archivo) != esperadas:
            raise ValueError("Cada imagen de Jamaica debe evaluarse exactamente una vez.")
        todas.append(pred)
    columnas = ["archivo", "y_real", "probabilidad", "prediccion", "modelo", "modo", "fold", "umbral"]
    return (pd.concat(todas, ignore_index=True) if todas else pd.DataFrame(columns=columnas)), pd.DataFrame(estado)


def tabla_umbrales(predicciones, n=2000):
    """Umbral por grupos, comparado contra 0.5 en CNN y 0.25 en detectores.

    *_cv evalúa la elección del umbral dejando un grupo afuera; no equivale a una
    CV anidada del entrenamiento (los modelos de otros folds comparten datos).
    """
    filas = []
    for (modelo, modo), g in predicciones.groupby(["modelo", "modo"]):
        fila = umbral.tabla_jamaica(g).iloc[0].to_dict()
        base = float(g.umbral.iloc[0])
        y, p = g.y_real.to_numpy(int), g.probabilidad.to_numpy(float)
        pred_cv, _ = umbral.evaluar_umbral_por_grupos(y, p, umbral.grupos_jamaica(g.archivo))
        referencia = g.assign(modelo="referencia", modo="base", prediccion=(p >= base).astype(int))
        ajustada = g.assign(modelo="referencia", modo="ajustado", prediccion=pred_cv)
        diferencia = comparacion.diferencias(pd.concat([referencia, ajustada]), ["referencia/ajustado"],
            "referencia/base", metrica="balanced_accuracy", n=n).iloc[0]
        fila.update(umbral_referencia=base, balanced_accuracy_referencia=umbral.balanced_accuracy(y, p >= base),
            diferencia_cv=diferencia.diferencia, diferencia_cv_inf=diferencia["inf"], diferencia_cv_sup=diferencia["sup"],
            mejora_clara=bool(diferencia["inf"] > 0))
        filas.append(fila)
    return pd.DataFrame(filas)
