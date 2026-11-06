"""Preparación de detección y conversión a clasificación, sin importar PyTorch."""
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import clasificacion, split


def preparar_dataset(rutas, destino):
    """Copia derivada RGB y YOLO de una clase; conserva originales y split.

    La auditoría compartida solo permite excluir Z01/línea 1. Conserva cajas
    pequeñas y confianza original en cajas_derivadas.csv. El destino debe ser
    fuera del dataset original. Reutiliza una copia íntegra del mismo manifiesto.
    """
    destino = Path(destino).resolve()
    original = rutas.datos.resolve()
    if destino == original or destino.is_relative_to(original) or original.is_relative_to(destino):
        raise ValueError("El derivado debe estar fuera del dataset original.")
    tabla, cajas = clasificacion.preparar_tabla(rutas)
    if set(tabla.split) != {"train", "val"} or tabla.archivo.duplicated().any():
        raise ValueError("El manifiesto debe tener train/val y nombres únicos.")
    if not tabla.archivo.map(lambda n: Path(n).name == n and n.endswith(".png")).all():
        raise ValueError("Nombres de imagen inválidos en el manifiesto.")
    huella = clasificacion.huella_tabla(tabla)
    registro = destino / "preparado.json"
    if destino.exists():
        if not registro.is_file():
            raise FileExistsError(f"Derivado incompleto: usar otro destino local: {destino}")
        previo = json.loads(registro.read_text(encoding="utf-8"))
        if previo["huella"] != huella or previo["destino"] != str(destino):
            raise ValueError("El derivado corresponde a otros datos; usar otro destino local.")
        for nombre, digest in previo["archivos"].items():
            archivo = destino / nombre
            if not archivo.is_file() or hashlib.sha256(archivo.read_bytes()).hexdigest() != digest:
                raise ValueError(f"El derivado fue modificado o está incompleto: {nombre}")
        return tabla, cajas
    incluidas = cajas.loc[cajas.incluida]
    for conjunto in ("train", "val"):
        (destino / "images" / conjunto).mkdir(parents=True)
        (destino / "labels" / conjunto).mkdir(parents=True)
    listas = {"train": [], "val": [], "jamaica": []}
    for fila in tabla.itertuples(index=False):
        imagen = destino / "images" / fila.split / fila.archivo
        with Image.open(rutas.imagenes / fila.archivo) as original_img:
            original_img.convert("RGB").save(imagen)
        seleccion = incluidas.loc[incluidas.archivo == fila.archivo]
        lineas = ["0 " + " ".join(f"{v:.10f}" for v in (c.xc, c.yc, c.w, c.h))
                  for c in seleccion.itertuples(index=False)]
        etiqueta = destino / "labels" / fila.split / (Path(fila.archivo).stem + ".txt")
        etiqueta.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
        listas[fila.split].append(imagen.as_posix())
        if fila.split == "val" and split.es_jamaica(fila.archivo):
            listas["jamaica"].append(imagen.as_posix())
    if not listas["jamaica"]:
        raise ValueError("No hay imágenes de Jamaica en val.")
    for nombre, imagenes in listas.items():
        (destino / f"{nombre}.txt").write_text("\n".join(imagenes) + "\n", encoding="utf-8")
    tabla.to_csv(destino / "manifiesto.csv", index=False)
    cajas.to_csv(destino / "cajas_derivadas.csv", index=False)
    # JSON es un subconjunto de YAML: no requiere otro lector/escritor en CI.
    for nombre, lista_val in (("datos", "val"), ("datos_jamaica", "jamaica")):
        contenido = dict(path=destino.as_posix(), train="train.txt", val=f"{lista_val}.txt",
                         names={0: "panel"})
        (destino / f"{nombre}.yaml").write_text(json.dumps(contenido, indent=2), encoding="utf-8")
    archivos = {p.relative_to(destino).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destino.rglob("*") if p.is_file()}
    registro.write_text(json.dumps(dict(huella=huella, destino=str(destino), archivos=archivos)),
                        encoding="utf-8")
    return tabla, cajas


def resumir_predicciones(predicciones, archivos):
    """Máxima confianza de panel por imagen (0 sin cajas) y cajas normalizadas.

    Es un puntaje de detección, no una probabilidad calibrada de presencia.
    Exige una salida por imagen, incluso para negativos, y alinea por nombre.
    """
    nombres = [Path(a).name for a in archivos]
    if len(nombres) != len(set(nombres)) or not nombres:
        raise ValueError("Se requieren nombres únicos y no vacíos.")
    esperados = set(nombres)
    puntajes, filas = {}, []
    for resultado in predicciones:
        nombre = Path(resultado.path).name
        if nombre not in esperados or nombre in puntajes:
            raise ValueError(f"Predicción inesperada o repetida: {nombre}")
        cajas = resultado.boxes
        if cajas is None:
            raise ValueError(f"Falta la salida de detección para {nombre}")
        conf = np.asarray(cajas.conf.cpu(), dtype=float).reshape(-1)
        clases = np.asarray(cajas.cls.cpu(), dtype=float).reshape(-1)
        xywh = np.asarray(cajas.xywhn.cpu(), dtype=float).reshape(-1, 4)
        if (len(conf) != len(clases) or len(conf) != len(xywh)
                or not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any()
                or not np.isfinite(xywh).all() or not (clases == 0).all()):
            raise ValueError(f"Cajas inválidas o clase distinta de panel en {nombre}")
        puntajes[nombre] = float(conf.max()) if len(conf) else 0.0
        filas.extend(dict(archivo=nombre, clase=0, confianza=float(p),
                          xc=float(c[0]), yc=float(c[1]), w=float(c[2]), h=float(c[3]))
                     for p, c in zip(conf, xywh))
    if set(puntajes) != esperados:
        raise ValueError("Faltan predicciones para imágenes de val.")
    return np.array([puntajes[n] for n in nombres]), filas


UMBRALES_IOU = np.linspace(0.5, 0.95, 10)


def leer_cajas(txt) -> np.ndarray:
    """Cajas (n, 4) xc, yc, w, h normalizadas de un .txt YOLO de una clase; (0, 4) si está vacío."""
    lineas = [l.split() for l in Path(txt).read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(len(v) != 5 or v[0] != "0" for v in lineas):
        raise ValueError(f"Se esperaban líneas 'clase xc yc w h' con clase 0 en {txt}")
    return np.array([[float(x) for x in v[1:]] for v in lineas]).reshape(-1, 4)


def _iou(reales: np.ndarray, predichas: np.ndarray) -> np.ndarray:
    def esquinas(c):
        return np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], 1)
    a, b = esquinas(reales), esquinas(predichas)
    ancho = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    alto = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    interseccion = ancho * alto
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return interseccion / (area(a)[:, None] + area(b)[None, :] - interseccion + 1e-7)


def _aciertos(reales: np.ndarray, predichas: np.ndarray) -> np.ndarray:
    """(m, 10) qué predicción es correcta en cada umbral de IoU, con el emparejado de Ultralytics."""
    correctas = np.zeros((len(predichas), len(UMBRALES_IOU)), dtype=bool)
    if not len(reales) or not len(predichas):
        return correctas
    iou = _iou(reales, predichas)
    for j, umbral in enumerate(UMBRALES_IOU):
        real, pred = np.nonzero(iou >= umbral)
        if not len(real):
            continue
        pares = np.stack([real, pred, iou[real, pred]], 1)
        if len(pares) > 1:
            # Mismo orden que Ultralytics (DetectionValidator.match_predictions): IoU descendente, una
            # caja real por predicción y después una predicción por caja real.
            pares = pares[pares[:, 2].argsort()[::-1]]
            pares = pares[np.unique(pares[:, 1], return_index=True)[1]]
            pares = pares[np.unique(pares[:, 0], return_index=True)[1]]
        correctas[pares[:, 1].astype(int), j] = True
    return correctas


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Área bajo la curva precisión-recall con interpolación de 101 puntos, como Ultralytics."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    integrar = getattr(np, "trapezoid", None) or np.trapz  # trapz en NumPy < 2
    return float(integrar(np.interp(x, mrec, mpre), x))


def map_cajas(reales: dict, predichas) -> dict:
    """mAP50, mAP50-95 y la precisión y el recall del F1 máximo, igual que `model.val` de Ultralytics.

    `reales`: {archivo: cajas (n, 4) normalizadas}, con todas las imágenes evaluadas (también las vacías).
    `predichas`: filas con archivo, confianza, xc, yc, w, h (las de `resumir_predicciones`).
    Sirve para medir sin escribir el .yaml y las carpetas que necesita `model.val`.
    """
    por_imagen = {}
    for fila in predichas:
        if fila["archivo"] not in reales:
            raise ValueError(f"Predicción de una imagen sin etiqueta: {fila['archivo']}")
        por_imagen.setdefault(fila["archivo"], []).append(fila)
    aciertos, confianzas = [], []
    for archivo, cajas in reales.items():
        filas = por_imagen.get(archivo, [])
        caja = np.array([[f["xc"], f["yc"], f["w"], f["h"]] for f in filas]).reshape(-1, 4)
        aciertos.append(_aciertos(np.asarray(cajas).reshape(-1, 4), caja))
        confianzas += [f["confianza"] for f in filas]
    n_reales = sum(len(c) for c in reales.values())
    if not n_reales:
        raise ValueError("No hay cajas reales para evaluar.")
    if not confianzas:
        return dict(map50=0.0, map50_95=0.0, precision=0.0, recall=0.0)
    aciertos, confianzas = np.concatenate(aciertos), np.array(confianzas)
    orden = np.argsort(-confianzas, kind="stable")
    aciertos, confianzas = aciertos[orden], confianzas[orden]
    tp, fp = np.cumsum(aciertos, 0), np.cumsum(1 - aciertos, 0)
    recall = tp / (n_reales + 1e-16)
    precision = tp / (tp + fp)
    ap = np.array([_ap(recall[:, j], precision[:, j]) for j in range(len(UMBRALES_IOU))])
    # Precisión y recall en el puntaje que maximiza el F1 suavizado (Ultralytics, ap_per_class).
    x = np.linspace(0, 1, 1000)
    r = np.interp(-x, -confianzas, recall[:, 0], left=0)
    p = np.interp(-x, -confianzas, precision[:, 0], left=1)
    f1 = 2 * p * r / (p + r + 1e-16)
    ventana = round(len(f1) * 0.1 * 2) // 2 + 1
    relleno = np.ones(ventana // 2)
    f1 = np.convolve(np.concatenate((relleno * f1[0], f1, relleno * f1[-1])), np.ones(ventana) / ventana, mode="valid")
    i = int(f1.argmax())
    return dict(map50=float(ap[0]), map50_95=float(ap.mean()), precision=float(p[i]), recall=float(r[i]))
