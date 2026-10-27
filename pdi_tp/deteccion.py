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
