"""Auditoría de solo lectura del dataset y del split compartido.

Los errores impiden considerar un archivo válido; las advertencias requieren revisión.
Los duplicados se buscan por bytes y por píxeles RGB idénticos (no por similitud).
"""
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import split
from .entorno import Rutas


@dataclass
class Informe:
    imagenes: pd.DataFrame
    cajas: pd.DataFrame
    incidencias: pd.DataFrame
    duplicados: pd.DataFrame
    controles: pd.DataFrame
    resumen: dict

    def guardar(self, destino: Path) -> None:
        """Exporta los resultados fuera del dataset original."""
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        for nombre in ("imagenes", "cajas", "incidencias", "duplicados", "controles"):
            getattr(self, nombre).to_csv(destino / f"{nombre}.csv", index=False, encoding="utf-8")
        (destino / "resumen.json").write_text(
            json.dumps(self.resumen, ensure_ascii=False, indent=2), encoding="utf-8",
        )


def validar(rutas: Rutas, splits_dir: Path | None = None, clases=(0, 1, 2),
            dimension=(256, 256), minimo_px=3.0, tolerancia=1e-6) -> Informe:
    """Revisa archivos, YOLO, geometría, duplicados y separación del split.

    `clases` enumera los IDs esperados, no interpreta su significado.
    `minimo_px` solo señala cajas para revisión; no las elimina.
    `tolerancia` se aplica a los bordes de las cajas por redondeo de coordenadas.
    Una etiqueta faltante o inválida nunca se interpreta como imagen negativa.
    """
    splits_dir = Path(splits_dir) if splits_dir is not None else split.SPLITS_DIR
    # iterdir falla explícitamente si la carpeta falta o no es accesible.
    archivos_img = sorted(p for p in rutas.imagenes.iterdir() if p.is_file())
    archivos_lbl = sorted(p for p in rutas.labels.iterdir() if p.is_file())
    imagenes = [p for p in archivos_img if p.suffix.lower() == ".png"]
    labels = [p for p in archivos_lbl if p.suffix.lower() == ".txt"]
    if not imagenes:
        raise ValueError(f"No hay imágenes PNG en {rutas.imagenes}.")
    incidencias, filas_img, filas_caja, duplicados = [], [], [], []

    def anotar(nivel, control, archivo, detalle, linea=None):
        incidencias.append(dict(nivel=nivel, control=control, archivo=archivo,
                                linea=linea, detalle=detalle))

    for p in archivos_img + archivos_lbl:
        esperado = ".png" if p.parent == rutas.imagenes else ".txt"
        if p.suffix != esperado:
            anotar("advertencia", "extension_inesperada", p.name,
                   f"Se espera {esperado} en minúsculas; revisar compatibilidad con el lector.")

    # Se conserva cada línea del split para detectar repeticiones y entradas inválidas.
    conjuntos = {}
    for nombre in ("train", "val"):
        archivo = splits_dir / f"{nombre}.txt"
        try:
            lineas = archivo.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            anotar("error", "split_no_legible", archivo.name, str(exc))
            lineas = []
        entradas = []
        for numero, linea in enumerate(lineas, 1):
            valor = linea.strip()
            if not valor:
                continue
            if "/" in valor or "\\" in valor or not valor.endswith(".png"):
                anotar("error", "split_entrada_invalida", archivo.name, valor, numero)
            else:
                entradas.append(valor)
        for valor, cantidad in Counter(entradas).items():
            if cantidad > 1:
                anotar("error", "split_entrada_repetida", valor, f"{nombre}: {cantidad} veces")
        conjuntos[nombre] = set(entradas)
        if not entradas:
            anotar("error", "split_vacio", archivo.name, "No contiene imágenes válidas.")

    nombres_img = {p.name for p in imagenes}
    for nombre, entradas in conjuntos.items():
        for faltante in sorted(entradas - nombres_img):
            anotar("error", "split_imagen_inexistente", faltante, f"Referenciada en {nombre}.")
    for nombre in sorted(conjuntos["train"] & conjuntos["val"]):
        anotar("error", "imagen_en_ambos_splits", nombre, "Figura en train y val.")
    for nombre in sorted(nombres_img - conjuntos["train"] - conjuntos["val"]):
        anotar("error", "imagen_sin_split", nombre, "No figura en train ni val.")

    por_stem = {}
    for label in labels:
        por_stem.setdefault(label.stem, []).append(label)
    stems_img = {p.stem for p in imagenes}
    for label in labels:
        if label.stem not in stems_img:
            anotar("error", "etiqueta_sin_imagen", label.name, "No hay PNG con el mismo nombre base.")

    for imagen in imagenes:
        pertenencia = [s for s in ("train", "val") if imagen.name in conjuntos[s]]
        fila = dict(archivo=imagen.name, fuente=split.fuente(imagen.stem),
                    split="+".join(pertenencia), legible=False, ancho=None, alto=None,
                    modo=None, sha256=None, sha256_rgb=None, etiqueta=None,
                    etiqueta_valida=False, n_lineas=0, n_cajas_validas=0,
                    con_paneles=None, grupo=None, sha256_etiqueta=None)
        try:
            fila["sha256"] = hashlib.sha256(imagen.read_bytes()).hexdigest()
            with Image.open(imagen) as im:
                im.verify()
            with Image.open(imagen) as im:
                im.load()
                fila.update(legible=True, ancho=im.width, alto=im.height, modo=im.mode)
                rgb = im.convert("RGB")
                contenido = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
                fila["sha256_rgb"] = hashlib.sha256(contenido).hexdigest()
                if im.format != "PNG":
                    anotar("error", "formato_imagen", imagen.name, f"Formato real: {im.format}.")
                if im.size != tuple(dimension):
                    anotar("error", "dimension_imagen", imagen.name,
                           f"{im.size}; se espera {tuple(dimension)}.")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            anotar("error", "imagen_no_legible", imagen.name, str(exc))

        try:
            split.grupos_espaciales([imagen.stem])
            fila["ubicacion_reconocida"] = True
        except ValueError as exc:
            fila["ubicacion_reconocida"] = False
            anotar("error", "ubicacion_no_reconocida", imagen.name, str(exc))

        candidatas = por_stem.get(imagen.stem, [])
        if len(candidatas) != 1:
            control = "etiqueta_faltante" if not candidatas else "etiqueta_ambigua"
            anotar("error", control, imagen.name, f"Se encontraron {len(candidatas)} etiquetas.")
            filas_img.append(fila)
            continue
        label = candidatas[0]
        fila["etiqueta"] = label.name
        try:
            contenido = label.read_bytes()
            fila["sha256_etiqueta"] = hashlib.sha256(contenido).hexdigest()
            lineas = contenido.decode("utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            anotar("error", "etiqueta_no_legible", imagen.name, str(exc))
            filas_img.append(fila)
            continue
        etiqueta_valida = True
        for numero, linea in enumerate(lineas, 1):
            if not linea.strip():
                continue
            fila["n_lineas"] += 1
            campos = linea.split()
            if len(campos) != 5:
                anotar("error", "columnas_yolo", imagen.name,
                       f"{len(campos)} valores; se esperan 5: {linea}", numero)
                etiqueta_valida = False
                continue
            try:
                valores = np.array([float(v) for v in campos])
            except ValueError:
                anotar("error", "valor_no_numerico", imagen.name, linea, numero)
                etiqueta_valida = False
                continue
            if not np.isfinite(valores).all():
                anotar("error", "valor_no_finito", imagen.name, linea, numero)
                etiqueta_valida = False
                continue
            clase, xc, yc, w, h = valores
            errores = []
            if not clase.is_integer() or clase < 0:
                errores.append(("clase_invalida", f"ID no entero o negativo: {clase}"))
            elif int(clase) not in clases:
                errores.append(("clase_no_admitida", f"ID {int(clase)}; esperados {list(clases)}."))
            if not (0 <= xc <= 1 and 0 <= yc <= 1):
                errores.append(("centro_fuera_rango", f"Centro: ({xc}, {yc})."))
            if w <= 0 or h <= 0:
                errores.append(("dimension_caja_no_positiva", f"Ancho={w}, alto={h}."))
            if w > 1 or h > 1:
                errores.append(("dimension_caja_fuera_rango", f"Ancho={w}, alto={h}."))
            limites = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
            if w > 0 and h > 0 and (min(limites[:2]) < -tolerancia
                                    or max(limites[2:]) > 1 + tolerancia):
                errores.append(("caja_fuera_imagen", f"Bordes normalizados: {limites}."))
            for control, detalle in errores:
                anotar("error", control, imagen.name, detalle, numero)
            valida = not errores
            etiqueta_valida &= valida
            fila["n_cajas_validas"] += int(valida)
            w_px = w * fila["ancho"] if fila["legible"] else None
            h_px = h * fila["alto"] if fila["legible"] else None
            chica = fila["legible"] and w > 0 and h > 0 and (w_px < minimo_px or h_px < minimo_px)
            if chica:
                anotar("advertencia", "caja_pequena", imagen.name,
                       f"{w_px:.4f} × {h_px:.4f} px; umbral de revisión: {minimo_px} px.", numero)
            filas_caja.append(dict(archivo=imagen.name, linea=numero, clase=clase,
                                   xc=xc, yc=yc, w=w, h=h, w_px=w_px, h_px=h_px,
                                   valida=valida, pequena=chica, split=fila["split"]))
        fila["etiqueta_valida"] = bool(etiqueta_valida)
        if etiqueta_valida:
            fila["con_paneles"] = fila["n_lineas"] > 0
        filas_img.append(fila)

    tabla_img = pd.DataFrame(filas_img)
    tabla_cajas = pd.DataFrame(filas_caja, columns=[
        "archivo", "linea", "clase", "xc", "yc", "w", "h", "w_px", "h_px", "valida", "pequena", "split",
    ])
    reconocidas = tabla_img["ubicacion_reconocida"]
    if reconocidas.any():
        grupos = split.grupos_espaciales([Path(n).stem for n in tabla_img.loc[reconocidas, "archivo"]])
        tabla_img.loc[reconocidas, "grupo"] = grupos
    for grupo, filas in tabla_img.dropna(subset=["grupo"]).groupby("grupo"):
        lados = {s for n in filas["archivo"] for s, entradas in conjuntos.items() if n in entradas}
        if len(lados) > 1:
            anotar("error", "grupo_espacial_entre_splits", filas.iloc[0]["archivo"],
                   f"Grupo {int(grupo)}: {', '.join(filas['archivo'])}")

    # Píxeles RGB: detecta también copias con distinta compresión o modo de color.
    for tipo, columna in (("bytes", "sha256"), ("pixeles_rgb", "sha256_rgb")):
        for huella, filas in tabla_img.dropna(subset=[columna]).groupby(columna):
            if len(filas) < 2:
                continue
            nombres = sorted(filas["archivo"])
            lados = {s for n in nombres for s, entradas in conjuntos.items() if n in entradas}
            cruza = len(lados) > 1
            duplicados.append(dict(tipo=tipo, hash=huella, cantidad=len(nombres),
                                   archivos=" | ".join(nombres), entre_splits=cruza))
            # Una sola incidencia por grupo visual; los duplicados por bytes ya están incluidos.
            if tipo == "pixeles_rgb":
                anotar("error" if cruza else "advertencia",
                       "duplicado_entre_splits" if cruza else "imagen_duplicada",
                       nombres[0], " | ".join(nombres))
                anotaciones = []
                for nombre in nombres:
                    if not bool(filas.loc[filas["archivo"] == nombre, "etiqueta_valida"].iloc[0]):
                        continue
                    cajas = tabla_cajas.loc[tabla_cajas["archivo"] == nombre, ["clase", "xc", "yc", "w", "h"]]
                    anotaciones.append(tuple(sorted(map(tuple, cajas.to_numpy()))))
                if len(set(anotaciones)) > 1:
                    anotar("advertencia", "duplicado_anotaciones_distintas", nombres[0],
                           "Los píxeles son idénticos pero las anotaciones difieren: " + " | ".join(nombres))

    tabla_inc = pd.DataFrame(incidencias, columns=["nivel", "control", "archivo", "linea", "detalle"])
    tabla_dup = pd.DataFrame(duplicados, columns=["tipo", "hash", "cantidad", "archivos", "entre_splits"])
    controles = []
    familias = {
        "Correspondencia y archivos": {"extension_inesperada", "etiqueta_faltante", "etiqueta_ambigua", "etiqueta_sin_imagen", "etiqueta_no_legible"},
        "Imágenes": {"imagen_no_legible", "dimension_imagen", "formato_imagen"},
        "Formato y clases YOLO": {"columnas_yolo", "valor_no_numerico", "valor_no_finito", "clase_invalida", "clase_no_admitida"},
        "Geometría de cajas": {"centro_fuera_rango", "dimension_caja_no_positiva", "dimension_caja_fuera_rango", "caja_fuera_imagen"},
        "Cajas pequeñas": {"caja_pequena"},
        "Duplicados exactos": {"imagen_duplicada", "duplicado_entre_splits", "duplicado_anotaciones_distintas"},
        "Integridad del split": {"split_no_legible", "split_entrada_invalida", "split_entrada_repetida", "split_vacio", "split_imagen_inexistente", "imagen_en_ambos_splits", "imagen_sin_split"},
        "Separación espacial": {"ubicacion_no_reconocida", "grupo_espacial_entre_splits"},
    }
    for control, codigos in familias.items():
        casos = tabla_inc[tabla_inc["control"].isin(codigos)]
        errores = int((casos["nivel"] == "error").sum())
        avisos = int((casos["nivel"] == "advertencia").sum())
        controles.append(dict(control=control, estado="error" if errores else "revisar" if avisos else "ok",
                              errores=errores, advertencias=avisos))
    resumen = dict(
        imagenes=len(tabla_img), etiquetas=len(labels),
        imagenes_legibles=int(tabla_img["legible"].sum()),
        etiquetas_validas=int(tabla_img["etiqueta_valida"].sum()),
        imagenes_con_paneles=int((tabla_img["con_paneles"] == True).sum()),
        imagenes_sin_paneles=int((tabla_img["con_paneles"] == False).sum()),
        imagenes_etiqueta_desconocida=int(tabla_img["con_paneles"].isna().sum()),
        cajas_numericas_finitas=len(tabla_cajas), cajas_validas=int(tabla_cajas["valida"].sum()),
        cajas_pequenas=int(tabla_cajas["pequena"].sum()),
        cajas_dimension_no_positiva=int(((tabla_cajas["w"] <= 0) | (tabla_cajas["h"] <= 0)).sum()),
        grupos_espaciales=int(tabla_img["grupo"].nunique()),
        train=len(conjuntos["train"]), val=len(conjuntos["val"]),
        grupos_duplicados_bytes=int((tabla_dup["tipo"] == "bytes").sum()),
        grupos_duplicados_rgb=int((tabla_dup["tipo"] == "pixeles_rgb").sum()),
        grupos_duplicados_rgb_entre_splits=int(((tabla_dup["tipo"] == "pixeles_rgb") & tabla_dup["entre_splits"]).sum()),
        errores=int((tabla_inc["nivel"] == "error").sum()),
        advertencias=int((tabla_inc["nivel"] == "advertencia").sum()),
        parametros=dict(clases=list(clases), dimension=list(dimension), minimo_px=minimo_px, tolerancia=tolerancia),
    )
    return Informe(tabla_img, tabla_cajas, tabla_inc, tabla_dup, pd.DataFrame(controles), resumen)


def dibujar_ejemplos(informe: Informe, rutas: Rutas, archivos, maximo=8):
    """Dibuja cajas numéricas incluso si exceden los bordes; no usa el lector estricto."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    archivos = list(dict.fromkeys(archivos))[:maximo]
    if not archivos:
        print("No hay ejemplos para esta selección.")
        return None
    columnas = min(4, len(archivos))
    fig, axs = plt.subplots(int(np.ceil(len(archivos) / columnas)), columnas,
                            figsize=(4 * columnas, 4 * int(np.ceil(len(archivos) / columnas))), squeeze=False)
    for ax in axs.flat:
        ax.set_axis_off()
    for ax, nombre in zip(axs.flat, archivos):
        try:
            with Image.open(rutas.imagenes / nombre) as im:
                rgb = im.convert("RGB")
                ancho, alto = rgb.size
                ax.imshow(rgb)
            cajas = informe.cajas[informe.cajas["archivo"] == nombre]
            for c in cajas.itertuples():
                color = "red" if not c.valida else "orange" if c.pequena else "lime"
                ax.add_patch(Rectangle(((c.xc - c.w / 2) * ancho, (c.yc - c.h / 2) * alto),
                                       c.w * ancho, c.h * alto, fill=False, edgecolor=color, linewidth=1.2))
                ax.text(c.xc * ancho, c.yc * alto, f"L{c.linea}/C{c.clase:g}", color=color, fontsize=7)
            ax.set_xlim(-5, ancho + 5)
            ax.set_ylim(alto + 5, -5)
        except (OSError, ValueError):
            ax.text(0.1, 0.5, "Imagen no disponible", transform=ax.transAxes)
        ax.set_title(nombre[:34] + ("…" if len(nombre) > 34 else ""), fontsize=8)
    fig.suptitle("Verde: caja válida · Naranja: menor a 3 px · Rojo: caja inválida", fontsize=11)
    fig.tight_layout()
    return fig
