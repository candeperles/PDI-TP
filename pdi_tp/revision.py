"""Selección reproducible y galerías de revisión; no cambia las anotaciones."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from .entorno import Rutas
from .validacion import Informe


def seleccionar_casos(informe: Informe, semilla=42, negativos_por_estrato=3) -> pd.DataFrame:
    """Incluye todas las cajas raras/pequeñas/inválidas y una muestra de negativos.

    Negativos: hasta tres imágenes por combinación fuente/split. La muestra es
    exploratoria y no sirve para estimar una tasa poblacional de errores.
    """
    cajas = informe.cajas.merge(informe.imagenes[["archivo", "fuente"]], on="archivo")
    bloques = []
    for categoria, prefijo, seleccion in (
        ("clase_minoritaria", "R", cajas["clase"].isin([1, 2])),
        ("caja_pequena", "S", cajas["pequena"]),
        ("caja_invalida", "Z", ~cajas["valida"]),
    ):
        tabla = cajas.loc[seleccion].sort_values(["archivo", "linea"]).copy()
        tabla["categoria"] = categoria
        tabla["caso"] = [f"{prefijo}{i:02}" for i in range(1, len(tabla) + 1)]
        bloques.append(tabla)
    negativos = informe.imagenes[informe.imagenes["con_paneles"] == False]
    muestras = [g.sample(n=min(negativos_por_estrato, len(g)), random_state=semilla)
                for _, g in negativos.groupby(["fuente", "split"], sort=True)]
    if muestras:
        tabla = pd.concat(muestras).sort_values(["fuente", "split", "archivo"]).copy()
        tabla["categoria"] = "muestra_sin_paneles"
        tabla["caso"] = [f"N{i:02}" for i in range(1, len(tabla) + 1)]
        bloques.append(tabla)
    columnas = ["caso", "categoria", "archivo", "linea", "clase", "split", "fuente", "w_px", "h_px"]
    casos = pd.concat(bloques, ignore_index=True).reindex(columns=columnas)
    casos["linea"] = casos["linea"].astype("Int64")
    casos["clase"] = casos["clase"].astype("Int64")
    return casos


def exportar_galerias(casos: pd.DataFrame, informe: Informe, rutas: Rutas, destino: Path) -> list[Path]:
    """Genera páginas de seis casos con imagen completa y zoom de la caja.

    El zoom conserva contexto (al menos 24 px) y usa píxeles sin suavizado.
    No se interpreta el zoom como información de mayor resolución.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    casos.to_csv(destino / "casos.csv", index=False, encoding="utf-8")
    archivos = []
    for categoria, grupo in casos.groupby("categoria", sort=False):
        for inicio in range(0, len(grupo), 6):
            pagina = grupo.iloc[inicio:inicio + 6]
            negativo = categoria == "muestra_sin_paneles"
            columnas = 3 if negativo else 4
            filas = int(np.ceil(len(pagina) / (3 if negativo else 2)))
            fig, axs = plt.subplots(filas, columnas, figsize=(4 * columnas, 4 * filas), squeeze=False)
            for ax in axs.flat:
                ax.set_axis_off()
            for indice, caso in enumerate(pagina.itertuples()):
                with Image.open(rutas.imagenes / caso.archivo) as im:
                    rgb = np.asarray(im.convert("RGB"))
                alto, ancho = rgb.shape[:2]
                ax = axs.flat[indice] if negativo else axs.flat[indice * 2]
                ax.imshow(rgb, interpolation="nearest")
                if negativo:
                    ax.set_title(f"{caso.caso} · {caso.fuente}/{caso.split}\nEtiqueta vacía", fontsize=10)
                    continue
                otras = informe.cajas[informe.cajas["archivo"] == caso.archivo]
                for c in otras.itertuples():
                    ax.add_patch(Rectangle(((c.xc-c.w/2)*ancho, (c.yc-c.h/2)*alto),
                                           c.w*ancho, c.h*alto, fill=False, edgecolor="lime", linewidth=0.7))
                c = otras.loc[otras["linea"] == caso.linea].iloc[0]
                x0, y0 = (c.xc-c.w/2)*ancho, (c.yc-c.h/2)*alto
                w, h = c.w*ancho, c.h*alto
                ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="red", linewidth=1.5))
                ax.set_xlim(-2, ancho+2)
                ax.set_ylim(alto+2, -2)
                ax.set_title(f"{caso.caso} · C{caso.clase}/L{caso.linea} · {caso.split}", fontsize=10)
                zoom = axs.flat[indice*2+1]
                zoom.imshow(rgb, interpolation="nearest")
                zoom.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="red", linewidth=1.5))
                margen = 8
                lado_x, lado_y = max(24, w+2*margen), max(24, h+2*margen)
                cx, cy = c.xc*ancho, c.yc*alto
                zoom.set_xlim(max(-2, cx-lado_x/2), min(ancho+2, cx+lado_x/2))
                zoom.set_ylim(min(alto+2, cy+lado_y/2), max(-2, cy-lado_y/2))
                zoom.set_title(f"Contexto ampliado · {w:.2f} × {h:.2f} px", fontsize=10)
            numero = inicio//6+1
            fig.suptitle(f"{categoria} · página {numero} · IDs completos en casos.csv", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .97))
            archivo = destino / f"{categoria}_{numero:02}.png"
            fig.savefig(archivo, dpi=100, bbox_inches="tight")
            plt.close(fig)
            archivos.append(archivo)
    return archivos
