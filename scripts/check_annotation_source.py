"""Coteja las clases 1/2 locales con las etiquetas originales de Figshare v3.

Uso: python scripts/check_annotation_source.py
Requiere red. Guarda evidencia en resultados/revision_anotaciones/.
No modifica datos ni descarga imágenes nuevas.
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdi_tp import entorno, revision, validacion  # noqa: E402

URL = "https://api.figshare.com/v2/articles/22081091"
SHA256_LABELS = "9a24d5657d7d0e27d5b083f7749067fefa376f03189293ff8c05b2b3c9916929"


def cotejar():
    rutas = entorno.preparar("revision_anotaciones", usar_drive=False)
    salida = rutas.resultados
    informe = validacion.validar(rutas)
    casos = revision.seleccionar_casos(informe)
    with urllib.request.urlopen(URL, timeout=60) as respuesta:
        meta = json.load(respuesta)
    if meta.get("version") != 3:
        raise ValueError("Cambió la versión de Figshare; revisar la procedencia antes de continuar.")
    (salida / "figshare_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    recurso = next(f for f in meta["files"] if f["name"] == "labels.zip")
    archivo = salida / "labels.zip"
    if not archivo.is_file():
        with urllib.request.urlopen(recurso["download_url"], timeout=120) as respuesta:
            archivo.write_bytes(respuesta.read())
    contenido = archivo.read_bytes()
    if hashlib.sha256(contenido).hexdigest() != SHA256_LABELS:
        raise ValueError("El ZIP de etiquetas no coincide con la fuente cotejada.")
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        originales = {Path(n).name: z.read(n).decode("utf-8") for n in z.namelist()
                      if n.endswith(".txt") and "hd" in n}
    filas = []
    for caso in casos[casos["categoria"] == "clase_minoritaria"].itertuples():
        stem, primera, segunda = Path(caso.archivo).stem.rsplit("_", 2)
        original = originales.get(stem + ".txt")
        coincidencias = []
        caja = informe.cajas[(informe.cajas["archivo"] == caso.archivo)
                            & (informe.cajas["linea"] == caso.linea)].iloc[0]
        if original is not None:
            valores = np.loadtxt(io.StringIO(original), ndmin=2)
            # Hipótesis de recorte: grilla 4×4; se verifican ambas convenciones de ejes.
            for ix, iy in sorted({(int(primera), int(segunda)), (int(segunda), int(primera))}):
                for numero, (clase, xc, yc, w, h) in enumerate(valores, 1):
                    x0, y0 = max(xc-w/2, ix/4), max(yc-h/2, iy/4)
                    x1, y1 = min(xc+w/2, (ix+1)/4), min(yc+h/2, (iy+1)/4)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    transformada = [clase, (x0+x1)*2-ix, (y0+y1)*2-iy, (x1-x0)*4, (y1-y0)*4]
                    local = caja[["clase", "xc", "yc", "w", "h"]].to_numpy(dtype=float)
                    if np.allclose(transformada, local, atol=3e-6, rtol=0):
                        coincidencias.append((numero, ix, iy))
        numero, ix, iy = coincidencias[0] if coincidencias else (None, None, None)
        filas.append(dict(caso=caso.caso, archivo=caso.archivo, linea_local=int(caso.linea),
                          clase=int(caso.clase), etiqueta_fuente=stem + ".txt", linea_fuente=numero,
                          columna_recorte=ix, fila_recorte=iy, coincide=bool(coincidencias)))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(salida / "cotejo_clases.csv", index=False, encoding="utf-8")
    print(tabla.to_string(index=False))
    if tabla.empty or not tabla["coincide"].all():
        raise ValueError("Hay anotaciones sin correspondencia con la fuente original.")
    print(f"Cotejo completo: {len(tabla)}/{len(tabla)} cajas coinciden en ID y geometría.")


if __name__ == "__main__":
    cotejar()
