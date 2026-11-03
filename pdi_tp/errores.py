"""Revisión visual de predicciones guardadas; no entrena ni escribe resultados.

Las cuentas por caja son un diagnóstico a confianza/IoU fijas, no reemplazan el
mAP de Ultralytics. Una caja cercana con IoU insuficiente sigue siendo FP y FN.
"""
import textwrap

import numpy as np
import pandas as pd

from . import comparacion, datos, resultados, split

COORDENADAS = ["xc", "yc", "w", "h"]
COLUMNAS_CAJAS = ["archivo", "clase", "confianza", *COORDENADAS]
COLORES = dict(referencia="#0072B2", correcta="#009E73", falsa="#D62728",
               desajustada="#E69F00", omitida="#CC79A7")


def _validar_cajas(cajas):
    cajas = np.asarray(cajas, dtype=float).reshape(-1, 4)
    if (not np.isfinite(cajas).all() or (cajas[:, :2] < 0).any()
            or (cajas[:, :2] > 1).any() or (cajas[:, 2:] <= 0).any()
            or (cajas[:, 2:] > 1 + 1e-6).any()):
        raise ValueError("Las cajas deben ser finitas, normalizadas y de tamaño positivo.")
    return cajas


def _esquinas(cajas):
    return np.clip(np.concatenate([cajas[:, :2] - cajas[:, 2:] / 2,
                                   cajas[:, :2] + cajas[:, 2:] / 2], axis=1), 0, 1)


def iou_cajas(anotadas, predichas):
    """Matriz de IoU (referencias × predicciones), en coordenadas normalizadas."""
    reales, propuestas = _esquinas(_validar_cajas(anotadas)), _esquinas(_validar_cajas(predichas))
    minimo = np.maximum(reales[:, None, :2], propuestas[None, :, :2])
    maximo = np.minimum(reales[:, None, 2:], propuestas[None, :, 2:])
    interseccion = np.maximum(maximo - minimo, 0).prod(axis=2)
    area_real = (reales[:, 2:] - reales[:, :2]).prod(axis=1)
    area_pred = (propuestas[:, 2:] - propuestas[:, :2]).prod(axis=1)
    union = area_real[:, None] + area_pred[None, :] - interseccion
    return np.divide(interseccion, union, out=np.zeros_like(union), where=union > 0)


def diagnosticar_cajas(anotadas, predichas, *, confianza=0.25, iou_min=0.5, iou_cercana=0.1):
    """Empareja por confianza descendente, sin reutilizar una referencia.

    Entre las cajas restantes, marca pares cercanos por IoU descendente. Son
    candidatos a mala localización, no una confirmación de que sean el mismo objeto.
    Los índices P/R de la vista son estables dentro del umbral elegido.
    """
    if not 0 <= confianza <= 1 or not 0 < iou_cercana < iou_min <= 1:
        raise ValueError("Se requiere confianza en [0, 1] y 0 < IoU cercana < IoU mínima <= 1.")
    reales = _validar_cajas(anotadas)
    valores = predichas[["confianza", *COORDENADAS]].to_numpy(dtype=float)
    if not np.isfinite(valores).all() or ((valores[:, 0] < 0) | (valores[:, 0] > 1)).any():
        raise ValueError("Confianzas inválidas en las predicciones.")
    _validar_cajas(valores[:, 1:])
    seleccion = predichas.loc[predichas.confianza >= confianza].sort_values(
        "confianza", ascending=False, kind="stable").reset_index(drop=True)
    propuestas = seleccion[COORDENADAS].to_numpy(float)
    ious = iou_cajas(reales, propuestas)
    libres = set(range(len(reales)))
    coincidencias = []
    for p in range(len(propuestas)):
        if not libres:
            break
        r = max(sorted(libres), key=lambda r: ious[r, p])
        if ious[r, p] >= iou_min:
            coincidencias.append((r, p, float(ious[r, p])))
            libres.remove(r)
    acertadas = {p for _, p, _ in coincidencias}
    sobrantes = set(range(len(propuestas))) - acertadas
    candidatos = sorted(((float(ious[r, p]), r, p) for r in libres for p in sobrantes
                         if iou_cercana <= ious[r, p] < iou_min), key=lambda c: (-c[0], c[1], c[2]))
    cercanas, usadas_r, usadas_p = [], set(), set()
    for valor, r, p in candidatos:
        if r not in usadas_r and p not in usadas_p:
            cercanas.append((r, p, valor))
            usadas_r.add(r)
            usadas_p.add(p)
    return dict(anotadas=reales, predichas=seleccion, coincidencias=coincidencias, cercanas=cercanas,
                omitidas=sorted(libres - usadas_r), falsas=sorted(sobrantes - usadas_p),
                tp=len(coincidencias), fp=len(sobrantes), fn=len(libres),
                confianza=confianza, iou_min=iou_min, iou_cercana=iou_cercana)


def cargar_revision(rutas, tabla, *, incluir_detectores_anteriores=False):
    """Lee el formato común y cajas_val.csv; distingue ausencia de cajas de archivo faltante.

    Solo admite CNN comun_v2. Por defecto los detectores deben tener imgsz=640.
    Un detector de otra resolución puede inspeccionarse explícitamente, sin usarlo
    para concluir diferencias de arquitectura. Todo fallo queda en el inventario.
    """
    corridas = resultados.cargar_corridas(rutas)
    predicciones = resultados.cargar_predicciones(rutas)
    val = tabla.loc[tabla.split == "val"].set_index("archivo")
    filas, admitidas, cajas = [], [], {}
    for _, corrida in corridas.iterrows():
        nombre = f"{corrida['modelo']}/{corrida['modo']}"
        familia = comparacion.familia(corrida["modelo"])
        receta = corrida.get("params.receta")
        es_detector = familia == "detector"
        lado = receta.get("imgsz") if isinstance(receta, dict) else None
        descripcion = f"{lado} px" if es_detector else receta
        motivo = ""
        if not corrida["completa"]:
            motivo = "Corrida incompleta"
        elif familia == "referencia":
            motivo = "Referencia trivial: sin cajas que revisar"
        elif not es_detector and receta != comparacion.RECETA_VIGENTE:
            motivo = "CNN con receta anterior"
        elif es_detector and lado != 640 and not incluir_detectores_anteriores:
            motivo = "Detector con resolución anterior o sin registrar"
        seleccion = predicciones.loc[(predicciones.modelo == corrida["modelo"])
                                     & (predicciones.modo == corrida["modo"])].copy()
        if not motivo:
            try:
                _validar_predicciones(seleccion, val, float(corrida["umbral"]))
                if es_detector:
                    archivo = rutas.resultados_equipo / nombre / "cajas_val.csv"
                    if not archivo.is_file():
                        raise ValueError("Falta cajas_val.csv; no equivale a cero detecciones")
                    propuestas = pd.read_csv(archivo)
                    _validar_exportacion(propuestas, seleccion)
                    cajas[nombre] = propuestas
            except (ValueError, OSError, KeyError) as error:
                motivo = str(error)
        if not motivo:
            admitidas.append(seleccion.assign(corrida=nombre))
        filas.append(dict(corrida=nombre, familia=familia, receta=descripcion, fecha=corrida.get("fecha"),
                          disponible=not motivo, motivo=motivo, umbral=corrida.get("umbral"),
                          confianza_minima=receta.get("conf", 0.001) if es_detector and isinstance(receta, dict) else 0))
    columnas = ["corrida", "familia", "receta", "fecha", "disponible", "motivo", "umbral", "confianza_minima"]
    inventario = pd.DataFrame(filas, columns=columnas)
    pred = pd.concat(admitidas, ignore_index=True) if admitidas else pd.DataFrame(
        columns=[*resultados.COLUMNAS_PREDICCIONES, "modelo", "modo", "corrida"])
    return inventario, pred, cajas


def _validar_predicciones(pred, val, umbral):
    if (pred.archivo.duplicated().any() or set(pred.archivo) != set(val.index)
            or len(pred) != len(val)):
        raise ValueError("Las predicciones no cubren exactamente las mismas imágenes de val")
    alineadas = pred.set_index("archivo").loc[val.index]
    valores = alineadas[["y_real", "probabilidad", "prediccion", "acierto"]].to_numpy(float)
    if (not np.isfinite(valores).all() or not 0 <= umbral <= 1
            or not np.isin(valores[:, [0, 2, 3]], [0, 1]).all()
            or ((valores[:, 1] < 0) | (valores[:, 1] > 1)).any()):
        raise ValueError("Predicciones binarias o probabilidades inválidas")
    if not np.array_equal(valores[:, 0], val.y_original.to_numpy(int)):
        raise ValueError("La etiqueta guardada no coincide con la referencia actual")
    if (not np.array_equal(valores[:, 2], (valores[:, 1] >= umbral).astype(int))
            or not np.array_equal(valores[:, 3], (valores[:, 0] == valores[:, 2]).astype(int))):
        raise ValueError("Decisión o acierto incompatibles con el umbral guardado")


def _validar_exportacion(propuestas, pred):
    if set(COLUMNAS_CAJAS) - set(propuestas.columns):
        raise ValueError("cajas_val.csv no tiene las columnas del formato común")
    if not set(propuestas.archivo) <= set(pred.archivo):
        raise ValueError("cajas_val.csv contiene imágenes ajenas a val")
    valores = propuestas[["clase", "confianza", *COORDENADAS]].to_numpy(float)
    if not np.isfinite(valores).all() or not (valores[:, 0] == 0).all():
        raise ValueError("Las cajas deben tener clase única panel (0) y valores finitos")
    if ((valores[:, 1] < 0) | (valores[:, 1] > 1)).any():
        raise ValueError("Confianzas inválidas en cajas_val.csv")
    _validar_cajas(valores[:, 2:])
    maximos = propuestas.groupby("archivo").confianza.max().reindex(pred.archivo, fill_value=0)
    if not np.allclose(maximos.to_numpy(float), pred.probabilidad.to_numpy(float), atol=1e-6, rtol=0):
        raise ValueError("Las cajas y el puntaje por imagen no corresponden a la misma exportación")


def revisar_imagenes(tabla, anotaciones, cajas_por_corrida, predicciones, *,
                     confianza=0.25, iou_min=0.5, iou_cercana=0.1):
    """Devuelve el índice de casos por imagen y los diagnósticos que usa el visor."""
    filas, diagnosticos = [], {}
    for fila in tabla.loc[tabla.split == "val"].itertuples(index=False):
        reales = anotaciones.loc[(anotaciones.archivo == fila.archivo) & anotaciones.incluida, COORDENADAS]
        por_detector = {}
        for corrida, propuestas in cajas_por_corrida.items():
            por_detector[corrida] = diagnosticar_cajas(reales.to_numpy(float),
                propuestas.loc[propuestas.archivo == fila.archivo], confianza=confianza,
                iou_min=iou_min, iou_cercana=iou_cercana)
        diagnosticos[fila.archivo] = por_detector
        pred = predicciones.loc[predicciones.archivo == fila.archivo]
        filas.append(dict(archivo=fila.archivo, fuente=fila.fuente, jamaica=split.es_jamaica(fila.archivo),
            pendiente_revision=bool(fila.pendiente_revision), y_real=int(fila.y_original),
            omitidas=sum(len(d["omitidas"]) for d in por_detector.values()),
            falsas=sum(len(d["falsas"]) for d in por_detector.values()),
            desajustadas=sum(len(d["cercanas"]) for d in por_detector.values()),
            errores_cajas=sum(d["fp"] + d["fn"] for d in por_detector.values()),
            errores_clasificacion=int((pred.prediccion != pred.y_real).sum())))
    return pd.DataFrame(filas), diagnosticos


def zonas_error(diagnosticos):
    """Zonas normalizadas para ampliar exactamente el mismo recorte en todos los modelos."""
    zonas = []
    for corrida, diagnostico in diagnosticos.items():
        reales = _esquinas(diagnostico["anotadas"])
        propuestas = _esquinas(diagnostico["predichas"][COORDENADAS].to_numpy(float))
        for r, p, iou in diagnostico["cercanas"]:
            limite = np.r_[np.minimum(reales[r, :2], propuestas[p, :2]),
                           np.maximum(reales[r, 2:], propuestas[p, 2:])]
            zonas.append((f"{corrida}: P{p + 1}/R{r + 1} cercana, IoU {iou:.2f}", limite))
        for tipo, indices, limites in (("anotación omitida R", diagnostico["omitidas"], reales),
                                        ("detección sin referencia P", diagnostico["falsas"], propuestas)):
            zonas.extend((f"{corrida}: {tipo}{i + 1}", limites[i]) for i in indices)
    return zonas


def limites_zoom(limites, ancho, alto, margen=12, minimo=48):
    """Recorte en píxeles, con contexto y sin ampliar más allá de la imagen original."""
    x0, y0, x1, y1 = np.asarray(limites, dtype=float) * [ancho, alto, ancho, alto]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lado_x, lado_y = max(minimo, x1 - x0 + 2 * margen), max(minimo, y1 - y0 + 2 * margen)
    return max(0, cx - lado_x / 2), max(0, cy - lado_y / 2), min(ancho, cx + lado_x / 2), min(alto, cy + lado_y / 2)


def mostrar_imagen(rutas, archivo, anotadas, diagnosticos, *, zoom=None, mostrar_aciertos=False, titulo=""):
    """Referencia y cada detector lado a lado: imagen completa y, si se elige, un zoom común.

    Las coordenadas se escalan al tamaño de la imagen leída, nunca al imgsz de YOLO.
    Devuelve una figura en memoria, sin escribir en el dataset ni en Drive.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    imagen = datos.leer_imagen(rutas.imagenes / archivo)
    alto, ancho = imagen.shape[:2]
    reales = _validar_cajas(anotadas)
    columnas = len(diagnosticos) + 1
    filas = 2 if zoom is not None else 1
    figura, ejes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4.8 * filas), squeeze=False)

    def caja(ax, coordenadas, color, etiqueta, estilo="-", grosor=1.6, interior=False):
        xc, yc, w, h = coordenadas * [ancho, alto, ancho, alto]
        x, y = xc - w / 2, yc - h / 2
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                               linewidth=grosor, linestyle=estilo))
        if not etiqueta:
            return
        derecha = xc > ancho / 2
        texto = ax.text(min(ancho, x + w) if derecha else max(0, x), max(1, y), etiqueta,
                fontsize=7, color="white", va="top" if y < 15 or interior else "bottom",
                ha="right" if derecha else "left", clip_on=True,
                bbox=dict(facecolor="black", edgecolor=color, alpha=0.85, pad=1))
        # Las etiquetas no deben achicar los paneles para entrar en tight_layout.
        texto.set_in_layout(False)

    for fila in range(filas):
        for columna, ax in enumerate(ejes[fila]):
            ax.imshow(imagen, interpolation="nearest")
            ax.set_xlim(0, ancho)
            ax.set_ylim(alto, 0)
            ax.set_axis_off()
            if columna == 0:
                for r, c in enumerate(reales):
                    caja(ax, c, COLORES["referencia"], f"R{r + 1}", "--")
                nombre = f"Anotación de referencia\n{len(reales)} cajas · {ancho}×{alto} px"
            else:
                corrida, d = list(diagnosticos.items())[columna - 1]
                for r, c in enumerate(reales):
                    # Siempre mostrar todas las referencias, aun ocultando los aciertos.
                    ax.add_patch(Rectangle(((c[0] - c[2]/2)*ancho, (c[1] - c[3]/2)*alto),
                        c[2]*ancho, c[3]*alto, fill=False, edgecolor=COLORES["referencia"],
                        linestyle="--", linewidth=0.7, alpha=0.6))
                for r in d["omitidas"]:
                    caja(ax, reales[r], COLORES["omitida"], f"R{r + 1} omitida", "--", 2, interior=True)
                estados = {p: ("correcta", f"R{r + 1} IoU={iou:.2f}") for r, p, iou in d["coincidencias"]}
                for r, p, iou in d["cercanas"]:
                    estados[p] = ("desajustada", f"R{r + 1} IoU={iou:.2f}")
                    caja(ax, reales[r], COLORES["desajustada"], "", "--")
                    c = d["predichas"].iloc[p]
                    ax.plot([reales[r, 0]*ancho, c.xc*ancho], [reales[r, 1]*alto, c.yc*alto],
                            color=COLORES["desajustada"], linewidth=1, linestyle=":")
                estados.update({p: ("falsa", "sin coincidencia") for p in d["falsas"]})
                for p, propuesta in d["predichas"].iterrows():
                    estado, detalle = estados[p]
                    if estado == "correcta" and not mostrar_aciertos:
                        continue
                    caja(ax, propuesta[COORDENADAS].to_numpy(float), COLORES[estado],
                         f"P{p + 1} {propuesta.confianza:.2f}\n{detalle}")
                nombre = (corrida.replace("/", "\n") + f"\nTP {d['tp']} · FP {d['fp']} · FN {d['fn']}"
                          + f" | conf≥{d['confianza']:.2f}, IoU≥{d['iou_min']:.2f}")
                if not mostrar_aciertos:
                    nombre += "\nCajas correctas ocultas"
            if fila:
                x0, y0, x1, y1 = limites_zoom(zoom, ancho, alto)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
                nombre = "Misma zona ampliada · píxeles originales"
            ax.set_title(nombre, fontsize=10)
    leyenda = [Line2D([], [], color=COLORES[tipo], linestyle="--" if tipo in ("referencia", "omitida") else "-",
                      label=nombre) for tipo, nombre in (("referencia", "Referencia"), ("correcta", "Correcta"),
                        ("falsa", "Sin coincidencia / duplicada"), ("desajustada", "Cercana, IoU insuficiente"),
                        ("omitida", "Anotación sin caja cercana"))]
    figura.suptitle(titulo + "\n" + textwrap.fill(archivo, 100), fontsize=11)
    figura.legend(handles=leyenda, loc="lower center", ncol=min(3, len(leyenda)), fontsize=9)
    figura.tight_layout(rect=(0, 0.09, 1, 0.94), h_pad=3.2, w_pad=1.2)
    return figura
