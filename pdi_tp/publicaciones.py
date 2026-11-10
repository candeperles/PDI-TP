"""Copias de consulta con salidas: protocolo acordado para la entrega del #54."""
import copy
import base64
import hashlib
import json
import re
from datetime import datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path

LIMITE_BYTES = 5 * 1024 * 1024
VERSION = "ejecutado_v1"
NUMEROS = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "15", "18"}


def texto(valor):
    return valor if isinstance(valor, str) else "".join(valor)


def huella(contenido):
    return hashlib.sha256(contenido).hexdigest()


def huella_original(ruta):
    """SHA estable entre Windows y CI: UTF-8 con saltos de línea LF."""
    return huella(Path(ruta).read_text(encoding="utf-8").encode("utf-8"))


class _TablaSegura(HTMLParser):
    """Conserva texto y tablas; excluye scripts, estilos, enlaces e imágenes."""
    permitidas = {"table", "thead", "tbody", "tr", "th", "td", "div", "span", "p", "br",
                  "pre", "b", "strong", "i", "em", "h1", "h2", "h3"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.partes, self.oculto = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.oculto += 1
        elif not self.oculto and tag in self.permitidas:
            self.partes.append(f"<{tag}>")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.oculto = max(0, self.oculto - 1)
        elif not self.oculto and tag in self.permitidas and tag != "br":
            self.partes.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.oculto:
            self.partes.append(escape(data))


def limpiar_html(valor):
    parser = _TablaSegura()
    parser.feed(texto(valor))
    return "".join(parser.partes)


def _origen(raiz, relativa):
    ruta = (raiz / relativa).resolve()
    if not ruta.is_relative_to((raiz / "notebooks").resolve()) or ruta.parent != raiz / "notebooks":
        raise ValueError("El origen debe estar directamente en notebooks/.")
    if ruta.stem[:2] not in NUMEROS or ruta.suffix != ".ipynb":
        raise ValueError("Notebook fuera del alcance acordado en #54.")
    return ruta


def _fuente(celda, familia, origen):
    fuente = texto(celda.get("source", []))
    if origen.name == "18_evaluacion_test.ipynb" and familia == "clasificadores":
        fuente = fuente.replace('FAMILIA = "clasificadores"', 'FAMILIA = "yolo26n"')
    return fuente


def _comparar(nb, original, familia, origen):
    if familia not in (None, "clasificadores", "yolo26n"):
        raise ValueError("Familia no autorizada.")
    if familia and origen.name != "18_evaluacion_test.ipynb":
        raise ValueError("Solo el test admite seleccionar FAMILIA.")
    if origen.name == "18_evaluacion_test.ipynb" and familia:
        codigo = "\n".join(texto(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
        if re.findall(r'^FAMILIA = "([^"]+)"', codigo, re.M) != [familia]:
            raise ValueError("La familia declarada no coincide con la ejecutada.")
    if len(nb["cells"]) != len(original["cells"]):
        raise ValueError("Cantidad de celdas distinta del original.")
    for i, (c, o) in enumerate(zip(nb["cells"], original["cells"])):
        if c["cell_type"] != o["cell_type"] or _fuente(c, familia, origen) != texto(o["source"]):
            raise ValueError(f"Celda {i}: fuente distinta del original.")


def cabecera(meta):
    entorno = json.dumps(meta["entorno"], ensure_ascii=False, sort_keys=True)
    return ("# Copia de consulta — no editar ni usar para entrenar\n\n"
            f"Original: `{meta['origen']}` · SHA-256 (UTF-8, saltos LF): `{meta['sha256_origen']}`.\n\n"
            f"Tipo: **{meta['tipo']}**. Fecha de ejecución/consulta: {meta['fecha_ejecucion'] or 'no registrada'}. "
            f"Commit de ejecución: `{meta['commit_ejecucion'] or 'no registrado'}`.\n\n"
            f"Entorno de ejecución: `{entorno}`. Publicación: {meta['fecha_publicacion']}. "
            f"Commit de referencia del código: `{meta['commit_referencia']}`.\n\n"
            f"Preparación del insumo: {meta.get('preparacion', 'sin celdas omitidas')}.\n\n"
            f"Celdas ejecutadas (índices del original): {meta['celdas_ejecutadas']}. "
            f"Familia del test: {meta.get('familia') or 'no aplica'}.\n\n"
            "Una consulta de corridas guardadas muestra resultados reales ya existentes; "
            "no reproduce los logs del entrenamiento. Val eligió checkpoints y se usa para reportar. "
            "Se quitaron widgets y formatos activos; en test se quitaron todas las imágenes.\n")


def preparar(nb, *, raiz, origen, commit, tipo="ejecucion_completa", familia=None):
    raiz = Path(raiz).resolve()
    ruta = _origen(raiz, origen)
    original = json.loads(ruta.read_text(encoding="utf-8"))
    _comparar(nb, original, familia, ruta)
    copia = copy.deepcopy(nb)
    ejecucion = nb.get("metadata", {}).get("ejecucion_entrega", {})
    indices = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and c.get("execution_count") is not None]
    meta = dict(version=VERSION, origen=ruta.relative_to(raiz).as_posix(),
                sha256_origen=huella_original(ruta), commit_referencia=commit,
                fecha_publicacion=datetime.now().astimezone().isoformat(timespec="seconds"),
                fecha_ejecucion=ejecucion.get("fecha"), commit_ejecucion=ejecucion.get("commit"),
                entorno=ejecucion.get("entorno", {"estado": "no registrado en la sesión original"}),
                tipo=tipo, familia=familia, celdas_ejecutadas=indices,
                preparacion=ejecucion.get("preparacion", "sin celdas omitidas"))
    copia["metadata"] = {k: v for k, v in copia.get("metadata", {}).items() if k in ("kernelspec", "language_info")}
    copia["metadata"]["publicacion"] = meta
    es_test = ruta.stem.startswith("18_")
    for celda in copia["cells"]:
        celda["metadata"] = {}
        celda.pop("attachments", None)
        if celda["cell_type"] != "code":
            continue
        nuevas = []
        for salida in celda.get("outputs", []):
            clase = salida.get("output_type")
            if clase == "error":
                raise ValueError("No publicar ejecuciones con errores.")
            if clase == "stream":
                nuevas.append({"output_type": clase, "name": salida["name"], "text": salida["text"]})
            elif clase in {"display_data", "execute_result"}:
                data = {}
                for mime, valor in salida.get("data", {}).items():
                    if mime == "text/html":
                        data[mime] = limpiar_html(valor)
                    elif mime == "text/plain" or (not es_test and mime in {"image/png", "image/jpeg"}):
                        data[mime] = valor
                if data:
                    item = dict(output_type=clase, data=data, metadata={})
                    if clase == "execute_result":
                        item["execution_count"] = salida.get("execution_count")
                    nuevas.append(item)
        celda["outputs"] = nuevas
    copia["cells"].insert(0, dict(cell_type="markdown", metadata={}, source=cabecera(meta).splitlines(True)))
    problemas = validar(copia, raiz=raiz)
    if problemas:
        raise ValueError("; ".join(problemas))
    return copia


def serializar(nb):
    contenido = (json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if len(contenido) > LIMITE_BYTES:
        raise ValueError("La publicación supera 5 MiB; reducir las figuras en el original y volver a ejecutar.")
    return contenido


def validar(nb, *, raiz):
    """Usado por el publicador y por CI; no modifica ni ejecuta código."""
    try:
        raiz = Path(raiz).resolve()
        meta = nb["metadata"]["publicacion"]
        if meta["version"] != VERSION or nb.get("nbformat") != 4:
            raise ValueError("Formato de publicación inválido.")
        ruta = _origen(raiz, meta["origen"])
        if huella_original(ruta) != meta["sha256_origen"]:
            raise ValueError("Cambió el original; volver a generar la copia.")
        if not re.fullmatch(r"[a-f0-9]{40}", meta["commit_referencia"]):
            raise ValueError("Falta un commit de referencia válido.")
        if nb["cells"][0]["cell_type"] != "markdown" or texto(nb["cells"][0]["source"]) != cabecera(meta):
            raise ValueError("Cabecera de procedencia ausente o alterada.")
        cuerpo = dict(nb, cells=nb["cells"][1:])
        original = json.loads(ruta.read_text(encoding="utf-8"))
        _comparar(cuerpo, original, meta.get("familia"), ruta)
        codigos = {i for i, c in enumerate(cuerpo["cells"]) if c["cell_type"] == "code"}
        ejecutadas = {i for i in codigos if cuerpo["cells"][i].get("execution_count") is not None}
        if sorted(ejecutadas) != meta["celdas_ejecutadas"]:
            raise ValueError("La lista de celdas ejecutadas no coincide.")
        if meta["tipo"] == "ejecucion_completa":
            esperadas = codigos
        elif meta["tipo"] == "consulta_guardada" and ruta.stem[:2] in {"05", "06", "07", "08", "09", "10"}:
            esperadas = {i for i in codigos if original["cells"][i].get("metadata", {}).get("pdi_tp", {}).get("consulta_entrega") is True}
        else:
            raise ValueError("Tipo de ejecución no autorizado.")
        if not esperadas or ejecutadas != esperadas:
            raise ValueError("Ejecución incompleta o celdas fuera de la consulta declarada.")
        if set(nb["metadata"]) - {"publicacion", "kernelspec", "language_info"}:
            raise ValueError("No se publica estado de widgets.")
        for i, c in enumerate(cuerpo["cells"]):
            if c.get("metadata"):
                raise ValueError("Metadata de celda no autorizada.")
            if c.get("attachments"):
                raise ValueError("No se publican adjuntos.")
            for s in c.get("outputs", []):
                if s.get("metadata"):
                    raise ValueError("Metadata de salida no autorizada.")
                if i not in ejecutadas or s.get("output_type") not in {"stream", "display_data", "execute_result"}:
                    raise ValueError("Salida sin ejecución o con error/formato no autorizado.")
                for mime, valor in s.get("data", {}).items():
                    permitidos = {"text/plain", "text/html"} | (set() if ruta.stem.startswith("18_") else {"image/png", "image/jpeg"})
                    if mime not in permitidos or (mime == "text/html" and texto(valor) != limpiar_html(valor)):
                        raise ValueError("Salida activa o imagen de test no autorizada.")
                    if mime.startswith("image/"):
                        binario = base64.b64decode(texto(valor), validate=True)
                        if not binario.startswith(b"\x89PNG\r\n\x1a\n" if mime == "image/png" else b"\xff\xd8\xff"):
                            raise ValueError("Imagen sin cabecera PNG/JPEG válida.")
        serializar(nb)
        return []
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [str(error)]
