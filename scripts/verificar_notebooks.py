"""Verifica que los notebooks del repo cumplan las normas del equipo (ver AGENTS.md).

Corre en el hook pre-commit (.githooks/) y en GitHub Actions. Sale con código 1 si algo no cumple.
Uso: python scripts/verificar_notebooks.py [notebook ...]   (sin argumentos: todos los de notebooks/)

Normas:
1. Sin salidas ni contadores de ejecución (se suben limpios).
2. `INTEGRANTE` sin nombres fijos: solo "nombre" (o leerlo de PDI_INTEGRANTE / input()).
3. Un solo mecanismo de Drive y nada de borrados: prohibido montar Drive o usar la API por fuera de
   pdi_tp.entorno, y prohibidas las formas viejas de guardar resultados.
4. Todo notebook que entrena un modelo guarda con `resultados.guardar_corrida(...)`
   (carpeta resultados/<modelo>/<modo>/ con pesos, métricas y predicción contra el real).
5. Cada notebook está enlazado en el README con su botón de Colab.

Excepciones: notebooks que no entrenan ni generan resultados y quedan por compatibilidad. Se listan en
EXCEPCIONES con el motivo; se les sigue exigiendo 1, 2 y 5. No agregar excepciones sin acuerdo del equipo.
"""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
COLAB = "https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/"

PROHIBIDOS = {
    "drive.mount(": "montar Drive a mano: usar entorno.preparar(INTEGRANTE)",
    "googleapiclient": "API de Drive: usar entorno.preparar (drive.mount) y resultados.guardar_corrida",
    "googleapis.com/auth/drive": "permisos de la API de Drive",
    "RespaldoDrive": "guardado por API de Drive",
    "guardar_experimento(": "forma vieja de guardar: usar resultados.guardar_corrida",
    "cargar_experimentos(": "forma vieja de leer: usar resultados.cargar_corridas",
    "'trashed': True": "mandar archivos a la papelera desde código",
    '"trashed": True': "mandar archivos a la papelera desde código",
    "rmtree(": "borrar carpetas desde un notebook",
}
# Herramientas viejas que se conservan pero NO son para entrenar: no se les aplica la regla 3.
EXCEPCIONES = {
    "notebooks/06_ordenar_resultados_resnet50.ipynb":
        "herramienta vieja: ordena en Drive corridas de ResNet50 hechas con la API (formato anterior); "
        "no entrena ni genera resultados",
}
# Señales de que el notebook entrena un modelo.
ENTRENA = ("keras.applications", "torchvision.models", "from ultralytics", ".fit(", ".train(", "torch.optim")
INTEGRANTE_OK = re.compile(r"""(["']nombre["']|PDI_INTEGRANTE|^input\()""")


def verificar(ruta: Path, readme: str) -> list[str]:
    try:
        nb = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"no es un notebook JSON válido: {error}"]
    if nb.get("nbformat") != 4:
        return ["no es nbformat 4"]
    problemas = []
    codigo = []
    for i, celda in enumerate(nb.get("cells", [])):
        if celda.get("cell_type") != "code":
            continue
        fuente = "".join(celda.get("source", []))
        codigo.append(fuente)
        if celda.get("outputs") or celda.get("execution_count") is not None:
            problemas.append(f"celda {i}: tiene salidas (Colab: Editar → Borrar todos los resultados)")
        for linea in fuente.splitlines():
            m = re.match(r"\s*INTEGRANTE\s*=\s*(.+)", linea)
            if m and not INTEGRANTE_OK.search(m.group(1).split("#")[0].strip()):
                problemas.append(f"celda {i}: INTEGRANTE con un nombre fijo ({m.group(1).strip()}); dejar \"nombre\"")
    todo = "\n".join(codigo)
    relativa = ruta.resolve().relative_to(RAIZ).as_posix()
    if relativa not in EXCEPCIONES:
        for patron, motivo in PROHIBIDOS.items():
            if patron in todo:
                problemas.append(f"usa `{patron}`: {motivo}")
        if any(s in todo for s in ENTRENA) and "resultados.guardar_corrida(" not in todo:
            problemas.append("entrena un modelo pero no guarda con resultados.guardar_corrida(...)")
    if COLAB + relativa not in readme:
        problemas.append(f"no está enlazado en el README (falta el botón de Colab a {relativa})")
    return problemas


def main(argumentos: list[str]) -> int:
    rutas = [Path(a) for a in argumentos] or sorted(
        p for p in (RAIZ / "notebooks").rglob("*.ipynb")
        if "local" not in p.relative_to(RAIZ / "notebooks").parts and ".ipynb_checkpoints" not in p.parts)
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    fallas = 0
    for ruta in rutas:
        problemas = verificar(ruta, readme)
        nombre = ruta.resolve().relative_to(RAIZ)
        if problemas:
            fallas += 1
            print(f"✗ {nombre}")
            for p in problemas:
                print(f"    - {p}")
        elif nombre.as_posix() in EXCEPCIONES:
            print(f"⚠ {nombre} (excepción: {EXCEPCIONES[nombre.as_posix()]})")
        else:
            print(f"✓ {nombre}")
    if fallas:
        print(f"\n{fallas} notebook(s) no cumplen las normas del equipo (ver AGENTS.md).")
        return 1
    print("\nTodos los notebooks cumplen las normas.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
