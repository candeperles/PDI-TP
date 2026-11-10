"""Verifica que los notebooks del repo cumplan las normas del equipo (ver AGENTS.md).

Corre en el hook pre-commit (.githooks/) y en GitHub Actions. Sale con código 1 si algo no cumple.
Uso: python scripts/verificar_notebooks.py [notebook ...]   (sin argumentos: todos los de notebooks/)

Normas:
1. Sin salidas ni contadores de ejecución (se suben limpios).
2. Sin `INTEGRANTE`: los resultados se guardan por modelo y modo, no se pide el nombre de nadie.
3. Un solo mecanismo de Drive y nada de borrados: prohibido montar Drive o usar la API por fuera de
   pdi_tp.entorno, y prohibidas las formas viejas de guardar resultados.
4. Todo notebook que entrena un modelo guarda con `resultados.guardar_corrida(...)`
   (carpeta resultados/<modelo>/<modo>/ con pesos, métricas y predicción contra el real).
   El notebook 16 tiene el protocolo adicional cv_jamaica_v1 acordado para el #43:
   pesos, particiones y predicciones fuera de fold en analisis/cv_jamaica, con integridad y tests.
   Los que no entrenan no montan Drive (`entorno.preparar(usar_drive=False)`): solo los modelos crean
   carpetas en la carpeta compartida. Excepción: los que solo leen resultados del equipo (comparación
   final) usan `entorno.preparar(solo_lectura=True)`, y entonces no pueden entrenar ni escribir archivos.
5. Cada notebook está enlazado en el README con su botón de Colab.

Las reglas se aplican a todos los notebooks. No agregar excepciones sin acuerdo del equipo.
"""
import ast
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from pdi_tp import publicaciones
COLAB = "https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/"
# Flujo adicional acordado para el #43: solo el notebook 16 puede usarlo.
FLUJOS_CV = {"notebooks/16_validacion_cruzada_jamaica.ipynb": "cv_jamaica_v1"}

PROHIBIDOS = {
    "INTEGRANTE": "ya no se pide el nombre de quien corre: borrar INTEGRANTE y llamar entorno.preparar()",
    "drive.mount(": "montar Drive a mano: usar entorno.preparar()",
    "googleapiclient": "API de Drive: usar entorno.preparar (drive.mount) y resultados.guardar_corrida",
    "googleapis.com/auth/drive": "permisos de la API de Drive",
    "RespaldoDrive": "guardado por API de Drive",
    "guardar_experimento(": "forma vieja de guardar: usar resultados.guardar_corrida",
    "cargar_experimentos(": "forma vieja de leer: usar resultados.cargar_corridas",
    "'trashed': True": "mandar archivos a la papelera desde código",
    '"trashed": True': "mandar archivos a la papelera desde código",
    "rmtree(": "borrar carpetas desde un notebook",
}
# Señales de que el notebook entrena un modelo. Importar Ultralytics no cuenta: cargar un YOLO guardado para
# predecir no es entrenar, y entrenarlo ya se detecta con `.train(`.
ENTRENA = ("keras.applications", "torchvision.models", ".fit(", ".train(", "torch.optim")
SOLO_LECTURA = "solo_lectura=True"
# Lo que un notebook de solo lectura no puede hacer: crear o modificar archivos.
ESCRITURA = ("guardar_corrida(", "carpeta_corrida(", ".mkdir(", ".to_csv(", ".to_json(", ".write_text(",
             ".write_bytes(", ".savefig(", ".save(", ".rename(", ".unlink(", "shutil.", "open(",
             "cv_jamaica.preparar_fold(", "cv_jamaica.guardar_fold(")


def verificar_cv(nb, todo, relativa):
    """Reconoce llamadas reales del protocolo; comentarios y strings no cuentan.

    Los controles de pesos, particiones, receta, commit, puntajes y lectura se
    ejecutan en pdi_tp.cv_jamaica (con tests independientes de TensorFlow).
    """
    flujo = nb.get("metadata", {}).get("pdi_tp", {}).get("flujo")
    if flujo is None:
        return False, []
    if FLUJOS_CV.get(relativa) != flujo:
        return False, ["flujo de CV no autorizado para este notebook"]
    try:
        arbol = ast.parse(todo)
    except SyntaxError:
        return False, ["el notebook de CV debe contener código Python válido"]
    llamadas = {ast.unparse(n.func) for n in ast.walk(arbol) if isinstance(n, ast.Call)}
    requeridas = {"split.folds_jamaica", "cv_jamaica.preparar_fold", "cv_jamaica.fold_completo",
                  "cv_jamaica.guardar_fold", "cv_jamaica.cargar_predicciones"}
    problemas = [f"CV sin llamada a {nombre}(...)" for nombre in sorted(requeridas - llamadas)]
    if {"resultados.guardar_corrida", "resultados.carpeta_corrida"} & llamadas:
        problemas.append("CV no puede escribir corridas comunes: usar solo cv_jamaica")
    preparaciones = [n for n in ast.walk(arbol) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "entorno.preparar"]
    if len(preparaciones) != 1 or not any(
            k.arg == "solo_lectura" and ast.unparse(k.value) == "not ENTRENAR"
            for llamada in preparaciones for k in llamada.keywords):
        problemas.append("CV debe usar entorno.preparar(solo_lectura=not ENTRENAR)")
    return not problemas, problemas


def verificar(ruta: Path, readme: str) -> list[str]:
    try:
        nb = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"no es un notebook JSON válido: {error}"]
    if nb.get("nbformat") != 4:
        return ["no es nbformat 4"]
    if ruta.resolve().is_relative_to(RAIZ / "ejecutados"):
        problemas = publicaciones.validar(nb, raiz=RAIZ)
        if ruta.stat().st_size > publicaciones.LIMITE_BYTES:
            problemas.append("la copia supera 5 MiB")
        if ruta.resolve().relative_to(RAIZ).as_posix() not in readme:
            problemas.append("la copia ejecutada no está enlazada en el README")
        return problemas
    problemas = []
    codigo = []
    for i, celda in enumerate(nb.get("cells", [])):
        if celda.get("cell_type") != "code":
            continue
        fuente = "".join(celda.get("source", []))
        codigo.append(fuente)
        if celda.get("outputs") or celda.get("execution_count") is not None:
            problemas.append(f"celda {i}: tiene salidas (Colab: Editar → Borrar todos los resultados)")
    todo = "\n".join(codigo)
    relativa = ruta.resolve().relative_to(RAIZ).as_posix()
    for patron, motivo in PROHIBIDOS.items():
        if patron in todo:
            problemas.append(f"usa `{patron}`: {motivo}")
    entrena = any(s in todo for s in ENTRENA)
    es_cv, problemas_cv = verificar_cv(nb, todo, relativa)
    problemas.extend(problemas_cv)
    if entrena and not es_cv and "resultados.guardar_corrida(" not in todo:
        problemas.append("entrena un modelo pero no guarda con resultados.guardar_corrida(...)")
    if SOLO_LECTURA in todo:
        if entrena:
            problemas.append("entorno.preparar(solo_lectura=True) es para notebooks que no entrenan")
        problemas += [f"usa `{e}` pero abrió los resultados en solo lectura" for e in ESCRITURA if e in todo]
    elif not entrena and "entorno.preparar(" in todo and "usar_drive=False" not in todo:
        problemas.append("no entrena pero monta Drive: usar entorno.preparar(usar_drive=False), o "
                         "entorno.preparar(solo_lectura=True) si solo lee resultados")
    if COLAB + relativa not in readme:
        problemas.append(f"no está enlazado en el README (falta el botón de Colab a {relativa})")
    return problemas


def main(argumentos: list[str]) -> int:
    rutas = [Path(a) for a in argumentos] or sorted(
        p for p in (RAIZ / "notebooks").rglob("*.ipynb")
        if "local" not in p.relative_to(RAIZ / "notebooks").parts and ".ipynb_checkpoints" not in p.parts)
    if not argumentos:
        rutas += sorted((RAIZ / "ejecutados").rglob("*.ipynb"))
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
        else:
            print(f"✓ {nombre}")
    if fallas:
        print(f"\n{fallas} notebook(s) no cumplen las normas del equipo (ver AGENTS.md).")
        return 1
    print("\nTodos los notebooks cumplen las normas.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
