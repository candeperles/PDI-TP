"""Ejecuta la exploración con un kernel limpio del Python que invoca este script.

Uso desde la raíz: python scripts/check_environment.py
La copia ejecutada y las versiones quedan en resultados/verificacion/.
No modifica el notebook original ni regenera el split.
"""
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent


def verificar() -> None:
    salida = REPO_ROOT / "resultados" / "verificacion"
    salida.mkdir(parents=True, exist_ok=True)
    os.environ["PDI_INTEGRANTE"] = "verificacion"
    os.environ["PYTHONUTF8"] = "1"
    os.environ.setdefault("IPYTHONDIR", str(salida / "ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(salida / "runtime"))

    notebook = nbformat.read(REPO_ROOT / "notebooks" / "01_exploracion.ipynb", as_version=4)
    for celda in notebook.cells:
        if celda.cell_type == "code":
            celda.outputs = []
            celda.execution_count = None

    kernel = KernelManager(kernel_name="python3")
    # Usa este entorno virtual, incluso si hay otros kernels instalados en la PC.
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    cliente = NotebookClient(
        notebook, km=kernel, timeout=180,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    print("Ejecutando la exploración completa en un kernel limpio...", flush=True)
    try:
        cliente.execute()
    finally:
        # También conserva el error si alguna celda falla; no anuncia éxito en ese caso.
        try:
            nbformat.write(notebook, salida / "01_exploracion.ejecutado.ipynb")
        finally:
            if kernel.has_kernel:
                kernel.shutdown_kernel(now=True)
    paquetes = ["numpy", "pandas", "Pillow", "matplotlib", "scikit-learn",
                "ipykernel", "notebook", "nbclient", "nbformat"]
    registro = {
        "python": sys.version,
        "plataforma": platform.platform(),
        "ejecutable": sys.executable,
        "paquetes": {p: version(p) for p in paquetes},
        "celdas_ejecutadas": sum(c.cell_type == "code" for c in notebook.cells),
    }
    (salida / "entorno.json").write_text(
        json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    for celda in notebook.cells:
        for resultado in celda.get("outputs", []):
            if resultado.output_type == "stream" and resultado.name == "stdout":
                print(resultado.text, end="")
    print(f"\nVerificación completada. Evidencia en {salida}")


if __name__ == "__main__":
    verificar()
