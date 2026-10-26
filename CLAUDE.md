# CLAUDE.md

Las normas del repo están en `AGENTS.md` y valen igual para Claude Code:

@AGENTS.md

## Específico de Claude Code

- Hablar con el equipo en castellano rioplatense.
- Para editar un `.ipynb`, cambiar solo las celdas necesarias y conservar el formato del archivo (sin
  salidas, mismos ids de celda); después correr `python scripts/verificar_notebooks.py`.
- No ejecutar entrenamientos largos en local sin pedirlo: el trabajo pesado corre en Colab.
- El flujo de rama + PR de `AGENTS.md` es automático: crear la rama, subirla y abrir el PR no necesita
  confirmación (es la forma de trabajo del repo). Confirmar con el usuario antes del **merge** y antes de
  borrar o renombrar un notebook.
