# AGENTS.md

Normas para trabajar en este repo. Valen para agentes de IA (Claude Code, Codex, Cursor, Gemini, etc.) y
para las personas del equipo. Si una tarea choca con alguna de estas normas, frenar y preguntar.

## Contexto

- TP de Procesamiento de Imágenes (grupo 5, 4 integrantes): clasificar imágenes aéreas en con/sin paneles
  solares y detectar los paneles, y comparar ambos enfoques. Consigna: `docs/Consigna_Paneles_Solares.pdf`;
  resumen: `docs/alcance.md`.
- Se entrena en **Google Colab con GPU T4**. Los resultados van a una carpeta compartida de Google Drive
  (`Mi unidad/PDI-TP/`); el repo tiene solo código, docs y el split.
- **Leer el cartel del README antes de trabajar en clasificación:** hoy los modelos no son comparables entre
  sí (recetas distintas, y EfficientNet hay que volver a correrlo tras corregir sus bugs).

## Reglas duras (las controla `scripts/verificar_notebooks.py`)

No desactivar, aflojar ni esquivar el verificador. Si algo legítimo no entra en las reglas, preguntar.

1. **Notebooks sin salidas** ni contadores de ejecución.
2. **`INTEGRANTE = "nombre"`** en el repo. Nunca subir un notebook con un nombre de persona.
3. **Drive se usa de una sola manera:** `entorno.preparar(INTEGRANTE)` (que hace `drive.mount`).
   Prohibido: la API de Drive (`googleapiclient`, scopes de `auth/drive`), montar Drive a mano,
   **borrar o mandar a la papelera archivos de Drive** y `rmtree` en notebooks.
4. **Todo modelo entrenado se guarda con el formato común** de `pdi_tp/resultados.py`:

   ```python
   carpeta = resultados.carpeta_corrida(rutas, "<modelo>", "<modo>", reiniciar=REPETIR)
   # ... entrenar y guardar los pesos DENTRO de `carpeta` (modelo.keras / modelo.pt / modelo.joblib) ...
   resultados.guardar_corrida(rutas, carpeta, archivos=archivos_val, probabilidad=prob, params=config)
   ```

   - Una carpeta por **modelo** y **modo** (`resultados/mobilenetv2/imagenet_congelado/`), nunca por persona.
     `<modelo>` y `<modo>`: minúsculas, números, `-`, `_` o `.`.
   - Contiene pesos, `metricas.json`, `predicciones_val.csv` (predicción contra el real de las 162 imágenes de
     val) y `config.json`. `guardar_corrida` calcula las métricas y frena si falta algo: no escribir esos
     archivos a mano ni calcular métricas por otro lado para registrarlas.
   - Para comparar: `resultados.cargar_corridas(rutas)` y `resultados.cargar_predicciones(rutas)`.
   - Modos ya completos se saltean (`resultados.corrida_completa`); `REPETIR = True` archiva la carpeta
     anterior (`<modo>__anterior_<fecha>`), no la borra.
5. **Cada notebook está en la tabla del README** con su botón de Colab
   (`https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/<ruta>`).

Única excepción documentada (`EXCEPCIONES` en el verificador): `06_ordenar_resultados_resnet50`, herramienta
vieja que ordena corridas de ResNet50 hechas con la API de Drive. No usarla como modelo para notebooks nuevos
ni agregar excepciones sin acuerdo del equipo.

## Datos y evaluación

- El dataset no está en git (`data/` ignorado; se baja con `scripts/download_data.py`). **No modificar los
  datos originales.** Las correcciones se documentan en `docs/criterios_datos.md` y se aplican en memoria.
- **El split es fijo** (`splits/train.txt` / `val.txt`, 738/162, agrupado por zona espacial). No
  regenerarlo ni usar otra división sin acuerdo explícito del equipo.
- Una sola clase de detección, `panel` (los IDs 0/1/2 de `solarpanels_hd` son confianza de anotación).
  N06/N16 están pendientes de revisión: ver `docs/criterios_datos.md`.
- Val se usa para elegir checkpoints y para reportar: decirlo siempre al mostrar resultados.
- **No inventar ni estimar métricas.** Si algo no se corrió, decirlo. No afirmar que un modelo es mejor sin
  resultados reales del formato común.
- Para que la comparación entre arquitecturas valga, cambiar solo la arquitectura y mantener la receta
  (tamaño, augmentation, pesos de clase, épocas, paciencia, LR). Todo hiperparámetro va en `params`.

## Código

- Código que usa más de un notebook → `pdi_tp/`, con tests en `tests/` (sin depender del dataset ni de
  TensorFlow, para que corran en CI).
- Antes de commitear: `python scripts/verificar_notebooks.py` y `python -m unittest discover -s tests`.
- Dependencias: `requirements.txt` (exploración, versiones fijas) y `requirements-training.txt`
  (TensorFlow). No mezclarlos: las versiones de NumPy del primero rompen TensorFlow.
- Todo en castellano: nombres de variables y funciones, comentarios, markdown y mensajes. Seguir el estilo
  del código que ya existe.
- Notebooks extra, fuera de la consigna, en `notebooks/extras/`, marcados como **EXTRA** en el título y en el
  README.
- No tocar notebooks de otro integrante (ver `git log`) sin que lo pidan; si hace falta, avisar qué cambia.

## Git: rama + pull request, siempre y sin que lo pidan

`main` está protegida: no acepta pushes directos y solo une PRs con el chequeo `verificar` en verde.
**Todo cambio que hagas en el repo sigue este flujo automáticamente**, aunque el usuario solo diga
"subilo", "commitealo" o "pushealo":

1. Partir de `main` actualizado y crear una rama: `git switch main && git pull` y
   `git switch -c <tipo>/<descripcion-corta>` (tipos: `notebook/`, `codigo/`, `datos/`, `docs/`, `fix/`).
   Si ya estás en una rama de trabajo propia para esta tarea, seguir en ella.
2. Hacer los cambios, correr `python scripts/verificar_notebooks.py` y `python -m unittest discover -s tests`,
   y commitear archivos puntuales (nunca `git add .`).
3. Subir la rama: `git push -u origin <rama>`.
4. Abrir el PR hacia `main`: `gh pr create --base main --fill`, completando la plantilla
   (`.github/pull_request_template.md`). Sin `gh`, dar al usuario el link que imprime `git push`.
5. Esperar el chequeo (`gh pr checks --watch`). Si falla, corregir en la misma rama y volver a subir.
6. **El merge no lo hace el agente por su cuenta:** con el chequeo en verde, preguntar al usuario.
   - Si el PR toca `pdi_tp/`, `splits/`, `scripts/`, `.github/`, `.githooks/`, `docs/criterios_datos.md` o un
     notebook de otro integrante, avisar que necesita **revisión de otra persona** antes del merge.
   - Si no, con el OK del usuario: `gh pr merge --squash --delete-branch`.

Además:

- No commitear `data/`, `resultados/`, `resultados_locales/` ni `notebooks/local/`.
- Mensajes de commit y títulos de PR en castellano, en infinitivo, diciendo qué cambia ("Agregar ...",
  "Corregir ...").
- Nunca push a `main`, nunca `--force` sobre `main` ni sobre ramas ajenas; en la rama propia, solo
  `--force-with-lease` después de un `git pull --rebase origin main`. No borrar ramas ajenas.
- Hook local: `git config core.hooksPath .githooks` (corre el verificador antes de cada commit).
  En GitHub corre `.github/workflows/verificar.yml` en cada push.

## Comandos útiles

```bash
python scripts/download_data.py            # bajar el dataset a data/grupo_5/
python scripts/verificar_notebooks.py      # normas de los notebooks
python -m unittest discover -s tests -v    # tests
gh pr create --base main --fill             # abrir el PR de la rama actual
gh pr checks --watch                       # esperar el chequeo verificar
```
