# Validación del dataset grupo_5

Auditoría realizada el 25/10/2026 sobre la copia descargada de `dataset-v1`, con
`notebooks/02_validacion_datos.ipynb` y `pdi_tp/validacion.py`.
No se modificaron imágenes, etiquetas ni archivos del split.

## Resultado

**Hay una caja con ancho cero que necesita revisión antes de entrenar.**
El resto de los controles estructurales pasó. También hay cajas diminutas para
revisión visual; no se filtraron automáticamente.

| Control | Resultado |
|---|---|
| Correspondencia imagen/etiqueta | 900 imágenes y 900 etiquetas; ninguna faltante ni huérfana. |
| Lectura y formato | Las 900 imágenes son PNG legibles. |
| Dimensiones | Todas son de 256 × 256 px. |
| Modo de color | 846 RGB y 54 en paleta P; el lector compartido convierte a RGB. |
| Formato YOLO y clases | Las 2669 filas tienen cinco valores numéricos finitos e IDs enteros en {0, 1, 2}. |
| Geometría | 2668 cajas válidas y una con ancho cero. No hay cajas fuera de los bordes con tolerancia de 1e-6 normalizado. |
| Cajas pequeñas positivas | 37 cajas en 32 imágenes tienen ancho o alto menor a 3 px. Son advertencias. |
| Duplicados exactos | Ningún grupo duplicado por bytes ni por píxeles RGB. |
| Cobertura del split | 738 imágenes en train y 162 en val; sin repeticiones, referencias inexistentes, imágenes sin asignar ni nombres compartidos. |
| Separación espacial | 187 grupos; ninguno compartido entre train y val según la heurística actual. |

## Anotación inválida

- Imagen: `solarpanels_hd_1__x0_13014_y0_24247_dxdy_832_1_2.png`.
- Etiqueta del mismo nombre base, **línea 1**.
- Split: **train**.
- Valores interpretados: clase 0, centro (0, 0.793269), ancho 0 y alto 0.413462.
- Tamaño en píxeles: **0 × 105.846272**. No define un rectángulo con área positiva.

La imagen tiene otras siete cajas válidas. La etiqueta completa queda marcada como
inválida y su estado binario como desconocido en el reporte conservador; no se
reclasifica como negativa. Por eso el resumen de validación cuenta 760 imágenes
con etiqueta positiva íntegramente válida, 139 negativas y una por revisar.
Esto no cambia las 761 imágenes con etiquetas no vacías informadas por la exploración.

Acción pendiente: revisar esta línea sobre la imagen original y acordar cómo
tratarla en una copia derivada del dataset. No se eliminó, corrigió ni excluyó
la anotación, la imagen o su asignación al split.

## Cajas pequeñas

| Fuente | Train | Val | Total |
|---|---:|---:|---:|
| PV01 | 19 | 10 | 29 |
| solarpanels_hd | 5 | 3 | 8 |
| Total | 24 | 13 | 37 |

La exploración contaba 38 cajas con ancho o alto menor a 3 px porque incluía la
caja de ancho cero. La validación separa **37 cajas de tamaño positivo** de
**una caja degenerada**. El umbral de 3 px es un criterio de revisión, no una
regla de limpieza ya acordada.

## Archivos generados

El notebook guarda en `rutas.resultados / "validacion_datos"`:

- `imagenes.csv`: dimensiones, modo, split, grupo, estado de la etiqueta y hashes de originales.
- `cajas.csv`: valores YOLO, línea original, tamaño en píxeles y validez de cada fila numérica finita.
- `incidencias.csv`: nivel, control, archivo, línea y detalle; una caja puede tener más de una incidencia.
- `duplicados.csv`: grupos de copias exactas y si cruzan train/val; queda con cabeceras si no hay casos.
- `controles.csv` y `resumen.json`: estado de cada familia de controles y cantidades.
- PNG de ejemplos de la anotación inválida, cajas pequeñas y clases minoritarias.

El verificador local usa `resultados/verificacion/validacion_datos/` y guarda
la copia ejecutada del notebook en `resultados/verificacion/`. Estos resultados
no se suben a Git; cada integrante puede reproducirlos. En Colab se guardan en Drive.

## Límites y pendientes

- Los IDs 0, 1 y 2 se aceptan porque son los que usa el repositorio. Su significado
  no está confirmado; las 17 cajas de clases 1 y 2 siguen pendientes de revisión semántica.
- Los hashes detectan copias exactas; no descartan imágenes casi iguales,
  transformadas o recortes solapados.
- La separación espacial usa las reglas basadas en nombres de `pdi_tp/split.py`.
  Que los grupos no se crucen no demuestra independencia geográfica absoluta.
- Los controles numéricos no prueban que las cajas estén completas ni que cada
  anotación corresponda visualmente a un panel. Hace falta revisión visual del equipo.
- Antes de entrenar: resolver la anotación degenerada y acordar si se conservan
  todas las cajas pequeñas. Mantener los originales y documentar cualquier derivación.

## Verificación del código

Seis pruebas sintéticas cubren datos sanos y etiquetas vacías, etiquetas faltantes,
filas malformadas/no numéricas/no finitas, IDs inválidos, geometría y tolerancia,
duplicados visuales con distintas anotaciones entre splits, cobertura/repeticiones
y separación espacial, imágenes corruptas, dimensiones incorrectas y etiquetas huérfanas.

Comando: `python -m unittest discover -s tests -v`.
