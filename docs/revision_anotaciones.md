# Revisión de anotaciones dudosas

Revisión visual realizada por Codex el 25/10/2026. El registro por caso está en
[revision_anotaciones.csv](revision_anotaciones.csv); las decisiones están en
[criterios_datos.md](criterios_datos.md). No se alteraron datos ni splits.

## Fuentes comprobadas

La consigna local no incluye un diccionario de clases. La release
[`dataset-v1`](https://github.com/candeperles/PDI-TP/releases/tag/dataset-v1) tampoco.
El nombre `solarpanels_hd` permitió localizar las
[etiquetas originales de Clark, Figshare v3](https://figshare.com/articles/dataset/Solar_Panel_Object_Labels/22081091)
y el [artículo de Clark y Pacifici (2023)](https://www.nature.com/articles/s41597-023-02539-8).
Este último explica la confianza asociada a los IDs; no son tecnologías diferentes.

Se descargó `labels.zip` v3 y se comprobó su SHA-256:
`9a24d5657d7d0e27d5b083f7749067fefa376f03189293ff8c05b2b3c9916929`.
Las 17 cajas locales de IDs 1/2 coinciden con cajas de esa fuente en ID y geometría,
con tolerancia absoluta de 3e-6 normalizado, al intersectar con un recorte de una
grilla 4×4 y renormalizar. El cotejo encontró columna = segundo sufijo y fila = primero.
Esto comprueba esos casos; no demuestra la procedencia de todas las otras fuentes.

Evidencia versionada: [cotejo_clases.csv](cotejo_clases.csv).
Reproducción opcional con acceso a Internet: `python scripts/check_annotation_source.py`.
Los archivos descargados quedan fuera de Git en `resultados/revision_anotaciones/`.

## Casos revisados

| Grupo | Cobertura | Observación |
|---|---|---|
| R01–R17 | Todas las 17 cajas de IDs 1/2 en 15 imágenes | Varias estructuras son ambiguas a esta resolución. La apariencia no permite reclasificarlas como termotanques, lucernarios u otra tecnología con certeza. Se preserva la incertidumbre original. |
| S01–S37 | Todas las 37 cajas positivas menores a 3 px en 32 imágenes | Son fragmentos en el borde; la distancia al borde es ≤0.000128 px, dentro del redondeo. Varias contienen muy poca información independiente. |
| Z01 | Caja inválida de entrenamiento | Ancho cero; la imagen tiene siete cajas válidas adicionales. |
| N01–N18 | 18 de las 139 imágenes con etiqueta vacía | Tres por combinación fuente/split disponible, semilla 42. PV01 no tiene negativos. |

En las 16 imágenes negativas restantes de la muestra no identifiqué evidencia
visual clara de paneles. Esto no confirma que estén libres de omisiones.
En dos casos marqué una duda, sin reclasificarlos:

| Caso | Imagen | Motivo |
|---|---|---|
| N06 · geo/val | `2165022_unknown_18.160237711023147_-77.49053387589882_0_1.png` | Gran cubierta clara con patrón regular; no se distingue con certeza entre cubierta/lucernarios y módulos. |
| N16 · solarpanels_hd/val | `solarpanels_hd_1__x0_10114_y0_26273_dxdy_832_3_1.png` | Estructura rectangular clara sobre cubierta; identidad no resoluble en este recorte. |

La segunda revisión debe usar contexto o evidencia adicional y registrar cualquier
cambio de referencia. Esta muestra exploratoria no estima una tasa de falsos negativos
ni garantiza la corrección de las otras 121 imágenes con etiqueta vacía.

## Cómo mirar los casos juntos

Ejecutar `notebooks/03_revision_anotaciones.ipynb`. Genera imágenes completas y zooms
con IDs vinculados al registro CSV, para comparar observaciones y añadir una segunda
revisión. Un zoom no agrega resolución ni confirma por sí solo la identidad del objeto.
Los PNG y la copia ejecutada quedan en resultados, fuera de Git (en Drive desde Colab).

La revisión informa criterios globales antes de los modelos. No se eligieron umbrales
por resultados de validación ni se cambió el split después de evaluar un modelo.
