# Criterios de datos · v1

Definidos en la revisión del 25/10/2026, antes de entrenar. Son criterios de trabajo
de esta revisión; no implican confirmación visual independiente de cada objeto.
Los originales `dataset-v1` y el split **738/162 se conservan sin cambios**.

## Qué se etiqueta

El objetivo es detectar presencia y ubicación de paneles según las anotaciones
entregadas. Una caja puede abarcar un conjunto de módulos: no se subdivide en
paneles individuales ni se añaden cajas por apariencia sin evidencia suficiente.

En `solarpanels_hd`, los IDs son **confianza de la anotación**:

| ID original | Significado |
|---|---|
| 0 | Confianza alta |
| 1 | Confianza moderada |
| 2 | Confianza baja |

Lo define el [artículo original, sección Technical Validation](https://www.nature.com/articles/s41597-023-02539-8).
El [cotejo con Figshare](cotejo_clases.csv) confirmó ID y geometría de las 17 cajas
locales de clases 1/2. Esta interpretación de confianza no se extrapola al ID 0
de `PV01`, `geo` o `kingston`, cuya confianza de origen no está documentada aquí.

## Decisiones para la futura preparación

- **Una clase de detección: `panel`.** Mapear IDs 0/1/2 a 0 en una copia derivada,
  conservando `clase_original`, fuente y confianza disponible como metadatos.
  Mantener también las anotaciones de baja confianza, identificadas como tales;
  no convertirlas en negativos ni afirmar que están confirmadas visualmente.
- **Conservar las 37 cajas pequeñas positivas**, incluidos los recortes del borde.
  El umbral de 3 px solo sirve para revisión. Eliminarlas sin más dejaría cuatro
  imágenes positivas sin cajas. No completar las partes que quedaron fuera del recorte.
- **Excluir solo la caja de ancho cero Z01 del derivado.** Mantener su imagen y sus
  otras siete cajas. No inventar un ancho mínimo ni descartar la imagen completa.
- **Clasificación:** positiva si queda al menos una caja válida; negativa si la
  etiqueta original está vacía y no hay duda registrada. Faltantes, etiquetas ilegibles
  o casos ambiguos quedan pendientes, sin reclasificación automática.
- **Negativos N06 y N16:** conservar la anotación original, marcarlos para segunda
  revisión antes de fijar el conjunto de evaluación. No corregir solo val por intuición.
- **Split:** mantenerlo. Cualquier cambio debe justificarse en un registro, ser
  compartido por el equipo y aplicarse antes de comparar experimentos.

## Aplicación y pendientes

Los criterios se aplican ahora al manifiesto binario y los metadatos de cajas de
`04_clasificacion_ResNet50.ipynb`, sin modificar originales. No se generó un dataset
derivado de detección. La auditoría original sigue informando una anotación inválida.
Pendientes: segunda revisión de N06/N16 con mayor contexto y confirmar con la
cátedra si esperan conservar la incertidumbre de origen como etiquetas separadas.
La fusión definida aquí representa la tarea física de detectar paneles.

Evidencia, muestra y límites: [revisión de anotaciones](revision_anotaciones.md).
