# Galería web de demostración

Vista del visor con cuatro ejemplos: caja desplazada, panel omitido, falsa detección y coincidencia.
Permite filtrar, recorrer imágenes, comparar lado a lado, superponer cajas y ampliar la misma zona.
**Las predicciones son simuladas: no son salidas ni métricas de ningún modelo.** Las anotaciones de
referencia corresponden a las cuatro imágenes identificadas en los ejemplos.

## Abrir en la computadora

1. Clonar o descargar el repo completo y bajar el dataset con `python scripts/download_data.py`
   desde la raíz del proyecto.
2. Abrir `docs/galeria/index.html` con Chrome, Edge o Firefox. También se puede usar **Ctrl + O** en el
   navegador y elegir ese archivo. No hace falta instalar JavaScript ni levantar un servidor.
3. Las imágenes se buscan en `data/grupo_5/images/`. Si están en otro lugar, usar **Elegir carpeta de
   imágenes** y seleccionar la carpeta `images`, o **Elegir archivos** y seleccionar los PNG originales.

Mantener `index.html`, `galeria.css` y `galeria.js` juntos. GitHub muestra el código HTML; para usar el
visor hay que abrir la copia local en el navegador. No se configura hosting ni GitHub Pages.

Las fotos se leen en el navegador y no se suben a ningún servicio. No están incrustadas en el HTML ni
versionadas en Git. Los archivos seleccionados se buscan por nombre; solo se cargan estos cuatro:

- `kingston_145_3_0_12.png`
- `kingston_145_3_0_14.png`
- `kingston2_1_33_0.png`
- `kingston_145_3_0_22.png`

Se esperan los originales de 256 × 256 px. El IoU mostrado es el solapamiento geométrico del ejemplo
simulado y su referencia, no un resultado de evaluación.

## Revisar errores reales

Usar [`13_galeria_errores.ipynb`](../../notebooks/13_galeria_errores.ipynb) en
[Colab](https://colab.research.google.com/github/candeperles/PDI-TP/blob/main/notebooks/13_galeria_errores.ipynb).
Ese notebook sigue disponible: lee las corridas guardadas en Drive en solo lectura, sin volver a
entrenar ni hacer inferencia. Val también se usa para elegir checkpoints y para reportar resultados.
