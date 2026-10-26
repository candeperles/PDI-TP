## Qué cambia

<!-- Una o dos líneas: qué y por qué. -->

## Chequeos

- [ ] `python scripts/verificar_notebooks.py` pasa (notebooks sin salidas, `INTEGRANTE = "nombre"`, guardado común)
- [ ] `python -m unittest discover -s tests` pasa
- [ ] Si entrena un modelo: guarda con `resultados.guardar_corrida` en `resultados/<modelo>/<modo>/`
- [ ] Si agrega un notebook: está en la tabla del README con su botón de Colab

## ¿Necesita revisión de otra persona antes del merge?

- [ ] Sí: toca `pdi_tp/`, `splits/`, `scripts/`, `.github/`, `.githooks/`, `docs/criterios_datos.md` o un notebook de otro integrante
- [ ] No: quien abre el PR lo une cuando el chequeo está en verde
