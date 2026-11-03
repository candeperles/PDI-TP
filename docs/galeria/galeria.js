/* Demostración visual: las predicciones son simuladas, no resultados de modelos. */
(() => {
  'use strict';

  const raiz = document.getElementById('pdi-galeria-demo');
  const aplicacion = raiz.querySelector('.pdi-app');
  const ejemplos = JSON.parse(document.getElementById('pdi-datos-demo').textContent);
  const buscar = selector => raiz.querySelector(selector);
  const buscarTodos = selector => Array.from(raiz.querySelectorAll(selector));
  const espacioSvg = 'http://www.w3.org/2000/svg';
  const estado = {
    caso: ejemplos[0].id, filtro: 'todos', vista: 'comparar',
    zoom: false, referencias: true, predicciones: true, etiquetas: true,
  };
  const imagenes = new Map();
  const solicitudes = new Map();
  const direccionesTemporales = new Set();
  let cargando = 0;
  let avisoCarga = '';

  const filtrados = () => ejemplos.filter(caso => estado.filtro === 'todos' || caso.id === estado.filtro);
  const actual = () => ejemplos.find(caso => caso.id === estado.caso);

  function crearSvg(etiqueta, atributos = {}, texto) {
    const elemento = document.createElementNS(espacioSvg, etiqueta);
    Object.entries(atributos).forEach(([nombre, valor]) => elemento.setAttribute(nombre, String(valor)));
    if (texto !== undefined) elemento.textContent = texto;
    return elemento;
  }

  function vista(caso) {
    if (!estado.zoom) return [0, 0, 256, 256];
    const limites = caso.foco.map(valor => valor * 256);
    const lado = Math.min(256, Math.max(100,
      Math.max(limites[2] - limites[0], limites[3] - limites[1]) * 1.45));
    return [
      Math.min(256 - lado, Math.max(0, (limites[0] + limites[2] - lado) / 2)),
      Math.min(256 - lado, Math.max(0, (limites[1] + limites[3] - lado) / 2)),
      lado, lado,
    ];
  }

  function dibujarImagen(destino, caso, conPrediccion) {
    if (!imagenes.has(caso.id)) {
      const vacio = document.createElement('div');
      vacio.className = 'pdi-empty-image';
      const titulo = document.createElement('strong');
      titulo.textContent = 'Imagen pendiente';
      const nombre = document.createElement('span');
      nombre.textContent = caso.archivo;
      const ayuda = document.createElement('span');
      ayuda.textContent = 'Elegí la carpeta images del dataset o este archivo PNG.';
      vacio.append(titulo, nombre, ayuda);
      destino.replaceChildren(vacio);
      return;
    }

    const encuadre = vista(caso);
    const ancho = Math.min(destino.clientWidth || 320, destino.clientHeight || 320);
    const dibujo = crearSvg('svg', {
      viewBox: encuadre.join(' '), role: 'img', preserveAspectRatio: 'xMidYMid meet',
      'aria-label': caso.nombre + (conPrediccion ? ' con cajas simuladas' : ' con anotaciones originales'),
    });
    dibujo.appendChild(crearSvg('title', {}, caso.nombre + (estado.zoom ? ' · zona ampliada' : '')));
    dibujo.appendChild(crearSvg('image', { href: imagenes.get(caso.id), x: 0, y: 0, width: 256, height: 256 }));

    function dibujarCaja(coordenadas, tipo, indice) {
      const [centroX, centroY, anchura, altura] = coordenadas.map(valor => valor * 256);
      const izquierda = centroX - anchura / 2;
      const arriba = centroY - altura / 2;
      dibujo.appendChild(crearSvg('rect', {
        x: izquierda, y: arriba, width: anchura, height: altura, class: 'pdi-box pdi-box-' + tipo,
      }));
      if (!estado.etiquetas) return;
      const tamanoLetra = 11 * encuadre[2] / ancho;
      const etiqueta = (tipo === 'ref' ? 'R' : 'P') + (indice + 1) + (tipo === 'pred' ? ' · demo' : '');
      const anchoEtiqueta = etiqueta.length * tamanoLetra * 0.58 + tamanoLetra * 0.7;
      const altoEtiqueta = tamanoLetra * 1.6;
      const posicionX = Math.max(encuadre[0] + 1,
        Math.min(izquierda, encuadre[0] + encuadre[2] - anchoEtiqueta - 1));
      const propuestaY = tipo === 'ref' ? arriba - altoEtiqueta - 1 : arriba + altura + 1;
      const posicionY = Math.max(encuadre[1] + 1,
        Math.min(propuestaY, encuadre[1] + encuadre[3] - altoEtiqueta - 1));
      dibujo.appendChild(crearSvg('rect', {
        x: posicionX, y: posicionY, width: anchoEtiqueta, height: altoEtiqueta,
        rx: tamanoLetra * 0.2, class: 'pdi-box-label-bg',
      }));
      dibujo.appendChild(crearSvg('text', {
        x: posicionX + tamanoLetra * 0.35, y: posicionY + tamanoLetra * 1.12,
        'font-size': tamanoLetra, class: 'pdi-box-label',
      }, etiqueta));
    }

    if (estado.referencias) caso.referencia.forEach((caja, indice) => dibujarCaja(caja, 'ref', indice));
    if (conPrediccion && estado.predicciones) {
      if (caso.id === 'desplazada' && estado.referencias) {
        const referencia = caso.referencia[caso.principal];
        const prediccion = caso.prediccion[caso.principal];
        dibujo.appendChild(crearSvg('line', {
          x1: referencia[0] * 256, y1: referencia[1] * 256,
          x2: prediccion[0] * 256, y2: prediccion[1] * 256, class: 'pdi-connector',
        }));
      }
      caso.prediccion.forEach((caja, indice) => dibujarCaja(caja, 'pred', indice));
    }
    destino.replaceChildren(dibujo);
  }

  function mostrar() {
    const caso = actual();
    const casos = filtrados();
    aplicacion.style.setProperty('--pdi-pred', caso.color);
    raiz.dataset.caso = caso.id;
    raiz.dataset.vista = estado.vista;
    raiz.dataset.zoom = String(estado.zoom);
    buscar('.pdi-total').textContent = String(casos.length);
    buscar('.pdi-thumbnails').replaceChildren(...casos.map(ejemplo => {
      const boton = document.createElement('button');
      boton.type = 'button';
      boton.className = 'pdi-thumbnail';
      boton.setAttribute('aria-pressed', String(ejemplo.id === estado.caso));
      boton.setAttribute('aria-label', ejemplo.nombre);
      const miniatura = document.createElement(imagenes.has(ejemplo.id) ? 'img' : 'span');
      if (imagenes.has(ejemplo.id)) {
        miniatura.src = imagenes.get(ejemplo.id);
        miniatura.alt = '';
      } else {
        miniatura.className = 'pdi-thumbnail-empty';
        miniatura.textContent = 'PNG';
        miniatura.setAttribute('aria-hidden', 'true');
      }
      const texto = document.createElement('span');
      texto.className = 'pdi-thumbnail-info';
      const nombre = document.createElement('span');
      nombre.className = 'pdi-thumbnail-name';
      nombre.textContent = ejemplo.nombre;
      const detalle = document.createElement('span');
      detalle.className = 'pdi-thumbnail-meta';
      detalle.textContent = ejemplo.fuente + ' · ' + ejemplo.referencia.length + ' anot.';
      texto.append(nombre, detalle);
      boton.append(miniatura, texto);
      boton.addEventListener('click', () => { estado.caso = ejemplo.id; mostrar(); });
      return boton;
    }));
    buscar('.pdi-case-title').textContent = caso.nombre;
    buscar('.pdi-filename').textContent = caso.archivo;
    buscar('.pdi-page').textContent = (casos.findIndex(ejemplo => ejemplo.id === estado.caso) + 1) + ' / ' + casos.length;
    buscar('.pdi-panels').dataset.mode = estado.vista;
    buscar('.pdi-pred-title').textContent = estado.vista === 'superponer'
      ? 'Anotación + predicción simulada' : 'Predicción simulada';
    buscarTodos('.pdi-view').forEach(elemento =>
      elemento.setAttribute('aria-pressed', String(elemento.dataset.view === estado.vista)));
    buscar('.pdi-zoom').setAttribute('aria-pressed', String(estado.zoom));
    buscar('.pdi-zoom-label').textContent = estado.zoom ? 'Imagen completa' : 'Acercar al error';
    buscar('.pdi-real-count').textContent = caso.referencia.length + (caso.referencia.length === 1 ? ' caja' : ' cajas');
    buscar('.pdi-pred-count').textContent = caso.prediccion.length + (caso.prediccion.length === 1 ? ' caja simulada' : ' cajas simuladas');
    buscar('.pdi-zoom-value').textContent = (256 / vista(caso)[2]).toFixed(1) + '×';
    buscar('.pdi-detail-text').textContent = caso.detalle;
    buscar('.pdi-overlap').textContent = caso.iou === null
      ? (caso.id === 'omitida' ? 'Sin predicción para R1' : 'Sin anotación de referencia')
      : 'IoU del ejemplo ' + caso.iou.toFixed(2);
    buscar('.pdi-filter').value = estado.filtro;
    buscar('.pdi-check-ref').checked = estado.referencias;
    buscar('.pdi-check-pred').checked = estado.predicciones;
    buscar('.pdi-check-label').checked = estado.etiquetas;
    buscar('.pdi-file-status').textContent = avisoCarga || (cargando
      ? 'Buscando imágenes…'
      : imagenes.size + ' de ' + ejemplos.length + ' imágenes disponibles. '
        + (imagenes.size < ejemplos.length ? 'Elegí la carpeta images del dataset para cargar las que faltan.' : 'Todo listo para explorar.'));
    dibujarImagen(buscar('.pdi-stage-ref'), caso, false);
    dibujarImagen(buscar('.pdi-stage-pred'), caso, true);
  }

  // Las imágenes se leen desde el disco local. No se suben a ningún servicio.
  function cargarImagen(caso, direccion, temporal = false) {
    const solicitud = {};
    solicitudes.set(caso.id, solicitud);
    const foto = new Image();
    cargando += 1;
    const terminar = correcta => {
      cargando -= 1;
      if (solicitudes.get(caso.id) !== solicitud) {
        if (temporal) URL.revokeObjectURL(direccion);
        mostrar();
        return;
      }
      if (correcta && foto.naturalWidth === 256 && foto.naturalHeight === 256) {
        const anterior = imagenes.get(caso.id);
        if (direccionesTemporales.delete(anterior)) URL.revokeObjectURL(anterior);
        imagenes.set(caso.id, direccion);
        if (temporal) direccionesTemporales.add(direccion);
      } else {
        if (temporal) {
          URL.revokeObjectURL(direccion);
          avisoCarga = 'No se pudo cargar ' + caso.archivo + '. Se espera el PNG original de 256 × 256 px.';
        }
      }
      mostrar();
    };
    foto.onload = () => terminar(true);
    foto.onerror = () => terminar(false);
    foto.src = direccion;
  }

  function elegirArchivos(evento) {
    const archivos = Array.from(evento.target.files || []);
    avisoCarga = '';
    let encontrados = 0;
    for (const caso of ejemplos) {
      const archivo = archivos.find(elemento => elemento.name === caso.archivo);
      if (archivo) {
        encontrados += 1;
        cargarImagen(caso, URL.createObjectURL(archivo), true);
      }
    }
    if (!encontrados) avisoCarga = 'La selección no contiene las imágenes de estos ejemplos. Elegí data/grupo_5/images/.';
    evento.target.value = '';
    mostrar();
  }

  buscarTodos('.pdi-view').forEach(elemento => elemento.addEventListener('click', () => {
    estado.vista = elemento.dataset.view;
    mostrar();
  }));
  buscar('.pdi-filter').addEventListener('change', evento => {
    estado.filtro = evento.target.value;
    if (!filtrados().some(caso => caso.id === estado.caso)) estado.caso = filtrados()[0].id;
    mostrar();
  });
  buscar('.pdi-zoom').addEventListener('click', () => { estado.zoom = !estado.zoom; mostrar(); });
  for (const [selector, clave] of [
    ['.pdi-check-ref', 'referencias'], ['.pdi-check-pred', 'predicciones'], ['.pdi-check-label', 'etiquetas'],
  ]) buscar(selector).addEventListener('change', evento => { estado[clave] = evento.target.checked; mostrar(); });
  for (const [selector, paso] of [['.pdi-prev', -1], ['.pdi-next', 1]]) {
    buscar(selector).addEventListener('click', () => {
      const casos = filtrados();
      const indice = casos.findIndex(caso => caso.id === estado.caso);
      estado.caso = casos[(indice + paso + casos.length) % casos.length].id;
      mostrar();
    });
  }
  buscar('.pdi-choose-folder').addEventListener('click', () => buscar('.pdi-folder-input').click());
  buscar('.pdi-choose-files').addEventListener('click', () => buscar('.pdi-files-input').click());
  buscar('.pdi-folder-input').addEventListener('change', elegirArchivos);
  buscar('.pdi-files-input').addEventListener('change', elegirArchivos);
  mostrar();
  if (globalThis.ResizeObserver) {
    new ResizeObserver(() => {
      dibujarImagen(buscar('.pdi-stage-ref'), actual(), false);
      dibujarImagen(buscar('.pdi-stage-pred'), actual(), true);
    }).observe(buscar('.pdi-panels'));
  }
  for (const caso of ejemplos) cargarImagen(caso, '../../data/grupo_5/images/' + caso.archivo);
})();
