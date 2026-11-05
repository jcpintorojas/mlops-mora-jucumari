'use strict';
// Laboratorio local: consulta los créditos del CSV de entrenamiento y compara la predicción con lo que pasó.
// Solo existe si se define DATOS_CSV (scripts/laboratorio.sh). En Kubernetes no se activa y la imagen no lleva datos.
const fs = require('fs');
const express = require('express');
const { predecir, CAMPOS } = require('./riesgo');

const FIN_TRAIN = '2024-06-30';
const FIN_TEST = '2025-09-30';
const CATEGORICAS = Object.keys(CAMPOS).filter((n) => CAMPOS[n].opciones);
const NUMERICAS = Object.keys(CAMPOS).filter((n) => !CAMPOS[n].opciones);
const ETIQUETAS = {
  monto_total: 'Monto de la venta', pct_inicial: 'Cuota inicial', n_lineas: 'Productos en la nota', n_creditos_previos: 'Créditos anteriores',
  monto_previo: 'Monto de créditos anteriores', n_previos_maduros: 'Anteriores con resultado conocido', n_previos_mora30: 'Anteriores atrasados',
  max_dias_atraso_previo: 'Mayor atraso anterior', saldo_abierto_previo: 'Saldo que aún debía', meses_desde_primera_venta: 'Meses como cliente',
  n_contado_previos: 'Compras al contado', plazo_cat: 'Plazo', atraso_previo: 'Cómo pagó su último crédito', con_plan: 'Plan de cuotas',
};

// El CSV viene de SQL con COPY: los valores usados no llevan comas ni comillas.
function leerCsv(ruta) {
  const [cabecera, ...lineas] = fs.readFileSync(ruta, 'utf8').trim().split('\n');
  const columnas = cabecera.split(',');
  return lineas.map((l) => {
    const v = l.split(',');
    if (v.length !== columnas.length) throw new Error(`fila con ${v.length} columnas, se esperaban ${columnas.length}`);
    return Object.fromEntries(columnas.map((c, i) => [c, v[i]]));
  });
}

function periodoDe(f) {
  if (f.regularizada === 't') return 'excluido';
  if (f.madura !== 't') return 'sin_resultado';
  if (f.fecha_venta <= FIN_TRAIN) return 'entrenamiento';
  return f.fecha_venta <= FIN_TEST ? 'prueba' : 'posterior';
}

function aCredito(f) {
  const credito = {};
  for (const n of NUMERICAS) credito[n] = Number(f[n]);
  for (const n of CATEGORICAS) credito[n] = f[n];
  return { id: f.credito_id, cliente: f.cliente_id, fecha: f.fecha_venta, periodo: periodoDe(f), real: f.mora30 === '' ? null : Number(f.mora30), credito };
}

// AUC por rangos, con empates: la probabilidad de que un atrasado reciba más riesgo que uno que pagó.
function auc(filas) {
  const orden = [...filas].sort((a, b) => a.p - b.p);
  let i = 0; let suma = 0; let pos = 0;
  while (i < orden.length) {
    let j = i;
    while (j < orden.length && orden[j].p === orden[i].p) j += 1;
    for (let k = i; k < j; k += 1) if (orden[k].y === 1) { suma += (i + j + 1) / 2; pos += 1; }
    i = j;
  }
  const neg = orden.length - pos;
  return pos && neg ? (suma - (pos * (pos + 1)) / 2) / (pos * neg) : null;
}

function resumen(items, umbral) {
  const periodos = ['entrenamiento', 'prueba', 'posterior', 'sin_resultado'];
  return periodos.map((periodo) => {
    const g = items.filter((c) => c.periodo === periodo);
    const conReal = g.filter((c) => c.real !== null);
    const atrasos = conReal.filter((c) => c.real === 1).length;
    const media = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);
    return {
      periodo, creditos: g.length, con_resultado: conReal.length, atrasos,
      atraso_real: conReal.length ? atrasos / conReal.length : null,
      prob_media: media(g.map((c) => c.prob)),
      aciertos: conReal.length ? conReal.filter((c) => (c.prob >= umbral ? 1 : 0) === c.real).length / conReal.length : null,
      auc: atrasos >= 10 ? auc(conReal.map((c) => ({ p: c.prob, y: c.real }))) : null,
    };
  });
}

function crearLaboratorio(ruta, estado, umbral) {
  const filas = leerCsv(ruta).map(aCredito);
  let cache = null; let versionCache = null;
  const items = () => {
    if (!estado.modelo) return null;
    if (versionCache !== estado.version) {
      cache = filas.map((c) => ({ ...c, ...(({ probabilidad, aportes }) => ({ prob: probabilidad, aportes }))(predecir(estado.modelo, c.credito)) }));
      versionCache = estado.version;
    }
    return cache;
  };
  const corto = (c) => ({ id: c.id, cliente: c.cliente, fecha: c.fecha, periodo: c.periodo, monto: c.credito.monto_total, plazo: c.credito.plazo_cat, atraso_previo: c.credito.atraso_previo, probabilidad: c.prob, prediccion: c.prob >= umbral ? 'riesgo_alto' : 'riesgo_normal', real: c.real });
  const detalle = (c) => ({
    ...corto(c), datos: c.credito,
    modelo_lo_vio: c.periodo === 'entrenamiento',
    acerto: c.real === null ? null : (c.prob >= umbral ? 1 : 0) === c.real,
    aportes: [...c.aportes].sort((a, b) => Math.abs(b.aporte) - Math.abs(a.aporte)).map((a) => ({ ...a, etiqueta: ETIQUETAS[a.variable], valor: c.credito[a.variable] })),
    modelo: { version: estado.version, run_id: estado.runId },
  });
  const buscar = (lista, campo, id) => (typeof id === 'string' && id.length >= 8 ? lista.filter((c) => c[campo].startsWith(id)) : []);

  const r = express.Router();
  r.use((_req, res, next) => { if (!items()) return res.status(503).json({ detail: 'modelo no cargado' }); return next(); });
  r.get('/resumen', (_req, res) => res.json({ version: estado.version, run_id: estado.runId, umbral, periodos: resumen(items(), umbral) }));
  r.get('/creditos', (req, res) => {
    let l = items();
    if (req.query.periodo) l = l.filter((c) => c.periodo === req.query.periodo);
    if (req.query.resultado === '0' || req.query.resultado === '1') l = l.filter((c) => c.real === Number(req.query.resultado));
    if (req.query.orden === 'riesgo') l = [...l].sort((a, b) => b.prob - a.prob);
    else if (req.query.orden === 'recientes') l = [...l].sort((a, b) => b.fecha.localeCompare(a.fecha));
    const limite = Math.min(Math.max(Number(req.query.limite) || 30, 1), 200);
    res.json({ total: l.length, creditos: l.slice(0, limite).map(corto) });
  });
  r.get('/creditos/:id', (req, res) => {
    const e = buscar(items(), 'id', req.params.id);
    if (e.length !== 1) return res.status(404).json({ detail: 'crédito no encontrado; usa al menos 8 caracteres de su identificador' });
    return res.json(detalle(e[0]));
  });
  r.get('/clientes/:id', (req, res) => {
    const e = buscar(items(), 'cliente', req.params.id);
    if (!e.length) return res.status(404).json({ detail: 'cliente no encontrado; usa al menos 8 caracteres de su identificador' });
    return res.json({ cliente: e[0].cliente, creditos: e.length, historial: e.map(corto) });
  });
  return { router: r, filas: filas.length };
}

module.exports = { crearLaboratorio, leerCsv, periodoDe, auc, resumen };
