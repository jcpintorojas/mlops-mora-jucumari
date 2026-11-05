'use strict';
// Servicio de inferencia (Fase 2). Al arrancar carga models:/mora-credito@champion desde MLflow.
// Cada respuesta indica qué versión y qué run la produjo, y qué Pod la atendió.
const os = require('os');
const express = require('express');
const { validar, predecir } = require('./riesgo');
const { cargarPorAlias, versionServidor } = require('./mlflow');
const { ENDPOINTS, EJEMPLO, pagina } = require('./docs');
const { crearLaboratorio } = require('./laboratorio');
const paginaLaboratorio = require('./laboratorio_pagina');

const BASE = (process.env.MLFLOW_TRACKING_URI || 'http://mlflow-mora:5000').replace(/\/$/, '');
const NOMBRE = process.env.MODELO_NOMBRE || 'mora-credito';
const ALIAS = process.env.MODELO_ALIAS || 'champion';
const MLFLOW_ESPERADO = process.env.MLFLOW_VERSION || '3.14.0';
const POD = process.env.POD_NAME || os.hostname();
const UMBRAL = Number(process.env.UMBRAL_RIESGO || '0.5');
const PUERTO = Number(process.env.PORT || '8000');

const estado = { modelo: null, version: null, runId: null, error: null, mlflow: null };

// Mientras no cargue, /ready responde 503 y el Service no le manda tráfico. Reintenta sin reiniciar el proceso.
async function cargar() {
  try {
    const servidor = await versionServidor(BASE);
    if (servidor !== MLFLOW_ESPERADO) throw new Error(`MLflow ${servidor}, se esperaba ${MLFLOW_ESPERADO}`);
    const r = await cargarPorAlias(BASE, NOMBRE, ALIAS);
    Object.assign(estado, { modelo: r.modelo, version: r.version, runId: r.runId, error: null, mlflow: servidor });
    console.log(`[${POD}] modelo ${NOMBRE}@${ALIAS} -> versión ${r.version} (run ${r.runId})`);
  } catch (e) {
    estado.error = e.message;
    console.log(`[${POD}] no se pudo cargar el modelo: ${e.message}`);
    setTimeout(cargar, 5000).unref();
  }
}

const app = express();
app.disable('x-powered-by');
app.use(express.json({ limit: '16kb' }));

const PAGINA = pagina();
app.get('/', (_req, res) => res.type('html').send(PAGINA));
app.get('/endpoints', (_req, res) => res.json({ endpoints: ENDPOINTS, ejemplo_predict: EJEMPLO }));
app.get('/health', (_req, res) => res.json({ estado: 'vivo', pod: POD }));
app.get('/ready', (_req, res) => {
  if (!estado.modelo) return res.status(503).json({ detail: `modelo no cargado: ${estado.error}` });
  return res.json({ estado: 'listo', pod: POD, version: estado.version });
});
app.get('/info', (_req, res) => res.json({
  pod: POD, modelo: NOMBRE, alias: ALIAS, version: estado.version, run_id: estado.runId,
  umbral: UMBRAL, servicio: 'node', node: process.version, mlflow_servidor: estado.mlflow,
}));
app.post('/predict', (req, res) => {
  if (!estado.modelo) return res.status(503).json({ detail: 'modelo no cargado' });
  const errores = validar(req.body);
  if (errores.length) return res.status(422).json({ detail: errores });
  const { probabilidad, razones } = predecir(estado.modelo, req.body);
  return res.json({
    probabilidad_mora: Number(probabilidad.toFixed(4)),
    prediccion: probabilidad >= UMBRAL ? 'riesgo_alto' : 'riesgo_normal',
    razones,
    modelo: { nombre: NOMBRE, alias: ALIAS, version: estado.version, run_id: estado.runId },
    pod: POD,
  });
});
// Laboratorio local: solo con DATOS_CSV. En Kubernetes no se define, y estas rutas no existen.
if (process.env.DATOS_CSV) {
  const lab = crearLaboratorio(process.env.DATOS_CSV, estado, UMBRAL);
  console.log(`[${POD}] laboratorio activo: ${lab.filas} créditos`);
  app.get('/laboratorio', (_req, res) => res.type('html').send(paginaLaboratorio));
  app.use('/laboratorio', lab.router);
}
// JSON mal formado: error del cliente, no del servicio.
app.use((err, _req, res, _next) => res.status(err.status || 500).json({ detail: err.status ? 'JSON inválido' : 'error interno' }));

const servidor = app.listen(PUERTO, '0.0.0.0', () => { console.log(`[${POD}] escuchando en ${PUERTO}`); cargar(); });
// Kubernetes avisa con SIGTERM: termina lo que está atendiendo antes de salir.
for (const s of ['SIGTERM', 'SIGINT']) process.on(s, () => servidor.close(() => process.exit(0)));
