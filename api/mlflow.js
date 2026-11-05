'use strict';
// Cliente mínimo de la API REST de MLflow. Pide el modelo por alias, nunca por una ruta de archivo.
const ARTEFACTO = 'exportado/coeficientes.json';

async function json(url) {
  const r = await fetch(url, { signal: AbortSignal.timeout(10000) });
  if (!r.ok) throw new Error(`${url} respondió ${r.status}`);
  return r.json();
}

async function cargarPorAlias(base, nombre, alias) {
  const q = new URLSearchParams({ name: nombre, alias });
  const { model_version: v } = await json(`${base}/api/2.0/mlflow/registered-models/alias?${q}`);
  const a = new URLSearchParams({ path: ARTEFACTO, run_uuid: v.run_id });
  const modelo = await json(`${base}/get-artifact?${a}`);
  if (modelo.formato !== 1 || modelo.run_id !== v.run_id) throw new Error('el artefacto no corresponde a esta versión');
  return { modelo, version: v.version, runId: v.run_id };
}

async function versionServidor(base) {
  const r = await fetch(`${base}/version`, { signal: AbortSignal.timeout(10000) });
  if (!r.ok) throw new Error(`/version respondió ${r.status}`);
  return (await r.text()).trim();
}

module.exports = { cargarPorAlias, versionServidor };
