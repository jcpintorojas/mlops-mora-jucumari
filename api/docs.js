'use strict';
// Documentación de la API. Los parámetros de /predict salen de CAMPOS: lo que se documenta es lo que se valida.
const { CAMPOS } = require('./riesgo');

const DESCRIPCION = {
  monto_total: ['Total de la venta a crédito, en Bs', 45000],
  pct_inicial: ['Parte pagada el día de la venta, de 0 a 1', 0],
  n_lineas: ['Productos distintos en la nota', 3],
  n_creditos_previos: ['Créditos anteriores del cliente', 6],
  monto_previo: ['Suma de esos créditos, en Bs', 90000],
  n_previos_maduros: ['Créditos anteriores cuyo resultado ya se conocía', 5],
  n_previos_mora30: ['De esos, cuántos se atrasaron más de 30 días', 3],
  max_dias_atraso_previo: ['Mayor atraso conocido al vender, en días', 120],
  saldo_abierto_previo: ['Lo que aún debe de créditos anteriores, en Bs', 15000],
  meses_desde_primera_venta: ['Meses desde su primera compra', 24],
  n_contado_previos: ['Compras al contado anteriores', 1],
  plazo_cat: ['Plazo en meses, o "dias" en notas antiguas', '1'],
  atraso_previo: ['Cómo pagó su último crédito con resultado conocido', 'si'],
  con_plan: ['Tiene plan de cuotas', 'no'],
};

function tipoDe(regla) {
  if (regla.opciones) return 'texto';
  return regla.entero ? 'entero' : 'número';
}

function reglaDe(regla) {
  if (regla.opciones) return `Uno de: ${regla.opciones.join(', ')}`;
  const min = regla.excluyente ? `mayor que ${regla.min}` : `desde ${regla.min}`;
  return regla.max !== undefined ? `${min} hasta ${regla.max}` : min;
}

const PARAMETROS = Object.entries(CAMPOS).map(([nombre, regla]) => ({
  nombre, tipo: tipoDe(regla), obligatorio: true, regla: reglaDe(regla), descripcion: DESCRIPCION[nombre][0],
}));
const EJEMPLO = Object.fromEntries(Object.keys(CAMPOS).map((n) => [n, DESCRIPCION[n][1]]));

const ENDPOINTS = [
  { metodo: 'GET', ruta: '/', resumen: 'Esta página.', parametros: [], respuestas: [{ codigo: 200, descripcion: 'Página HTML con la documentación', campos: [] }] },
  { metodo: 'GET', ruta: '/endpoints', resumen: 'Esta misma documentación, en JSON.', parametros: [], respuestas: [{ codigo: 200, descripcion: 'Lista de endpoints, parámetros y respuestas', campos: [['endpoints', 'lista', 'Un elemento por endpoint'], ['ejemplo_predict', 'objeto', 'Crédito inventado para probar /predict']] }] },
  { metodo: 'GET', ruta: '/health', resumen: 'Liveness: indica si el proceso responde. Lo usa Kubernetes para reiniciar el contenedor.', parametros: [], respuestas: [{ codigo: 200, descripcion: 'El proceso está vivo', campos: [['estado', 'texto', 'Siempre "vivo"'], ['pod', 'texto', 'Pod que atendió']] }] },
  { metodo: 'GET', ruta: '/ready', resumen: 'Readiness: indica si el modelo está cargado. Sin esto, el Service no le manda tráfico.', parametros: [], respuestas: [{ codigo: 200, descripcion: 'Listo para atender', campos: [['estado', 'texto', 'Siempre "listo"'], ['pod', 'texto', 'Pod que atendió'], ['version', 'texto', 'Versión del modelo cargada']] }, { codigo: 503, descripcion: 'Modelo sin cargar', campos: [['detail', 'texto', 'Motivo']] }] },
  { metodo: 'GET', ruta: '/info', resumen: 'Qué modelo sirve este Pod y de qué run salió.', parametros: [], respuestas: [{ codigo: 200, descripcion: 'Datos del servicio', campos: [['pod', 'texto', 'Pod que atendió'], ['modelo', 'texto', 'Nombre en el Model Registry'], ['alias', 'texto', 'Alias cargado'], ['version', 'texto', 'Versión del registro'], ['run_id', 'texto', 'Run de MLflow que la produjo'], ['umbral', 'número', 'Probabilidad desde la que se marca riesgo alto'], ['servicio', 'texto', 'Siempre "node"'], ['node', 'texto', 'Versión de Node.js'], ['mlflow_servidor', 'texto', 'Versión del servidor MLflow']] }] },
  { metodo: 'POST', ruta: '/predict', resumen: 'Estima la probabilidad de que el crédito siga impago 30 días después de vencer. Es un aviso: nunca aprueba ni niega.', parametros: PARAMETROS, respuestas: [{ codigo: 200, descripcion: 'Predicción', campos: [['probabilidad_mora', 'número', 'De 0 a 1, con 4 decimales'], ['prediccion', 'texto', '"riesgo_alto" desde el umbral; si no, "riesgo_normal"'], ['razones', 'lista', 'Hasta 3 variables que más subieron el riesgo'], ['modelo', 'objeto', 'nombre, alias, version y run_id'], ['pod', 'texto', 'Pod que atendió']] }, { codigo: 400, descripcion: 'JSON mal formado', campos: [['detail', 'texto', '"JSON inválido"']] }, { codigo: 422, descripcion: 'Datos inválidos', campos: [['detail', 'lista', 'Un mensaje por campo']] }, { codigo: 503, descripcion: 'Modelo sin cargar', campos: [['detail', 'texto', 'Motivo']] }] },
];

const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function tabla(cabeceras, filas) {
  return `<div class="t"><table><thead><tr>${cabeceras.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${filas.map((f) => `<tr>${f.map((v, i) => `<td${i === 0 ? ' class="c"' : ''}>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function seccion(e, i) {
  const params = e.parametros.length
    ? `<h3>Parámetros <small>JSON en el cuerpo, todos obligatorios</small></h3>${tabla(['Nombre', 'Tipo', 'Valores', 'Qué es'], e.parametros.map((p) => [p.nombre, p.tipo, p.regla, p.descripcion]))}`
    : '<p class="n">No recibe parámetros.</p>';
  const resp = e.respuestas.map((r) => `<h3><span class="cod c${String(r.codigo)[0]}">${r.codigo}</span> ${esc(r.descripcion)}</h3>${r.campos.length ? tabla(['Campo', 'Tipo', 'Qué es'], r.campos) : ''}`).join('');
  const cuerpo = e.metodo === 'POST' ? `<label for="c${i}">Cuerpo de ejemplo (inventado)</label><textarea id="c${i}" rows="17" spellcheck="false">${esc(JSON.stringify(EJEMPLO, null, 2))}</textarea>` : '';
  return `<section id="e${i}"><h2><span class="m ${e.metodo.toLowerCase()}">${e.metodo}</span> <code>${e.ruta}</code></h2><p>${esc(e.resumen)}</p>${params}<h3 class="r">Qué devuelve</h3>${resp}<div class="probar">${cuerpo}<button type="button" data-m="${e.metodo}" data-r="${e.ruta}" data-c="c${i}" data-s="s${i}">Probar</button><pre id="s${i}" hidden></pre></div></section>`;
}

function pagina() {
  return `<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>API de mora · Jucumari</title><style>
:root{color-scheme:light;--a:#1f3864;--t:#1d2433;--m:#55617a;--l:#d9dfea;--f:#f4f6fa;--v:#2e7d4f;--z:#2e5597;--r:#b3261e;--am:#9a6200}
*{box-sizing:border-box}body{margin:0;background:var(--f);color:var(--t);font:15px/1.5 system-ui,sans-serif}header{background:var(--a);color:#fff;padding:18px 24px}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;opacity:.85;font-size:14px}main{max-width:1000px;margin:18px auto;padding:0 16px}nav,section{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:16px 20px;margin-bottom:16px}nav a{display:inline-block;margin:3px 14px 3px 0;color:var(--z);text-decoration:none}nav a:hover{text-decoration:underline}h2{margin:0 0 6px;font-size:19px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}h3{margin:16px 0 6px;font-size:14px}h3 small{font-weight:400;color:var(--m)}h3.r{border-top:1px solid var(--l);padding-top:12px}p{margin:4px 0}.n{color:var(--m)}code,.c,textarea,pre{font-family:ui-monospace,Menlo,Consolas,monospace}.m{font:600 13px ui-monospace,monospace;color:#fff;padding:3px 9px;border-radius:5px}.get{background:var(--z)}.post{background:var(--v)}.cod{font:600 12px ui-monospace,monospace;padding:2px 7px;border-radius:5px;color:#fff}.c2{background:var(--v)}.c4{background:var(--am)}.c5{background:var(--r)}.t{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th{text-align:left;color:var(--m);font-weight:600;border-bottom:2px solid var(--l);padding:5px 8px}td{border-bottom:1px solid var(--l);padding:6px 8px;vertical-align:top}.c{white-space:nowrap}.probar{margin-top:14px}label{display:block;font-size:13px;color:var(--m);margin-bottom:4px}textarea{width:100%;font-size:13px;padding:8px;border:1px solid var(--l);border-radius:6px}button{margin-top:8px;border:0;border-radius:6px;padding:8px 16px;background:var(--z);color:#fff;font-size:14px;cursor:pointer}button:hover{background:var(--a)}button:focus-visible,textarea:focus-visible,a:focus-visible{outline:3px solid #7a9fd6;outline-offset:2px}pre{margin:10px 0 0;background:#0d1f33;color:#dce8f2;padding:12px;border-radius:6px;overflow-x:auto;font-size:13px}
</style></head><body><header><h1>API de mora de créditos · Jucumari</h1><p>Servicio de inferencia en Node.js. Carga el modelo desde MLflow por su alias.</p></header><main><nav aria-label="Endpoints">${ENDPOINTS.map((e, i) => `<a href="#e${i}"><b>${e.metodo}</b> ${e.ruta}</a>`).join('')}</nav>${ENDPOINTS.map(seccion).join('')}</main><script>
document.querySelectorAll('button[data-r]').forEach(function(b){b.addEventListener('click',async function(){var s=document.getElementById(b.dataset.s);s.hidden=false;s.textContent='Consultando…';try{var o={method:b.dataset.m};if(b.dataset.m==='POST'){o.headers={'Content-Type':'application/json'};o.body=document.getElementById(b.dataset.c).value}var r=await fetch(b.dataset.r,o);var t=await r.text();var c=r.headers.get('content-type')||'';if(c.includes('json'))t=JSON.stringify(JSON.parse(t),null,2);else t='Página HTML de '+t.length+' caracteres';s.textContent='HTTP '+r.status+'\\n'+t}catch(e){s.textContent='Error: '+e.message}})});
</script></body></html>`;
}

module.exports = { ENDPOINTS, EJEMPLO, pagina };
