'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { validar, predecir, CAMPOS } = require('../riesgo');
const { ENDPOINTS, EJEMPLO, pagina } = require('../docs');
const { auc, periodoDe, resumen } = require('../laboratorio');

const credito = { monto_total: 100, pct_inicial: 0, n_lineas: 1, n_creditos_previos: 0, monto_previo: 0, n_previos_maduros: 0, n_previos_mora30: 0, max_dias_atraso_previo: 0, saldo_abierto_previo: 0, meses_desde_primera_venta: 0, n_contado_previos: 0, plazo_cat: '1', atraso_previo: 'si', con_plan: 'no' };

test('calcula a mano una regresión logística', () => {
  const modelo = { intercepto: -1, numericas: [{ nombre: 'monto_total', media: 50, escala: 25, coef: 0.5 }], categoricas: [{ nombre: 'atraso_previo', categorias: { si: 1, no: -1 } }] };
  const r = predecir(modelo, credito); // z = -1 + 0,5 x 2 + 1 = 1
  assert.ok(Math.abs(r.probabilidad - 0.7310585786) < 1e-9);
  assert.deepStrictEqual(r.razones, ['monto_total', 'atraso_previo']);
});

test('una categoría desconocida no aporta', () => {
  const modelo = { intercepto: 0, numericas: [], categoricas: [{ nombre: 'atraso_previo', categorias: { no: 3 } }] };
  assert.strictEqual(predecir(modelo, credito).probabilidad, 0.5);
});

test('valida el contrato', () => {
  assert.deepStrictEqual(validar(credito), []);
  assert.ok(validar({ ...credito, monto_total: 0 }).length);
  assert.ok(validar({ ...credito, pct_inicial: 2 }).length);
  assert.ok(validar({ ...credito, plazo_cat: '99' }).length);
  assert.ok(validar({ ...credito, n_lineas: 1.5 }).length);
  assert.ok(validar({ ...credito, monto_previo: '5' }).length);
  assert.ok(validar({}).length >= 14);
  assert.ok(validar([]).length);
});

test('la documentación cubre todo el contrato', () => {
  const predict = ENDPOINTS.find((e) => e.ruta === '/predict');
  assert.deepStrictEqual(predict.parametros.map((p) => p.nombre), Object.keys(CAMPOS));
  assert.deepStrictEqual(validar(EJEMPLO), []);
  const html = pagina();
  for (const e of ENDPOINTS) assert.ok(html.includes(`<code>${e.ruta}</code>`), e.ruta);
  for (const n of Object.keys(CAMPOS)) assert.ok(html.includes(n), n);
  assert.ok(!html.includes('undefined'));
});

test('calcula AUC con empates', () => {
  assert.strictEqual(auc([{ p: 0.1, y: 0 }, { p: 0.2, y: 0 }, { p: 0.8, y: 1 }, { p: 0.9, y: 1 }]), 1);
  assert.strictEqual(auc([{ p: 0.5, y: 0 }, { p: 0.5, y: 1 }]), 0.5);
  assert.strictEqual(auc([{ p: 0.9, y: 0 }, { p: 0.1, y: 1 }]), 0);
  assert.strictEqual(auc([{ p: 0.5, y: 1 }]), null);
});

test('clasifica periodos como el entrenamiento', () => {
  const f = (fecha_venta, madura = 't', regularizada = 'f') => ({ fecha_venta, madura, regularizada });
  assert.strictEqual(periodoDe(f('2024-06-30')), 'entrenamiento');
  assert.strictEqual(periodoDe(f('2024-07-01')), 'prueba');
  assert.strictEqual(periodoDe(f('2025-09-30')), 'prueba');
  assert.strictEqual(periodoDe(f('2025-10-01')), 'posterior');
  assert.strictEqual(periodoDe(f('2026-10-01', 'f')), 'sin_resultado');
  assert.strictEqual(periodoDe(f('2024-01-01', 't', 't')), 'excluido');
});

test('resume aciertos con el umbral', () => {
  const r = resumen([{ periodo: 'prueba', real: 1, prob: 0.7 }, { periodo: 'prueba', real: 0, prob: 0.7 }, { periodo: 'prueba', real: null, prob: 0.1 }], 0.5);
  const p = r.find((x) => x.periodo === 'prueba');
  assert.strictEqual(p.con_resultado, 2);
  assert.strictEqual(p.aciertos, 0.5);
});

// Se genera con src/exportar_coeficientes.py y no se versiona, porque trae los parámetros del modelo.
const paridad = path.join(__dirname, 'paridad.local.json');
test('coincide con scikit-learn en casos inventados', { skip: !fs.existsSync(paridad) }, () => {
  const { modelo, casos } = JSON.parse(fs.readFileSync(paridad));
  let maximo = 0;
  for (const c of casos) maximo = Math.max(maximo, Math.abs(predecir(modelo, c.credito).probabilidad - c.esperado));
  assert.ok(casos.length >= 200);
  assert.ok(maximo < 1e-9, `diferencia máxima ${maximo}`);
});
