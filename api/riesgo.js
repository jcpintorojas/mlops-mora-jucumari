'use strict';
// Lógica de predicción, sin red ni Express: se puede llevar tal cual a otro backend de Node.
// Replica el modelo de scikit-learn: estandariza números, codifica categorías y aplica la regresión logística.

const PLAZOS = ['1', '2-3', '4-6', '7+', 'dias'];
const HISTORIAL = ['si', 'no', 'sin_historial'];
const SI_NO = ['si', 'no'];

// Contrato de entrada. Un dato inválido recibe HTTP 422 y nunca llega al modelo.
const CAMPOS = {
  monto_total: { min: 0, excluyente: true },
  pct_inicial: { min: 0, max: 1 },
  n_lineas: { min: 0, entero: true },
  n_creditos_previos: { min: 0, entero: true },
  monto_previo: { min: 0 },
  n_previos_maduros: { min: 0, entero: true },
  n_previos_mora30: { min: 0, entero: true },
  max_dias_atraso_previo: { min: 0, entero: true },
  saldo_abierto_previo: { min: 0 },
  meses_desde_primera_venta: { min: 0, entero: true },
  n_contado_previos: { min: 0, entero: true },
  plazo_cat: { opciones: PLAZOS },
  atraso_previo: { opciones: HISTORIAL },
  con_plan: { opciones: SI_NO },
};

function validar(cuerpo) {
  if (cuerpo === null || typeof cuerpo !== 'object' || Array.isArray(cuerpo)) return ['el cuerpo debe ser un objeto JSON'];
  const errores = [];
  for (const [nombre, regla] of Object.entries(CAMPOS)) {
    const v = cuerpo[nombre];
    if (v === undefined || v === null) { errores.push(`${nombre}: es obligatorio`); continue; }
    if (regla.opciones) {
      if (!regla.opciones.includes(v)) errores.push(`${nombre}: debe ser uno de ${regla.opciones.join(', ')}`);
      continue;
    }
    if (typeof v !== 'number' || !Number.isFinite(v)) { errores.push(`${nombre}: debe ser un número`); continue; }
    if (regla.entero && !Number.isInteger(v)) errores.push(`${nombre}: debe ser entero`);
    if (regla.excluyente ? v <= regla.min : v < regla.min) errores.push(`${nombre}: fuera de rango`);
    if (regla.max !== undefined && v > regla.max) errores.push(`${nombre}: fuera de rango`);
  }
  return errores;
}

function predecir(modelo, credito) {
  const aportes = [];
  let z = modelo.intercepto;
  for (const v of modelo.numericas) {
    const a = v.coef * ((credito[v.nombre] - v.media) / v.escala);
    z += a; aportes.push({ variable: v.nombre, aporte: a });
  }
  for (const v of modelo.categoricas) {
    // Una categoría desconocida no aporta nada, igual que handle_unknown="ignore".
    const a = v.categorias[credito[v.nombre]] ?? 0;
    z += a; aportes.push({ variable: v.nombre, aporte: a });
  }
  const razones = aportes.filter((x) => x.aporte > 0).sort((x, y) => y.aporte - x.aporte).slice(0, 3).map((x) => x.variable);
  return { probabilidad: 1 / (1 + Math.exp(-z)), razones, aportes };
}

module.exports = { validar, predecir, CAMPOS };
