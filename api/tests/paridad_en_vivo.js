'use strict';
// Compara la API desplegada contra scikit-learn. La API redondea a 4 decimales. USO: node api/tests/paridad_en_vivo.js http://NODO:30092
const { casos } = require('./paridad.local.json');
(async () => {
  let maximo = 0; const pods = new Set();
  for (const c of casos) {
    const r = await fetch(`${process.argv[2]}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(c.credito) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json(); pods.add(d.pod);
    maximo = Math.max(maximo, Math.abs(d.probabilidad_mora - c.esperado));
  }
  console.log(`${casos.length} casos inventados · diferencia máxima con scikit-learn ${maximo.toFixed(6)} · ${pods.size} Pods`);
  if (maximo > 0.00005001) process.exit(1);
})().catch((e) => { console.error(e.message); process.exit(1); });
