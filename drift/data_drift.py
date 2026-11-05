"""Fase 4.1 — Detección de data drift: compara cada variable de un lote nuevo contra el baseline de entrenamiento.

Pruebas por tipo de variable (clase 7, lámina 9):
- Numéricas (montos, historial del cliente): Kolmogorov-Smirnov de dos muestras. Compara las
  distribuciones acumuladas completas sin suponer normalidad (los montos son muy asimétricos). Con 11 pruebas
  simultáneas se aplica la corrección de Bonferroni: alerta si p < 0,05 / 11 = 0,0045.
  Además se informa la magnitud del cambio (estadístico D) para no alarmar por diferencias ínfimas.
- Categóricas (plazo, cómo pagó antes, plan de cuotas): índice de estabilidad poblacional (PSI) por categoría.
  Umbrales de la industria (clase 7, láminas 9 y 11): PSI < 0,10 estable; 0,10–0,20 alerta menor; >= 0,20 drift.
Estado del lote: ROJO si al menos una variable está en rojo; AMARILLO si solo hay alertas menores; VERDE si no.
"""
# json guarda el resumen de resultados.
import json
# Path maneja rutas de archivos.
from pathlib import Path

# numpy calcula el PSI.
import numpy as np
# pandas maneja las tablas de datos.
import pandas as pd
# ks_2samp es la prueba de Kolmogorov-Smirnov para dos muestras.
from scipy.stats import ks_2samp

# Misma división que el entrenamiento: el baseline es el conjunto de entrenamiento.
import datos

# Nivel de significancia clásico: 5 %.
ALFA = 0.05
# Bonferroni: 0,05 / 11 = 0,0045, para evitar falsas alarmas por hacer 11 pruebas.
ALFA_BONF = ALFA / len(datos.NUMERICAS)
# Convención de la industria: 0,10 alerta menor y 0,20 drift.
PSI_ALERTA, PSI_DRIFT = 0.10, 0.20
# Valor muy pequeño que evita dividir entre cero o calcular ln(0).
EPS = 1e-4


# PSI = suma de (actual - referencia) x ln(actual / referencia), por categoría.
def psi(ref, act):
    # Todas las categorías presentes en cualquiera de las dos muestras.
    cats = sorted(set(ref.unique()) | set(act.unique()))
    # Proporción de cada categoría en la referencia.
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + EPS
    # Proporción de cada categoría en el lote nuevo.
    q = act.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + EPS
    # Aplica la fórmula del PSI.
    return float(np.sum((q - p) * np.log(q / p)))


# Compara cada variable del lote contra el baseline y devuelve una tabla.
def evaluar(referencia: pd.DataFrame, lote: pd.DataFrame) -> pd.DataFrame:
    filas = []
    # Variables numéricas: prueba KS.
    for v in datos.NUMERICAS:
        # Devuelve D (mayor distancia entre curvas) y el p-valor.
        r = ks_2samp(referencia[v], lote[v])
        # Rojo si es muy improbable que la diferencia sea por azar.
        estado = "ROJO" if r.pvalue < ALFA_BONF else "VERDE"
        filas.append({"variable": v, "tipo": "numérica", "prueba": "KS", "estadistico": round(r.statistic, 4),
                      "p_valor": float(f"{r.pvalue:.3g}"), "umbral": f"p < {ALFA_BONF:.4f}", "estado": estado})
    # Variables categóricas: PSI.
    for v in datos.CATEGORICAS:
        valor = psi(referencia[v], lote[v])
        # Rojo desde 0,20, amarillo desde 0,10; si no, verde.
        estado = "ROJO" if valor >= PSI_DRIFT else "AMARILLO" if valor >= PSI_ALERTA else "VERDE"
        filas.append({"variable": v, "tipo": "categórica", "prueba": "PSI", "estadistico": round(valor, 4),
                      "p_valor": None, "umbral": f"PSI >= {PSI_DRIFT}", "estado": estado})
    return pd.DataFrame(filas)


# El lote queda en rojo si al menos una variable está en rojo.
def estado_global(tabla):
    return "ROJO" if (tabla.estado == "ROJO").any() else "AMARILLO" if (tabla.estado == "AMARILLO").any() else "VERDE"


# Inyecta drift a propósito para probar que el detector funciona.
def lote_derivado(lote: pd.DataFrame, semilla=7) -> pd.DataFrame:
    """Simula lo que trae el cambio de cliente Java a Ionic: plan de cuotas en toda venta, plazos más largos y tickets más altos."""
    rng = np.random.default_rng(semilla)
    d = lote.copy()
    d["con_plan"] = "si"
    largo = rng.random(len(d)) < 0.40
    d.loc[largo, "plazo_cat"] = rng.choice(["4-6", "7+"], largo.sum())
    d["monto_total"] = d["monto_total"] * 1.5
    d["pct_inicial"] = (d["pct_inicial"] + rng.uniform(0.05, 0.25, len(d))).clip(upper=1)
    return d


def main():
    baseline, mismo_origen = datos.baseline_y_mismo_origen()
    salida = Path("drift/salidas"); salida.mkdir(parents=True, exist_ok=True)
    resumen = {}
    lotes = [("mismo_origen", mismo_origen), ("derivado", lote_derivado(mismo_origen))] + list(datos.lotes_reales().items())
    for nombre, lote in lotes:
        tabla = evaluar(baseline, lote)
        tabla.to_csv(salida / f"data_drift_{nombre}.csv", index=False)
        resumen[nombre] = {"filas": len(lote), "estado": estado_global(tabla), "variables_en_rojo": tabla[tabla.estado == "ROJO"].variable.tolist(),
                           "variables_en_amarillo": tabla[tabla.estado == "AMARILLO"].variable.tolist()}
        print(f"\n=== Lote '{nombre}' ({len(lote)} filas) vs baseline ({len(baseline)} filas) ===")
        print(tabla.to_string(index=False))
        print(f"ESTADO DEL LOTE: {resumen[nombre]['estado']}  · en rojo: {resumen[nombre]['variables_en_rojo']}")
    (salida / "data_drift_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
