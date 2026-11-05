"""Carga del dataset de créditos de Jucumari y división temporal (compartida por entrenamiento y drift).

Unidad: un crédito en el momento de otorgarlo. Etiqueta mora30: a los 30 días del vencimiento no estaba pagado.
Los datos NO viven en este repositorio: DATOS_RUTA apunta a un CSV seudonimizado fuera de Git.
"""
import hashlib
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RUTA = Path(os.environ.get("DATOS_RUTA", "/datos/creditos_2026-10-31.csv"))
SEMILLA = 42
OBJETIVO = "mora30"
# Corte temporal: se entrena con el pasado y se prueba con lo que vino después, como en producción.
FIN_TRAIN = "2024-06-30"
FIN_TEST = "2025-09-30"
TEST_SIZE = "temporal"
# Parte del periodo de entrenamiento reservada como "mismo origen" para la prueba de data drift en verde.
PROP_MISMO_ORIGEN = 0.20

CATEGORICAS = ["plazo_cat", "atraso_previo", "con_plan"]
NUMERICAS = ["monto_total", "pct_inicial", "n_lineas", "n_creditos_previos", "monto_previo", "n_previos_maduros",
             "n_previos_mora30", "max_dias_atraso_previo", "saldo_abierto_previo", "meses_desde_primera_venta",
             "n_contado_previos"]
VARIABLES = NUMERICAS + CATEGORICAS
# Ejemplo inventado para la firma del modelo: en MLflow nunca se guarda una fila real.
EJEMPLO = pd.DataFrame([{**{v: 1.0 for v in NUMERICAS}, "pct_inicial": 0.0, "plazo_cat": "1",
                         "atraso_previo": "sin_historial", "con_plan": "no"}])[VARIABLES]


def md5_datos():
    return hashlib.md5(RUTA.read_bytes()).hexdigest()


def tabla():
    df = pd.read_csv(RUTA, dtype={"plazo_cat": str}, parse_dates=["fecha_venta"], true_values=["t"], false_values=["f"])
    df[NUMERICAS] = df[NUMERICAS].astype(float)
    return df


def cargar():
    """X_train, X_test, y_train, y_test: solo créditos con resultado conocido y sin saldos regularizados a mano."""
    df = tabla()
    d = df[df.madura & ~df.regularizada]
    tr, te = d[d.fecha_venta <= FIN_TRAIN], d[(d.fecha_venta > FIN_TRAIN) & (d.fecha_venta <= FIN_TEST)]
    return tr[VARIABLES], te[VARIABLES], tr[OBJETIVO].astype(int), te[OBJETIVO].astype(int)


def baseline_y_mismo_origen():
    """Para drift: baseline y un lote del mismo periodo que nunca se usa como referencia."""
    X_train, _, y_train, _ = cargar()
    return train_test_split(X_train, test_size=PROP_MISMO_ORIGEN, random_state=SEMILLA, stratify=y_train)


def lotes_reales():
    """Lotes posteriores al entrenamiento, para medir drift real por periodo (incluye créditos sin etiqueta)."""
    df = tabla()
    p = df[df.fecha_venta > FIN_TRAIN]
    return {"2024-S2_2025-S1": p[p.fecha_venta <= "2025-06-30"][VARIABLES],
            "2025-S2_2026-S1": p[(p.fecha_venta > "2025-06-30") & (p.fecha_venta <= "2026-06-30")][VARIABLES],
            "2026-S2": p[p.fecha_venta > "2026-06-30"][VARIABLES]}
