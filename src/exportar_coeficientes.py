"""Exporta una versión logística del registro para servirla desde Node, sin Python en la API.

Guarda exportado/coeficientes.json como artefacto del mismo run y genera casos inventados con la probabilidad
que da scikit-learn, para comprobar que Node calcula lo mismo. El JSON de casos no se versiona.
USO: bash scripts/entrenar.sh src/exportar_coeficientes.py <version>
"""
import json
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

import datos

NOMBRE = "mora-credito"
version = sys.argv[1]
v = MlflowClient().get_model_version(NOMBRE, version)
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version}")
pre, reg = modelo.named_steps["pre"], modelo.named_steps["modelo"]
escala, ohe = pre.named_transformers_["num"], pre.named_transformers_["cat"]
coef = reg.coef_[0]
n = len(datos.NUMERICAS)
numericas = [{"nombre": x, "media": float(escala.mean_[i]), "escala": float(escala.scale_[i]), "coef": float(coef[i])} for i, x in enumerate(datos.NUMERICAS)]
categoricas, j = [], n
for nombre, cats in zip(datos.CATEGORICAS, ohe.categories_):
    categoricas.append({"nombre": nombre, "categorias": {str(c): float(coef[j + k]) for k, c in enumerate(cats)}})
    j += len(cats)
salida = {"formato": 1, "modelo": NOMBRE, "version": version, "run_id": v.run_id, "intercepto": float(reg.intercept_[0]), "numericas": numericas, "categoricas": categoricas}

with tempfile.TemporaryDirectory() as d, mlflow.start_run(run_id=v.run_id):
    p = Path(d) / "coeficientes.json"
    p.write_text(json.dumps(salida, indent=2))
    mlflow.log_artifact(str(p), "exportado")
print(f"versión {version} (run {v.run_id}) exportada: {len(coef)} coeficientes")

# Casos inventados: ninguna fila real.
rng = np.random.default_rng(datos.SEMILLA)
N = 300
X = pd.DataFrame({"monto_total": rng.uniform(100, 80000, N), "pct_inicial": rng.uniform(0, 1, N), "n_lineas": rng.integers(1, 8, N),
                  "n_creditos_previos": rng.integers(0, 40, N), "monto_previo": rng.uniform(0, 500000, N), "n_previos_maduros": rng.integers(0, 40, N),
                  "n_previos_mora30": rng.integers(0, 20, N), "max_dias_atraso_previo": rng.integers(0, 400, N), "saldo_abierto_previo": rng.uniform(0, 100000, N),
                  "meses_desde_primera_venta": rng.integers(0, 60, N), "n_contado_previos": rng.integers(0, 20, N),
                  "plazo_cat": rng.choice(["1", "2-3", "4-6", "7+", "dias"], N), "atraso_previo": rng.choice(["si", "no", "sin_historial"], N), "con_plan": rng.choice(["si", "no"], N)})
X = X[datos.VARIABLES]
for c in datos.NUMERICAS:
    X[c] = X[c].astype(float)
esperado = modelo.predict_proba(X)[:, 1]
casos = [{"credito": {k: (val if isinstance(val, str) else float(val)) for k, val in fila.items()}, "esperado": float(e)} for fila, e in zip(X.to_dict("records"), esperado)]
Path("api/tests/paridad.local.json").write_text(json.dumps({"modelo": salida, "casos": casos}))
print(f"{N} casos inventados para comprobar paridad")
