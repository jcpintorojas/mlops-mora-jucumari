"""Drift real por cohorte: mora y AUC del modelo por semestre de otorgamiento, con las etiquetas que ya se conocen.
Muestra el retraso de etiquetas: un crédito de 1 mes tarda 2 meses en dar resultado; uno de 12, 13 meses.
"""
import json
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

import datos

df = datos.tabla()
df = df[df.madura & ~df.regularizada & (df.fecha_venta > datos.FIN_TRAIN)].copy()
modelo = mlflow.pyfunc.load_model("models:/mora-credito@champion")
df["p"] = np.asarray(modelo.predict(df[datos.VARIABLES]))[:, 1]
df["semestre"] = df.fecha_venta.dt.year.astype(str) + "-S" + ((df.fecha_venta.dt.month > 6) + 1).astype(str)
tasas = None
X_train, _, y_train, _ = datos.cargar()
tasas = y_train.groupby(X_train["atraso_previo"]).mean()
df["r"] = df.atraso_previo.map(tasas)
filas = []
for s, g in df.groupby("semestre"):
    y = g[datos.OBJETIVO].astype(int)
    filas.append({"semestre": s, "n": len(g), "mora_real": round(float(y.mean()), 3), "prob_media": round(float(g.p.mean()), 3),
                  "auc": round(float(roc_auc_score(y, g.p)), 3) if y.nunique() == 2 and y.sum() >= 10 else None,
                  "auc_regla": round(float(roc_auc_score(y, g.r)), 3) if y.nunique() == 2 and y.sum() >= 10 else None,
                  "mora_por_historial": {k: round(float(v), 3) for k, v in y.groupby(g.atraso_previo).mean().items()}})
    print(filas[-1])
Path("drift/salidas").mkdir(parents=True, exist_ok=True)
Path("drift/salidas/cohortes.json").write_text(json.dumps(filas, indent=2))
