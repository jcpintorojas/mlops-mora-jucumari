"""Candidata v3: reentrenamiento tras el drift real. Etiqueta: terminó de pagar después del vencimiento.

Pasado -> entrena (hasta 2025-S1, con más peso a lo reciente) · 2025-S2 -> elige modelo y ajusta probabilidades ·
2026 -> evaluación final una sola vez. Registra la versión con alias @candidato: @champion no se mueve.
USO: bash scripts/entrenar.sh src/entrenar_v3.py
"""
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from skops.io import dumps, get_untrusted_types

import datos
from entrenar import con_preproceso

datos.RUTA = Path("/datos/creditos_v3_2026-10-31.csv")
ETIQUETA, S, VIDA_MEDIA = "atraso_venc", datos.SEMILLA, 365.0
FIN_TRAIN, FIN_VAL = "2025-06-30", "2025-12-31"

df = datos.tabla()
df = df[df.madura & ~df.regularizada]
tr, va, te = df[df.fecha_venta <= FIN_TRAIN], df[(df.fecha_venta > FIN_TRAIN) & (df.fecha_venta <= FIN_VAL)], df[df.fecha_venta > FIN_VAL]
y = lambda d: d[ETIQUETA].astype(int).to_numpy()
pesos = 0.5 ** ((tr.fecha_venta.max() - tr.fecha_venta).dt.days.to_numpy() / VIDA_MEDIA)
print(f"train {len(tr)} ({y(tr).sum()} atrasos) · validación {len(va)} ({y(va).sum()}) · prueba {len(te)} ({y(te).sum()})")

tasas = tr.assign(e=y(tr)).groupby("atraso_previo").e.mean()
regla = lambda d: d.atraso_previo.map(tasas).to_numpy()
CONFIG = [("regresion-logistica", "v3_logreg_C-0.1", lambda: LogisticRegression(C=0.1, max_iter=2000, random_state=S), {"C": 0.1}),
          ("regresion-logistica", "v3_logreg_C-1", lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=S), {"C": 1.0}),
          ("random-forest", "v3_rf_depth-4", lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=S, n_jobs=-1), {"n_estimators": 300, "max_depth": 4}),
          ("gradient-boosting", "v3_gb_depth-2", lambda: GradientBoostingClassifier(learning_rate=0.05, n_estimators=100, max_depth=2, subsample=0.8, random_state=S), {"learning_rate": 0.05, "n_estimators": 100, "max_depth": 2})]

mlflow.set_experiment("mora-credito-v3")
md5, res = datos.md5_datos(), []
for algoritmo, nombre, crear, params in CONFIG:
    with mlflow.start_run(run_name=nombre) as run:
        base = con_preproceso(crear()).fit(tr[datos.VARIABLES], y(tr), modelo__sample_weight=pesos)
        auc_val = roc_auc_score(y(va), base.predict_proba(va[datos.VARIABLES])[:, 1])
        # Ajusta las probabilidades a la tasa reciente usando solo validación.
        modelo = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(va[datos.VARIABLES], y(va))
        firma = infer_signature(datos.EJEMPLO, modelo.predict_proba(datos.EJEMPLO)[:, 1])
        mlflow.log_params({**params, "semilla": S, "vida_media_dias": VIDA_MEDIA, "etiqueta": ETIQUETA})
        mlflow.log_metric("roc_auc_validacion", auc_val)
        mlflow.set_tags({"algoritmo": algoritmo, "datos_md5": md5, "git_commit": os.environ.get("GIT_COMMIT", "desconocido")})
        mlflow.sklearn.log_model(sk_model=modelo, name="modelo", signature=firma, input_example=datos.EJEMPLO, pyfunc_predict_fn="predict_proba",
                                 skops_trusted_types=get_untrusted_types(data=dumps(modelo)))
        res.append((auc_val, nombre, run.info.run_id, modelo))
        print(f"{nombre:20s} AUC validación {auc_val:.4f}")

# Elección con validación, no con la prueba.
auc_val, nombre, run_id, modelo = max(res, key=lambda r: r[0])
yt, pm, pr = y(te), modelo.predict_proba(te[datos.VARIABLES])[:, 1], regla(te)
rng = np.random.default_rng(S); dif = []
while len(dif) < 2000:
    i = rng.integers(0, len(yt), len(yt))
    if yt[i].min() != yt[i].max():
        dif.append(roc_auc_score(yt[i], pm[i]) - roc_auc_score(yt[i], pr[i]))
marcados = pm >= np.quantile(pm, 0.75)
final = {"elegido": nombre, "run_id": run_id, "n_prueba": len(yt), "atrasos_prueba": int(yt.sum()), "tasa_real": float(yt.mean()), "prob_media_modelo": float(pm.mean()),
         "auc_modelo": float(roc_auc_score(yt, pm)), "auc_regla": float(roc_auc_score(yt, pr)), "dif_ic95": [float(x) for x in np.percentile(dif, [2.5, 97.5])],
         "brier_modelo": float(brier_score_loss(yt, pm)), "brier_tasa_constante": float(brier_score_loss(yt, np.full(len(yt), y(va).mean()))),
         "mora_en_top25": float(yt[marcados].mean()), "captura_top25": float(yt[marcados].sum() / yt.sum())}
with mlflow.start_run(run_name="regla_historial_v3"):
    mlflow.set_tag("algoritmo", "regla"); mlflow.log_metric("roc_auc_prueba_2026", final["auc_regla"])
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics({"roc_auc_prueba_2026": final["auc_modelo"], "brier_prueba_2026": final["brier_modelo"]})
v = mlflow.register_model(f"runs:/{run_id}/modelo", "mora-credito")
c = MlflowClient()
c.update_model_version("mora-credito", v.version, f"Candidata tras drift real: {nombre}, etiqueta {ETIQUETA}. No es @champion.")
c.set_registered_model_alias("mora-credito", "candidato", v.version)
final["version"] = v.version
print(json.dumps(final, indent=2))
Path("drift/salidas").mkdir(parents=True, exist_ok=True); Path("drift/salidas/v3.json").write_text(json.dumps(final, indent=2))
