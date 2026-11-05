"""Candidata v4: usa todos los créditos con resultado conocido hasta hoy.

Como el modelo final ya no deja datos futuros para probarlo, primero se evalúa el MISMO procedimiento hacia adelante
(walk-forward): en cada trimestre solo se usa lo que ya se sabía al empezarlo. Después se entrena el modelo final.
USO: DATOS_ARCHIVO=creditos_AAAA-MM-DD.csv bash scripts/entrenar.sh src/entrenar_v4.py
"""
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from skops.io import dumps, get_untrusted_types

import datos
from entrenar import con_preproceso

datos.RUTA = Path("/datos") / (os.environ.get("DATOS_ARCHIVO") or "creditos_2026-11-05.csv")
ETIQUETA, S, VIDA_MEDIA, VENTANA = "atraso_venc", datos.SEMILLA, 365.0, 180
df = datos.tabla()
df["fecha_resultado"] = pd.to_datetime(df.fecha_resultado)
df = df[df.madura & ~df.regularizada].copy()
df[ETIQUETA] = df[ETIQUETA].astype(int)
HOY = pd.Timestamp(datos.RUTA.stem.split("_")[-1])


def ajustar(conocido, fecha):
    """Entrena con lo viejo, dando más peso a lo reciente, y ajusta las probabilidades con los últimos 180 días."""
    corte = fecha - pd.Timedelta(days=VENTANA)
    base_d, cal_d = conocido[conocido.fecha_resultado < corte], conocido[conocido.fecha_resultado >= corte]
    pesos = 0.5 ** ((fecha - base_d.fecha_venta).dt.days.to_numpy() / VIDA_MEDIA)
    base = con_preproceso(LogisticRegression(C=0.1, max_iter=2000, random_state=S)).fit(base_d[datos.VARIABLES], base_d[ETIQUETA], modelo__sample_weight=pesos)
    modelo = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(cal_d[datos.VARIABLES], cal_d[ETIQUETA])
    return modelo, len(base_d), len(cal_d), int(cal_d[ETIQUETA].sum())


filas, acum = [], []
for ini in pd.date_range("2025-07-01", HOY, freq="QS"):
    fin = ini + pd.offsets.QuarterBegin(startingMonth=1)
    conocido = df[df.fecha_resultado < ini]
    te = df[(df.fecha_venta >= ini) & (df.fecha_venta < fin)]
    if te.empty:
        continue
    modelo, nb, nc, ac = ajustar(conocido, ini)
    tasas = conocido.groupby("atraso_previo")[ETIQUETA].mean()
    pm, pr = modelo.predict_proba(te[datos.VARIABLES])[:, 1], te.atraso_previo.map(tasas).fillna(conocido[ETIQUETA].mean()).to_numpy()
    y = te[ETIQUETA].to_numpy()
    acum.append(pd.DataFrame({"y": y, "pm": pm, "pr": pr, "t": str(ini.date())}))
    auc = lambda p: round(float(roc_auc_score(y, p)), 3) if 5 <= y.sum() <= len(y) - 5 else None
    filas.append({"trimestre": str(ini.date()), "creditos": len(y), "atrasos": int(y.sum()), "atraso_real": round(float(y.mean()), 3), "prob_media": round(float(pm.mean()), 3), "auc_modelo": auc(pm), "auc_regla": auc(pr), "train": nb, "ajuste": nc, "atrasos_ajuste": ac})
    print(filas[-1])

t = pd.concat(acum); yt, pm, pr = t.y.to_numpy(), t.pm.to_numpy(), t.pr.to_numpy()
rng = np.random.default_rng(S); dif = []
while len(dif) < 2000:
    i = rng.integers(0, len(yt), len(yt))
    dif.append(roc_auc_score(yt[i], pm[i]) - roc_auc_score(yt[i], pr[i]))
marca = pm >= np.quantile(pm, 0.75)
res = {"creditos": len(yt), "atrasos": int(yt.sum()), "atraso_real": float(yt.mean()), "prob_media": float(pm.mean()), "auc_modelo": float(roc_auc_score(yt, pm)), "auc_regla": float(roc_auc_score(yt, pr)),
       "dif_ic95": [float(x) for x in np.percentile(dif, [2.5, 97.5])], "atraso_en_top25": float(yt[marca].mean()), "captura_top25": float(yt[marca].sum() / yt.sum())}

modelo, nb, nc, ac = ajustar(df, HOY)
mlflow.set_experiment("mora-credito-v4")
with mlflow.start_run(run_name="v4_logreg_C-0.1_hasta_hoy") as run:
    mlflow.log_params({"C": 0.1, "semilla": S, "vida_media_dias": VIDA_MEDIA, "ventana_ajuste_dias": VENTANA, "etiqueta": ETIQUETA, "corte": str(HOY.date())})
    mlflow.log_metrics({"roc_auc_walk_forward": res["auc_modelo"], "roc_auc_regla_walk_forward": res["auc_regla"]})
    mlflow.set_tags({"algoritmo": "regresion-logistica", "datos_md5": datos.md5_datos(), "filas_base": nb, "filas_ajuste": nc, "git_commit": os.environ.get("GIT_COMMIT", "desconocido")})
    firma = infer_signature(datos.EJEMPLO, modelo.predict_proba(datos.EJEMPLO)[:, 1])
    mlflow.sklearn.log_model(sk_model=modelo, name="modelo", signature=firma, input_example=datos.EJEMPLO, pyfunc_predict_fn="predict_proba", skops_trusted_types=get_untrusted_types(data=dumps(modelo)))
    run_id = run.info.run_id
v = mlflow.register_model(f"runs:/{run_id}/modelo", "mora-credito")
c = MlflowClient()
c.update_model_version("mora-credito", v.version, f"Candidata con datos hasta {HOY.date()}. No es @champion.")
c.set_registered_model_alias("mora-credito", "candidato", v.version)
res.update(version=v.version, run_id=run_id, filas_base=nb, filas_ajuste=nc, atrasos_ajuste=ac, trimestres=filas)
print(json.dumps({k: x for k, x in res.items() if k != "trimestres"}, indent=2))
Path("drift/salidas").mkdir(parents=True, exist_ok=True); Path("drift/salidas/v4.json").write_text(json.dumps(res, indent=2))
