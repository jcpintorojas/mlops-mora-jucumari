"""¿El modelo supera de verdad a la regla de negocio? Bootstrap de la diferencia de AUC en prueba, por lo global y sin clientes vistos.
USO: bash scripts/entrenar.sh src/comparar_regla.py
"""
import json
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

import datos

S = np.random.default_rng(datos.SEMILLA)
X_train, X_test, y_train, y_test = datos.cargar()
modelo = mlflow.pyfunc.load_model("models:/mora-credito@champion")
pm = np.asarray(modelo.predict(X_test))[:, 1]
pr = X_test["atraso_previo"].map(y_train.groupby(X_train["atraso_previo"]).mean()).to_numpy()
y = y_test.to_numpy()
dif = []
for _ in range(2000):
    i = S.integers(0, len(y), len(y))
    dif.append(roc_auc_score(y[i], pm[i]) - roc_auc_score(y[i], pr[i]))
lo, hi = np.percentile(dif, [2.5, 97.5])
# Si marcamos el 25 % de créditos más riesgosos, cuántos atrasos reales se capturan.
corte = np.quantile(pm, 0.75)
marcados = pm >= corte
res = {"n_test": int(len(y)), "tasa_base": float(y.mean()), "auc_modelo": float(roc_auc_score(y, pm)), "auc_regla": float(roc_auc_score(y, pr)),
       "dif_auc": float(np.mean(dif)), "ic95": [float(lo), float(hi)], "pct_mora_en_top25": float(y[marcados].mean()), "captura_top25": float(y[marcados].sum() / y.sum())}
print(json.dumps(res, indent=2))
Path("drift/salidas").mkdir(parents=True, exist_ok=True)
Path("drift/salidas/comparar_regla.json").write_text(json.dumps(res, indent=2))
