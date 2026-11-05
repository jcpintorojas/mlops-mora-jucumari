"""Fase 1 — Entrena modelos de mora en créditos de Jucumari y registra cada ejecución en MLflow.

Cada combinación de algoritmo + hiperparámetros es un run. El piso a superar es la regla de negocio
"cómo pagó su crédito anterior" (run regla_historial), con tasas aprendidas solo del entrenamiento.
USO: bash scripts/entrenar.sh src/entrenar.py
"""
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from skops.io import dumps, get_untrusted_types
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import datos

EXPERIMENTO = "mora-credito"


def con_preproceso(modelo):
    """Codifica las categóricas y escala las numéricas dentro del propio modelo: la API recibe datos crudos."""
    pre = ColumnTransformer([("num", StandardScaler(), datos.NUMERICAS),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), datos.CATEGORICAS)])
    return Pipeline([("pre", pre), ("modelo", modelo)])


S = datos.SEMILLA
CONFIGURACIONES = [
    ("regresion-logistica", "logreg_C-0.01", lambda: LogisticRegression(C=0.01, max_iter=2000, random_state=S), {"C": 0.01}),
    ("regresion-logistica", "logreg_C-1.0", lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=S), {"C": 1.0}),
    ("random-forest", "rf_n-100_depth-4", lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=S, n_jobs=-1), {"n_estimators": 100, "max_depth": 4}),
    ("random-forest", "rf_n-300_depth-8", lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=S, n_jobs=-1), {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5}),
    ("gradient-boosting", "gb_lr-0.1_depth-2", lambda: GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=2, random_state=S), {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 2}),
    ("gradient-boosting", "gb_lr-0.05_depth-3", lambda: GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, max_depth=3, subsample=0.8, random_state=S), {"learning_rate": 0.05, "n_estimators": 200, "max_depth": 3, "subsample": 0.8}),
]


def metricas(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y, proba), "f1": f1_score(y, pred), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred)}


def etiquetas(algoritmo, md5, X_train, X_test):
    return {"algoritmo": algoritmo, "dataset": "jucumari-creditos-seudonimizado", "datos_md5": md5,
            "filas_train": len(X_train), "filas_test": len(X_test), "corte_train": datos.FIN_TRAIN,
            "git_commit": os.environ.get("GIT_COMMIT", "desconocido")}


def main():
    np.random.seed(S)
    X_train, X_test, y_train, y_test = datos.cargar()
    mlflow.set_experiment(EXPERIMENTO)
    md5 = datos.md5_datos()
    print(f"train {len(X_train)} ({y_train.mean():.1%} mora) · test {len(X_test)} ({y_test.mean():.1%} mora)")

    # Piso de negocio: la regla, sin modelo que empaquetar.
    tasas = y_train.groupby(X_train["atraso_previo"]).mean()
    with mlflow.start_run(run_name="regla_historial"):
        m = metricas(y_test, X_test["atraso_previo"].map(tasas).to_numpy())
        mlflow.log_params({f"tasa_{k}": round(v, 4) for k, v in tasas.items()})
        mlflow.log_metrics(m)
        mlflow.set_tags(etiquetas("regla", md5, X_train, X_test))
        print(f"{'regla_historial':22s} AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  tasas={tasas.round(3).to_dict()}")

    for algoritmo, nombre, crear, params in CONFIGURACIONES:
        with mlflow.start_run(run_name=nombre) as run:
            modelo = con_preproceso(crear()).fit(X_train, y_train)
            m = metricas(y_test, modelo.predict_proba(X_test)[:, 1])
            mlflow.log_params({**params, "semilla": S, "division": datos.TEST_SIZE})
            mlflow.log_metrics(m)
            mlflow.log_metric("roc_auc_train", roc_auc_score(y_train, modelo.predict_proba(X_train)[:, 1]))
            mlflow.set_tags(etiquetas(algoritmo, md5, X_train, X_test))
            firma = infer_signature(datos.EJEMPLO, modelo.predict_proba(datos.EJEMPLO)[:, 1])
            # input_example inventado: el artefacto no lleva ninguna fila real.
            mlflow.sklearn.log_model(sk_model=modelo, name="modelo", signature=firma, input_example=datos.EJEMPLO,
                                     pyfunc_predict_fn="predict_proba",
                                     skops_trusted_types=get_untrusted_types(data=dumps(modelo)))
            print(f"{nombre:22s} run={run.info.run_id[:8]}  AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
