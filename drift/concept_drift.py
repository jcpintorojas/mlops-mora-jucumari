"""Fase 4.2 — Simulación de concept drift y criterio de reentrenamiento.

Escenario: la empresa cambia su cobranza. Desde el lote 5, una parte creciente de los clientes que antes se
atrasaban pasa a pagar a tiempo, y los que tienen saldo abierto de créditos anteriores se atrasan más. Las ENTRADAS no cambian (P(X) igual): cambia la relación
entrada -> salida, P(Y|X). Por eso el monitor de data drift sigue en verde y solo la métrica lo revela.

Se usa el modelo en producción, cargado del registro: models:/mora-credito@champion.
Criterio de reentrenamiento (clase 7, lámina 11, disparador por rendimiento):
  umbral = AUC de referencia - 2 desviaciones estándar del AUC en lotes sanos de 60 (variación normal por azar);
  alarma de reentrenamiento si el AUC queda bajo el umbral en 3 lotes CONSECUTIVOS (evita reaccionar a un lote raro).
Cada lote se registra en MLflow (experimento "monitoreo-concept-drift", métrica por step = lote).
"""
# json guarda los resultados.
import json
# Path maneja rutas de archivos.
from pathlib import Path

import matplotlib
# Agg dibuja la gráfica sin pantalla, dentro del contenedor.
matplotlib.use("Agg")
# pyplot crea la gráfica.
import matplotlib.pyplot as plt
# mlflow carga el modelo y registra las métricas por lote.
import mlflow
import numpy as np
# roc_auc_score mide el rendimiento en cada lote.
from sklearn.metrics import roc_auc_score

# Misma división que el entrenamiento.
import datos
# Reutiliza el detector para mostrar que sigue en verde.
from data_drift import estado_global, evaluar

# 12 lotes de 60 muestreados de los 446 créditos de prueba, drift desde el lote 5 y alarma con 3 lotes seguidos.
N_LOTES, TAM, INICIO_DRIFT, CONSECUTIVOS = 12, 60, 5, 3
# Porcentaje de clientes afectados por lote: un cambio gradual.
INTENSIDAD = {5: 0.15, 6: 0.30, 7: 0.45, 8: 0.60, 9: 0.70, 10: 0.80, 11: 0.85, 12: 0.90}  # fracción de clientes afectados


def main():
    # Semilla fija: la simulación es repetible.
    rng = np.random.default_rng(datos.SEMILLA)
    _, X_test, _, y_test = datos.cargar()
    # Evalúa el modelo en producción, cargado por su alias.
    modelo = mlflow.pyfunc.load_model("models:/mora-credito@champion")
    # Guarda qué versión se está evaluando.
    version = mlflow.MlflowClient().get_model_version_by_alias("mora-credito", "champion")
    # Función corta: devuelve la probabilidad de mora.
    proba = lambda X: np.asarray(modelo.predict(X))[:, 1]

    # AUC de referencia en todo el conjunto de prueba.
    auc_ref = roc_auc_score(y_test, proba(X_test))
    # 200 muestras de 60 créditos: mide cuánto varía el AUC por puro azar.
    boot = [roc_auc_score(y_test.iloc[i], proba(X_test.iloc[i]))
            for i in (rng.choice(len(X_test), TAM, replace=False) for _ in range(200))]
    # Umbral = referencia - 2 desviaciones estándar: bajo él ya no parece azar.
    sigma = float(np.std(boot)); umbral = auc_ref - 2 * sigma

    # Mezcla los clientes para formar los lotes.
    orden = rng.permutation(len(X_test))
    # seguidos cuenta los lotes consecutivos bajo el umbral.
    filas, seguidos, alarma = [], 0, None
    # Experimento separado para el monitoreo.
    mlflow.set_experiment("monitoreo-concept-drift")
    # Toda la simulación queda en un run.
    with mlflow.start_run(run_name=f"simulacion-cambio-cobranza-v{version.version}"):
        # Registra el modelo evaluado y los valores del criterio.
        mlflow.log_params({"modelo": f"mora-credito v{version.version}", "run_modelo": version.run_id,
                           "auc_referencia": round(auc_ref, 4), "sigma_auc_lote": round(sigma, 4),
                           "umbral_auc": round(umbral, 4), "lotes_consecutivos": CONSECUTIVOS, "tam_lote": TAM})
        # Procesa los lotes en orden, como si llegaran en el tiempo.
        for k in range(1, N_LOTES + 1):
            # Índices de los clientes de este lote.
            idx = rng.choice(len(X_test), TAM, replace=False)  # solo hay 446 créditos: cada lote es una muestra
            # Las entradas X no se alteran; se copia y para poder cambiarla.
            X, y = X_test.iloc[idx], y_test.iloc[idx].to_numpy().copy()
            # Fracción de clientes afectados; 0 en los lotes 1 a 4.
            f = INTENSIDAD.get(k, 0.0)
            if f:
                # Regla nueva: la relación con el historial se invierte.
                nueva = ((X["atraso_previo"] != "si") & (X["saldo_abierto_previo"] == 0)).astype(int).to_numpy()
                # Elige al azar qué clientes siguen la regla nueva.
                afectado = rng.random(len(y)) < f
                # Concept drift: cambia la etiqueta, no las entradas. Cambia P(Y|X).
                y[afectado] = nueva[afectado]
            # El modelo predice sin saber que la regla cambió.
            p = proba(X)
            # Rendimiento real del lote, con sus etiquetas.
            auc = roc_auc_score(y, p)
            # Suma si está bajo el umbral; un lote sano reinicia la cuenta.
            seguidos = seguidos + 1 if auc < umbral else 0
            # Criterio de reentrenamiento: guarda el primer lote que lo cumple.
            if seguidos >= CONSECUTIVOS and alarma is None:
                alarma = k
            # Data drift contra el mismo periodo del que salen los lotes: X no cambia, solo la etiqueta.
            dd = estado_global(evaluar(X_test, X))
            filas.append({"lote": k, "fraccion_afectada": f, "auc": round(auc, 4), "bajo_umbral": auc < umbral,
                          "lotes_seguidos": seguidos, "tasa_real_mora": round(float(y.mean()), 3),
                          "prob_media_predicha": round(float(p.mean()), 3), "data_drift": dd})
            # step=k crea la gráfica temporal en MLflow.
            mlflow.log_metrics({"auc_lote": auc, "tasa_real_mora": y.mean(), "prob_media_predicha": p.mean()}, step=k)
            print(f"lote {k:2d}  afectados {f:4.0%}  AUC {auc:.4f}  {'BAJO UMBRAL' if auc < umbral else 'ok':11s}"
                  f"  seguidos {seguidos}  mora real {y.mean():.3f}  prob. media {p.mean():.3f}  data drift {dd}")
        # Registra en qué lote saltó la alarma.
        mlflow.log_metric("lote_alarma", alarma or 0)

        sal = Path("drift/salidas"); sal.mkdir(parents=True, exist_ok=True)
        lotes = [r["lote"] for r in filas]; aucs = [r["auc"] for r in filas]
        # Gráfica temporal que exige el enunciado.
        fig, ax = plt.subplots(figsize=(10, 5))
        # Sombrea los lotes con drift activo.
        ax.axvspan(INICIO_DRIFT - 0.5, N_LOTES + 0.5, color="#f4cccc", alpha=0.35, label="Concept drift activo")
        # La curva de AUC por lote.
        ax.plot(lotes, aucs, "o-", color="#1f3864", lw=2, label="ROC AUC por lote (modelo @champion)")
        # Líneas de referencia y de umbral.
        ax.axhline(auc_ref, color="#6aa84f", ls="--", label=f"AUC de referencia = {auc_ref:.3f}")
        ax.axhline(umbral, color="#cc0000", ls="--", label=f"Umbral = referencia − 2σ = {umbral:.3f}")
        # Marca el lote en que salta la alarma.
        if alarma:
            ax.axvline(alarma, color="#cc0000", lw=2)
            ax.annotate(f"ALARMA: reentrenar\n({CONSECUTIVOS} lotes seguidos bajo el umbral)", (alarma, umbral),
                        xytext=(alarma + 0.4, auc_ref + 0.02), arrowprops=dict(arrowstyle="->", color="#cc0000"), color="#cc0000")
        ax.set(xlabel="Lote de producción (60 créditos cada uno)", ylabel="ROC AUC",
               title="Degradación del modelo por concept drift", xticks=lotes)
        ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout(); fig.savefig(sal / "concept_drift.png", dpi=150)
        # Guarda la gráfica como artefacto del run.
        mlflow.log_artifact(str(sal / "concept_drift.png"))
        # Guarda los resultados para la evidencia.
        (sal / "concept_drift.json").write_text(json.dumps(
            {"auc_referencia": auc_ref, "sigma": sigma, "umbral": umbral, "lote_alarma": alarma, "lotes": filas},
            indent=2, ensure_ascii=False, default=bool))
        print(f"\nAUC referencia {auc_ref:.4f} · σ por lote {sigma:.4f} · umbral {umbral:.4f} · ALARMA DE REENTRENAMIENTO en el lote {alarma}")


if __name__ == "__main__":
    main()
