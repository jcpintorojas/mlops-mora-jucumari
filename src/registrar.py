"""Fase 1 — Registra modelos en el Model Registry de MLflow y marca la versión que se despliega.

- Primero la mejor regresión logística y después el mejor run de todo el experimento según ROC AUC.
- El alias "champion" apunta a la logística: Node sirve sus coeficientes (src/exportar_coeficientes.py) y empata con el mejor.
  El servicio carga models:/mora-credito@champion, nunca una ruta de archivo. Mover el alias cambia el modelo servido sin tocar código.
USO: bash scripts/entrenar.sh src/registrar.py
"""
# mlflow registra un modelo a partir de un run.
import mlflow
# MlflowClient administra versiones y alias.
from mlflow import MlflowClient

# Nombre estable del modelo en el registro.
NOMBRE = "mora-credito"
# Etiqueta que señala la versión en producción.
ALIAS = "champion"


# Busca los runs, ordenados por ROC AUC, y devuelve el mejor.
def mejor_run(cliente, exp_id, filtro=""):
    return cliente.search_runs([exp_id], filter_string=filtro, order_by=["metrics.roc_auc DESC"], max_results=1)[0]


def main():
    # Conexión al servidor (usa MLFLOW_TRACKING_URI).
    cliente = MlflowClient()
    # Localiza el experimento de entrenamiento.
    exp = cliente.get_experiment_by_name("mora-credito")
    # Versión 1: la mejor regresión logística, como modelo base.
    base = mejor_run(cliente, exp.experiment_id, "tags.algoritmo = 'regresion-logistica'")
    # Versión 2: el mejor AUC de todos los runs.
    mejor = mejor_run(cliente, exp.experiment_id, "tags.algoritmo != 'regla'")
    # Registra primero la base y después el mejor, en ese orden.
    for run, descripcion in [(base, "Modelo base: regresión logística"), (mejor, "Mejor ROC AUC del experimento")]:
        # Crea una versión nueva enlazada a su run: ese es el linaje.
        v = mlflow.register_model(f"runs:/{run.info.run_id}/modelo", NOMBRE)
        # Describe la versión para verla en la interfaz.
        cliente.update_model_version(NOMBRE, v.version, f"{descripcion} — run {run.data.tags['mlflow.runName']} "
                                     f"(AUC {run.data.metrics['roc_auc']:.4f})")
        # Muestra qué run quedó en cada versión.
        print(f"v{v.version} <- {run.data.tags['mlflow.runName']} run_id={run.info.run_id} AUC={run.data.metrics['roc_auc']:.4f}")
        # Recuerda la primera versión registrada: la logística.
        if run is base:
            elegida = v.version
    # Marca la logística como @champion: la que cargará la API.
    cliente.set_registered_model_alias(NOMBRE, ALIAS, elegida)
    # Describe el modelo registrado.
    cliente.update_registered_model(NOMBRE, "Probabilidad de mora de un crédito al otorgarlo (datos seudonimizados de Jucumari).")
    # Confirma a qué versión apunta el alias.
    print(f"alias @{ALIAS} -> versión {cliente.get_model_version_by_alias(NOMBRE, ALIAS).version}")


if __name__ == "__main__":
    main()
