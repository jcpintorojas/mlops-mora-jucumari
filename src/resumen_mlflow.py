"""Imprime los runs del experimento ordenados por ROC AUC y el estado del Model Registry (versiones, alias, run de origen).
USO: bash scripts/entrenar.sh src/resumen_mlflow.py
"""
# Solo lectura: este script no cambia nada en MLflow.
from mlflow import MlflowClient

c = MlflowClient()
# Localiza el experimento.
exp = c.get_experiment_by_name("mora-credito")
print(f"Experimento '{exp.name}' (id {exp.experiment_id})\n")
print(f"{'run':24s} {'run_id':34s} {'algoritmo':20s} {'roc_auc':>8s} {'f1':>7s} {'accuracy':>9s}")
# Lista los runs de mejor a peor ROC AUC.
for r in c.search_runs([exp.experiment_id], order_by=["metrics.roc_auc DESC"]):
    m, t = r.data.metrics, r.data.tags
    print(f"{t['mlflow.runName']:24s} {r.info.run_id:34s} {t['algoritmo']:20s} {m['roc_auc']:8.4f} {m['f1']:7.4f} {m['accuracy']:9.4f}")
print("\nModel Registry: mora-credito")
# Lee el modelo registrado y sus alias.
modelo = c.get_registered_model("mora-credito")
# Muestra cada versión con su run de origen.
for v in sorted(c.search_model_versions("name='mora-credito'"), key=lambda v: int(v.version)):
    # Busca qué alias apuntan a esta versión.
    alias = [a for a, num in modelo.aliases.items() if num == v.version]
    print(f"  versión {v.version}  run_id {v.run_id}  alias {alias or '-'}  · {v.description}")
# La versión que corresponde al modelo desplegado.
print(f"\nmodels:/mora-credito@champion -> versión {c.get_model_version_by_alias('mora-credito', 'champion').version}")
