"""Mueve el alias @champion de mora-credito a la versión indicada (rollback/promoción del modelo).
USO: bash scripts/entrenar.sh src/alias.py <version>
"""
# sys lee la versión indicada en la línea de comandos.
import sys

# MlflowClient administra el alias.
from mlflow import MlflowClient

# Conexión al servidor MLflow.
cliente = MlflowClient()
# Mueve @champion. Es el rollback o la promoción, sin cambiar código ni imagen.
cliente.set_registered_model_alias("mora-credito", "champion", sys.argv[1])
# Consulta el resultado para confirmarlo.
v = cliente.get_model_version_by_alias("mora-credito", "champion")
# Los Pods tomarán el cambio al reiniciarse (rollout restart).
print(f"@champion -> versión {v.version} (run {v.run_id})")
