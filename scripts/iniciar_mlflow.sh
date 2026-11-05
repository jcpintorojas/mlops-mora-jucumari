#!/usr/bin/env bash
# Levanta el servidor de MLflow (tracking + Model Registry + artifact store) en un contenedor.
# Se conecta a la red Docker "kind" para que los Pods del clúster lo encuentren como http://mlflow-mora:5000.
# Interfaz web: http://127.0.0.1:5001  (usar 127.0.0.1: "localhost" resuelve a IPv6 y la red de kind no lo publica)
# Detiene el script ante un error o una variable no definida.
set -euo pipefail
# Trabaja desde la raíz del proyecto.
cd "$(dirname "$0")/.."
# Carpeta donde se conservan la base de datos y los modelos.
mkdir -p mlflow-data
# Si el contenedor ya existe, lo enciende; si no, lo crea.
if docker ps -a --format '{{.Names}}' | grep -qx mlflow-mora; then
  docker start mlflow-mora >/dev/null
else
  # --network kind: los Pods lo encuentran como mlflow. -p: publica la interfaz. -v: conserva los datos.
  docker run -d --name mlflow-mora --network kind -p 5001:5000 \
    -u "$(id -u):$(id -g)" -v "$PWD/mlflow-data:/mlflow" \
    riesgo-mlflow:3.14.0 \
    mlflow server --host 0.0.0.0 --port 5000 \
      --backend-store-uri sqlite:////mlflow/mlflow.db \
      --artifacts-destination /mlflow/artifacts --serve-artifacts \
      --allowed-hosts "localhost,localhost:5001,127.0.0.1:5001,mlflow-mora,mlflow-mora:5000" >/dev/null
fi
# Espera hasta 60 s a que MLflow responda.
for i in $(seq 1 60); do curl -sf http://127.0.0.1:5001/health >/dev/null && break; sleep 1; done
echo "MLflow listo en http://127.0.0.1:5001 ($(curl -s http://127.0.0.1:5001/version))"
