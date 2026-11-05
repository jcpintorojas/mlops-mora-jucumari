#!/usr/bin/env bash
# Ejecuta un script de src/ dentro del entorno de experimentación, conectado al servidor MLflow.
# USO: bash scripts/entrenar.sh [src/entrenar.py]
# Detiene el script ante un error.
set -euo pipefail
# Trabaja desde la raíz del proyecto.
cd "$(dirname "$0")/.."
# Ejecuta Python en un contenedor temporal, en la red de MLflow, con el proyecto montado. Guarda el commit actual.
docker run --rm --network kind -u "$(id -u):$(id -g)" -e HOME=/tmp -e MLFLOW_TRACKING_URI=http://mlflow-mora:5000 \
  -e GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo sin-git)" \
  -e PYTHONPATH=/proyecto/src:/proyecto/drift -e DATOS_ARCHIVO="${DATOS_ARCHIVO:-}" -v "$PWD:/proyecto" -v "${DATOS_DIR:-/home/cesar/MifacodiDB/mora}:/datos:ro" riesgo-mlflow:3.14.0 python "${@:-src/entrenar.py}"
