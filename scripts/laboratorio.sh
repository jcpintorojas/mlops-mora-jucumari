#!/usr/bin/env bash
# Laboratorio solo local: consulta los créditos del CSV y compara predicción con resultado real.
# Publica únicamente en 127.0.0.1. Requiere MLflow encendido. Detener: docker rm -f mora-lab
set -euo pipefail
cd "$(dirname "$0")/.."
DATOS_DIR="${DATOS_DIR:-/home/cesar/MifacodiDB/mora}"
ARCHIVO="${DATOS_ARCHIVO:-creditos_2026-10-31.csv}"
bash scripts/iniciar_mlflow.sh >/dev/null
docker build -q -t mora-api:2.0 api >/dev/null
docker rm -f mora-lab >/dev/null 2>&1 || true
# Los datos se montan de solo lectura; no entran en la imagen.
docker run -d --name mora-lab --network kind -p 127.0.0.1:8010:8000 -u "$(id -u):$(id -g)" \
  -v "$DATOS_DIR:/datos:ro" -e DATOS_CSV="/datos/$ARCHIVO" mora-api:2.0 >/dev/null
for i in $(seq 1 30); do curl -sf http://127.0.0.1:8010/ready >/dev/null && break; sleep 1; done
echo "Laboratorio listo: http://127.0.0.1:8010/laboratorio"
