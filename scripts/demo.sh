#!/usr/bin/env bash
# Levanta TODO para la presentación con un solo comando (el clúster y MLflow ya deben haberse creado una vez).
#   bash scripts/demo.sh
# Falla ante variables no definidas, pero sigue si un paso opcional falla.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
echo "1) Clúster Kubernetes (kind 'mlops')"
# Enciende el nodo del clúster si estaba apagado.
docker start mlops-control-plane >/dev/null 2>&1 || true
# Espera a que el nodo esté Ready.
until kubectl get nodes 2>/dev/null | grep -q " Ready"; do sleep 2; done
kubectl get nodes
echo "2) Servidor MLflow"
# Tras reiniciar la PC, MLflow no arranca solo.
bash scripts/iniciar_mlflow.sh
echo "3) Pods de la API y de la interfaz"
# Crea Pods nuevos, que vuelven a pedir el modelo @champion.
kubectl -n mora rollout restart deployment/mora-api >/dev/null   # que vuelvan a leer @champion del registro
kubectl -n mora rollout status deployment/mora-api --timeout=300s
kubectl -n mora rollout status deployment/mora-ui --timeout=120s
while kubectl -n mora get pods --no-headers | grep -q Terminating; do sleep 2; done   # que no se vean Pods viejos
kubectl -n mora get pods -o wide
# Obtiene la IP del nodo para armar las direcciones.
NODO=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlops-control-plane)
echo
echo "Listo:"
echo "  MLflow (experimentos y Model Registry): http://127.0.0.1:5001"
echo "  Interfaz web (consume la API en K8s):   http://$NODO:30093"
echo "  API y sus endpoints:                   http://$NODO:30092/"
echo "  Prueba rápida de balanceo:"
echo "    for i in \$(seq 1 10); do curl -s -X POST http://$NODO:30092/predict -H 'Content-Type: application/json' -d @scripts/credito_alto.json; echo; done"
# Abre el navegador si el sistema lo permite.
command -v xdg-open >/dev/null && { xdg-open http://127.0.0.1:5001 >/dev/null 2>&1; xdg-open "http://$NODO:30093" >/dev/null 2>&1; } || true
