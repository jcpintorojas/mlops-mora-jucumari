#!/usr/bin/env bash
# Construye la imagen de la API, la carga en el clúster kind "mlops" y aplica todos los manifiestos.
# Requisitos: servidor MLflow arriba (scripts/iniciar_mlflow.sh) con el modelo registrado y alias @champion.
# Detiene el script ante un error.
set -euo pipefail
# Trabaja desde la raíz del proyecto.
cd "$(dirname "$0")/.."
# Permite encontrar kind y kubectl -n mora.
export PATH=$HOME/.local/bin:$PATH
# Construye la imagen de la API.
docker build -t mora-api:2.0 api
# Copia la imagen al nodo de kind; no hay un registro externo.
kind load docker-image mora-api:2.0 --name mlops
kubectl create namespace mora --dry-run=client -o yaml | kubectl apply -f -
# Crea o actualiza la API y su Service.
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
# Crea el ConfigMap con la página y la configuración de nginx.
kubectl -n mora create configmap mora-ui --from-file=ui/index.html --from-file=ui/default.conf --dry-run=client -o yaml | kubectl -n mora apply -f -
# Despliega la interfaz.
kubectl apply -f k8s/ui.yaml
# Reinicia los Pods para que tomen la imagen y el alias actuales.
kubectl -n mora rollout restart deployment/mora-api
# Espera a que las réplicas estén listas.
kubectl -n mora rollout status deployment/mora-api --timeout=300s
kubectl -n mora rollout status deployment/mora-ui --timeout=120s
# Espera a que terminen los Pods anteriores.
while kubectl -n mora get pods --no-headers | grep -q Terminating; do sleep 2; done
# Muestra el resultado.
kubectl -n mora get pods -o wide
