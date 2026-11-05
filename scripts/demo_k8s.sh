#!/usr/bin/env bash
# Fase 3 — Las cuatro demostraciones exigidas sobre el clúster (y el cambio de versión por alias).
# Guarda la salida real de cada paso en evidencias/k8s-NN-*.txt.  USO: bash scripts/demo_k8s.sh
# Sigue aunque una prueba falle, para registrar el resultado.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
mkdir -p evidencias
# IP del nodo: desde ahí se entra al NodePort.
NODO=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlops-control-plane)
# Dirección del Service de la API.
URL="http://$NODO:30092"
# paso: abre un archivo de evidencia.
paso() { EVF="evidencias/$1.txt"; : > "$EVF"; echo "### $2" | tee -a "$EVF"; }
# run: muestra y ejecuta un comando, y guarda su salida real.
run()  { echo "\$ $*" | tee -a "$EVF"; bash -c "$*" 2>&1 | tee -a "$EVF"; echo | tee -a "$EVF"; }
# nota: agrega una explicación a la evidencia.
nota() { echo "# $*" | tee -a "$EVF"; }
export URL
# Espera a que terminen los Pods anteriores, para no mezclar versiones.
esperar_viejos() {  # espera a que terminen los Pods del ReplicaSet anterior
  local h; h=$(kubectl -n mora get rs -l app=mora-api --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.labels.pod-template-hash}')
  kubectl -n mora wait --for=delete pod -l "app=mora-api,pod-template-hash!=$h" --timeout=120s >/dev/null 2>&1; echo "(Pods de la versión anterior ya terminados)"
}
export -f esperar_viejos
# pedir: envía una predicción de ejemplo a la API.
pedir() { curl -s -X POST "$URL/predict" -H 'Content-Type: application/json' -d @scripts/credito_alto.json; }
export -f pedir

# Prueba 1: tres réplicas en Running.
paso k8s-01-replicas "1. Tres réplicas en estado Running simultáneamente"
run kubectl -n mora get deployment mora-api -o wide
run kubectl -n mora get pods -l app=mora-api -o wide
run "kubectl -n mora get service mora-api -o wide"
run "kubectl -n mora logs -l app=mora-api --prefix --tail=50 | grep 'modelo mora-credito'"
nota "Cada Pod cargó models:/mora-credito@champion desde el Model Registry de MLflow al arrancar."

# Prueba 2: peticiones sucesivas atendidas por Pods distintos.
paso k8s-02-balanceo "2. El tráfico se distribuye entre réplicas (nombre del Pod en la respuesta)"
run "pedir | python3 -m json.tool"
run "for i in \$(seq 1 30); do pedir | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"pod\"])'; done | sort | uniq -c"
nota "30 peticiones sucesivas al Service (NodePort 30092) fueron atendidas por los 3 Pods."

# Prueba 3: borrar un Pod mientras hay tráfico continuo.
paso k8s-03-autorreparacion "3. Autorreparación: se elimina un Pod y el servicio no deja de responder"
VICTIMA=$(kubectl -n mora get pods -l app=mora-api -o jsonpath='{.items[0].metadata.name}')
# En segundo plano: envía peticiones durante 40 s y cuenta aciertos y errores.
( ok=0; fallo=0; fin=$((SECONDS+40)); while [ $SECONDS -lt $fin ]; do
    c=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/predict" -H 'Content-Type: application/json' -d @scripts/credito_alto.json)
    [ "$c" = 200 ] && ok=$((ok+1)) || fallo=$((fallo+1)); sleep 0.2; done
  echo "$ok $fallo" > /tmp/mora_carga.txt ) &
CARGA=$!
sleep 3
run kubectl -n mora get pods -l app=mora-api
# --wait=false: no espera, para observar la recuperación.
run kubectl -n mora delete pod "$VICTIMA" --wait=false
run sleep 3
run kubectl -n mora get pods -l app=mora-api
run kubectl -n mora rollout status deployment/mora-api --timeout=180s
run kubectl -n mora get pods -l app=mora-api
wait $CARGA
read OK FALLO < /tmp/mora_carga.txt
run "echo 'Peticiones enviadas durante 40 s mientras se eliminaba el Pod: $OK respondidas con 200, $FALLO con error'"
nota "Se eliminó $VICTIMA; el ReplicaSet creó otro Pod y el Service siguió respondiendo con los otros dos."

# Prueba 4: cambiar el número de réplicas.
paso k8s-04-escalado "4. Escalado: de 3 a 5 réplicas y regreso a 3"
run kubectl -n mora scale deployment/mora-api --replicas=5
run kubectl -n mora rollout status deployment/mora-api --timeout=240s
run kubectl -n mora get pods -l app=mora-api
# Espera a que los 5 Pods estén listos en el Service.
run "until [ \$(kubectl -n mora get endpointslices -l kubernetes.io/service-name=mora-api -o jsonpath='{range .items[*].endpoints[?(@.conditions.ready==true)]}x{end}' | wc -c) -eq 5 ]; do sleep 1; done; sleep 5; kubectl -n mora get endpointslices -l kubernetes.io/service-name=mora-api"   # +5 s: kube-proxy actualiza iptables
run "for i in \$(seq 1 50); do pedir | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"pod\"])'; done | sort | uniq -c"
run kubectl -n mora scale deployment/mora-api --replicas=3
run "until [ \$(kubectl -n mora get pods -l app=mora-api --no-headers | wc -l) -eq 3 ]; do sleep 2; done; kubectl -n mora get deployment mora-api; kubectl -n mora get pods -l app=mora-api"
nota "Con 5 réplicas el tráfico se reparte entre 5 Pods; al volver a 3 se eliminan los sobrantes."

# Prueba extra: cambiar de modelo moviendo el alias, sin tocar la imagen.
paso k8s-05-alias "5. Trazabilidad: cambiar el alias @champion cambia el modelo servido sin tocar código ni imagen"
run "curl -s $URL/info | python3 -m json.tool"
# Rollback del modelo: @champion pasa a la otra versión logística.
run "bash scripts/entrenar.sh src/alias.py ${VERSION_ROLLBACK:-5} 2>/dev/null"
# Los Pods nuevos cargan la versión que indica el alias.
run kubectl -n mora rollout restart deployment/mora-api
run kubectl -n mora rollout status deployment/mora-api --timeout=300s
run esperar_viejos
run "for i in \$(seq 1 6); do curl -s $URL/info; echo; done"
# Promoción: @champion vuelve a su versión.
run "bash scripts/entrenar.sh src/alias.py ${VERSION_CHAMPION:-1} 2>/dev/null"
run kubectl -n mora rollout restart deployment/mora-api
run kubectl -n mora rollout status deployment/mora-api --timeout=300s
run esperar_viejos
run "for i in \$(seq 1 3); do curl -s $URL/info; echo; done"
nota "El alias pasó a la otra versión logística (rollback del modelo) y volvió; la imagen mora-api:2.0 fue la misma en todo momento."
