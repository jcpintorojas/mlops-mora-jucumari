# Operación

## Arrancar
```bash
export PATH=$HOME/.local/bin:$PATH
bash scripts/demo.sh
```
Enciende el clúster y MLflow, reinicia los Pods e imprime las direcciones. Usa `127.0.0.1`, no `localhost`.

| Servicio | Dirección |
|---|---|
| MLflow | http://127.0.0.1:5001 |
| Interfaz | http://IP_DEL_NODO:30093 |
| API y sus endpoints | http://IP_DEL_NODO:30092/ |

## Comandos
```bash
kubectl -n mora get pods                                    # estado
kubectl -n mora logs -l app=mora-api --tail=20              # qué modelo cargó cada Pod
bash scripts/demo_k8s.sh                                    # regenera las evidencias de Kubernetes
bash scripts/entrenar.sh -m pytest -v drift/test_drift.py   # data drift
bash scripts/entrenar.sh drift/concept_drift.py             # concept drift
bash scripts/entrenar.sh drift/cohortes.py                  # drift real
```

## Laboratorio solo local
```bash
bash scripts/laboratorio.sh      # http://127.0.0.1:8010/laboratorio
docker rm -f mora-lab            # detener
```
Muestra créditos seudonimizados: no lo uses en la defensa ni en capturas.

## Cambiar de modelo
Solo sirven regresiones logísticas exportadas (versiones 1 y 5):
```bash
bash scripts/entrenar.sh src/exportar_coeficientes.py <version>   # si es nueva
bash scripts/entrenar.sh src/alias.py <version>
kubectl -n mora rollout restart deployment/mora-api
```

## Problemas
| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| Pods sin estar listos, 0/1 | MLflow apagado o sin artefacto exportado | `bash scripts/iniciar_mlflow.sh`; ver `kubectl -n mora logs` |
| `MLflow x, se esperaba 3.14.0` | Servidor de otra versión | Usar la imagen `riesgo-mlflow:3.14.0` |
| `el artefacto no corresponde` | `@champion` en una versión sin exportar | Exportarla o volver el alias a la 1 |
| La IP del nodo cambió | Se reinició la PC | Usar la que imprime `demo.sh` |
| `FileNotFoundError` en `/datos` | Falta el CSV | Revisar `/home/cesar/MifacodiDB/mora` o definir `DATOS_DIR` |
