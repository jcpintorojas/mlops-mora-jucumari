# MLOps de mora de créditos en Jucumari

Proyecto final del módulo **ML-Ops y Puesta en Producción** (UAGRM-SOE) · **Julio Cesar Pinto Rojas** - **David Valverde Andia**

Estima, al otorgar una venta a crédito en el ERP Jucumari, la probabilidad de que siga impaga 30 días después de vencer. Python entrena, MLflow versiona y una API en **Node.js** sirve el modelo en Kubernetes.

| Fase | Qué hay | Dónde |
|---|---|---|
| 1. MLflow | 7 runs (6 modelos + regla de negocio), `mora-credito` con 5 versiones, `@champion` en la v1 y `@candidato` en la v4 | `src/` |
| 2. Docker | API en Node.js que carga `mora-credito@champion` y responde con versión, run y Pod | `api/` |
| 3. Kubernetes | 3 réplicas en el namespace `mora`, balanceo, autorreparación, escalado y rollback por alias | `k8s/`, `scripts/demo_k8s.sh` |
| 4. Drift | Data drift (KS + PSI, pytest verde/rojo), concept drift simulado y drift real por cohorte | `drift/` |
| Extra | Interfaz web en el clúster que consume la API | `ui/` |

**Resultado principal:** el pipeline funciona de punta a punta, pero el modelo **no está listo para avisar en ventas reales**. Tras reentrenar, las versiones 3 y 4 tampoco superaron a la regla y no se promovieron. Ver [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) y [`docs/OPERACION.md`](docs/OPERACION.md).

## Datos
Los datos **no están en este repositorio**. Es un CSV seudonimizado (sin nombres, CI, NIT, teléfonos ni direcciones) en una carpeta externa, montada de solo lectura en `/datos`. `sql/` documenta cómo se obtiene. Los artefactos de MLflow usan un `input_example` inventado.

## Puesta en marcha
```bash
bash scripts/iniciar_mlflow.sh                          # http://127.0.0.1:5001
bash scripts/entrenar.sh src/entrenar.py
bash scripts/entrenar.sh src/registrar.py
bash scripts/entrenar.sh src/exportar_coeficientes.py 1 # deja el modelo listo para Node
bash scripts/desplegar.sh
bash scripts/demo.sh                                    # para la presentación
```
Convive con el proyecto UCI: usa otro contenedor de MLflow (`mlflow-mora`, puerto 5001), otro namespace (`mora`) y otros NodePort (API 30092, UI 30093).
