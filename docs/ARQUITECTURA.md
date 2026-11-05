# Arquitectura y decisiones

## 1. Problema
- **Unidad:** un crédito en el momento de otorgarlo. **Etiqueta `mora30`:** a vencimiento + 30 días los pagos vigentes no cubrían el total.
- **Variables:** solo lo conocido al vender: monto, plazo, inicial y comportamiento anterior del cliente. Se excluyen `venta.saldo`, `fechavenc`, `estadopagado` y datos de cobranza, porque se actualizan después (fuga).
- **División temporal:** entrenamiento hasta 30/06/2024 (687 créditos, 47,5 % de mora), prueba del 01/07/2024 al 30/09/2025 (446, 38,3 %). Una división aleatoria mezclaría futuro y pasado.
- **Piso a superar:** regla de negocio "cómo pagó su crédito anterior", con tasas aprendidas del entrenamiento.

## 2. MLflow: modelo desplegado
`mora-credito@champion` → **versión 1**, run `f8411f02f6934b789549c0c85ed7e9b8` (`logreg_C-1.0`).

| Run | Versión | AUC en prueba | F1 |
|---|---|---|---|
| rf_n-300_depth-8 | 2 | 0,6378 | 0,5337 |
| **logreg_C-1.0** | **1, champion** | **0,6373** | **0,5671** |
| logreg_C-0.01 | 5, para el rollback | 0,6336 | 0,5391 |
| rf_n-100_depth-4 | | 0,6274 | 0,5425 |
| gb_lr-0.05_depth-3 | | 0,5980 | 0,5198 |
| regla_historial | | 0,5930 | 0,5201 |
| gb_lr-0.1_depth-2 | | 0,5663 | 0,4892 |

**Por qué la logística y no el bosque:** empatan en AUC (0,0005), tiene mejor F1, se puede explicar y Node la calcula sin Python. El bosque fue champion hasta que la API pasó a Node.

Diferencia contra la regla: +0,044, **IC 95 % de −0,011 a +0,098: incluye el cero**, y el modelo se eligió mirando este mismo conjunto de prueba. No se afirma que la supere.

## 3. API en Node.js y Docker
**Decisión:** Python entrena y Node sirve. El backend del ERP ya es Node, y así el servicio no suma otro lenguaje en producción.

```
scikit-learn → MLflow (run + versión + alias) → exportado/coeficientes.json → Node
```

- `src/exportar_coeficientes.py` guarda medias, escalas, coeficientes e intercepto como artefacto **del mismo run**.
- Al arrancar, la API pregunta a MLflow por REST qué versión tiene `@champion` y descarga su artefacto. No usa una ruta de archivo; comprueba que el artefacto corresponda a ese run. Si falla, `/ready` responde 503 y reintenta.
- `api/riesgo.js` no depende de red ni de Express, para poder llevarlo al backend del ERP.
- **Documentación:** `GET /` muestra cada endpoint, sus parámetros y sus respuestas, con un botón para probarlo; `GET /endpoints` da lo mismo en JSON. Los parámetros de `/predict` salen del mismo contrato que valida la API, y una prueba verifica que coincidan.
- **Paridad comprobada:** en 300 casos inventados, Node coincide con scikit-learn (diferencia < 1e-9 en la prueba unitaria; 0,00005 en vivo, por el redondeo a 4 decimales). Evidencia: `evidencias/api-0*.txt`.

**Límite de esta decisión:** solo se pueden servir regresiones logísticas sin calibrar. El bosque y las versiones 3 y 4 no tienen exportación; si `@champion` apuntara a ellas, los Pods no quedarían listos y seguirían atendiendo los anteriores.

**Dependencias fijadas:** `node:22.23.4-slim`, `express` con `package-lock.json`. La API no instala MLflow: lo consume por REST, y exige que el servidor sea la versión fijada (`MLFLOW_VERSION=3.14.0`, igual que `docker/mlflow/requirements.txt`). Usuario sin privilegios (`node`). Atiende el apagado de Kubernetes (`SIGTERM`).

**Laboratorio solo local:** `scripts/laboratorio.sh` levanta la misma imagen con el CSV montado de solo lectura, publicada únicamente en `127.0.0.1:8010`. Permite consultar créditos y clientes seudonimizados y comparar predicción con resultado real. Las rutas solo existen con `DATOS_CSV`: en Kubernetes responden 404, y la imagen no lleva datos. Node recalculó sobre los datos reales de prueba el AUC de MLflow (0,6373) y sus aciertos (55,16 %).

## 4. Kubernetes
Tres réplicas, `readinessProbe` sobre el modelo cargado, `RollingUpdate` con `maxUnavailable: 0`, 250m de CPU y 128 Mi. Evidencia real en `evidencias/k8s-*.txt`: 30 peticiones repartidas en 3 Pods; 186 respuestas 200 y 0 errores al borrar un Pod; escalado a 5 y regreso a 3; rollback moviendo el alias a la versión 5 y volviendo a la 1, con la misma imagen.

## 5. Data drift
- **Numéricas: Kolmogorov-Smirnov.** Compara las distribuciones completas sin suponer normalidad, y los montos son muy asimétricos. Umbral: p < 0,05/11 = 0,0045 (corrección de Bonferroni, porque se hacen 11 pruebas y sin ella habría falsas alarmas).
- **Categóricas: PSI.** Mide cuánto cambia el peso de cada categoría, y no necesita que la variable tenga orden. Umbrales: 0,10 alerta y 0,20 drift, convención de la industria vista en la clase 7.

Mismo origen: **VERDE**. Lote derivado (simula el cambio de Java a Ionic): **ROJO** en las 4 variables alteradas. `pytest`: 3 de 3. Lotes reales posteriores: **ROJO en los tres**; lo esperable, porque el historial del cliente crece con el tiempo.

## 6. Concept drift
**Simulado:** desde el lote 5 se invierte la relación con el historial por un cambio de cobranza. Umbral = AUC de referencia − 2σ = 0,498 (σ medida con 200 muestras de 60). Con 3 lotes seguidos bajo el umbral salta la alarma: **lote 10**. Son muestras de 446 créditos, no lotes independientes.

**Real (lo más importante del proyecto):**

| Semestre | Créditos con resultado | Mora real | Prob. media del modelo | AUC modelo | AUC regla |
|---|---|---|---|---|---|
| 2024-S2 | 249 | 45,0 % | 55,9 % | 0,684 | 0,609 |
| 2025-S1 | 138 | 34,8 % | 56,2 % | 0,647 | 0,589 |
| 2025-S2 | 164 | 8,5 % | 56,9 % | 0,518 | 0,632 |
| 2026-S1 | 230 | 2,2 % | 40,2 % | sin datos suficientes | sin datos suficientes |
| 2026-S2 | 50 | 4,0 % | 26,4 % | sin datos suficientes | sin datos suficientes |

En créditos a 1 mes, el tiempo promedio hasta cada pago pasó de 50-90 días (hasta 2025-T2) a 29 en 2025-T3 y a 14-21 desde 2025-T4; los últimos trimestres pueden subir cuando lleguen más pagos (`evidencias/drift-05-tiempo-de-pago.txt`). **La causa no está determinada:** puede ser un cambio real de cobranza o de condiciones, o un cambio en cómo se registran los datos. Hay que preguntarlo al negocio antes de reentrenar. Además, en cohortes recientes solo maduraron los plazos cortos (sesgo de selección), y 2025-S2 tiene apenas 14 atrasos.

**Criterio de reentrenamiento:** AUC bajo referencia − 2σ durante 3 lotes consecutivos. **Retraso de etiquetas:** real: 2 meses en plazo de 1 mes y 13 meses en plazo de 12. Mientras tanto se vigilan data drift, distribución de las predicciones y el pago de la primera cuota, y se guarda cada predicción con su `run_id` para evaluarla cuando llegue la etiqueta.

## 7. Reentrenamiento
**Versión 3:** etiqueta nueva, "terminó de pagar después del vencimiento", porque casi no quedan atrasos de más de 30 días; modelo y regla se comparan con ella. Entrena hasta 2025-S1, elige y ajusta con 2025-S2, y prueba con 2026 (280 créditos, 39 atrasos). AUC 0,426 contra 0,495 de la regla; IC de la diferencia de −0,183 a +0,048. Atraso real 13,9 % y probabilidad media 17,1 %: corrige el nivel, no el orden.

**Versión 4:** con todo lo conocido al 05/11/2026 (1.523 créditos con resultado). Evaluada hacia adelante, cada trimestre solo con lo que ya se sabía:

| Trimestre | Créditos | Atrasos | Atraso real | Prob. media | AUC modelo | AUC regla |
|---|---|---|---|---|---|---|
| 2025-T3 | 59 | 19 | 32,2 % | 73,6 % | 0,238 | 0,539 |
| 2025-T4 | 105 | 9 | 8,6 % | 59,8 % | 0,691 | 0,679 |
| 2026-T1 | 135 | 11 | 8,1 % | 32,6 % | 0,360 | 0,554 |
| 2026-T2 | 95 | 17 | 17,9 % | 11,4 % | 0,609 | 0,454 |
| 2026-T3 | 54 | 12 | 22,2 % | 5,9 % | 0,560 | 0,506 |
| **Total** | **448** | **68** | **15,2 %** | **36,7 %** | **0,494** | **0,523** |

**Decisión: no se promueven.** Con estas variables, ni el modelo ni la regla ordenan los atrasos recientes mejor que el azar. Son pocos atrasos, y debe confirmarse con más datos. `@candidato` queda en la versión 4 y `@champion` en la 1.

## 8. Limitaciones
`pago.fecha` es editable y `pago` no guarda fecha de registro, los pagos anulados no se rebobinan, las ventas anuladas quedan fuera y la etiqueta usa una regla reconstruida de plazos (meses o días). Solo se usó una empresa, con poca información del cliente: sin ingresos ni ocupación. Es un modelo de asociación, no explica causas.

## 9. Privacidad
Datos seudonimizados con md5 y sal guardada fuera del repo, CSV fuera de Git y carpeta con permisos restringidos. En la defensa solo se muestran agregados; no se abre el CSV. Seudonimizar no es anonimizar: un monto raro podría identificar a un cliente, por eso no se publican filas. El repositorio debe ser privado.

## 10. Producción: diseño, no desplegado
kind es solo para la demostración. El servidor de producción (EC2 t3.small) no aloja Kubernetes. Python sigue fuera: entrena en otra máquina y publica coeficientes.

| Etapa | Qué sale | Condición |
|---|---|---|
| 1. Ficha del cliente | Durante la venta: créditos anteriores, cuánto debe y su mayor atraso. Sin puntaje | Ninguna: solo muestra hechos |
| 2. Silencioso | El backend de Node calcula y guarda el riesgo sin mostrarlo | Explicar la caída desde 2025 |
| 3. Aviso | Lo ven usuarios elegidos; nunca bloquea | Superar a la regla y al azar con suficientes atrasos |

Reglas: apagado por defecto por empresa, guardia y privilegio en el endpoint, timeout corto, y la venta sigue si falla.

## Reparto de trabajo
Trabajo individual: todos los componentes (modelo, MLflow, Docker, Kubernetes, drift, interfaz y documentación) son de Julio Cesar Pinto Rojas, con asistencia de IA para programar y documentar.
