# Servicio de inferencia en Node.js. El modelo NO va dentro de la imagen:
# se descarga al arrancar desde el Model Registry de MLflow (models:/mora-credito@champion).
FROM node:22.23.4-slim
ENV NODE_ENV=production
WORKDIR /app
# La lista de dependencias va antes que el código para aprovechar la caché de capas.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY server.js riesgo.js mlflow.js docs.js laboratorio.js laboratorio_pagina.js ./
# Indica dónde está MLflow, su versión fijada y qué modelo y alias cargar.
ENV MLFLOW_TRACKING_URI=http://mlflow-mora:5000 MLFLOW_VERSION=3.14.0 MODELO_NOMBRE=mora-credito MODELO_ALIAS=champion
# La imagen trae el usuario node, sin privilegios de administrador.
USER node
EXPOSE 8000
# Sin npm de por medio: node recibe directamente la señal de apagado de Kubernetes.
CMD ["node", "server.js"]
