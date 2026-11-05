# Imagen del entorno de experimentación (servidor MLflow + entrenamiento + drift).
# No tiene CMD: el comando se indica al ejecutarla (mlflow server o python ...).

# Parte de una imagen con Linux mínimo y Python 3.12.
FROM python:3.12-slim

# Evita archivos .pyc, muestra los logs al instante y no guarda la caché de pip.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Crea la carpeta /proyecto y ejecuta desde ahí los comandos del contenedor.
WORKDIR /proyecto

# Copia la lista de librerías a una carpeta temporal de la imagen.
COPY requirements.txt /tmp/requirements.txt

# Instala MLflow, scikit-learn y las demás librerías con sus versiones fijadas.
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Documenta que MLflow usa el puerto 5000. Se publica con -p al ejecutar el contenedor.
EXPOSE 5000
