# Utilizamos una imagen base con Ubuntu y CUDA ya preinstalado por NVIDIA (Ubuntu resulta práctico para compatibilidad con cuGraph).
# Con esto tenemos una buena base para desarrollo de GPU en C++ y Python.
# Usamos la versión más reciente de CUDA compatible con cuGraph y cuDF, indicada por el número de versión.

# CUDA 12.4.0 en Ubuntu 22.04.
FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

LABEL maintainer="Tu Equipo <tu.email@example.com>"
LABEL description="Entorno de desarrollo para cuGraph en C++ y Python"

# ====================================================================
# PASO 1: INSTALAR DEPENDENCIAS GENERALES Y DE C++
# ====================================================================

# Actualizar lista de paquetes e instalar herramientas básicas de C++
# build-essential: compilador g++, make
# cmake: para construir proyectos C++ más complejos
# git: para clonar repositorios (aunque los archivos se copian, es útil para el entorno, y tabajamos en gran medida usando git, así que por si acaso)
# libboost-all-dev: Boost es una librería común para C++ que cuGraph puede usar.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    # Limpieza de caché de APT para reducir tamaño de imagen
    && rm -rf /var/lib/apt/lists/*

# (Conceptual) Instalación de cuGraph C++
# ATENCIÓN: Esta sección es conceptual ya que no se pudo probar al no tener el hardware
# ni un entorno de desarrollo de cuGraph C++ completo.
# De proporcionarse un repositorio APT específico para cuGraph C++, se agregaría aquí.
# Por ahora, simulamos la instalación de la librería C++ principal si estuviera disponible.
# Si existen paquetes binarios para Ubuntu, serían preferibles a compilar desde fuente.
# Ejemplo de instalación de paquete binario (verificando el nombre real del paquete):
# RUN apt-get update && apt-get install -y librapids-cxx-dev=24.04 # Ajustar versión si es necesario
#
# O si fuera por fuente (esto puede ser MUY complejo y lento, y NO se ha probado):
# RUN git clone --recursive https://github.com/rapidsai/cugraph.git /opt/cugraph_src \
#     && mkdir /opt/cugraph_src/cpp/build \
#     && cd /opt/cugraph_src/cpp/build \
#     && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/cugraph_cxx -DCUGRAPH_BUILD_TESTS=OFF \
#     && make -j$(nproc) \
#     && make install \
#     && rm -rf /opt/cugraph_src # Limpiar fuentes después de la instalación


# ====================================================================
# PASO 2: INSTALAR PYTHON Y LIBRERÍAS DE CUDf/CUGRAPH PYTHON
# ====================================================================

# Instalar Python y pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Asegurarse de que 'pip' apunte a python3 (mismo que usamos en Colab)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Instalar librerías de Python para cuGraph, cuDF, NetworkX, Matplotlib, PyVis, fueron todas las que usamos
# NOTA: Los índices extra son cruciales para cuDF/cuGraph en Colab/Docker.
# Asegúrate de usar las versiones compatibles con CUDA 12 (ej. -cu12).
RUN pip install --no-cache-dir \
    cudf-cu12 --extra-index-url=https://pypi.nvidia.com \
    cugraph-cu12 --extra-index-url=https://pypi.nvidia.com \
    networkx \
    matplotlib \
    pyvis

# ====================================================================
# PASO 3: COPIAR CÓDIGO FUENTE DEL PROYECTO
# ====================================================================

# Crear un directorio de trabajo para el proyecto dentro del contenedor
WORKDIR /app

# Copiar todo el contenido del directorio local (donde está el Dockerfile)
# a /app dentro del contenedor. Esto incluirá src/, notebooks/, data/ y el readme.md.
COPY . /app/

# ====================================================================
# PASO 4: COMPILAR CÓDIGOS C++ (Conceptual para la entrega, NO se ha probado)
# ====================================================================

# Cambiar al directorio donde están los códigos C++
WORKDIR /app/src/

# Compilar cada proyecto C++ individualmente.
# Esto asume que cada subdirectorio tiene un main.cpp y un CMakeLists.txt (o Makefile), pero tal vez no alcanzamos a tenerlos para la hora de la entrega.:
# RUN make -C erdos_renyi_network
# RUN make -C watts_strogatz_network
# RUN make -C real_network

# Si se usa CMake (más robusto para proyectos C++):
RUN mkdir -p erdos_renyi_network/build && \
    cmake -S erdos_renyi_network -B erdos_renyi_network/build && \
    cmake --build erdos_renyi_network/build

RUN mkdir -p watts_strogatz_network/build && \
    cmake -S watts_strogatz_network -B watts_strogatz_network/build && \
    cmake --build watts_strogatz_network/build

RUN mkdir -p real_network/build && \
    cmake -S real_network -B real_network/build && \
    cmake --build real_network/build

# ====================================================================
# PASO 5: DEFINIR COMANDO DE INICIO (Opcional, para un punto de entrada por defecto)
# ====================================================================

# Intentamos definir un comando por defecto. Pensamos que sería útil un script
# que muestre cómo se ejecutarían los diferentes programas.
# Probablemente no alcanzamos a crear un script simple en la raíz del proyecto (ej. run_examples.sh)
# que liste cómo ejecutar cada programa C++ y Python.
# La idea era que si el profesor lo ejecutaba, esto mostraría un "tour" conceptual.

# CMD ["/bin/bash", "/app/run_examples.sh"]
# Por el tiempo lo dejamos solamente en el README.md, pero si se crea el script, se puede descomentar esta línea.