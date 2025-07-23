# Network Analysis with cuGraph

Este documento describe la estructura y las instrucciones para ejecutar el proyecto de análisis de redes complejas utilizando `cuGraph`. El objetivo es demostrar la capacidad de `cuGraph` para acelerar el procesamiento de grafos mediante el uso de GPUs.

## 🔗 Enlaces Importantes

* **Video de Exposición del Proyecto:** [(https://www.youtube.com/watch?v=PwLIpvTRyp8)]
* **Repositorio de GitHub (para referencia y control de versiones):** [(https://github.com/Benn-code/Network-Analysis-with-CuGraph)]

## 🚀 Estructura del Proyecto

El archivo comprimido `proyecto_cugraph.zip` contiene la siguiente estructura:

## 🛠️ Entorno de Ejecución con Docker

Hemos proporcionado un `Dockerfile` que permite construir un entorno completo con `cuGraph` para C++ y Python. Esto encapsula todas las dependencias necesarias.

### Requisitos Previos

Para utilizar este `Dockerfile`, necesitarás tener instalados:

* **Docker Desktop** (Windows/macOS) o **Docker Engine** (Linux).
* Una **GPU NVIDIA compatible con CUDA** y los drivers apropiados instalados en tu sistema host.

### 1. Construir la Imagen Docker

Desde la raíz del directorio descomprimido (donde se encuentra el `Dockerfile`), abre una terminal y ejecuta el siguiente comando:

```bash
docker build -t cugraph-project-env .

Este proceso descargará la imagen base de NVIDIA CUDA e instalará todas las dependencias y librerías necesarias para C++ y Python, incluyendo cuGraph. Este paso puede tardar varios minutos dependiendo de tu conexión a internet.

2. Ejecutar Programas C++ (Demostración Conceptual)
Nota Importante: Dado que no se tuvo acceso a un clúster HPC con GPU dedicada para pruebas exhaustivas, la ejecución de estos programas C++ es de carácter conceptual para esta entrega. El Dockerfile instala las dependencias y compila el código, demostrando la estructura de un proyecto C++ con cuGraph.

Asumiendo que la compilación fue exitosa durante la construcción de la imagen Docker (ver paso 1), los ejecutables se encontrarían dentro del contenedor en sus respectivos directorios build dentro de /app/src/.

Para ejecutar un programa C++, puedes usar:

# Ejemplo de ejecución del programa Erdos-Renyi:
docker run --rm --gpus all cugraph-project-env /app/src/erdos_renyi_network/build/my_erdos_renyi_program

# Ejemplo de ejecución del programa Watts-Strogatz:
docker run --rm --gpus all cugraph-project-env /app/src/watts_strogatz_network/build/my_watts_strogatz_program

# Ejemplo de ejecución del programa de la Red Real:
docker run --rm --gpus all cugraph-project-env /app/src/real_network/build/my_real_network_program

--rm: Elimina el contenedor una vez que termina su ejecución.

--gpus all: Permite que el contenedor acceda a todas las GPUs disponibles en el sistema host.

my_erdos_renyi_program (y similares) son nombres de ejemplo de los ejecutables. Verifiquen los nombres exactos definidos en los CMakeLists.txt o Makefiles de cada subproyecto C++.
3. Ejecutar Programas Python
Este contenedor también está configurado con Python y todas las librerías necesarias de cuGraph y visualización. Los scripts Python ejecutables se encuentran en la carpeta src/python_scripts/.

Para ejecutar un script Python dentro del contenedor:

Inicia una sesión interactiva de Bash en el contenedor:

docker run -it --rm --gpus all cugraph-project-env /bin/bash

Una vez dentro del contenedor, navega al directorio de los scripts Python y ejecuta el que desees:

cd /app/src/python_scripts/

# Para ejecutar el script de visualización con Matplotlib (genera PNG):
python3 run_erdos_renyi_1.py

# Para ejecutar el script de visualización interactiva con PyVis (genera HTML):
python3 run_erdos_renyi_2.py

# Para ejecutar otros scripts de los compañeros:
python3 run_watts_strogatz.py
python3 run_real_network.py

# Los archivos de salida (ej. erdos_renyi_matplotlib.png, erdos_renyi_interactive.html)
# se generarán en el directorio actual dentro del contenedor (/app/src/python_scripts/).
# Para acceder a ellos desde tu sistema host, necesitarías copiarlos fuera del contenedor
# antes de que se elimine (ver `docker cp` o usar volúmenes, que no se explican aquí para simplicidad).