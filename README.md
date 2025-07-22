# Proyecto
Aquí les dejo una breve explicación de cómo crear sus propias ramas y cómo podría ser el flujo de trabajo. 

## Paso 1: Clonar el Repositorio de GitHub a tu Máquina Local

    Esto crea una copia del repositorio de la nube a sus computadores. Solo necesitan hacer esto una vez. Evidentemente deben hacerlo fuera de la carpeta de sus otros repositorios para no crear conflictos.

    Aquí en Github hacen clic en el botón "Code".

    Seleccionan la pestaña "HTTPS" (es la más común y sencilla).

    Luego copian la URL que aparece (será algo como https://github.com/TuUsuario/Network-Analysis-with-cuGraph.git).

    Van a su terminal (sea en el computador de la universidad o en uno propio (Linux/macOS) o Git Bash (Windows)).

    Navegan hasta la carpeta donde quieran almacenar el proyecto (ej. cd Documents/Projects).

    Ejecutan el comando git clone seguido de la URL que copiaron:

    Ejemplo:
    git clone https://github.com/TuUsuario/Network-Analysis-with-cuGraph.git
    Esto creará una nueva carpeta llamada Network-Analysis-with-cuGraph en ese directorio actual.

## Paso 2: Entrar a esa carpeta y crear sus propias ramas

    Bueno, entran a la carpeta Network-Analysis-with-cuGraph

## Paso 3: Crear y Cambiar a una Nueva Rama de Desarrollo

    Me parece que lo más práctico en este caso será que cada quien tenga una rama con su nombre. 
    Estando en esa carpeta usan el siguiente comando:
    
    git checkout -b feature/Sunombre

    En mi caso usé git checkout -b feature/Benjamin

## Paso 4: Crear y poner sus archivos 

    Luego ya crean sus archivos y lo que quieran. 
    Usan el git add y el git commit -m "..." de siempre
    Pero luego usan

    git push origin feature/sunombre

## Paso 5: Hacer un pull reques 

    Una vez que sus cambios están en su rama en GitHub, normalmente no se fusionas directamente a main. En su lugar, abren un "Pull Request" (PR) en GitHub.

    Van al repositorio en GitHub en el navegador.

    Verán un mensaje que dice que su rama tiene nuevos cambios. Habrá un botón para "Compare & pull request" o "New pull request". Hacen clic en él.

    Escribe un título y una descripción para el PR, explicando qué cambios hicieron.

    Asignan a los demás compañeros como revisores. Así podemos ver los cambios, hacer comentarios y aprobarlos.

    Una vez aprobado, pueden hacer clic en "Merge pull request" para fusionar su rama a main.
