#Código de Generación y Visualización Red Aleatoria Erdos-Renyi con PageRank
# run_erdos_renyi_1.py (para Matplotlib) (modificado respecto al de Colab)

import cugraph
import cudf
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# --- 1. Parámetros para la red Erdos-Renyi ---
# N = Número de nodos (vértices)
N = 50 # Un tamaño manejable para visualizar fácilmente
# P = Probabilidad de que exista una arista entre dos nodos
P = 0.1 # Ajusta esto para hacer la red más o menos densa

# --- 2. Generación de la red Erdos-Renyi con NetworkX (CPU) ---
# NetworkX es ideal para generar grafos sintéticos pequeños y luego convertirlos
G_nx = nx.erdos_renyi_graph(n=N, p=P, seed=42) # 'seed' para reproducibilidad

# --- 3. Convertir a formato cuGraph (GPU) ---
# Primero, extraemos las aristas de NetworkX en un DataFrame de cuDF
sources = [u for u, v in G_nx.edges()]
destinations = [v for u, v in G_nx.edges()]

# Si el grafo es pequeño y queremos todas las aristas
if not G_nx.edges(): # Manejar caso de grafo vacío
    print("El grafo Erdos-Renyi generado está vacío (quizás P es muy bajo o N muy pequeño).")
    # Crear un DataFrame vacío si no hay aristas
    edges_df = cudf.DataFrame({'source': cudf.Series(dtype=int), 'destination': cudf.Series(dtype=int)})
else:
    edges_df = cudf.DataFrame({
        'source': cudf.Series(sources, dtype=np.int32), # Asegúrate del tipo de dato
        'destination': cudf.Series(destinations, dtype=np.int32)
    })


# Crear el objeto grafo de cuGraph
G_cugraph = cugraph.Graph()
# Importante: para Erdos-Renyi, generalmente es no dirigido
G_cugraph.from_cudf_edgelist(edges_df, source='source', destination='destination', renumber=True)

# Después de renumber=True, los IDs de los vértices pueden cambiar.
# Se puede obtener el mapeo si se necesita para la visualización posterior (si los IDs originales importan)
# Si no se necesita el mapeo, y solo se visualizan los IDs re-numerados, no hay problema.

# --- 4. Aplicar un algoritmo de cuGraph (Ej: PageRank) ---
# El resultado es un DataFrame de cuDF con las columnas 'vertex' (id del nodo) y 'pagerank' (el valor).
pr_df = cugraph.pagerank(G_cugraph)

# --- 5. Preparar datos para visualización (convertir a CPU) ---
# cuGraph trabaja en GPU, NetworkX y Matplotlib en CPU. Necesitamos mover los datos.
# Convertir el grafo de cuGraph a NetworkX para visualización
# NOTA: Si el grafo es muy grande, esta conversión a CPU puede ser lenta o agotar la RAM.
# Para grafos pequeños (N=50-200), está bien.

# Convertir el DataFrame de aristas de cuDF a Pandas (CPU)
edges_pandas = edges_df.to_pandas()
G_nx_display = nx.Graph() # Crear un nuevo objeto NetworkX para visualizar
G_nx_display.add_edges_from(edges_pandas[['source', 'destination']].values)

# Preparar los puntajes de PageRank para colorear los nodos
# Asegurarse de que los IDs de los vértices coincidan entre pr_df y G_nx_display
pr_pandas = pr_df.to_pandas()
# Si se usó renumber=True en G_cugraph.from_cudf_edgelist, los IDs en pr_df son los re-numerados.
# Si G_nx_display se crea con los IDs originales, hay que mapearlos.
# Para este ejemplo simple, asumamos que los IDs coinciden o que los nuevos IDs son válidos para la visualización.
pr_dict = pr_pandas.set_index('vertex')['pagerank'].to_dict()

# Mapear los PageRank a los nodos del grafo para visualización
node_colors = [pr_dict.get(node, 0.0) for node in G_nx_display.nodes()]


# --- 6. Visualizar el grafo con Matplotlib ---
plt.figure(figsize=(10, 8))

# Posicionamiento de los nodos (layout)
pos = nx.spring_layout(G_nx_display, seed=42) # Usar el mismo seed para reproducibilidad visual

# Dibujar los nodos
# 'node_color' se basa en los valores de PageRank para colorear los nodos
# 'cmap' es el mapa de colores (ej. 'viridis', 'plasma', 'jet')
# 'vmin', 'vmax' establecen el rango de valores para el mapa de colores
nodes = nx.draw_networkx_nodes(G_nx_display, pos, node_color=node_colors, cmap=plt.cm.viridis,
                               node_size=500, alpha=0.9) # alpha es la transparencia

# Dibujar las aristas
nx.draw_networkx_edges(G_nx_display, pos, alpha=0.3, width=1.0)

# Dibujar las etiquetas de los nodos (IDs)
nx.draw_networkx_labels(G_nx_display, pos, font_size=8, font_color='black')

# Agregar una barra de color para la leyenda de PageRank
if nodes: # Asegurarse de que hay nodos para la barra de color
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                              norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([]) # Necesario para Matplotlib < 3.3
=======
import numpy as np # Necesitamos numpy para verificar NaN

# --- 1. Parámetros para la red Erdos-Renyi ---
N = 50
P = 0.1

# --- 2. Generación de la red Erdos-Renyi con NetworkX (CPU) ---
G_nx = nx.erdos_renyi_graph(n=N, p=P, seed=42)

# --- 3. Convertir a formato cuGraph (GPU) ---
sources = [u for u, v in G_nx.edges()]
destinations = [v for u, v in G_nx.edges()]

if not G_nx.edges():
    print("El grafo Erdos-Renyi generado está vacío.")
    edges_df = cudf.DataFrame({'source': cudf.Series(dtype=np.int32), 'destination': cudf.Series(dtype=np.int32)})
else:
    edges_df = cudf.DataFrame({
        'source': cudf.Series(sources, dtype=np.int32),
        'destination': cudf.Series(destinations, dtype=np.int32)
    })

G_cugraph = cugraph.Graph()
G_cugraph.from_cudf_edgelist(edges_df, source='source', destination='destination', renumber=True)

# --- 4. Aplicar PageRank ---
pr_df = cugraph.pagerank(G_cugraph)

# --- 5. Preparar datos para visualización (convertir a CPU) ---
edges_pandas = edges_df.to_pandas()
G_nx_display = nx.Graph()
G_nx_display.add_edges_from(edges_pandas[['source', 'destination']].values)

pr_pandas = pr_df.to_pandas()
pr_dict = pr_pandas.set_index('vertex')['pagerank'].to_dict()
node_colors = [pr_dict.get(node, 0.0) for node in G_nx_display.nodes()]

# --- 6. Visualizar el grafo con Matplotlib y GUARDAR LA IMAGEN ---
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_nx_display, seed=42)

nodes = nx.draw_networkx_nodes(G_nx_display, pos, node_color=node_colors, cmap=plt.cm.viridis,
                               node_size=500, alpha=0.9)
nx.draw_networkx_edges(G_nx_display, pos, alpha=0.3, width=1.0)
nx.draw_networkx_labels(G_nx_display, pos, font_size=8, font_color='black')

if nodes:
    vmin = min(node_colors) if node_colors else 0
    vmax = max(node_colors) if node_colors else 1
    if vmin == vmax: vmax = vmin + 1e-9 # Evitar error si todos los valores son iguales
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('PageRank Score')

plt.title(f'Red Aleatoria Erdos-Renyi (N={N}, P={P}) con PageRank')

plt.axis('off') # Desactivar los ejes
plt.show()

plt.axis('off')

# Guardar la figura en un archivo
plt.savefig("erdos_renyi_matplotlib.png")
print("Imagen de la red generada y guardada como 'erdos_renyi_matplotlib.png'")
# plt.show() # No usar plt.show() en scripts que no se ejecutan en un entorno interactivo

