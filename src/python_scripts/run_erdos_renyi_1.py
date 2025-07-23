# run_erdos_renyi_1.py (para Matplotlib) (modificado respecto al de Colab)

import cugraph
import cudf
import networkx as nx
import matplotlib.pyplot as plt
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
plt.axis('off')

# Guardar la figura en un archivo
plt.savefig("erdos_renyi_matplotlib.png")
print("Imagen de la red generada y guardada como 'erdos_renyi_matplotlib.png'")
# plt.show() # No usar plt.show() en scripts que no se ejecutan en un entorno interactivo