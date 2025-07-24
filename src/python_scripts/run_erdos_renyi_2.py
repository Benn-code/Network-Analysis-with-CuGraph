# run_erdos_renyi_2.py (para PyVis) (modificado respecto al de Colab, pues incluía lineas que dependían de Colab)

from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np # Necesitamos numpy para verificar NaN

# Importar las variables necesarias del primer script o re-ejecutar la lógica
# Dado que se ejecutarán independientemente, es mejor replicar la lógica principal
# o pasar los datos entre ellos si fueran un solo script.
# Para simplicidad y que cada script sea autónomo, replicamos la parte de la generación
# del grafo y PageRank.

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


# Crear una red PyVis a partir del grafo NetworkX para interactividad
# Usamos cdn_resources='in_line' para que todo el JS/CSS esté en el HTML (más portable)
net = Network(notebook=False, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='in_line')

# Crear un mapa de colores normalizado para PageRank
node_colors_filtered = [c for c in node_colors if not np.isnan(c)]
if node_colors_filtered:
    vmin = min(node_colors_filtered)
    vmax = max(node_colors_filtered)
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis
else:
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.viridis

# Añadir nodos y aristas al objeto PyVis
for node in G_nx_display.nodes():
    pr_score = pr_dict.get(node, 0.0)

    if np.isnan(pr_score):
        hex_color = "#808080"
    else:
        rgba_color = cmap(norm(pr_score))
        hex_color = mcolors.to_hex(rgba_color)

    net.add_node(int(node), label=str(node), title=f"Node {node} (PageRank: {pr_score:.4f})",
                 color=hex_color, value=float(pr_score))

edges_for_pyvis = [(int(u), int(v)) for u, v in G_nx_display.edges()]
net.add_edges(edges_for_pyvis)

# Generar y GUARDAR el grafo interactivo HTML
output_filename = "erdos_renyi_interactive.html"
net.show(output_filename)
print(f"Grafo interactivo generado y guardado como '{output_filename}'")