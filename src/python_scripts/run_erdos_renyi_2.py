 Código de Generación y Visualización interactiva Red Aleatoria Erdos-Renyi con PageRank
# run_erdos_renyi_2.py (para PyVis) (modificado respecto al de Colab, pues incluía lineas que dependían de Colab)

from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np # Necesitamos numpy para verificar NaN

# Crear una red PyVis a partir del grafo NetworkX para interactividad
net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local')

# Crear un mapa de colores normalizado para PageRank
node_colors_filtered = [c for c in node_colors if not np.isnan(c)] # Eliminar NaNs
if node_colors_filtered: # Asegurarse de que hay valores válidos para la normalización
    vmin = min(node_colors_filtered)
    vmax = max(node_colors_filtered)
    # Si todos los valores son iguales, ajusta vmax para evitar división por cero en Normalize
    if vmin == vmax:
        vmax = vmin + 1e-9 # Pequeño offset para evitar error de normalización
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis
else: # Caso para un grafo vacío o sin nodos con PageRank válido
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.viridis


# Añadir nodos y aristas al objeto PyVis
for node in G_nx_display.nodes():
    pr_score = pr_dict.get(node, 0.0) # Obtener PageRank

    # Manejar posibles NaNs en pr_score antes de normalizar
    if np.isnan(pr_score):
        hex_color = "#808080" # Un color gris si PageRank es NaN
    else:
        # Mapear el puntaje de PageRank a un color usando el mapa de colores
        rgba_color = cmap(norm(pr_score)) # Obtener color en formato RGBA (0-1)
        hex_color = mcolors.to_hex(rgba_color) # Convertir a formato HEX

    # La CLAVE está aquí: asegúrate de que 'node' sea un entero nativo de Python
    # Convertimos explícitamente el 'node' a int()
    net.add_node(int(node), label=str(node), title=f"Node {node} (PageRank: {pr_score:.4f})",
                 color=hex_color, value=float(pr_score)) # También convierte value a float para JSON

# Generar y mostrar el grafo interactivo
net.show("erdos_renyi_interactive.html")

# Para descargar el HTML generado (en mi caso no se pudo visualizar directamente en Colab, así que tocó descargarlo)
# from google.colab import files (no estoy seguro de que funcione en Docker, pero es útil en Colab)
files.download("erdos_renyi_interactive.html")
=======
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
