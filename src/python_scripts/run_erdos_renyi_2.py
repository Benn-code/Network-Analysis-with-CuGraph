# Código de Generación y Visualización interactiva Red Aleatoria Erdos-Renyi con PageRank
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