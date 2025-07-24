//la visualicacion por pyvis es unicamente de colab

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/graph_io.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>

int main() {
    raft::handle_t handle;

    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;

    // Grafo Karate Club (34 nodos, 78 aristas)
    std::vector<std::pair<vertex_t, vertex_t>> edges = {
        {0,1},{0,2},{0,3},{0,4},{0,5},{0,6},{0,7},{0,8},{0,10},{0,11},{0,12},{0,13},
        {0,17},{0,19},{0,21},{0,31},{1,2},{1,3},{1,7},{1,13},{1,17},{1,19},{1,21},{1,30},
        {2,3},{2,7},{2,27},{2,28},{2,32},{3,7},{4,6},{4,10},{5,6},{5,10},{5,16},
        {6,16},{8,30},{8,32},{8,33},{9,33},{13,33},{14,32},{14,33},{15,32},{15,33},
        {18,32},{18,33},{19,33},{20,32},{20,33},{22,32},{22,33},{23,25},{23,27},
        {23,29},{23,32},{23,33},{24,25},{24,27},{24,31},{25,31},{26,29},{26,33},
        {27,33},{28,31},{28,33},{29,32},{29,33},{30,32},{30,33},{31,32},{31,33},
        {32,33}
    };

    size_t num_edges = edges.size();
    rmm::device_uvector<vertex_t> d_srcs(num_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dsts(num_edges, handle.get_stream());

    std::vector<vertex_t> h_srcs, h_dsts;
    for (auto [u, v] : edges) {
        h_srcs.push_back(u);
        h_dsts.push_back(v);
    }

    raft::update_device(d_srcs.data(), h_srcs.data(), num_edges, handle.get_stream());
    raft::update_device(d_dsts.data(), h_dsts.data(), num_edges, handle.get_stream());

    // Crear grafo no ponderado y no dirigido
    auto [graph, edge_weights, renumber_map, d_renumber_map] =
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t>(
            handle,
            d_srcs,
            d_dsts,
            std::nullopt,
            cugraph::graph_properties_t{false, false},
            true  // renumber
        );

    // PageRank
    auto pagerank = cugraph::pagerank(handle, graph);
    std::cout << "\n=== PageRank ===\n";
    for (size_t i = 0; i < pagerank.size(); ++i) {
        std::cout << "Nodo " << i << ": " << pagerank.data()[i] << "\n";
    }

    // Louvain
    auto [parts, modularity] = cugraph::louvain(handle, graph);
    std::cout << "\n=== Louvain ===\n";
    std::cout << "Modularidad: " << modularity << "\n";
    for (size_t i = 0; i < parts.size(); ++i) {
        std::cout << "Nodo " << i << ": comunidad " << parts.data()[i] << "\n";
    }

    // Conteo de triángulos
    auto triangle_counts = cugraph::triangle_count(handle, graph);
    std::cout << "\n=== Conteo de Triángulos ===\n";
    int total_triangles = 0;
    for (size_t i = 0; i < triangle_counts.size(); ++i) {
        std::cout << "Nodo " << i << ": " << triangle_counts.data()[i] << " triángulos\n";
        total_triangles += triangle_counts.data()[i];
    }
    std::cout << "Total de triángulos (ajustado): " << total_triangles / 3 << "\n";

    return 0;
}
