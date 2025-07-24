#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/graph_io.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <tuple>

int main() {
    raft::handle_t handle;

    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;

    std::string filepath = "facebook_combined.txt";

    // Leer archivo como lista de aristas (source, destination)
    std::vector<vertex_t> srcs, dsts;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo.\n";
        return 1;
    }

    vertex_t s, d;
    while (file >> s >> d) {
        srcs.push_back(s);
        dsts.push_back(d);
    }
    file.close();

    size_t num_edges = srcs.size();
    rmm::device_uvector<vertex_t> d_srcs(num_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> d_dsts(num_edges, handle.get_stream());

    raft::update_device(d_srcs.data(), srcs.data(), num_edges, handle.get_stream());
    raft::update_device(d_dsts.data(), dsts.data(), num_edges, handle.get_stream());

    // Crear grafo no ponderado no dirigido
    auto [graph, edge_weights] = cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t>(
        handle,
        d_srcs,
        d_dsts,
        std::nullopt,
        cugraph::graph_properties_t{false, false},
        true // renumber
    );

    // LOUVAIN
    auto [louvain_labels, louvain_modularity] = cugraph::louvain(handle, graph);

    std::cout << "Modularidad Louvain: " << louvain_modularity << std::endl;

    // PAGERANK
    auto pagerank_scores = cugraph::pagerank(handle, graph);

    std::cout << "Primeros 10 resultados de PageRank:\n";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Nodo: " << i << " Score: " << pagerank_scores.data()[i] << "\n";
    }

    // JACCARD
    auto jaccard_results = cugraph::jaccard_coefficients(handle, graph, std::nullopt);

    std::cout << "Primeros 10 pares similares por Jaccard:\n";
    auto h_firsts = jaccard_results.first.first;
    auto h_seconds = jaccard_results.first.second;
    auto h_coeffs = jaccard_results.second;

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "(" << h_firsts.data()[i] << ", " << h_seconds.data()[i] << ") -> " << h_coeffs.data()[i] << "\n";
    }

    return 0;
}
