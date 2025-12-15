#include "hnswlib.h"
#include "../includes/memory_utils.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cstdint>

// -------------------- Loaders --------------------
std::vector<float> load_fvec(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir: " + path);

    f.seekg(0, std::ios::end);
    size_t size = f.tellg() / sizeof(float);
    f.seekg(0, std::ios::beg);

    std::vector<float> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    return data;
}

std::vector<uint64_t> load_u64vec(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir: " + path);

    f.seekg(0, std::ios::end);
    size_t size = f.tellg() / sizeof(uint64_t);
    f.seekg(0, std::ios::beg);

    std::vector<uint64_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size * sizeof(uint64_t));
    return data;
}

// -------------------- MAIN --------------------
int main(int argc, char **argv) {
    if (argc < 7) {
        std::cerr << "Uso:\n";
        std::cerr << argv[0]
                  << " index.bin queries.bin queries_ids.bin dim k efSearch\n";
        return 1;
    }

    std::string index_path = argv[1];
    std::string query_path = argv[2];
    std::string query_ids_path = argv[3];
    int dim = std::stoi(argv[4]);
    int k = std::stoi(argv[5]);
    int efS = std::stoi(argv[6]);

    std::cout << "=== CONFIGURACIÓN ===\n";
    std::cout << "Índice: " << index_path << "\n";
    std::cout << "Queries: " << query_path << "\n";
    std::cout << "IDs queries: " << query_ids_path << "\n";
    std::cout << "Dimensión: " << dim << "\n";
    std::cout << "k (vecinos): " << k << "\n";
    std::cout << "efSearch: " << efS << "\n";

    // Cargar queries e IDs
    std::cout << "\nCargando datos...\n";
    auto queries = load_fvec(query_path);
    auto q_ids = load_u64vec(query_ids_path);
    
    std::cout << "Queries cargadas: " << queries.size() << " floats\n";
    std::cout << "IDs cargados: " << q_ids.size() << " IDs\n";

    // Calcular cuántas queries hay automáticamente
    size_t Q = queries.size() / dim;
    std::cout << "Queries a procesar: " << Q << "\n";
    
    // Ajustar IDs si es necesario
    if (q_ids.size() < Q) {
        std::cout << "ADVERTENCIA: Menos IDs (" << q_ids.size() 
                  << ") que queries (" << Q << "). Usando IDs disponibles.\n";
        Q = q_ids.size();
    } else if (q_ids.size() > Q) {
        std::cout << "ADVERTENCIA: Más IDs (" << q_ids.size() 
                  << ") que queries (" << Q << "). Usando primeras " << Q << " IDs.\n";
    }

    std::cout << "\nCargando índice...\n";
    
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path);
    index.setEf(efS);

    MemoryMonitor::print_memory_usage("Inicio");

    std::vector<double> latencies(Q);

    std::cout << "\nEjecutando " << Q << " queries (SECUENCIAL)...\n";

    auto start_total = std::chrono::steady_clock::now();

    for (size_t i = 0; i < Q; i++) {
        const float *q = queries.data() + i * dim;

        auto start = std::chrono::steady_clock::now();
        auto results = index.searchKnn(q, k);
        auto end = std::chrono::steady_clock::now();

        latencies[i] =
            std::chrono::duration<double, std::milli>(end - start).count();

        // Limpiar resultados
        while (!results.empty()) results.pop();
        
        // Mostrar progreso cada 10000 queries
        if (i % 10000 == 0 && i > 0) {
            std::cout << "Progreso: " << i << "/" << Q 
                      << " (" << (i * 100 / Q) << "%)" << std::endl;
        }
    }

    auto end_total = std::chrono::steady_clock::now();

    MemoryMonitor::print_memory_usage("Después de queries");

    // -------------------- MÉTRICAS --------------------
    double total_time =
        std::chrono::duration<double>(end_total - start_total).count();

    double avg_latency =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / Q;

    double qps = Q / total_time;

    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies[Q * 0.50];
    double p95 = latencies[Q * 0.95];
    double p99 = latencies[Q * 0.99];

    // -------------------- RESULTADOS --------------------
    std::cout << "\n=== RESULTADOS ===\n";
    std::cout << "Queries procesadas: " << Q << "\n";
    std::cout << "Tiempo total: " << total_time << " s\n";
    std::cout << "QPS: " << qps << "\n";
    std::cout << "Latencia promedio: " << avg_latency << " ms\n";
    std::cout << "P50: " << p50 << " ms\n";
    std::cout << "P95: " << p95 << " ms\n";
    std::cout << "P99: " << p99 << " ms\n";

    // -------------------- CSV --------------------
    std::ofstream csv("basic_query_metrics.csv");
    csv << "query_id,latency_ms\n";
    for (size_t i = 0; i < Q; i++) {
        csv << q_ids[i] << "," << latencies[i] << "\n";
    }
    csv.close();

    // Guardar resumen también
    std::ofstream summary("basic_query_summary.csv");
    summary << "metric,value\n";
    summary << "queries," << Q << "\n";
    summary << "dimension," << dim << "\n";
    summary << "k," << k << "\n";
    summary << "efSearch," << efS << "\n";
    summary << "total_time_s," << total_time << "\n";
    summary << "qps," << qps << "\n";
    summary << "avg_latency_ms," << avg_latency << "\n";
    summary << "p50_ms," << p50 << "\n";
    summary << "p95_ms," << p95 << "\n";
    summary << "p99_ms," << p99 << "\n";
    summary << "peak_rss_mb," << MemoryMonitor::get_peak_rss_mb() << "\n";
    summary.close();

    std::cout << "\nMétricas guardadas en:\n";
    std::cout << "1. basic_query_metrics.csv\n";
    std::cout << "2. basic_query_summary.csv\n";

    return 0;
}