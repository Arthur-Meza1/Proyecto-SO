#include "../includes/memory_utils.hpp"
#include "hnswlib.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <thread>
#include <vector>
#include <cstdint>

struct ThreadStats {
    size_t queries = 0;
};

class RealQueryOptimizer {
private:
    hnswlib::HierarchicalNSW<float>& index;
    int dim;
    int num_threads;

public:
    RealQueryOptimizer(hnswlib::HierarchicalNSW<float>& idx, int d, int t)
        : index(idx), dim(d), num_threads(t) {}

    std::vector<float> load_queries(const std::string& file) {
        std::ifstream f(file, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("No se puede abrir queries.bin");

        size_t size = f.tellg();
        f.seekg(0);

        std::vector<float> q(size / sizeof(float));
        f.read(reinterpret_cast<char*>(q.data()), size);
        return q;
    }

    std::vector<uint64_t> load_query_ids(const std::string& file) {
        std::ifstream f(file, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("No se puede abrir query_ids.bin");

        size_t size = f.tellg();
        f.seekg(0);

        std::vector<uint64_t> ids(size / sizeof(uint64_t));
        f.read(reinterpret_cast<char*>(ids.data()), size);
        return ids;
    }

    static void pin_cpu(int id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(id % std::thread::hardware_concurrency(), &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    void run(
        const std::vector<float>& queries,
        const std::vector<uint64_t>& query_ids,
        int k,
        int ef,
        std::vector<double>& latencies,
        std::vector<uint64_t>& processed_ids,
        std::vector<ThreadStats>& stats
    ) {
        index.setEf(ef);
        size_t n = std::min(queries.size() / dim, query_ids.size());
        latencies.resize(n);
        processed_ids.resize(n);
        stats.resize(num_threads);

        std::atomic<size_t> counter{0};
        std::vector<std::thread> threads;

        auto worker = [&](int tid) {
            pin_cpu(tid);
            while (true) {
                size_t i = counter.fetch_add(1);
                if (i >= n) break;

                auto t0 = std::chrono::high_resolution_clock::now();
                auto res = index.searchKnn(queries.data() + i * dim, k);
                while (!res.empty()) res.pop();
                auto t1 = std::chrono::high_resolution_clock::now();

                latencies[i] =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                processed_ids[i] = query_ids[i];
                stats[tid].queries++;
            }
        };

        for (int i = 0; i < num_threads; i++)
            threads.emplace_back(worker, i);
        for (auto& t : threads) t.join();
    }
};

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Uso:\n"
                  << argv[0]
                  << " <index.bin> <queries.bin> <query_ids.bin> <dim> <k> <ef> <threads>\n";
        std::cerr << "\nEjemplo:\n"
                  << argv[0] << " indice.bin queries.bin query_ids.bin 128 10 200 12\n";
        return 1;
    }

    std::string index_file = argv[1];
    std::string queries_file = argv[2];
    std::string query_ids_file = argv[3];
    int dim = std::stoi(argv[4]);
    int k = std::stoi(argv[5]);
    int ef = std::stoi(argv[6]);
    int threads = std::stoi(argv[7]);

    std::cout << "=== CONFIGURACIÓN MEJORADA ===\n";
    std::cout << "Índice: " << index_file << "\n";
    std::cout << "Queries: " << queries_file << "\n";
    std::cout << "IDs queries: " << query_ids_file << "\n";
    std::cout << "Dimensión: " << dim << "\n";
    std::cout << "k (vecinos): " << k << "\n";
    std::cout << "efSearch: " << ef << "\n";
    std::cout << "Threads: " << threads << "\n";

    MemoryMonitor::print_memory_usage("Inicio");

    // Cargar índice
    std::cout << "\nCargando índice...\n";
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_file);

    // Crear optimizador y cargar datos
    RealQueryOptimizer opt(index, dim, threads);
    
    std::cout << "Cargando queries...\n";
    auto queries = opt.load_queries(queries_file);
    
    std::cout << "Cargando IDs de queries...\n";
    auto query_ids = opt.load_query_ids(query_ids_file);

    MemoryMonitor::print_memory_usage("Datos cargados");

    // Verificar consistencia
    size_t num_queries = queries.size() / dim;
    size_t num_ids = query_ids.size();
    
    std::cout << "\n=== VERIFICACIÓN ===\n";
    std::cout << "Queries calculadas: " << num_queries << " (a partir de " 
              << queries.size() << " floats / dim " << dim << ")\n";
    std::cout << "IDs disponibles: " << num_ids << "\n";
    
    size_t Q = std::min(num_queries, num_ids);
    std::cout << "Queries a procesar: " << Q << "\n";
    
    if (num_queries > num_ids) {
        std::cout << "ADVERTENCIA: Más queries que IDs. Usando solo " << Q << " queries.\n";
    } else if (num_ids > num_queries) {
        std::cout << "ADVERTENCIA: Más IDs que queries. Usando solo " << Q << " IDs.\n";
    }

    // Ejecutar queries
    std::cout << "\n=== EJECUTANDO QUERIES (MULTITHREAD) ===\n";
    std::vector<double> latencies;
    std::vector<uint64_t> processed_ids;
    std::vector<ThreadStats> thread_stats;

    auto t0 = std::chrono::high_resolution_clock::now();
    opt.run(queries, query_ids, k, ef, latencies, processed_ids, thread_stats);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_time = std::chrono::duration<double>(t1 - t0).count();

    // Métricas
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies[latencies.size() * 0.50];
    double p95 = latencies[latencies.size() * 0.95];
    double p99 = latencies[latencies.size() * 0.99];
    double qps = latencies.size() / total_time;

    // Resultados
    std::cout << "\n=== RESULTADOS ===\n";
    std::cout << "Queries procesadas: " << latencies.size() << "\n";
    std::cout << "Threads utilizados: " << threads << "\n";
    std::cout << "Tiempo total: " << total_time << " s\n";
    std::cout << "QPS (consultas por segundo): " << qps << "\n";
    std::cout << "Latencia promedio: " << avg << " ms\n";
    std::cout << "P50 (mediana): " << p50 << " ms\n";
    std::cout << "P95: " << p95 << " ms\n";
    std::cout << "P99: " << p99 << " ms\n";
    
    // Distribución por thread
    std::cout << "\n=== DISTRIBUCIÓN POR THREAD ===\n";
    for (size_t i = 0; i < thread_stats.size(); i++) {
        std::cout << "Thread " << i << ": " << thread_stats[i].queries 
                  << " queries (" 
                  << (thread_stats[i].queries * 100.0 / latencies.size()) 
                  << "%)\n";
    }

    // Guardar resultados
    std::cout << "\n=== GUARDANDO RESULTADOS ===\n";
    
    // 1. Latencias con IDs reales
    std::ofstream qf("improved_query_metrics.csv");
    qf << "query_id,latency_ms\n";
    for (size_t i = 0; i < latencies.size(); i++) {
        qf << processed_ids[i] << "," << latencies[i] << "\n";
    }
    qf.close();
    std::cout << "1. improved_query_metrics.csv - Latencias con IDs\n";
    
    // 2. Stats por hilo
    std::ofstream tf("thread_stats.csv");
    tf << "thread,queries,percentage\n";
    for (size_t i = 0; i < thread_stats.size(); i++) {
        double percentage = (thread_stats[i].queries * 100.0) / latencies.size();
        tf << i << "," << thread_stats[i].queries << "," << percentage << "\n";
    }
    tf.close();
    std::cout << "2. thread_stats.csv - Distribución por thread\n";

    // 3. Resumen de métricas
    std::ofstream sf("improved_summary_metrics.csv");
    sf << "metric,value\n";
    sf << "queries," << latencies.size() << "\n";
    sf << "threads," << threads << "\n";
    sf << "dimension," << dim << "\n";
    sf << "k," << k << "\n";
    sf << "efSearch," << ef << "\n";
    sf << "total_time_s," << total_time << "\n";
    sf << "qps," << qps << "\n";
    sf << "avg_latency_ms," << avg << "\n";
    sf << "real_avg_latency_ms," << (total_time * 1000.0 / latencies.size()) << "\n";
    sf << "p50_ms," << p50 << "\n";
    sf << "p95_ms," << p95 << "\n";
    sf << "p99_ms," << p99 << "\n";
    sf << "peak_rss_mb," << MemoryMonitor::get_peak_rss_mb() << "\n";
    sf.close();
    std::cout << "3. improved_summary_metrics.csv - Resumen completo\n";

    MemoryMonitor::print_memory_usage("Fin");
    
    return 0;
}