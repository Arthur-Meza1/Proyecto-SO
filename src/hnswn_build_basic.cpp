#include "../includes/memory_utils.hpp"
#include "hnswlib.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cstdint>

/* =======================
   CARGA DE BINARIOS
   ======================= */

std::vector<float> load_fvec(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("No se pudo abrir: " + path);

    size_t size = f.tellg();
    if (size % sizeof(float) != 0)
        throw std::runtime_error("Archivo embeddings corrupto");

    f.seekg(0, std::ios::beg);

    std::vector<float> data(size / sizeof(float));
    f.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

std::vector<uint64_t> load_u64(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("No se pudo abrir: " + path);

    size_t size = f.tellg();
    if (size % sizeof(uint64_t) != 0)
        throw std::runtime_error("Archivo ids corrupto");

    f.seekg(0, std::ios::beg);

    std::vector<uint64_t> data(size / sizeof(uint64_t));
    f.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

/* =======================
   MAIN
   ======================= */

int main(int argc, char **argv) {
    if (argc < 7) {
        std::cerr << "Uso:\n"
                  << argv[0]
                  << " <embeddings.bin> <ids.bin> <dim> <M> <efConstruction> <output_index.bin>\n";
        return 1;
    }

    std::string emb_path = argv[1];
    std::string ids_path = argv[2];
    int dim = std::stoi(argv[3]);
    int M = std::stoi(argv[4]);
    int efC = std::stoi(argv[5]);
    std::string out_path = argv[6];

    MemoryMonitor::print_memory_usage("Inicio");

    /* =======================
       CARGA DE DATOS
       ======================= */

    auto embeddings = load_fvec(emb_path);
    auto ids = load_u64(ids_path);

    if (embeddings.size() % dim != 0)
        throw std::runtime_error("El archivo embeddings no es múltiplo de dim");

    size_t N = embeddings.size() / dim;

    if (ids.size() != N)
        throw std::runtime_error("Cantidad de embeddings e IDs no coincide");

    std::cout << "=== BUILD HNSW BÁSICO ===\n";
    std::cout << "Elementos: " << N << "\n";
    std::cout << "Dimensión: " << dim << "\n";
    std::cout << "M: " << M << "\n";
    std::cout << "efConstruction: " << efC << "\n";

    MemoryMonitor::print_memory_usage("Datos cargados");

    /* =======================
       CONSTRUCCIÓN DEL ÍNDICE
       ======================= */

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, N, M, efC);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::ofstream progress("build_progress.csv");
    progress << "inserted\n";

    for (size_t i = 0; i < N; i++) {
        index.addPoint(&embeddings[i * dim], ids[i]);

        if (i % 10000 == 0) {
            progress << i << "\n";
            std::cout << "Insertados: " << i << "/" << N << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double build_time =
        std::chrono::duration<double>(t1 - t0).count();

    progress.close();

    MemoryMonitor::print_memory_usage("Después de construir índice");

    /* =======================
       GUARDAR ÍNDICE
       ======================= */

    index.saveIndex(out_path);
    std::cout << "Índice guardado en: " << out_path << "\n";

    /* =======================
       MÉTRICAS
       ======================= */

    double throughput = N / build_time;

    std::ofstream summary("build_summary_metrics.csv");
    summary << "metric,value\n";
    summary << "elements," << N << "\n";
    summary << "dimension," << dim << "\n";
    summary << "M," << M << "\n";
    summary << "efConstruction," << efC << "\n";
    summary << "build_time_s," << build_time << "\n";
    summary << "throughput_vectors_per_s," << throughput << "\n";
    summary << "peak_rss_mb," << MemoryMonitor::get_peak_rss_mb() << "\n";
    summary.close();

    std::cout << " Métricas guardadas:\n";
    std::cout << " - build_summary_metrics.csv\n";
    std::cout << " - build_progress.csv\n";

    MemoryMonitor::print_memory_usage("Fin");
    return 0;
}
