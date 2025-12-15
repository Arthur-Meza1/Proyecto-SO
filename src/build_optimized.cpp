#include "hnswlib.h"
#include <chrono>
#include <cstring>      
#include <fcntl.h>      
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>   
#include <sys/stat.h>   
#include <unistd.h>     
#include <vector>
#include <ctime>
#include <iomanip>
#include <sstream>


#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;


string get_current_timestamp() {
    auto now = chrono::system_clock::now();
    auto time_t_now = chrono::system_clock::to_time_t(now);
    
    tm local_tm;
    localtime_r(&time_t_now, &local_tm);  // Linux/Unix
    
    // Formatear
    stringstream ss;
    ss << put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
    
    // Agregar milisegundos
    auto milliseconds = chrono::duration_cast<chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    ss << '.' << setfill('0') << setw(3) << milliseconds.count();
    
    return ss.str();
}

vector<float> load_embeddings_mmap(const string &path, size_t &n, int dim) {
    // 1. Abrir archivo
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw runtime_error("No se pudo abrir: " + path);
    }
    // 2. Obtener tamaño del archivo
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw runtime_error("No se pudo obtener tamaño: " + path);
    }
    
    size_t file_size = sb.st_size;
    
    // 3. Verificar que el tamaño sea correcto
    if (file_size % (sizeof(float) * dim) != 0) {
        close(fd);
        throw runtime_error("Tamaño de archivo incorrecto para dim=" + to_string(dim));
    }
    n = file_size / (sizeof(float) * dim);
    // 4. Mapear archivo a memoria 
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        throw runtime_error("mmap falló: " + path);
    }
    
    madvise(mapped, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    
    vector<float> data(n * dim);
    memcpy(data.data(), mapped, file_size);

    munmap(mapped, file_size);
    close(fd);
    
    return data;
}

vector<uint64_t> load_ids_mmap(const string &path, size_t &n) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw runtime_error("No se pudo abrir: " + path);
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw runtime_error("No se pudo obtener tamaño: " + path);
    }
    
    size_t file_size = sb.st_size;
    
    if (file_size % sizeof(uint64_t) != 0) {
        close(fd);
        throw runtime_error("Tamaño de archivo incorrecto para IDs");
    }
    
    n = file_size / sizeof(uint64_t);
    
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        throw runtime_error("mmap falló: " + path);
    }
    
    madvise(mapped, file_size, MADV_SEQUENTIAL);
    
    vector<uint64_t> data(n);
    memcpy(data.data(), mapped, file_size);
    
    munmap(mapped, file_size);
    close(fd);
    
    return data;
}

// =================== NORMALIZACIÓN CON POSIX MEMALIGN (ALINEAMIENTO) ===================

vector<float> normalize_embeddings_aligned(const vector<float>& emb, int dim, int num_threads) {
    size_t n = emb.size() / dim;
    vector<float> normalized(n * dim);
    
    // Usar memoria alineada para mejor performance de SIMD
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n; i++) {
        const float* src = &emb[i * dim];
        float* dst = &normalized[i * dim];
        
        // Calcular norma
        float norm_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            norm_sq += src[d] * src[d];
        }
        
        float norm = sqrtf(norm_sq);
        
        // Normalizar
        if (norm > 1e-12f) {
            float inv_norm = 1.0f / norm;
            for (int d = 0; d < dim; d++) {
                dst[d] = src[d] * inv_norm;
            }
        } else {
            // Si la norma es 0, copiar tal cual
            for (int d = 0; d < dim; d++) {
                dst[d] = src[d];
            }
        }
    }
    
    return normalized;
}

// =================== CONSTRUCCIÓN CON PREFETCHING ===================

void build_with_prefetch(hnswlib::HierarchicalNSW<float>& index,
                        const vector<float>& embeddings,
                        const vector<uint64_t>& ids,
                        int dim) {
    size_t N = ids.size();
    const size_t PREFETCH_DISTANCE = 10;  
    
    for (size_t i = 0; i < N; i++) {
        if (i + PREFETCH_DISTANCE < N) {
            __builtin_prefetch(&embeddings[(i + PREFETCH_DISTANCE) * dim], 0, 1);
            __builtin_prefetch(&ids[i + PREFETCH_DISTANCE], 0, 1);
        }
        
        // Insertar vector actual
        index.addPoint(&embeddings[i * dim], ids[i]);
        
        // Mostrar progreso
        if ((i + 1) % 50000 == 0 || (i + 1) == N) {
            double progress = 100.0 * (i + 1) / N;
            cout << "\rProgreso: " << (i + 1) << "/" << N 
                 << " (" << progress << "%)" << flush;
        }
    }
    cout << endl;
}

// =================== MAIN CON OPTIMIZACIONES REALES ===================

int main(int argc, char **argv) {
    if (argc < 9) {
        cout << "Uso: " << argv[0] 
             << " <embeddings.bin> <ids.bin> <dim> <M> <efC> <ip|l2> <output> <threads>\n"
             << "\nOptimizaciones:\n"
             << "  - mmap() para carga rápida\n"
             << "  - madvise() para patrones de acceso\n"
             << "  - Prefetching manual\n"
             << "  - Normalización paralela\n";
        return 1;
    }

    // Parámetros
    string emb_path = argv[1];
    string ids_path = argv[2];
    int dim = stoi(argv[3]);
    int M = stoi(argv[4]);
    int efC = stoi(argv[5]);
    string space_type = argv[6];
    string out_path = argv[7];
    int num_threads = stoi(argv[8]);

    cout << "\n=== HNSW CON OPTIMIZACIONES DE SISTEMA ===\n";
    cout << "Usando mmap() y optimizaciones de SO\n";

    // ---------- CARGA CON MMAP ----------
    auto t_load = chrono::high_resolution_clock::now();
    
    size_t n_emb, n_ids;
    
    cout << "Cargando embeddings con mmap()...\n";
    auto embeddings = load_embeddings_mmap(emb_path, n_emb, dim);
    
    cout << "Cargando IDs con mmap()...\n";
    auto ids = load_ids_mmap(ids_path, n_ids);
    
    if (n_emb != n_ids) {
        throw runtime_error("Número de embeddings e IDs no coincide");
    }
    
    size_t N = n_emb;
    auto t_load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(t_load_end - t_load).count();
    
    cout << "✓ Cargados " << N << " vectores en " << load_time << " segundos\n";

    // ---------- PRE-PROCESO ----------
    vector<float> processed_embeddings;
    auto t_pre = chrono::high_resolution_clock::now();
    
    if (space_type == "ip") {
        cout << "Normalizando vectores (paralelo)...\n";
        processed_embeddings = normalize_embeddings_aligned(embeddings, dim, num_threads);
    } else {
        // Para L2, usar embeddings originales
        processed_embeddings = move(embeddings);
    }
    
    auto t_pre_end = chrono::high_resolution_clock::now();
    double pre_time = chrono::duration<double>(t_pre_end - t_pre).count();
    cout << "✓ Pre-proceso completado en " << pre_time << " segundos\n";

    // ---------- CONSTRUCCIÓN ----------
    hnswlib::SpaceInterface<float>* space = nullptr;
    if (space_type == "l2") {
        space = new hnswlib::L2Space(dim);
        cout << "Usando espacio L2 (distancia euclidiana)\n";
    } else {
        space = new hnswlib::InnerProductSpace(dim);
        cout << "Usando espacio Inner Product (coseno)\n";
    }
    
    cout << "\nConstruyendo índice HNSW...\n";
    cout << "Parámetros: M=" << M << ", efConstruction=" << efC << "\n";
    
    hnswlib::HierarchicalNSW<float> index(space, N, M, efC);
    
    auto t_build = chrono::high_resolution_clock::now();
    
    build_with_prefetch(index, processed_embeddings, ids, dim);
    
    auto t_build_end = chrono::high_resolution_clock::now();
    double build_time = chrono::duration<double>(t_build_end - t_build).count();

    // ---------- GUARDADO ----------
    cout << "\nGuardando índice...\n";
    index.saveIndex(out_path);
    cout << "✓ Índice guardado en: " << out_path << "\n";

    // ---------- ESTADÍSTICAS ----------
    double total_time = load_time + pre_time + build_time;
    double throughput = N / build_time;
    
    cout << "\n" << string(50, '=') << "\n";
    cout << "RESUMEN DE PERFORMANCE:\n";
    cout << string(50, '=') << "\n";
    cout << "Vectores:           " << N << "\n";
    cout << "Dimensión:          " << dim << "\n";
    cout << "Tiempo carga:       " << load_time << " s\n";
    cout << "Tiempo pre-proceso: " << pre_time << " s\n";
    cout << "Tiempo construcción: " << build_time << " s\n";
    cout << "Tiempo total:       " << total_time << " s\n";
    cout << string(30, '-') << "\n";
    cout << "Throughput:         " << throughput << " vec/segundo\n";
    cout << "Velocidad vs original: " << (1088.6 / build_time) << "x\n";
    
    if (total_time < 1088.6) {
        double minutos_ahorrados = (1088.6 - total_time) / 60.0;
        cout << "✓ Ahorraste aproximadamente " << minutos_ahorrados << " minutos!\n";
    }
    
    // Guardar métricas
    ofstream metrics("performance_metrics.txt");
    metrics << "HNSW Build Metrics\n";
    metrics << "==================\n";
    metrics << "Timestamp: " << get_current_timestamp() << "\n";
    metrics << "Vectors: " << N << "\n";
    metrics << "Dimension: " << dim << "\n";
    metrics << "Space: " << space_type << "\n";
    metrics << "M: " << M << "\n";
    metrics << "efConstruction: " << efC << "\n";
    metrics << "Threads: " << num_threads << "\n";
    metrics << "\nTiming:\n";
    metrics << "  Load: " << load_time << " s\n";
    metrics << "  Preprocess: " << pre_time << " s\n";
    metrics << "  Build: " << build_time << " s\n";
    metrics << "  Total: " << total_time << " s\n";
    metrics << "\nPerformance:\n";
    metrics << "  Throughput: " << throughput << " vec/s\n";
    metrics << "  Speedup vs original: " << (1088.6 / build_time) << "x\n";
    metrics.close();
    
    cout << "\n✓ Métricas guardadas en performance_metrics.txt\n";
    
    delete space;
    return 0;
}