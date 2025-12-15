#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <stdexcept>
#include <iostream>

class HNSWUtils {
public:
    // Generar datos sintéticos
    static std::vector<float> generate_synthetic_data(size_t num_vectors, int dim, int seed = 42) {
        std::vector<float> data(num_vectors * dim);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
        
        #pragma omp parallel for
        for (size_t i = 0; i < num_vectors * dim; ++i) {
            data[i] = distrib(rng);
        }
        return data;
    }
    
    // Guardar embeddings en binario
    static void save_embeddings_bin(const std::string& filename, 
                                   const std::vector<float>& data) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        
        size_t num_vectors = data.size();
        file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(data.data()), 
                  data.size() * sizeof(float));
    }
    
    // Cargar embeddings binarios
    static std::vector<float> load_embeddings_bin(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        
        size_t num_vectors;
        file.read(reinterpret_cast<char*>(&num_vectors), sizeof(size_t));
        
        std::vector<float> data(num_vectors);
        file.read(reinterpret_cast<char*>(data.data()), 
                 num_vectors * sizeof(float));
        return data;
    }
};