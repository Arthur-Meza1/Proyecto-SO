#include "hnswlib.h"
#include "memory_utils.hpp"
#include "timing.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  std::cout << "=== PRUEBA DE CONSTRUCCIÓN HNSW ===\n";

  // Parámetros pequeños para prueba
  int dim = 128;
  size_t num_elements = 1000000;
  int num_threads = 2;
  int M = 16;
  int ef_construction = 100;

  std::cout << "Configuración: " << num_elements << " elementos, "
            << num_threads << " threads\n";

  std::cout << "Generando datos...\n";
  std::vector<float> data(num_elements * dim);
  for (size_t i = 0; i < num_elements * dim; i++) {
    data[i] = static_cast<float>(i) / (num_elements * dim);
  }
  std::cout << "Datos generados\n";

  std::cout << "Creando índice...\n";
  hnswlib::L2Space space(dim);
  hnswlib::HierarchicalNSW<float> index(&space, num_elements, M,
                                        ef_construction);
  std::cout << "Índice creado\n";

  // 3. Insertar puntos
  std::cout << "Insertando puntos...\n";
  for (size_t i = 0; i < num_elements; i++) {
    float *vector = data.data() + i * dim;
    index.addPoint(vector, i);

    if (i % 100 == 0) {
      std::cout << "Insertado " << i << "/" << num_elements << "\n";
    }
  }

  std::cout << "Todos los puntos insertados\n";

  // 4. Guardar
  index.saveIndex("debug_index.bin");
  std::cout << "Índice guardado como debug_index.bin\n";

  std::cout << "PRUEBA EXITOSA!\n";
  return 0;
}