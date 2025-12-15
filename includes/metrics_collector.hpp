#pragma once
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

class MetricsCollector {
private:
  std::map<std::string, std::vector<double>> metrics;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
  MetricsCollector() { start_time = std::chrono::high_resolution_clock::now(); }

  void record(const std::string &metric, double value) {
    metrics[metric].push_back(value);
  }

  template <typename TimePoint>
  void record_latency(const std::string &operation, TimePoint start) {
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    record(operation + "_latency_ms", ms);
  }

  void print_summary() {
    std::cout << "\n=== RESUMEN DE MÉTRICAS ===\n";
    for (const auto &[metric, values] : metrics) {
      if (values.empty())
        continue;

      double sum = std::accumulate(values.begin(), values.end(), 0.0);
      double max = *std::max_element(values.begin(), values.end());
      double min = *std::min_element(values.begin(), values.end());
      double avg = sum / values.size();

      // Calcular percentiles
      std::vector<double> sorted = values;
      std::sort(sorted.begin(), sorted.end());
      double p50 = sorted[sorted.size() * 0.5];
      double p95 = sorted[sorted.size() * 0.95];
      double p99 = sorted[sorted.size() * 0.99];

      std::cout << metric << ": " << avg << " ms (min: " << min
                << ", max: " << max << ", p50: " << p50 << ", p95: " << p95
                << ", p99: " << p99 << ", n: " << values.size() << ")\n";
    }
  }

  void save_to_csv(const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error abriendo archivo: " << filename << "\n";
      return;
    }

    // Encabezados
    file << "timestamp";
    for (const auto &[metric, _] : metrics) {
      file << "," << metric;
    }
    file << "\n";

    // Datos
    size_t max_points = 0;
    for (const auto &[_, values] : metrics) {
      if (values.size() > max_points)
        max_points = values.size();
    }

    for (size_t i = 0; i < max_points; ++i) {
      file << i;
      for (const auto &[metric, values] : metrics) {
        file << "," << (i < values.size() ? values[i] : 0.0);
      }
      file << "\n";
    }

    std::cout << "Métricas guardadas en: " << filename << "\n";
  }

  std::vector<double> get_metric(const std::string &metric) const {
    auto it = metrics.find(metric);
    if (it != metrics.end())
      return it->second;
    return {};
  }
};