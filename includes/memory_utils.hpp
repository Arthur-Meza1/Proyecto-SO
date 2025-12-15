#pragma once
#include <iostream>
#include <string>

#ifdef __linux__
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#endif

class MemoryMonitor {
public:
  static size_t get_peak_rss_kb() {
#ifdef __linux__
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return (size_t)rusage.ru_maxrss;
#elif _WIN32
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)(info.PeakWorkingSetSize / 1024);
#else
    return 0; // No soportado en otros sistemas
#endif
  }

  static size_t get_peak_rss_mb() { return get_peak_rss_kb() / 1024; }

  static size_t get_current_rss_kb() {
#ifdef __linux__
    FILE *file = fopen("/proc/self/statm", "r");
    if (!file)
      return 0;

    long size, resident, share, text, lib, data, dt;
    if (fscanf(file, "%ld %ld %ld %ld %ld %ld %ld", &size, &resident, &share,
               &text, &lib, &data, &dt) != 7) {
      fclose(file);
      return 0;
    }
    fclose(file);

    long page_size_kb = sysconf(_SC_PAGESIZE) / 1024; // CORREGIDO: _SC_PAGESIZE
    return resident * page_size_kb;
#elif _WIN32
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)(info.WorkingSetSize / 1024);
#else
    return 0;
#endif
  }

  static void print_memory_usage(const std::string &phase) {
#ifdef __linux__
    std::cout << "[MEMORY] " << phase << " - Peak RSS: " << get_peak_rss_mb()
              << " MB, Current: " << (get_current_rss_kb() / 1024) << " MB"
              << std::endl;
#else
    std::cout << "[MEMORY] " << phase << " - Peak RSS: " << get_peak_rss_mb()
              << " MB" << std::endl;
#endif
  }
};