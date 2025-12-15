cmake_minimum_required(VERSION 3.15)
project(HNSW_OS_Optimized CXX)

# Configuración básica
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Optimizaciones GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -DNDEBUG -fopenmp)
endif()

# Includes - RUTA CORRECTA
include_directories(
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/external/hnswlib/hnswlib  
)

# OpenMP
find_package(OpenMP REQUIRED)

# Ejecutables
add_executable(hnsw_build_optimized src/build_optimized.cpp)
target_link_libraries(hnsw_build_optimized OpenMP::OpenMP_CXX pthread)

add_executable(hnsw_query_optimized src/query_optimized.cpp)
target_link_libraries(hnsw_query_optimized OpenMP::OpenMP_CXX pthread)

add_executable(hnswn_build_basic src/hnswn_build_basic.cpp)

add_executable(hnswn_query_basic src/hnsw_query_basic.cpp)

# Información
message(STATUS "=====================================")
message(STATUS "HNSW Optimized - Build Ready")
message(STATUS "OpenMP: YES")
message(STATUS "Includes: external/hnswlib/hnswlib")  
message(STATUS "=====================================")