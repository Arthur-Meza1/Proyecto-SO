
echo "=== Configurando proyecto HNSW Optimization ==="

mkdir -p data/inputs data/outputs build

if [ ! -d "external/hnswlib" ]; then
    echo "Error: hnswlib no encontrado en external/hnswlib"
    echo "Ejecuta: git submodule update --init --recursive"
    exit 1
fi

echo "Estructura creada correctamente"
echo "Para compilar:"
echo "  cd build && cmake .. && make -j4"