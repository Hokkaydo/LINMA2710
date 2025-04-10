CXX = mpic++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mtune=native -ffast-math -funroll-loops 
TARGET = distributedtests
OBJ = matrix.o distributedmatrix.o distributedtests.o mlp_sgd_distributed.o globals.o
BENCH_OBJ = benchmark.o matrix.o distributedmatrix.o
HEADERS = abstractmatrix.hpp matrix.hpp distributedmatrix.hpp globals.hpp
NPROC = 4
OUT_DIR = ./
$(shell mkdir -p $(OUT_DIR))

all:
	$(MAKE) clean && $(MAKE) run

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

matrix.o: matrix.cpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp
	ls -la

distributedmatrix.o: distributedmatrix.cpp distributedmatrix.hpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c distributedmatrix.cpp
	ls -la

distributedtests.o: distributedtests.cpp distributedmatrix.hpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c distributedtests.cpp

mlp_sgd_distributed.o: mlp_sgd_distributed.cpp globals.hpp abstractmatrix.hpp matrix.hpp distributedmatrix.hpp
	$(CXX) $(CXXFLAGS) -c mlp_sgd_distributed.cpp

globals.o: globals.cpp globals.hpp mlp_sgd_distributed.cpp
	$(CXX) $(CXXFLAGS) -c globals.cpp
	ls -la

run: $(TARGET)
	mpirun -np $(NPROC) ./$(TARGET)

build-benchmark: benchmark.cpp $(HEADERS) $(BENCH_OBJ)
	@$(CXX) $(CXXFLAGS) -o benchmark benchmark.cpp matrix.o distributedmatrix.o
	@echo "Using $(NPROC) processes."

benchmark: build-benchmark
	@echo "Running benchmark..."
	@mpirun -np $(NPROC) ./benchmark > $(OUT_DIR)/procs_$(NPROC).csv
	@echo "Benchmarking completed."
	@echo "Results saved to $(OUT_DIR)/procs_$(NPROC).csv"
	
nsys: build-benchmark
	@echo "Running nsys..."
	@echo "Using $(NPROC) processes."
	@nsys profile --trace=mpi,cuda,osrt,cublas,nvtx --mpi-impl=openmpi --stats=true mpirun -n $(NPROC) ./benchmark > $(OUT_DIR)/procs_$(NPROC).txt
	@echo "NSYS profiling completed."

clean:
	rm -f $(OBJ) $(TARGET) *.o benchmark

.PHONY: all run clean benchmark
