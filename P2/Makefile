CXX = mpic++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mtune=native -ffast-math -funroll-loops 
TARGET = distributedtests
OBJ = matrix.o distributedmatrix.o distributedtests.o mlp_sgd_distributed.o globals.o
HEADERS = abstractmatrix.hpp matrix.hpp distributedmatrix.hpp globals.hpp
NPROC = 4
BENCH_OUT_DIR = ./
TAU_OUT_DIR = ./

all:
	$(MAKE) clean && $(MAKE) run

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

matrix.o: matrix.cpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp

distributedmatrix.o: distributedmatrix.cpp distributedmatrix.hpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c distributedmatrix.cpp

distributedtests.o: distributedtests.cpp distributedmatrix.hpp matrix.hpp abstractmatrix.hpp
	$(CXX) $(CXXFLAGS) -c distributedtests.cpp

mlp_sgd_distributed.o: mlp_sgd_distributed.cpp globals.hpp abstractmatrix.hpp matrix.hpp distributedmatrix.hpp
	$(CXX) $(CXXFLAGS) -c mlp_sgd_distributed.cpp

globals.o: globals.cpp globals.hpp mlp_sgd_distributed.cpp
	$(CXX) $(CXXFLAGS) -c globals.cpp

run: $(TARGET)
	mpirun -np $(NPROC) ./$(TARGET)

benchmark: benchmark.cpp $(HEADERS) $(OBJ)
	@$(CXX) $(CXXFLAGS) -o benchmark benchmark.cpp matrix.o distributedmatrix.o
	@echo "Running benchmark..."
	@echo "Using $(NPROC) processes."
	@mpirun -np $(NPROC) ./benchmark > $(BENCH_OUT_DIR)benchmark_results_$(NPROC)_procs.csv
	@echo "Benchmarking completed."
	@echo "Results saved to $(BENCH_OUT_DIR)benchmark_results_$(NPROC)_procs.csv"

tau: benchmark.cpp $(HEADERS) $(OBJ)
	@$(CXX) $(CXXFLAGS) -o benchmark benchmark.cpp matrix.o distributedmatrix.o
	@echo "Running tau..."
	@echo "Using $(NPROC) processes."
	@mpirun -np $(NPROC) tau_exec ./benchmark > $(TAU_OUT_DIR)tau_results_$(NPROC)_procs.txt
	@echo "Tau completed."
	@echo "Results saved to $(TAU_OUT_DIR)tau_results_$(NPROC)_procs.txt."
clean:
	rm -f $(OBJ) $(TARGET) *.o benchmark

.PHONY: all run clean benchmark
