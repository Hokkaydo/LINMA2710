CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp -mavx2 -mfma -ffast-math -march=native -mtune=native -DNDEBUG
TARGET = tests
SRC = mlp_sgd.cpp tests.cpp globals.cpp matrix.cpp
BENCH_SRC = mlp_sgd.cpp globals.cpp matrix.cpp bench-runner.cpp
BENCH_M = 100:1000:100 
BENCH_N = 100:1000:100
BENCH_K = 100:1000:100
BENCH_OUT = bench-results.csv
BENCH_TARGET = bench-runner

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

bench: $(BENCH_TARGET) benchmark.py
	./$(BENCH_TARGET) $(BENCH_M) $(BENCH_N) $(BENCH_K) > $(BENCH_OUT)
	python3 benchmark.py
	
.PHONY: all run clean
