CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O0
TARGET = tests
SRC = mlp_sgd.cpp tests.cpp globals.cpp matrix.cpp bench-runner.cpp
BENCH_SRC = mlp_sgd.cpp globals.cpp matrix.cpp bench-runner.cpp
BENCH_M = 100:1000:100 
BENCH_N = 100:1000:100
BENCH_K = 100:1000:100
BENCH_OUT = bench-results.csv
BENCH_TARGET = bench-runner

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

bench-runner: $(BENCH_SRC)
	$(CXX) $(CXXFLAGS) -o $(BENCH_TARGET) $(BENCH_SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f $(BENCH_TARGET)
	rm -f $(BENCH_OUT)

bench: $(BENCH_TARGET) benchmark.py
	./$(BENCH_TARGET) $(BENCH_M) $(BENCH_N) $(BENCH_K) > $(BENCH_OUT)
	python3 benchmark.py $(BENCH_OUT)
	
.PHONY: all run clean
