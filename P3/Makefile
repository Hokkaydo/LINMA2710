CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 
TARGET = main
SRC = matrix_opencl.cpp mlp_sgd.cpp globals.cpp
LIBS = -lOpenCL

all:
	$(MAKE) clean && $(MAKE) run

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET)$(SUFFIX) $(SRC) main.cpp $(LIBS)

run: $(TARGET)
	./$(TARGET)

fast: benchmark.cpp
	$(CXX) $(CXXFLAGS) -DFAST_MATMUL -o benchmark$(SUFFIX) benchmark.cpp $(LIBS) $(SRC)

naive: benchmark.cpp
	$(CXX) $(CXXFLAGS) -o benchmark$(SUFFIX) benchmark.cpp $(LIBS) $(SRC)

clean:
	rm -f $(TARGET)
	rm -f benchmark


.PHONY: all run clean
