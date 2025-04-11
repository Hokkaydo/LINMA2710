#include "distributedmatrix.hpp"
#include <iostream>
#include <math.h>
#include <chrono>

int main(int argc, char** argv) {
    // Initialize MPI
    if( argc < 2 ) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }
    int size = atoi(argv[1]);
    if( size <= 0 ) {
        std::cerr << "Size must be a positive integer." << std::endl;
        return 1;
    }
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Benchmark time multiply between big matrices for multiple sizes and number of processes
    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    Matrix A(size, size);
    Matrix B(size, size);
    
    // Initialize matrices with random values
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < size; ++k) {
            A.set(j, k, (double)rand() / (double)RAND_MAX);
            B.set(j, k, (double)rand() / (double)RAND_MAX);
        }
    }
    
    // Create distributed matrices
    DistributedMatrix distA(A, numProcesses);
    DistributedMatrix distB(B, numProcesses);
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Perform multiplication
    Matrix res = distA.multiplyTransposed(distB);
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);

    if (rank == 0) {
        std::cout << size << "," << duration.count() << std::endl;
    }
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}