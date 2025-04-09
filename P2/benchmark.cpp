#include "distributedmatrix.hpp"
#include <iostream>
#include <math.h>

int main(int argc, char** argv) {
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Benchmark time multiply between big matrices for multiple sizes and number of processes
    int sizeExpRangeEnd = 4; // 10^5
    int sizeExpRangeStart = 1; // 10^1
    int maxIterations = 20;
    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    if (rank == 0) {
        std::cout << "Size,Time" << std::endl;
    }
    for (int i = 0; i < maxIterations; ++i) {
        int size = (int)pow(10, (sizeExpRangeEnd - sizeExpRangeStart) * (double)(i+1) / (double)maxIterations + sizeExpRangeStart);
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
        double startTime = MPI_Wtime();
        
        // Perform multiplication
        Matrix res = distA.multiplyTransposed(distB);
        
        // End timing
        double endTime = MPI_Wtime();
        
        if (rank == 0) {
            std::cout << size << "," << (endTime - startTime) << std::endl;
        }
    }
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}