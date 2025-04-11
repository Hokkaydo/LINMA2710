#include "distributedmatrix.hpp"
#include <iostream>
#include <math.h>
#include <chrono>

int bench(int size, int numProcesses) {
    Matrix A(size, size);
    Matrix B(size, size);
    
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < size; ++k) {
            A.set(j, k, (double)rand() / (double)RAND_MAX);
            B.set(j, k, (double)rand() / (double)RAND_MAX);
        }
    }
    
    DistributedMatrix distA(A, numProcesses);
    DistributedMatrix distB(B, numProcesses);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    Matrix res = distA.multiplyTransposed(distB);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);

    return duration.count();
}

int main(int argc, char** argv) {

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank;
    int numProcesses;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    if (argc > 1) {
        int size = atoi(argv[1]);
        int time = bench(size, numProcesses);
        if (rank == 0) {
            std::cout << size << "," << time << std::endl;
        }
    } else {
        if (rank == 0) {
            std::cout << "Size,Time" << std::endl;
        }
        int sizeExpRangeEnd = 3;
        int sizeExpRangeStart = 1; 
        int maxIterations = 20;

        for (int i = 0; i < maxIterations; ++i) {
            int size = (int)pow(10, (sizeExpRangeEnd - sizeExpRangeStart) * (double)(i+1) / (double)maxIterations + sizeExpRangeStart);
            int time = bench(size, numProcesses);
            if (rank == 0) {
                std::cout << size << "," << time << std::endl;
            }
        }
    }

    MPI_Finalize();
    
    return 0;
}