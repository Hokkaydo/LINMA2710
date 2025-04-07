#include "distributedmatrix.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Benchmark time multiply between big matrices for multiple sizes and number of processes
    int sizes[] = {1000, 2000, 3000, 4000, 5000};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    if (rank == 0) {
        std::cout << "Size,Time" << std::endl;
    }
    for (int i = 0; i < numSizes; ++i) {
        int size = sizes[i];
        Matrix A(size, size);
        Matrix B(size, size);
        
        // Initialize matrices with random values
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                A.set(j, k, static_cast<double>(rand()) / RAND_MAX);
                B.set(j, k, static_cast<double>(rand()) / RAND_MAX);
            }
        }
        
        // Create distributed matrices
        DistributedMatrix distA(A, numProcesses);
        DistributedMatrix distB(B, numProcesses);
        
        // Start timing
        double startTime = MPI_Wtime();
        
        // Perform multiplication
        Matrix result = distA.multiplyTransposed(distB);
        
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