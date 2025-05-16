#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "matrix_opencl.hpp"

std::vector<float> fill_random(int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(rows*cols);
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./matrix_mul_exec <size> <runs>" << std::endl;
        return 1;
    }

    int size = std::stoi(argv[1]);
    int runs = std::stoi(argv[2]);

    // 1. --- OpenCL Setup ---
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if (devices.empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            return 1;
        }
    }
    cl::Device device = devices.front();

    cl::Context context(device);
    cl_int err;
    cl_command_queue cq = clCreateCommandQueue(context(), device(), CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << err << std::endl;
        exit(1);
    }
    cl::CommandQueue queue(cq, true);


    std::vector<cl::Device> devices_to_init = {device};
    try {
        MatrixCL::initializeKernels(context, devices_to_init);
    } catch (const std::exception& e) {
        // Catching std::exception here because initializeKernels wraps cl::Error
        std::cerr << "FATAL ERROR during kernel initialization: " << e.what() << std::endl;
        // If the error was a BuildError, the log should have been printed
        // by the loadAndBuildProgram function within initializeKernels.
        return 1;
    }

    std::chrono::duration<double, std::milli> total_time(0);
    for (int i = 0; i < runs; ++i) {

        // 2. --- Matrix Multiplication ---
        std::vector<float> dataA = fill_random(size, size);
        std::vector<float> dataB = fill_random(size, size);

        MatrixCL A(size, size, context, queue, &dataA);
        MatrixCL B(size, size, context, queue, &dataB);
        
        auto start = std::chrono::high_resolution_clock::now();

        #ifdef FAST_MATMUL
            MatrixCL C = A.fast_matrix_mul(B);
        #else
            MatrixCL C = A * B;
        #endif
            queue.finish();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time += elapsed;
    }
    std::cout << total_time.count() / runs << std::endl;
    return 0;
}
