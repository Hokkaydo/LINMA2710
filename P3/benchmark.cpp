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
    if (argc < 2) {
        std::cerr << "Usage: ./matrix_mul_exec <size>" << std::endl;
        return 1;
    }

    int size = std::stoi(argv[1]);

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
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE); // Keep profiling enabled

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

    std::vector<float> dataA = fill_random(size, size);
    std::vector<float> dataB = fill_random(size, size);

    MatrixCL A(size, size, context, queue, &dataA);
    MatrixCL B(size, size, context, queue, &dataB);

#ifdef FAST_MATMUL
    MatrixCL C = A.fast_matrix_mul(B);
#else
    MatrixCL C = A * B;
    queue.finish();
#endif
    queue.finish();
    return 0;
}
