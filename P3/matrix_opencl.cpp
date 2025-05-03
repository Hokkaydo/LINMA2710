// P3/matrix_opencl.cpp
#include "matrix_opencl.hpp"
#include <iostream> // For error reporting during kernel build
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream> // For building kernel source string
#include <memory> 
#include <mutex>  

// ---------------------------------------------------------------------------
// Static Member Definitions
// ---------------------------------------------------------------------------
std::shared_ptr<KernelCache> MatrixCL::kernels_ = nullptr;

// ---------------------------------------------------------------------------
// Helper Function: Load and Build OpenCL Program (Used only during init)
// ---------------------------------------------------------------------------
cl::Program loadAndBuildProgram(cl::Context context,
                                const std::vector<cl::Device>& devices,
                                const std::string& sourceCode,
                                const std::string& kernel_name_for_error)
{
    cl::Program program(context, sourceCode);
    try {
        program.build(devices);
    } catch (const cl::BuildError& err) {
        std::cerr << "OpenCL Build Error for kernel source '" << kernel_name_for_error << "':\n"
                  << err.what() << "(" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog()) {
            std::cerr << "Device " << pair.first.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
            std::cerr << pair.second << std::endl;
        }
        throw;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error during program build for '" << kernel_name_for_error << "': "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    }
    return program;
}

// ---------------------------------------------------------------------------
// OpenCL Kernel Source Code Strings
// ---------------------------------------------------------------------------
const std::string kernel_source_fill = R"(
    __kernel void fill(__global float* matrix, float value, int rows, int cols) {
        int idx = get_global_id(0);
        int total_elements = rows * cols;
        if (idx < total_elements) {
            matrix[idx] = value;
        }
    }
)";
const std::string kernel_source_add = R"(
    __kernel void add(__global const float* A, __global const float* B, __global float* C, int rows, int cols) {
        int idx = get_global_id(0);
        int total_elements = rows * cols;
        if (idx < total_elements) {
            C[idx] = A[idx] + B[idx];
        }
    }
)";
const std::string kernel_source_sub_mul = R"(
    __kernel void sub_mul(__global float* A, __global const float* B, float scalar, int rows, int cols) {
        int idx = get_global_id(0);
        int total_elements = rows * cols;
        if (idx < total_elements) {
            A[idx] -= scalar * B[idx];
        }
    }
)";
const std::string kernel_source_transpose = R"(
    __kernel void transpose(__global const float* A, __global float* B, int A_rows, int A_cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < A_rows && col < A_cols) {
            B[col * A_rows + row] = A[row * A_cols + col];
        }
    }
)";
const std::string kernel_source_matrix_mul = R"(
    __kernel void matrix_mul(__global const float* A, __global const float* B, __global float* C, int A_rows, int A_cols, int B_cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < A_rows && col < B_cols) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                sum += A[row * A_cols + k] * B[k * B_cols + col];
            }
            C[row * B_cols + col] = sum;
        }
    }
)";
const std::string kernel_source_sigmoid = R"(
    __kernel void sigmoid(__global const float* input, __global float* output, int rows, int cols) {
        int idx = get_global_id(0);
        int total_elements = rows * cols;
        if (idx < total_elements) {
            float x = input[idx];
            output[idx] = 1.0f / (1.0f + exp(-x));
        }
    }
)";
const std::string kernel_source_sigmoid_backward = R"(
    __kernel void sigmoid_backward(__global float* grad_acc, __global const float* input, __global const float* out_grad, int rows, int cols) {
        int idx = get_global_id(0);
        int total_elements = rows * cols;
        if (idx < total_elements) {
            float x = input[idx];
            float s = 1.0f / (1.0f + exp(-x));
            grad_acc[idx] += out_grad[idx] * s * (1.0f - s);
        }
    }
)";
const std::string kernel_source_bce_elementwise = R"(
     __kernel void bce_elementwise(__global const float* predictions, __global const float* targets, __global float* elementwise_loss, int rows, int cols, float epsilon) {
        int idx = get_global_id(0); 
        int total_elements = rows * cols;
        if (idx < total_elements) {
            float pred = predictions[idx]; 
            float targ = targets[idx];
            float a = fmax(pred, epsilon);
            float b = fmax(1.0f - pred, epsilon);
            elementwise_loss[idx] = -(targ * log(a) + (1.0f - targ) * log(b));
        }
    }
)";
const std::string kernel_source_bce_backward = R"(
    __kernel void bce_backward(__global float* grad_acc, __global const float* predictions, __global const float* targets, int rows, int cols, float epsilon, float inv_num_elements) {
        int idx = get_global_id(0); 
        int total_elements = rows * cols;
        if (idx < total_elements) {
            float pred = predictions[idx]; 
            float targ = targets[idx];
            float a = fmax(pred, epsilon); 
            float b = fmax(1.0f - pred, epsilon);
            float bce_grad = -(targ / a - (1.0f - targ) / b);
            grad_acc[idx] += inv_num_elements * bce_grad;
        }
    }
)";

// ---------------------------------------------------------------------------
// KernelCache Implementation
// ---------------------------------------------------------------------------
void KernelCache::compileKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    if (initialized) return; // Already compiled

    std::cout << "Compiling OpenCL kernels..." << std::endl;
    try {
        cl::Program prog_fill = loadAndBuildProgram(context, devices, kernel_source_fill, "fill");
        kernel_fill = cl::Kernel(prog_fill, "fill");

        cl::Program prog_add = loadAndBuildProgram(context, devices, kernel_source_add, "add");
        kernel_add = cl::Kernel(prog_add, "add");

        cl::Program prog_sub_mul = loadAndBuildProgram(context, devices, kernel_source_sub_mul, "sub_mul");
        kernel_sub_mul = cl::Kernel(prog_sub_mul, "sub_mul");

        cl::Program prog_transpose = loadAndBuildProgram(context, devices, kernel_source_transpose, "transpose");
        kernel_transpose = cl::Kernel(prog_transpose, "transpose");

        cl::Program prog_matrix_mul = loadAndBuildProgram(context, devices, kernel_source_matrix_mul, "matrix_mul");
        kernel_matrix_mul = cl::Kernel(prog_matrix_mul, "matrix_mul");

        cl::Program prog_sigmoid = loadAndBuildProgram(context, devices, kernel_source_sigmoid, "sigmoid");
        kernel_sigmoid = cl::Kernel(prog_sigmoid, "sigmoid");

        cl::Program prog_sigmoid_bw = loadAndBuildProgram(context, devices, kernel_source_sigmoid_backward, "sigmoid_backward");
        kernel_sigmoid_backward = cl::Kernel(prog_sigmoid_bw, "sigmoid_backward");

        cl::Program prog_bce_ew = loadAndBuildProgram(context, devices, kernel_source_bce_elementwise, "bce_elementwise");
        kernel_bce_elementwise = cl::Kernel(prog_bce_ew, "bce_elementwise");

        cl::Program prog_bce_bw = loadAndBuildProgram(context, devices, kernel_source_bce_backward, "bce_backward");
        kernel_bce_backward = cl::Kernel(prog_bce_bw, "bce_backward");

        initialized = true;
        std::cout << "OpenCL kernels compiled successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to compile one or more OpenCL kernels. Aborting." << std::endl;
        throw; // Re-throw to potentially stop the program
    }
}

// ---------------------------------------------------------------------------
// MatrixCL Static Methods Implementation
// ---------------------------------------------------------------------------

// Ensures kernel cache is initialized exactly once.
void MatrixCL::initializeKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    try {
        // Only initialize if not already done
        if (!kernels_ || !kernels_->initialized) {
            std::cout << "Creating and compiling kernels directly..." << std::endl;
            kernels_ = std::make_shared<KernelCache>();
            kernels_->compileKernels(context, devices);
        }
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in direct kernel initialization: " 
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Exception in direct kernel initialization: " << e.what() << std::endl;
        throw;
    }
}


// ---------------------------------------------------------------------------
// MatrixCL Class Implementation
// ---------------------------------------------------------------------------

size_t MatrixCL::buffer_size_bytes() const {
    return static_cast<size_t>(rows_) * cols_ * sizeof(float);
}

// --- Constructors ---
// Creates a matrix initialized with zero elements or optional initial data
MatrixCL::MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data) : rows_(rows), cols_(cols), context_(context), queue_(queue) {
    buffer_ = cl::Buffer(context, CL_MEM_READ_WRITE, buffer_size_bytes());

    if (initial_data) {
        queue.enqueueWriteBuffer(buffer_, CL_TRUE, 0, buffer_size_bytes(), initial_data->data());
    } else {
        fill(0.0f);
    }
}

// Copy constructor (performs device-to-device copy)
MatrixCL::MatrixCL(const MatrixCL& other): rows_(other.rows_), cols_(other.cols_), context_(other.context_), queue_(other.queue_)
{
    buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, buffer_size_bytes());
    queue_.enqueueCopyBuffer(other.buffer_, buffer_, 0, 0, buffer_size_bytes());
}

// Copy assignment operator
MatrixCL& MatrixCL::operator=(const MatrixCL& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        context_ = other.context_;
        queue_ = other.queue_;
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, other.buffer_size_bytes());
        
        queue_.enqueueCopyBuffer(other.buffer_, buffer_, 0, 0, buffer_size_bytes());
    }
    return *this;
}

// Getters
int MatrixCL::numRows() const {
    return rows_;
}
int MatrixCL::numCols() const {
    return cols_;
}
cl::Context MatrixCL::getContext() const {
    return context_;
}
cl::CommandQueue MatrixCL::getQueue() const {
    return queue_;
}
const cl::Buffer& MatrixCL::getBuffer() const {
    return buffer_;
} // Read-only access to buffer

// Copy data from device buffer back to host in an std::vector
std::vector<float> MatrixCL::copyToHost() const {
    std::vector<float> host_data(rows_ * cols_);
    queue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, buffer_size_bytes(), host_data.data());
    return host_data;
}

// --- Operations (Must be implemented with OpenCL Kernels) ---
// Fill the entire matrix with a single value
void MatrixCL::fill(float value) {
    cl::Kernel kernel = kernels_->kernel_fill;
    kernel.setArg(0, buffer_); 
    kernel.setArg(1, value);
    kernel.setArg(2, rows_);
    kernel.setArg(3, cols_);

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_ * cols_), cl::NullRange);
}

// Addition: C = A + B
MatrixCL MatrixCL::operator+(const MatrixCL& other) const {
    MatrixCL result(rows_, cols_, context_, queue_);
    
    cl::Kernel kernel = kernels_->kernel_add; 
    kernel.setArg(0, buffer_);
    kernel.setArg(1, other.getBuffer());
    kernel.setArg(2, result.getBuffer()); 
    kernel.setArg(3, rows_);
    kernel.setArg(4, cols_);

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_ * cols_), cl::NullRange);
    
    return result;
}

// Matrix multiplication: C = A * B
MatrixCL MatrixCL::operator*(const MatrixCL& other) const {
    MatrixCL result(rows_, other.numCols(), context_, queue_);

    cl::Kernel kernel = kernels_->kernel_matrix_mul; 
    kernel.setArg(0, buffer_);
    kernel.setArg(1, other.getBuffer());
    kernel.setArg(2, result.getBuffer()); 
    kernel.setArg(3, rows_);
    kernel.setArg(4, cols_);
    kernel.setArg(5, other.numCols());

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_, cols_), cl::NullRange);

    return result;
}

// Transpose: returns a new Matrix that is the transpose (B = A^T)
MatrixCL MatrixCL::transpose() const {
    MatrixCL result(cols_, rows_, context_, queue_);
    
    cl::Kernel kernel = kernels_->kernel_transpose;
    kernel.setArg(0, buffer_);
    kernel.setArg(1, result.getBuffer());
    kernel.setArg(2, rows_);
    kernel.setArg(3, cols_);
    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_, cols_));
    
    return result;
}

// Subtract the product of a scalar and a given matrix: "this = this - scalar * other"
// Performs the operation in-place on 'this' matrix's buffer.
void MatrixCL::sub_mul(float scalar, const MatrixCL& other) {
    cl::Kernel kernel = kernels_->kernel_sub_mul; 

    kernel.setArg(0, buffer_); 
    kernel.setArg(1, other.getBuffer());
    kernel.setArg(2, scalar);
    kernel.setArg(3, rows_);
    kernel.setArg(4, cols_);

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_ * cols_));
}

// Applies sigmoid element-wise: Returns a matrix containing sigmoid(this)
MatrixCL MatrixCL::sigmoid() const {
    MatrixCL result(rows_, cols_, context_, queue_);
    
    cl::Kernel kernel = kernels_->kernel_sigmoid; 

    kernel.setArg(0, buffer_); 
    kernel.setArg(1, result.getBuffer()); 
    kernel.setArg(2, rows_);
    kernel.setArg(3, cols_);

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_ * cols_));

    return result;
}
// Calculates gradient for sigmoid and adds it to 'this' matrix (gradient accumulator).
void MatrixCL::sigmoid_backward(const MatrixCL& input_values, const MatrixCL& output_gradient) {
    cl::Kernel kernel = kernels_->kernel_sigmoid_backward;  

    kernel.setArg(0, this->buffer_);                        // gradient_accumulator
    kernel.setArg(1, input_values.getBuffer());             
    kernel.setArg(2, output_gradient.getBuffer());          
    kernel.setArg(3, rows_);
    kernel.setArg(4, cols_);

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_ * cols_));
}

// Calculates Binary Cross-Entropy Loss between the entries of 'this' matrix and the target matrix element-wise. Returns a MatrixCL containing the losses.
MatrixCL MatrixCL::binary_cross_entropy(const MatrixCL& targets) const {
    MatrixCL result(rows_, cols_, context_, queue_);

    const float epsilon = 1e-8f;
    cl::Kernel kernel = kernels_->kernel_bce_elementwise; 

    kernel.setArg(0, buffer_);                              // predictions
    kernel.setArg(1, targets.getBuffer());                  
    kernel.setArg(2, result.getBuffer());                   // elementwise_loss
    kernel.setArg(3, rows_);
    kernel.setArg(4, cols_);
    kernel.setArg(5, epsilon);                              

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows_ * cols_));

    return result;
}

void MatrixCL::binary_cross_entropy_backward(const MatrixCL& predictions, const MatrixCL& targets) {
    const size_t num_elements = rows_ * cols_;

    const float epsilon = 1e-8f;
    cl::Kernel kernel = kernels_->kernel_bce_backward;  

    kernel.setArg(0, this->buffer_);                    // gradient_accumulator
    kernel.setArg(1, predictions.getBuffer());          
    kernel.setArg(2, targets.getBuffer());              
    kernel.setArg(3, rows_);
    kernel.setArg(4, cols_);
    kernel.setArg(5, epsilon);
    kernel.setArg(6, 1.0f/num_elements); 

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_elements));
}