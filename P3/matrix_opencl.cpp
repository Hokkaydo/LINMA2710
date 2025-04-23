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
        int idx = get_global_id(0); int total_elements = rows * cols;
        if (idx < total_elements) {
            float pred = predictions[idx]; float targ = targets[idx];
            float denominator1 = max(pred + epsilon, epsilon); // Avoid exactly zero denominator
            float denominator2 = max(1.0f - pred + epsilon, epsilon);
            elementwise_loss[idx] = -(targ * log(denominator1) + (1.0f - targ) * log(denominator2));
        }
    }
)";
const std::string kernel_source_bce_backward = R"(
    __kernel void bce_backward(__global float* grad_acc, __global const float* predictions, __global const float* targets, int rows, int cols, float epsilon, float inv_num_elements) {
        int idx = get_global_id(0); int total_elements = rows * cols;
        if (idx < total_elements) {
            float pred = predictions[idx]; float targ = targets[idx];
            float denominator1 = max(pred + epsilon, epsilon); // Avoid exactly zero denominator
            float denominator2 = max(1.0f - pred + epsilon, epsilon);
            float bce_grad = -(targ / denominator1 - (1.0f - targ) / denominator2);
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
MatrixCL::MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    rows_ = rows;
    cols_ = cols;
    context_ = context;
    queue_ = queue;

    buffer_ = cl::Buffer(context, CL_MEM_READ_WRITE, buffer_size_bytes());

    if (initial_data) {
        if (initial_data->size() != static_cast<size_t>(rows) * cols) {
            throw std::invalid_argument("Initial data size does not match matrix dimensions.");
        }
        queue.enqueueWriteBuffer(buffer_, CL_TRUE, 0, buffer_size_bytes(), initial_data->data());
    } else {
        fill(0.0f);
    }
}

// Copy constructor (performs device-to-device copy)
MatrixCL::MatrixCL(const MatrixCL& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    context_ = other.context_;
    queue_ = other.queue_;
    buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, other.buffer_size_bytes(), nullptr);
    
    queue_.enqueueCopyBuffer(other.buffer_, buffer_, 0, 0, buffer_size_bytes());
}

// Destructor (cl::Buffer manages its own release via RAII)

// Copy assignment operator
MatrixCL& MatrixCL::operator=(const MatrixCL& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        context_ = other.context_;
        queue_ = other.queue_;
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, other.buffer_size_bytes(), nullptr);
        
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
    if (rows_ <= 0 || cols_ <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    try {
        cl::Kernel kernel = kernels_->kernel_fill;
        kernel.setArg(0, buffer_); 
        kernel.setArg(1, value);
        kernel.setArg(2, rows_);
        kernel.setArg(3, cols_);

        size_t global_work_size = static_cast<size_t>(rows_) * cols_;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during fill: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
        throw std::runtime_error("Error during fill: " + std::string(err.what()));
    }
}

// Addition: C = A + B
MatrixCL MatrixCL::operator+(const MatrixCL& other) const {
    if (rows_ != other.numRows() || cols_ != other.numCols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    if (context_() != other.getContext()() || queue_() != other.getQueue()()) {
        throw std::runtime_error("Cannot add matrices from different OpenCL contexts or queues.");
    }

    MatrixCL result(rows_, cols_, context_, queue_);
    try {
        cl::Kernel kernel = kernels_->kernel_add; 
        kernel.setArg(0, buffer_);
        kernel.setArg(1, other.getBuffer());
        kernel.setArg(2, result.getBuffer()); 
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);

        size_t global_work_size = static_cast<size_t>(rows_) * cols_;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during addition: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
        throw std::runtime_error("Error during addition: " + std::string(err.what()));
    }
    return result;
}

// Matrix multiplication: C = A * B
MatrixCL MatrixCL::operator*(const MatrixCL& other) const {
    if (cols_ != other.numRows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    if (context_() != other.getContext()() || queue_() != other.getQueue()()) {
        throw std::runtime_error("Cannot multiply matrices from different OpenCL contexts or queues.");
    }

    MatrixCL result(rows_, other.numCols(), context_, queue_);
    try {
        cl::Kernel kernel = kernels_->kernel_matrix_mul; 
        kernel.setArg(0, buffer_);
        kernel.setArg(1, other.getBuffer());
        kernel.setArg(2, result.getBuffer()); 
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);
        kernel.setArg(5, other.numCols());

        size_t global_work_size = static_cast<size_t>(rows_) * other.numCols();
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during matrix multiplication: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
        throw std::runtime_error("Error during matrix multiplication: " + std::string(err.what()));
    }
    return result;
}

// Transpose: returns a new Matrix that is the transpose (B = A^T)
MatrixCL MatrixCL::transpose() const {
    MatrixCL result(cols_, rows_, context_, queue_);
    try {
        cl::Kernel kernel = kernels_->kernel_transpose;
        kernel.setArg(0, buffer_);
        kernel.setArg(1, result.getBuffer());
        kernel.setArg(2, rows_);
        kernel.setArg(3, cols_);
        size_t global_work_size = static_cast<size_t>(rows_) * cols_;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
    } catch (const cl::Error& err) {    
        throw std::runtime_error("OpenCL error during transpose: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
        throw std::runtime_error("Error during transpose: " + std::string(err.what()));
    }
    return result;
}

// Subtract the product of a scalar and a given matrix: "this = this - scalar * other"
// Performs the operation in-place on 'this' matrix's buffer.
void MatrixCL::sub_mul(float scalar, const MatrixCL& other) {
    if (rows_ != other.numRows() || cols_ != other.numCols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    if (context_() != other.getContext()() || queue_() != other.getQueue()()) {
        throw std::runtime_error("Cannot subtract matrices from different OpenCL contexts or queues.");
    }

    try {
        cl::Kernel kernel = kernels_->kernel_sub_mul; 
        kernel.setArg(0, buffer_); 
        kernel.setArg(1, other.getBuffer());
        kernel.setArg(2, scalar);
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);

        size_t global_work_size = static_cast<size_t>(rows_) * cols_;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during subtraction: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
        throw std::runtime_error("Error during subtraction: " + std::string(err.what()));
    }
}

// Applies sigmoid element-wise: Returns a matrix containing sigmoid(this)
MatrixCL MatrixCL::sigmoid() const {
    MatrixCL result(rows_, cols_, context_, queue_);
    try {
        cl::Kernel kernel = kernels_->kernel_sigmoid; 
        kernel.setArg(0, buffer_); 
        kernel.setArg(1, result.getBuffer()); 
        kernel.setArg(2, rows_);
        kernel.setArg(3, cols_);

        size_t global_work_size = static_cast<size_t>(rows_) * cols_;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during sigmoid: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
        throw std::runtime_error("Error during sigmoid: " + std::string(err.what()));
    }
    return result;
}
// Calculates gradient for sigmoid and adds it to 'this' matrix (gradient accumulator).
void MatrixCL::sigmoid_backward(const MatrixCL& input_values, const MatrixCL& output_gradient) {
    if (rows_ != input_values.numRows() || cols_ != input_values.numCols() ||
        rows_ != output_gradient.numRows() || cols_ != output_gradient.numCols()) {
        throw std::invalid_argument("Matrix dimensions must match for sigmoid_backward.");
    }
    if (context_() != input_values.getContext()() || queue_() != input_values.getQueue()() ||
        context_() != output_gradient.getContext()() || queue_() != output_gradient.getQueue()()) {
         throw std::runtime_error("Cannot perform sigmoid backward update on matrices from different OpenCL contexts or queues.");
    }

    size_t num_elements = static_cast<size_t>(rows_) * cols_;
     if (num_elements == 0) return;

    try {
        cl::Kernel kernel = kernels_->kernel_sigmoid_backward; // Use cached kernel

        kernel.setArg(0, this->buffer_);            // gradient_accumulator (read-write)
        kernel.setArg(1, input_values.getBuffer());  // input_values (read-only)
        kernel.setArg(2, output_gradient.getBuffer()); // output_gradient (read-only)
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);

        size_t global_work_size = num_elements;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);

    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during sigmoid_backward: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
         throw std::runtime_error("Error during sigmoid_backward: " + std::string(err.what()));
    }
}

// Calculates Binary Cross-Entropy Loss between the entries of 'this' matrix and the target matrix element-wise. Returns a MatrixCL containing the losses.
MatrixCL MatrixCL::binary_cross_entropy(const MatrixCL& targets) const {
    if (rows_ != targets.numRows() || cols_ != targets.numCols()) {
        throw std::invalid_argument("Matrix dimensions must match for binary_cross_entropy.");
    }
    if (context_() != targets.getContext()() || queue_() != targets.getQueue()()) {
        throw std::runtime_error("Cannot calculate BCE on matrices from different OpenCL contexts or queues.");
    }

    MatrixCL result(rows_, cols_, context_, queue_);
    try {
        cl::Kernel kernel = kernels_->kernel_bce_elementwise; // Use cached kernel
        kernel.setArg(0, buffer_);            // predictions (read-only)
        kernel.setArg(1, targets.getBuffer()); // targets (read-only)
        kernel.setArg(2, result.getBuffer());  // elementwise_loss (write-only)
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);
        kernel.setArg(5, 1e-8f); // epsilon

        size_t global_work_size = static_cast<size_t>(rows_) * cols_;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);

    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during binary_cross_entropy: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
         throw std::runtime_error("Error during binary_cross_entropy: " + std::string(err.what()));
    }
    return result;
}

void MatrixCL::binary_cross_entropy_backward(const MatrixCL& predictions, const MatrixCL& targets) {
     if (rows_ != predictions.numRows() || cols_ != predictions.numCols() ||
        rows_ != targets.numRows() || cols_ != targets.numCols()) {
        throw std::invalid_argument("Matrix dimensions must match for binary_cross_entropy_backward.");
    }
    if (context_() != predictions.getContext()() || queue_() != predictions.getQueue()() ||
        context_() != targets.getContext()() || queue_() != targets.getQueue()()) {
         throw std::runtime_error("Cannot perform BCE backward update on matrices from different OpenCL contexts or queues.");
    }

    size_t num_elements = static_cast<size_t>(rows_) * cols_;
     if (num_elements == 0) return;

    const float epsilon = 1e-8f;
    const float inv_num_elements = 1.0f / static_cast<float>(num_elements);

    try {
        cl::Kernel kernel = kernels_->kernel_bce_backward; // Use cached kernel

        kernel.setArg(0, this->buffer_);            // gradient_accumulator (read-write)
        kernel.setArg(1, predictions.getBuffer());  // predictions (read-only)
        kernel.setArg(2, targets.getBuffer());      // targets (read-only)
        kernel.setArg(3, rows_);
        kernel.setArg(4, cols_);
        kernel.setArg(5, epsilon);
        kernel.setArg(6, inv_num_elements);

        size_t global_work_size = num_elements;
        queue_.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);

    } catch (const cl::Error& err) {
        throw std::runtime_error("OpenCL error during binary_cross_entropy_backward: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
    } catch (const std::runtime_error& err) {
         throw std::runtime_error("Error during binary_cross_entropy_backward: " + std::string(err.what()));
    }
}