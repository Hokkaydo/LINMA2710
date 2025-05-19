#include "matrix.hpp"
#include <stdexcept>
#include <cstring>
#include <immintrin.h>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
{
    // 64 byte alignment for AVX2 instructions
    size_t size = rows * cols * sizeof(double);
    size = size + 64 - size % 64;
    data = (double *)aligned_alloc(64, size);

    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols)
{
    // 64 byte alignment for AVX2 instructions
    size_t size = rows * cols * sizeof(double);
    size = size + 64 - size % 64;

    data = (double *)aligned_alloc(64, size);
    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
    memcpy(data, other.data, rows * cols * sizeof(double));
}

double Matrix::get(int i, int j) const
{
    return data[i * cols + j];
}

void Matrix::set(int i, int j, double value)
{
    data[i * cols + j] = value;
}

int Matrix::numRows() const
{
    return rows;
}

int Matrix::numCols() const
{
    return cols;
}

void Matrix::fill(double value)
{
    std::fill(data, data + rows * cols, value);
}

Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = data[i] + other.data[i];
    }
    return res;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = data[i] - other.data[i];
    }
    return res;
}

Matrix Matrix::operator*(double value) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = data[i] * value;
    }
    return res;
}

Matrix Matrix::mult_test(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    Matrix res = Matrix(rows, other.cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < other.cols; j++)
        {
            double sum = 0;
            for (int k = 0; k < cols; k++)
            {
                sum += data[i * cols + k] * other.data[k * other.cols + j];
            }
            res.data[i * other.cols + j] = sum;
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    Matrix res = Matrix(rows, other.cols);
    res.fill(0);
    const double* A = data;
    const Matrix B_transposed = other.transpose();
    const double* B_T = B_transposed.data;

    double* C = res.data;

    int M = rows;
    int N = other.cols;
    int K = cols;

    constexpr int BLOCK_SIZE = 64; // 64B cache line optimization

    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {

                for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ++ii) {
                    int jj;
                    for (jj = j; jj + 4 <= std::min(j + BLOCK_SIZE, N); jj += 4) {

                        int kk;
                        int k_max = std::min(k + BLOCK_SIZE, K);
                        
                        // Separate accumulators for each output column
                        __m256d acc0 = _mm256_setzero_pd();
                        __m256d acc1 = _mm256_setzero_pd();
                        __m256d acc2 = _mm256_setzero_pd();
                        __m256d acc3 = _mm256_setzero_pd();
                    
                        for (kk = k; kk + 4 <= k_max; kk += 4) {
                            __m256d a_vec = _mm256_loadu_pd(&A[ii * K + kk]);
                    
                            // Load columns of B_T
                            __m256d b_vec0 = _mm256_loadu_pd(&B_T[jj * K + kk]);
                            __m256d b_vec1 = _mm256_loadu_pd(&B_T[(jj + 1) * K + kk]);
                            __m256d b_vec2 = _mm256_loadu_pd(&B_T[(jj + 2) * K + kk]);
                            __m256d b_vec3 = _mm256_loadu_pd(&B_T[(jj + 3) * K + kk]);
                    
                            // Multiply and accumulate for each column
                            acc0 = _mm256_fmadd_pd(a_vec, b_vec0, acc0);
                            acc1 = _mm256_fmadd_pd(a_vec, b_vec1, acc1);
                            acc2 = _mm256_fmadd_pd(a_vec, b_vec2, acc2);
                            acc3 = _mm256_fmadd_pd(a_vec, b_vec3, acc3);
                        }
                    
                        // Store 0+1 on 0, 1 and 2+3 on 2,3
                        acc0 = _mm256_hadd_pd(acc0, acc0);
                        acc1 = _mm256_hadd_pd(acc1, acc1);
                        acc2 = _mm256_hadd_pd(acc2, acc2);
                        acc3 = _mm256_hadd_pd(acc3, acc3);

                        // Reduce 0 + 2
                        double temp[4];
                        temp[0] = ((double*)&acc0)[0] + ((double*)&acc0)[2]; 
                        temp[1] = ((double*)&acc1)[0] + ((double*)&acc1)[2];
                        temp[2] = ((double*)&acc2)[0] + ((double*)&acc2)[2];
                        temp[3] = ((double*)&acc3)[0] + ((double*)&acc3)[2];
                    
                        // Handle remaining elements
                        for (; kk < k_max; ++kk) {
                            temp[0] += A[ii * K + kk] * B_T[(jj + 0) * K + kk];
                            temp[1] += A[ii * K + kk] * B_T[(jj + 1) * K + kk];
                            temp[2] += A[ii * K + kk] * B_T[(jj + 2) * K + kk];
                            temp[3] += A[ii * K + kk] * B_T[(jj + 3) * K + kk];
                        }
                    
                        C[ii * N + jj + 0] += temp[0];
                        C[ii * N + jj + 1] += temp[1];
                        C[ii * N + jj + 2] += temp[2];
                        C[ii * N + jj + 3] += temp[3];
                    }
                    
                    // Handle remaining columns (if N is not a multiple of 4)
                    for (; jj < std::min(j + BLOCK_SIZE, N); jj++) {
                        double sum = C[ii * N + jj]; 
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, K); ++kk) {
                            sum += A[ii * K + kk] * B_T[jj * K + kk];
                        }
                        C[ii * N + jj] = sum;
                    }   
                }
            }
        }
    }
    return res;
}

Matrix Matrix::transpose() const
{
    Matrix transposed = Matrix(cols, rows);
    constexpr int block = 64;

    for (int i = 0; i < rows; i += 4)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int b = 0; b < block && i + b < rows; b++)
            {
                transposed.data[j * rows + i + b] = data[(i + b) * cols + j];
            }
        }
    }
    return transposed;
}

Matrix Matrix::apply(const std::function<double(double)> &func) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = func(data[i]);
    }
    return res;
}

void Matrix::sub_mul(double scalar, const Matrix &other)
{
    int size = rows * cols;
    for (int i = 0; i < size; i++)
    {
        data[i] -= scalar * other.data[i];
    }
}

Matrix::~Matrix()
{
    free(data);
}