#include "matrix.hpp"
#include <stdexcept>
#include <cstring>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
{
    // 64 byte alignment for AVX2 instructions
    size_t size = rows * cols * sizeof(double);
    data = (double *)malloc(size);

    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols)
{
    // 64 byte alignment for AVX2 instructions
    size_t size = rows * cols * sizeof(double);

    data = (double *)malloc(size);
    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
    memcpy(data, other.data, rows * cols * sizeof(double));
}

Matrix::Matrix(const double *data, int rows, int cols): rows(rows), cols(cols)
{
    size_t size = rows * cols * sizeof(double);

    this->data = (double *)malloc(size);
    if (this->data == nullptr)
    {
        throw std::bad_alloc();
    }
    memcpy(this->data, data, rows * cols * sizeof(double));
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

    for(int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B_T[j * K + k];
            }
            C[i * N + j] = sum;
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