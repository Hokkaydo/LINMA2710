#include "matrix.hpp"
#include <stdexcept>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
{
    // 32 byte alignment for AVX2 instructions
    // posix_memalign aligns size on a power of 2 boundary
    size_t size = rows * cols * sizeof(double);
    size = size + 32 - size % 32;
    data = (double *)aligned_alloc(32, size);

    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols)
{
    // 32 byte alignment for AVX2 instructions
    // posix_memalign aligns size on a multiple of 32 bytes
    size_t size = rows * cols * sizeof(double);
    size = size + 32 - size % 32;

    data = (double *)aligned_alloc(32, size);

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

#ifndef __AVX2__
Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = data[i] + other.data[i];
    }
    return res;
}
#else
Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);
    int size = rows * cols;

    for (int i = 0; i < size - (size % 4); i += 4)
    {
        __m256d a = _mm256_load_pd(data + i);
        __m256d b = _mm256_load_pd(other.data + i);
        __m256d c = _mm256_add_pd(a, b);
        _mm256_store_pd(res.data + i, c);
    }
    for (int i = size - (size % 4); i < size; i++)
    {
        res.data[i] = data[i] + other.data[i];
    }

    return res;
}
#endif

#ifndef __AVX2__
Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = data[i] - other.data[i];
    }
    return res;
}
#else
Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);
    int size = rows * cols;
    int aligned_size = size - (size % 4);
    for (int i = 0; i < aligned_size; i += 4)
    {
        __m256d a = _mm256_load_pd(data + i);
        __m256d b = _mm256_load_pd(other.data + i);
        __m256d c = _mm256_sub_pd(a, b);
        _mm256_store_pd(res.data + i, c);
    }

    for (int i = aligned_size; i < size; i++)
    {
        res.data[i] = data[i] - other.data[i];
    }
    return res;
}
#endif

#ifndef __AVX2__
Matrix Matrix::operator*(double value) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = data[i] * value;
    }
    return res;
}
#else
Matrix Matrix::operator*(double value) const
{
    Matrix res = Matrix(rows, cols);
    int size = rows * cols;
    int aligned_size = size - (size % 4);
    const __m256d value_vec = _mm256_set1_pd(value);

    for (int i = 0; i < aligned_size; i += 4)
    {
        __m256d a = _mm256_load_pd(data + i);
        __m256d b = _mm256_mul_pd(a, value_vec);
        _mm256_store_pd(res.data + i, b);
    }

    for (int i = aligned_size; i < size; i++)
    {
        res.data[i] = data[i] * value;
    }
    return res;
}
#endif

#ifndef __AVX2__
Matrix Matrix::operator*(const Matrix &other) const
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
#else

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
    const Matrix other_T = other.transpose();

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < other_T.rows; j++)
        {
            double final_sum = 0;
            int k = 0;

            if (cols - 4 > 0)
            {
                __m256d sum = _mm256_setzero_pd();
                for (; k < cols - 4; k += 4)
                {
                    __m256d a = _mm256_loadu_pd(data + i * cols + k);
                    __m256d b = _mm256_loadu_pd(other_T.data + j * other_T.cols + k);
                    sum = _mm256_fmadd_pd(a, b, sum);
                }
                double temp[4];
                _mm256_store_pd(temp, sum);
                final_sum = temp[0] + temp[1] + temp[2] + temp[3];
            }
            for (; k < cols; k++)
            {
                final_sum += data[i * cols + k] * other_T.data[j * other_T.cols + k];
            }
            res.data[i * other.cols + j] = final_sum;
        }
    }
    return res;
}
#endif

#ifndef __AVX2__
Matrix Matrix::transpose() const
{
    Matrix transposed = Matrix(cols, rows);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            transposed.data[j * rows + i] = data[i * cols + j];
        }
    }

    return transposed;
}
#else

Matrix Matrix::transpose() const
{
    Matrix transposed = Matrix(cols, rows);
    constexpr int block = 4;

    for (int i = 0; i < rows; i += block)
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

#endif

Matrix Matrix::apply(const std::function<double(double)> &func) const
{
    Matrix res = Matrix(rows, cols);

    for (int i = 0; i < rows * cols; i++)
    {
        res.data[i] = func(data[i]);
    }
    return res;
}

#ifndef __AVX2__
void Matrix::sub_mul(double scalar, const Matrix &other)
{
    for (int i = 0; i < rows * cols; i++)
    {
        data[i] -= scalar * other.data[i];
    }
}
#else

void Matrix::sub_mul(double scalar, const Matrix &other)
{
    int size = rows * cols;
    // int aligned_size = size - (size % 4);
    // const __m256d scalar_vec = _mm256_set1_pd(scalar);

    // for (int i = 0; i < aligned_size; i += 4)
    // {
    //     __m256d a = _mm256_load_pd(data + i);
    //     __m256d b = _mm256_load_pd(other.data + i);
    //     __m256d c = _mm256_fnmadd_pd(scalar_vec, b, a);
    //     _mm256_store_pd(data + i, c);
    // }

    for (int i = 0; i < size; i++)
    {
        data[i] -= scalar * other.data[i];
    }
}
#endif

Matrix::~Matrix()
{
    free(data);
}