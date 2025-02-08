#include "matrix.hpp"
#include <stdexcept>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
{
    data.resize(rows * cols);
}

Matrix::Matrix(const Matrix &other): rows(other.rows), cols(other.cols), data(other.data)
{
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
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            data[i * cols + j] = value;
        }
        
    }
}

Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            res.data[i * cols + j] = data[i * cols + j] + other.data[i * cols + j];
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix res = Matrix(rows, cols);
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            res.data[i * cols + j] = data[i * cols + j] - other.data[i * cols + j];
        }
    }
    return res;
}

Matrix Matrix::operator*(double value) const
{
    Matrix res = Matrix(rows, cols);
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            res.data[i * cols + j] = data[i * cols + j] * value;
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if(cols != other.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    Matrix res = Matrix(rows, other.cols);
    for(int i = 0; i < rows; i++)
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

Matrix Matrix::transpose() const
{
    Matrix transposed = Matrix(cols, rows);
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            transposed.data[j * rows + i] = data[i * cols + j];
        }
    }
    return transposed;
}

Matrix Matrix::apply(const std::function<double(double)> &func) const 
{
    Matrix res = Matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            res.data[i * cols + j] = func(data[i * cols + j]);
        }
        
    }
    return res;
}

void Matrix::sub_mul(double scalar, const Matrix &other)
{
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            data[i * cols + j] -= scalar * other.data[i * cols + j];
        }
    }
}