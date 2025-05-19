#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <cstdio>

class Matrix
{
private:
    int rows, cols;
    double* data;

public:
    // Constructors
    Matrix(int rows, int cols);
    Matrix(const Matrix &other);

    // Basic access
    double get(int i, int j) const;
    void set(int i, int j, double value);

    // Getters for the number of rows and columns
    int numRows() const;
    int numCols() const;

    // Fill the entire matrix with a single value.
    void fill(double value);

    // Elementary operations
    Matrix operator+(const Matrix &other) const; // Addition
    Matrix operator-(const Matrix &other) const; // Subtraction
    Matrix operator*(const Matrix &other) const; // Matrix multiplication
    Matrix operator*(double scalar) const;       // Scalar multiplication
    
    Matrix mult_test(const Matrix &other) const;
    // Transpose: returns a new Matrix that is the transpose.
    Matrix transpose() const;

    // Apply a function element–wise.
    Matrix apply(const std::function<double(double)> &func) const;

    // Subtract the product of a scalar and a given matrix: "this = this - scalar * other"
    void sub_mul(double scalar, const Matrix &other);

    // Add copy assignment operator (For students: you can ignore this)
    Matrix &operator=(const Matrix &other)
    {
        if (this != &other)
        {
            rows = other.rows;
            cols = other.cols;
            data = (double*) aligned_alloc(64, rows * cols * sizeof(double));
            if (data == nullptr)
            {
                throw std::bad_alloc();
            }
            memcpy(data, other.data, rows * cols * sizeof(double));
        }
        return *this;
    }
    
    ~Matrix();

};

#endif // MATRIX_H
