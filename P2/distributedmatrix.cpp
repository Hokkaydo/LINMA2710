#include "distributedmatrix.hpp"
#include <iostream>
// Constructor taking a Matrix in input and returning a DistributedMatrix
//      Assumes that MPI is already initialized
//      This constructor is called in parallel by all processes
//      Extract the columns that should be handled by this process in localData
DistributedMatrix::DistributedMatrix(const Matrix &matrix, int numProcesses) : globalRows(matrix.numRows()), globalCols(matrix.numCols()), numProcesses(numProcesses), localData(matrix.numRows(), 0) {
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    localCols = globalCols / numProcesses;
    localCols += (rank < globalCols % numProcesses) ? 1 : 0;

    startCol = 0;
    for (int i = 0; i < rank; i++)
    {
        startCol += globalCols / numProcesses;
        if (i < globalCols % numProcesses)
        {
            startCol++;
        }
    }

    this->localData = Matrix(matrix.numRows(), localCols);

    for (int i = 0; i < matrix.numRows(); i++)
    {
        for (int j = 0; j < localCols; j++)
        {
            localData.set(i, j, matrix.get(i, startCol + j));
        }
    }
}

// Copy constructor
DistributedMatrix::DistributedMatrix(const DistributedMatrix &other) : 
    globalRows(other.globalRows),
    globalCols(other.globalCols),
    localCols(other.localCols),
    startCol(other.startCol),
    numProcesses(other.numProcesses),
    rank(other.rank),
    localData(other.localData) {}

DistributedMatrix::DistributedMatrix(const Matrix &localData, int numProcesses, int startCol, int globalCols): 
    globalRows(localData.numRows()),
    globalCols(globalCols),
    localCols(localData.numCols()),
    startCol(startCol),
    numProcesses(numProcesses),
    localData(localData) { 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

int DistributedMatrix::numRows() const {
    return globalRows;
}
int DistributedMatrix::numCols() const {
    return globalCols;
}
double DistributedMatrix::get(int i, int j) const {
    if (j < 0 || j >= globalCols || i < 0 || i >= globalRows)
    {
        throw std::out_of_range("Column index out of range");
    }
    int localJ = localColIndex(j);
    if (localJ == -1)
    {
        throw std::out_of_range("Column index not local");
    }
    return localData.get(i, localJ);
}
void DistributedMatrix::set(int i, int j, double value) {
    if (j < 0 || j >= globalCols || i < 0 || i >= globalRows)
    {
        throw std::out_of_range("Column index out of range");
    }
    int localJ = localColIndex(j);
    if (localJ == -1)
    {
        throw std::out_of_range("Column index not local");
    }
    localData.set(i, localJ, value);
}

// Get the global column index (in the full distributed matrix) from a local index (in the localData matrix)
int DistributedMatrix::globalColIndex(int localColIndex) const {
    return startCol + localColIndex;
}

// Get the local column index (in localData) from a global index (in the full distributed matrix) (or -1 if not local)
int DistributedMatrix::localColIndex(int globalColIndex) const {
    if (globalColIndex < startCol || globalColIndex >= startCol + localCols)
    {
        return -1;
    }
    return globalColIndex - startCol;
}

// Get the process rank that owns a particular global column
int DistributedMatrix::ownerProcess(int globalColIndex) const {
    int process = 0;
    int col = 0;
    while(col < globalColIndex)
    {
        int localCol = globalCols / numProcesses;
        if (process < globalCols % numProcesses)
        {
            localCol++;
        }
        col += localCol;
        if(col <= globalColIndex)
        {
            process++;
        }
    }
    return process;
}

// Get the local data matrix
const Matrix &DistributedMatrix::getLocalData() const {
    return localData;
}

// Apply a function element-wise on the local data, returning the result as a new DistributedMatrix with the same partitioning of the columns across processes
DistributedMatrix DistributedMatrix::apply(const std::function<double(double)> &func) const {
    DistributedMatrix result(*this);
    result.localData = localData.apply(func);
    return result;
    
}

// Apply a binary function to two distributed matrices with the same columns' partitioning across processes (and keeps this partioning for the result)
DistributedMatrix DistributedMatrix::applyBinary(
    const DistributedMatrix &a,
    const DistributedMatrix &b,
    const std::function<double(double, double)> &func) {

    DistributedMatrix result(a);
    for (int i = 0; i < a.globalRows; i++)
    {
        for (int j = 0; j < a.localCols; j++)
        {
            result.localData.set(i, j, func(a.localData.get(i, j), b.localData.get(i, j)));
        }
    }
    return result;
}

// Matrix multiplication: DistributedMatrix * DistributedMatrix^T (returns a regular Matrix)
//      Can assume the same columns' partitioning across processes for the inputs
Matrix DistributedMatrix::multiplyTransposed(const DistributedMatrix &other) const {
    Matrix local = localData * other.localData.transpose();
    double *resultData = (double*) malloc(globalRows * other.globalRows * sizeof(double));
    MPI_Allreduce(local.getData(), resultData, globalRows*other.globalRows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Matrix result(resultData, globalRows, other.globalRows);
    free(resultData);
    return result;
}

// Return the sum of all the elements of the global matrix
double DistributedMatrix::sum() const {
    double sum = 0;
    for (int i = 0; i < localData.numRows(); i++)
    {
        for (int j = 0; j < localData.numCols(); j++)
        {
            sum += localData.get(i, j);
        }
    }
    double globalSum;
    MPI_Allreduce(&sum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalSum;
}

Matrix DistributedMatrix::gather() const {
    Matrix result(globalRows, globalCols);
    int nP;
    MPI_Comm_size(MPI_COMM_WORLD, &nP);

    int lcol[nP]; 
    int disp[nP];

    MPI_Allgather(&localCols, 1, MPI_INT, &lcol, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&startCol, 1, MPI_INT, &disp, 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < globalRows; i++){
        double * send_buffer = localData.getData() + i * localCols;
        double * rec_buffer = result.getData() + i * globalCols;

        MPI_Allgatherv(send_buffer, localCols, MPI_DOUBLE, rec_buffer, lcol, disp, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return result;
}


// Destructor
// DistributedMatrix::~DistributedMatrix() = default;

// Function for matrix * distributedmatrix multiplication
//      Assumes that the left matrix is already on all processes (no need to broadcast it)
//      Returns a DistributedMatrix with the same columns' partioning as the input right DistributedMatrix
DistributedMatrix multiply(const Matrix &left, const DistributedMatrix &right) {
    return DistributedMatrix(left*right.localData, right.numProcesses, right.startCol, right.globalCols);
}

// Synchronize the value of all processes so that after the call of this function,
// the value of the matrix on all process is the value before this call that the matrix had on the process for which `rank == src`.
void sync_matrix(Matrix *matrix, int rank, int src) {
    if (rank == src)
    {
        for (int i = 0; i < matrix->numRows(); i++)
        {
            for (int j = 0; j < matrix->numCols(); j++)
            {
                double value = matrix->get(i, j);
                MPI_Bcast(&value, 1, MPI_DOUBLE, src, MPI_COMM_WORLD);
                matrix->set(i, j, value);
            }
        }
    }
    else
    {
        for (int i = 0; i < matrix->numRows(); i++)
        {
            for (int j = 0; j < matrix->numCols(); j++)
            {
                double value;
                MPI_Bcast(&value, 1, MPI_DOUBLE, src, MPI_COMM_WORLD);
                matrix->set(i, j, value);
            }
        }
    }
}